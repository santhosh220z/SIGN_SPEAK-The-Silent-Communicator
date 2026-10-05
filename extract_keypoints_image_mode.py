"""
Extract MediaPipe keypoints from WLASL videos - IMAGE MODE (no timestamps)
Uses MediaPipe Tasks API with IMAGE running mode (no timestamp issues)
Maps video_id -> gloss -> class_idx using WLASL_v0.3.json + wlasl_class_list.txt
Saves .npy files to dataset/keypoints/ with format: {video_id}_{class_idx}_{split}.npy
"""
import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Config
VIDEO_DIR = Path("dataset/videos")
KEYPOINT_DIR = Path("dataset/keypoints")
WLASL_JSON = Path("dataset/WLASL_v0.3.json")
CLASS_LIST = Path("dataset/wlasl_class_list.txt")
MAX_FRAMES = 64

# Model paths
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker.task"
FACE_MODEL = "face_landmarker.task"

KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Load class mapping: gloss -> class_idx
class_map = {}
with open(CLASS_LIST) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[parts[1]] = int(parts[0])

# Load WLASL annotations: video_id -> gloss
video_to_gloss = {}
video_to_split = {}
with open(WLASL_JSON) as f:
    wlasl = json.load(f)
    for entry in wlasl:
        gloss = entry['gloss']
        for inst in entry['instances']:
            video_to_gloss[inst['video_id']] = gloss
            video_to_split[inst['video_id']] = inst['split']

print(f"Loaded {len(video_to_gloss)} video annotations")
print(f"Classes in class_list: {len(class_map)}")

# MediaPipe Tasks API - IMAGE MODE
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def create_landmarkers():
    """Create MediaPipe landmarkers in IMAGE mode (no timestamps needed)."""
    # Hand landmarker
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.IMAGE,  # IMAGE MODE - no timestamps
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    
    # Pose landmarker
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.IMAGE,  # IMAGE MODE
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    
    # Face landmarker
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.IMAGE,  # IMAGE MODE
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    
    return hand_landmarker, pose_landmarker, face_landmarker


def extract_keypoints(video_path, hand_landmarker, pose_landmarker, face_landmarker):
    """Extract (MAX_FRAMES, 543, 3) keypoints from video using IMAGE mode."""
    cap = cv2.VideoCapture(str(video_path))
    sequence = []
    
    while cap.isOpened() and len(sequence) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to MediaPipe Image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect landmarks - NO TIMESTAMPS NEEDED IN IMAGE MODE
        hand_results = hand_landmarker.detect(mp_image)
        pose_results = pose_landmarker.detect(mp_image)
        face_results = face_landmarker.detect(mp_image)
        
        frame_landmarks = np.full((543, 3), np.nan, dtype=np.float32)
        
        # Face (468 landmarks) - indices 0-467
        if face_results.face_landmarks:
            for idx, lm in enumerate(face_results.face_landmarks[0][:468]):
                frame_landmarks[idx] = [lm.x, lm.y, lm.z]
        
        # Left Hand (21 landmarks) - indices 468-488
        # Right Hand (21 landmarks) - indices 522-542
        if hand_results.hand_landmarks:
            for hand_idx, landmarks in enumerate(hand_results.hand_landmarks):
                if hand_idx < len(hand_results.handedness):
                    handedness = hand_results.handedness[hand_idx][0].category_name
                    if handedness == "Left":
                        base_idx = 468
                    else:
                        base_idx = 522  # Right hand
                    for idx, lm in enumerate(landmarks[:21]):
                        frame_landmarks[base_idx + idx] = [lm.x, lm.y, lm.z]
        
        # Pose (33 landmarks) - indices 489-521
        if pose_results.pose_landmarks:
            for idx, lm in enumerate(pose_results.pose_landmarks[0][:33]):
                frame_landmarks[489 + idx] = [lm.x, lm.y, lm.z]
        
        sequence.append(frame_landmarks)
    
    cap.release()
    
    if len(sequence) == 0:
        return None
    
    sequence = np.array(sequence, dtype=np.float32)
    
    # Pad or truncate to MAX_FRAMES
    if len(sequence) < MAX_FRAMES:
        pad = np.full((MAX_FRAMES - len(sequence), 543, 3), np.nan, dtype=np.float32)
        sequence = np.concatenate([sequence, pad], axis=0)
    else:
        sequence = sequence[:MAX_FRAMES]
    
    return sequence


# Process videos
video_files = list(VIDEO_DIR.glob("*.mp4"))
print(f"Found {len(video_files)} video files")

# Create landmarkers
print("Initializing MediaPipe landmarkers (IMAGE mode)...")
hand_landmarker, pose_landmarker, face_landmarker = create_landmarkers()

processed = 0
skipped = 0
errors = 0

for vid_path in tqdm(video_files, desc="Extracting keypoints"):
    video_id = vid_path.stem
    
    # Check if we have annotation for this video
    if video_id not in video_to_gloss:
        skipped += 1
        continue
    
    gloss = video_to_gloss[video_id]
    if gloss not in class_map:
        skipped += 1
        continue
    
    class_idx = class_map[gloss]
    split = video_to_split.get(video_id, 'train')
    
    # Output filename: {video_id}_{class_idx}_{split}.npy
    out_name = f"{video_id}_{class_idx}_{split}.npy"
    out_path = KEYPOINT_DIR / out_name
    
    if out_path.exists():
        processed += 1
        continue
    
    try:
        keypoints = extract_keypoints(vid_path, hand_landmarker, pose_landmarker, face_landmarker)
        if keypoints is not None:
            np.save(out_path, keypoints)
            processed += 1
        else:
            errors += 1
    except Exception as e:
        print(f"Error processing {video_id}: {e}")
        errors += 1

print(f"\nDone: {processed} processed, {skipped} skipped (no annotation/class), {errors} errors")

# Cleanup
hand_landmarker.close()
pose_landmarker.close()
face_landmarker.close()