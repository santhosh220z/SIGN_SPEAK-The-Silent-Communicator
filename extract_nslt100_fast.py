"""
Extract MediaPipe keypoints for NSLT-100 training set only - FAST VERSION
Uses only Hand + Pose landmarkers (skips slow face landmarker)
Processes only videos needed for NSLT-100 training (~2000 videos)
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
NSLT_100_JSON = Path("dataset/nslt_100.json")
CLASS_LIST = Path("dataset/wlasl_class_list.txt")
MAX_FRAMES = 64

# Model paths
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker.task"

KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Load class mapping: gloss -> class_idx
class_map = {}
with open(CLASS_LIST) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            class_map[parts[1]] = int(parts[0])

# Load NSLT-100 annotations - format: {video_id: {'subset': 'train', 'action': [class_idx, ...]}}
video_to_action = {}
video_to_split = {}
with open(NSLT_100_JSON) as f:
    nslt = json.load(f)
    for video_id, info in nslt.items():
        video_to_action[video_id] = info['action'][0]  # First action index is the class
        video_to_split[video_id] = info['subset']

print(f"Loaded {len(video_to_action)} NSLT-100 video annotations")
print(f"Classes in class_list: {len(class_map)}")

# MediaPipe Tasks API - IMAGE MODE (Hand + Pose only)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def create_landmarkers():
    """Create MediaPipe landmarkers - Hand + Pose only (fast)."""
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    
    return hand_landmarker, pose_landmarker


def extract_keypoints(video_path, hand_landmarker, pose_landmarker):
    """Extract (MAX_FRAMES, 543, 3) keypoints - Hand + Pose only (no face)."""
    cap = cv2.VideoCapture(str(video_path))
    sequence = []
    
    while cap.isOpened() and len(sequence) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        hand_results = hand_landmarker.detect(mp_image)
        pose_results = pose_landmarker.detect(mp_image)
        
        frame_landmarks = np.full((543, 3), np.nan, dtype=np.float32)
        
        # Left Hand (21) - indices 468-488
        # Right Hand (21) - indices 522-542
        if hand_results.hand_landmarks:
            for hand_idx, landmarks in enumerate(hand_results.hand_landmarks):
                if hand_idx < len(hand_results.handedness):
                    handedness = hand_results.handedness[hand_idx][0].category_name
                    if handedness == "Left":
                        base_idx = 468
                    else:
                        base_idx = 522
                    for idx, lm in enumerate(landmarks[:21]):
                        frame_landmarks[base_idx + idx] = [lm.x, lm.y, lm.z]
        
        # Pose (33) - indices 489-521
        if pose_results.pose_landmarks:
            for idx, lm in enumerate(pose_results.pose_landmarks[0][:33]):
                frame_landmarks[489 + idx] = [lm.x, lm.y, lm.z]
        
        sequence.append(frame_landmarks)
    
    cap.release()
    
    if len(sequence) == 0:
        return None
    
    sequence = np.array(sequence, dtype=np.float32)
    
    if len(sequence) < MAX_FRAMES:
        pad = np.full((MAX_FRAMES - len(sequence), 543, 3), np.nan, dtype=np.float32)
        sequence = np.concatenate([sequence, pad], axis=0)
    else:
        sequence = sequence[:MAX_FRAMES]
    
    return sequence


# Get only NSLT-100 video files that exist
video_files = list(VIDEO_DIR.glob("*.mp4"))
video_ids_in_nslt = set(video_to_action.keys())
nslt_video_files = [v for v in video_files if v.stem in video_ids_in_nslt]

print(f"Total videos: {len(video_files)}, NSLT-100 videos to process: {len(nslt_video_files)}")

# Create landmarkers
print("Initializing MediaPipe landmarkers (Hand + Pose only)...")
hand_landmarker, pose_landmarker = create_landmarkers()

processed = 0
skipped = 0
errors = 0

for vid_path in tqdm(nslt_video_files, desc="Extracting NSLT-100 keypoints"):
    video_id = vid_path.stem
    
    if video_id not in video_to_action:
        skipped += 1
        continue
    
    class_idx = video_to_action[video_id]
    # Verify class exists in class_map (by checking if any gloss maps to this idx)
    # For NSLT, action is already the class index
    split = video_to_split.get(video_id, 'train')
    
    out_name = f"{video_id}_{class_idx}_{split}.npy"
    out_path = KEYPOINT_DIR / out_name
    
    if out_path.exists():
        processed += 1
        continue
    
    try:
        keypoints = extract_keypoints(vid_path, hand_landmarker, pose_landmarker)
        if keypoints is not None:
            np.save(out_path, keypoints)
            processed += 1
        else:
            errors += 1
    except Exception as e:
        print(f"Error processing {video_id}: {e}")
        errors += 1

print(f"\nDone: {processed} processed, {skipped} skipped, {errors} errors")

hand_landmarker.close()
pose_landmarker.close()