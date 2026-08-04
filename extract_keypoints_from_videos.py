"""
Extract MediaPipe keypoints from WLASL videos in dataset/videos/
Maps video_id -> gloss -> class_idx using WLASL_v0.3.json + wlasl_class_list.txt
Saves .npy files to dataset/keypoints/
Uses MediaPipe legacy solutions API (v0.10.x)
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

# MediaPipe legacy Holistic
mp_holistic = mp.solutions.holistic

def extract_keypoints(video_path):
    """Extract (MAX_FRAMES, 543, 3) keypoints from video"""
    cap = cv2.VideoCapture(str(video_path))
    sequence = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        while cap.isOpened() and len(sequence) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            frame_landmarks = np.full((543, 3), np.nan, dtype=np.float32)
            
            # Face (468)
            if results.face_landmarks:
                for idx, lm in enumerate(results.face_landmarks.landmark[:468]):
                    frame_landmarks[idx] = [lm.x, lm.y, lm.z]
            
            # Left Hand (21) - indices 468-488
            if results.left_hand_landmarks:
                for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_landmarks[468 + idx] = [lm.x, lm.y, lm.z]
            
            # Pose (33) - indices 489-521
            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    frame_landmarks[489 + idx] = [lm.x, lm.y, lm.z]
            
            # Right Hand (21) - indices 522-542
            if results.right_hand_landmarks:
                for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_landmarks[522 + idx] = [lm.x, lm.y, lm.z]
            
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
        keypoints = extract_keypoints(vid_path)
        if keypoints is not None:
            np.save(out_path, keypoints)
            processed += 1
        else:
            errors += 1
    except Exception as e:
        print(f"Error processing {video_id}: {e}")
        errors += 1

print(f"\nDone: {processed} processed, {skipped} skipped (no annotation/class), {errors} errors")