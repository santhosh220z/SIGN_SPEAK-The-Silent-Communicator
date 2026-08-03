"""
WLASL Dataset - Stage 1 Optimized Parallel Keypoint Extractor
Extracts 543 MediaPipe landmarks (x, y, z) for WLASL videos and saves to numpy format.
Optimized with thread-local landmarker instance reuse for ultra-fast processing.
"""

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import cv2
import time
import threading
import numpy as np
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from mediapipe.tasks.python import vision, BaseOptions

BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
KEYPOINTS_DIR = DATASET_DIR / "keypoints"
JSON_PATH = DATASET_DIR / "WLASL_v0.3.json"

FACE_TASK = BASE_DIR / "face_landmarker.task"
POSE_TASK = BASE_DIR / "pose_landmarker.task"
HAND_TASK = BASE_DIR / "hand_landmarker.task"

KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_FRAMES = 64

# Thread local storage for landmarker models
thread_local = threading.local()

def get_thread_landmarkers():
    if not hasattr(thread_local, "landmarkers"):
        f_opt = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_TASK)),
            running_mode=vision.RunningMode.IMAGE
        )
        p_opt = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_TASK)),
            running_mode=vision.RunningMode.IMAGE
        )
        h_opt = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_TASK)),
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE
        )

        face_lm = vision.FaceLandmarker.create_from_options(f_opt)
        pose_lm = vision.PoseLandmarker.create_from_options(p_opt)
        hand_lm = vision.HandLandmarker.create_from_options(h_opt)
        thread_local.landmarkers = (face_lm, pose_lm, hand_lm)
    return thread_local.landmarkers


def process_single_video(video_id, bbox=None, frame_start=1, frame_end=-1):
    output_npy = KEYPOINTS_DIR / f"{video_id}.npy"
    if output_npy.exists():
        return video_id, True, "Already processed"
        
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        return video_id, False, "Video file missing"
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_id, False, "Corrupt video file"
        
    sequence = []
    current_frame = 1
    
    try:
        face_lm, pose_lm, hand_lm = get_thread_landmarkers()
    except Exception as e:
        cap.release()
        return video_id, False, f"Task Init Error: {e}"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_start > 1 and current_frame < frame_start:
            current_frame += 1
            continue
            
        if frame_end > 0 and current_frame > frame_end:
            break
            
        if bbox and len(bbox) == 4:
            ymin, xmin, ymax, xmax = bbox
            h, w, _ = frame.shape
            ymin, xmin = max(0, int(ymin)), max(0, int(xmin))
            ymax, xmax = min(h, int(ymax)), min(w, int(xmax))
            if ymax > ymin and xmax > xmin:
                frame = frame[ymin:ymax, xmin:xmax]
                
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        face_res = face_lm.detect(mp_image)
        pose_res = pose_lm.detect(mp_image)
        hand_res = hand_lm.detect(mp_image)

        frame_lm = np.full((543, 3), np.nan, dtype=np.float32)

        # 1. Face (0..467)
        if face_res and face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            for idx, lm in enumerate(face_res.face_landmarks[0][:468]):
                frame_lm[idx] = [lm.x, lm.y, lm.z]

        # 2. Hands: Left (468..488) & Right (522..542)
        if hand_res and hand_res.hand_landmarks and hand_res.handedness:
            for landmarks, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = handedness[0].category_name
                if label == "Left":
                    for idx, lm in enumerate(landmarks):
                        frame_lm[468 + idx] = [lm.x, lm.y, lm.z]
                elif label == "Right":
                    for idx, lm in enumerate(landmarks):
                        frame_lm[522 + idx] = [lm.x, lm.y, lm.z]

        # 3. Pose (489..521)
        if pose_res and pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
            for idx, lm in enumerate(pose_res.pose_landmarks[0][:33]):
                frame_lm[489 + idx] = [lm.x, lm.y, lm.z]
                
        sequence.append(frame_lm)
        current_frame += 1
        
        if len(sequence) >= MAX_FRAMES:
            break
            
    cap.release()
    
    if len(sequence) == 0:
        arr = np.full((MAX_FRAMES, 543, 3), np.nan, dtype=np.float32)
    else:
        arr = np.array(sequence, dtype=np.float32)
        if len(arr) < MAX_FRAMES:
            pad = np.full((MAX_FRAMES - len(arr), 543, 3), np.nan, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:MAX_FRAMES]
            
    np.save(output_npy, arr)
    return video_id, True, f"Success ({len(sequence)} frames)"


def run_extraction(max_workers=4):
    print(f"--- Stage 1: Fast Parallel Keypoint Extraction ({max_workers} threads) ---", flush=True)
    if not JSON_PATH.exists():
        print(f"Error: {JSON_PATH} not found!", flush=True)
        return
        
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    tasks = []
    for entry in data:
        gloss = entry['gloss']
        for inst in entry['instances']:
            video_id = inst['video_id']
            bbox = inst.get('bbox')
            f_start = inst.get('frame_start', 1)
            f_end = inst.get('frame_end', -1)
            output_npy = KEYPOINTS_DIR / f"{video_id}.npy"
            if not output_npy.exists():
                tasks.append((video_id, bbox, f_start, f_end))
            
    already_done = len(list(KEYPOINTS_DIR.glob("*.npy")))
    print(f"[*] Total videos remaining to extract: {len(tasks)} (Already extracted: {already_done})", flush=True)
    
    if len(tasks) == 0:
        print("[+] All keypoints already extracted!", flush=True)
        return

    start_time = time.time()
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, v_id, bbox, f_start, f_end): v_id for v_id, bbox, f_start, f_end in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            v_id, status, msg = future.result()
            if status:
                success_count += 1
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                fps = i / elapsed if elapsed > 0 else 0
                rem_sec = (len(tasks) - i) / fps if fps > 0 else 0
                print(f"[{i}/{len(tasks)}] Extracted {v_id} | Speed: {fps:.1f} vids/sec | Rem Time: {rem_sec/60:.1f} min", flush=True)
                
    print(f"\n[Stage 1 Complete] Keypoints extracted for {success_count} / {len(tasks)} videos in {time.time() - start_time:.1f}s.", flush=True)

if __name__ == "__main__":
    run_extraction(max_workers=4)
