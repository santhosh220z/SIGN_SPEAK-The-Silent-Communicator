"""Extract MediaPipe keypoints from ASL static gesture images.
Walks dataset/asl_dataset/<class>/*.jpeg, saves dataset/keypoints_asl/{class}_{idx}.npy
shaped (T, 543, 3) (T=8 repeated frames of one-shot keypoints), and writes
dataset/asl_labels.json in the NSLT schema {vid: {action:[idx], subset:...}}.
"""
import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

ASL_DIR = Path("dataset/asl_dataset")
KEYPOINT_DIR = Path("dataset/keypoints_asl")
LABELS_JSON = Path("dataset/asl_labels.json")
REPEATS = 8

KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_landmarkers():
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    return hand_landmarker, pose_landmarker, face_landmarker


def image_to_frame_lms(image_bgr, hand_landmarker, pose_landmarker, face_landmarker):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    hand_results = hand_landmarker.detect(mp_image)
    pose_results = pose_landmarker.detect(mp_image)
    face_results = face_landmarker.detect(mp_image)

    frame_lms = np.full((543, 3), np.nan, dtype=np.float32)
    if face_results.face_landmarks:
        for idx, lm in enumerate(face_results.face_landmarks[0][:468]):
            frame_lms[idx] = [lm.x, lm.y, lm.z]
    if hand_results.hand_landmarks:
        for hand_idx, landmarks in enumerate(hand_results.hand_landmarks):
            handedness = hand_results.handedness[hand_idx][0].category_name if hand_results.handedness else "Right"
            base_idx = 468 if handedness == "Left" else 522
            for idx, lm in enumerate(landmarks[:21]):
                frame_lms[base_idx + idx] = [lm.x, lm.y, lm.z]
    if pose_results.pose_landmarks:
        for idx, lm in enumerate(pose_results.pose_landmarks[0][:33]):
            frame_lms[489 + idx] = [lm.x, lm.y, lm.z]
    return frame_lms


def main():
    classes = sorted([d.name for d in ASL_DIR.iterdir() if d.is_dir()])
    print(f"{len(classes)} classes: {classes[:5]}...")
    hand_landmarker, pose_landmarker, face_landmarker = create_landmarkers()

    annos = {}
    for class_idx, cls in enumerate(classes):
        images = sorted((ASL_DIR / cls).glob("*.jpeg")) + sorted((ASL_DIR / cls).glob("*.jpg")) + sorted((ASL_DIR / cls).glob("*.png"))
        n = len(images)
        for i, img_path in enumerate(tqdm(images, desc=cls)):
            vid = f"asl_{cls}_{i}"
            # deterministic per-class 80/10/10 split
            r = i / max(n, 1)
            subset = "train" if r < 0.8 else ("val" if r < 0.9 else "test")
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            lms = image_to_frame_lms(frame, hand_landmarker, pose_landmarker, face_landmarker)
            if np.isnan(lms[468:489]).all() and np.isnan(lms[522:543]).all():
                continue  # no hands detected
            seq = np.repeat(lms[None, :, :], REPEATS, axis=0)
            np.save(KEYPOINT_DIR / f"{vid}.npy", seq)
            annos[vid] = {"action": [class_idx], "subset": subset}

    with open(LABELS_JSON, "w") as f:
        json.dump(annos, f)
    print(f"Saved {len(annos)} samples -> {KEYPOINT_DIR}, labels -> {LABELS_JSON}")
    hand_landmarker.close()
    pose_landmarker.close()
    face_landmarker.close()


if __name__ == "__main__":
    main()
