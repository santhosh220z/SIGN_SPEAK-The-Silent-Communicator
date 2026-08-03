"""
SIGN SPEAK - YOLO11s ASL Fine-Tuning Pipeline
Supports both static images (.jpg, .png) and video files (.mp4, .avi, .mov).
"""

import os
import sys
import glob
import cv2
import yaml
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# Global Configurations
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
RAW_VIDEOS_DIR = DATASET_DIR / "raw_videos"
RAW_IMAGES_DIR = DATASET_DIR / "raw_images"

PROCESSED_IMAGES_DIR = DATASET_DIR / "images"
PROCESSED_LABELS_DIR = DATASET_DIR / "labels"

MODEL_SAVE_DIR = BASE_DIR / "asl_yolo11s_output"


def extract_frames_from_videos(video_dir, output_dir, frame_stride=5):
    """
    Extracts frames from video files (.mp4, .avi, .mov, .mkv) at a regular stride.
    
    Args:
        video_dir (Path): Folder containing video files.
        output_dir (Path): Destination folder for extracted frames.
        frame_stride (int): Save 1 frame every N frames (e.g., stride 5 at 30fps = 6 fps saved).
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(ext)))
        video_files.extend(list(video_dir.glob(ext.upper())))
        
    if not video_files:
        print(f"[Info] No video files found in {video_dir}")
        return 0

    print(f"[Processing] Found {len(video_files)} video(s). Extracting frames (stride={frame_stride})...")
    total_saved = 0

    for vid_path in video_files:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[Warning] Could not open video: {vid_path.name}")
            continue

        video_name = vid_path.stem
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                frame_filename = f"{video_name}_f{frame_idx:06d}.jpg"
                save_path = output_dir / frame_filename
                cv2.imwrite(str(save_path), frame)
                saved_count += 1
                total_saved += 1

            frame_idx += 1

        cap.release()
        print(f"  └─ Extracted {saved_count} frames from {vid_path.name}")

    print(f"[Success] Extracted total {total_saved} frames to {output_dir}\n")
    return total_saved


def prepare_yolo_dataset(dataset_dir, val_split=0.2, classes_dict=None):
    """
    Organizes images and YOLO labels into train/ and val/ splits and generates data.yaml.
    """
    dataset_dir = Path(dataset_dir)
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"
    val_img_dir = dataset_dir / "val" / "images"
    val_lbl_dir = dataset_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect images
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not all_images:
        print(f"[Warning] No images found in {images_dir}. Please place images or frames inside.")
        return None

    random.shuffle(all_images)
    split_index = int(len(all_images) * (1.0 - val_split))
    train_files = all_images[:split_index]
    val_files = all_images[split_index:]

    print(f"[Dataset Split] Total: {len(all_images)} | Train: {len(train_files)} | Val: {len(val_files)}")

    def copy_pair(files, dst_img_dir, dst_lbl_dir):
        for img_file in files:
            shutil.copy(img_file, dst_img_dir / img_file.name)
            lbl_file = labels_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy(lbl_file, dst_lbl_dir / lbl_file.name)

    copy_pair(train_files, train_img_dir, train_lbl_dir)
    copy_pair(val_files, val_img_dir, val_lbl_dir)

    # Generate data.yaml configuration
    if classes_dict is None:
        classes_dict = {0: "asl_sign"} # Default single class or custom dict

    yaml_data = {
        'path': str(dataset_dir.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'names': classes_dict
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"[Success] Generated dataset configuration at: {yaml_path}\n")
    return yaml_path


def train_yolo11s(data_yaml_path, epochs=50, imgsz=640, batch_size=16):
    """
    Fine-tunes YOLO11s model on the prepared ASL dataset.
    """
    print("=" * 60)
    print("         STARTING YOLO11s ASL FINE-TUNING")
    print("=" * 60)

    # 1. Load Pretrained YOLO11 Small model
    model = YOLO("yolo11s.pt")

    # 2. Train / Fine-tune
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=0.01,
        lrf=0.01,
        fliplr=0.0,              # Disable horizontal flip if sign orientation matters
        mosaic=1.0,              # Mosaic augmentation for hand gesture robustness
        project=str(MODEL_SAVE_DIR),
        name="yolo11s_asl_run",
        exist_ok=True
    )

    print("\n[Validation] Evaluating fine-tuned YOLO11s model...")
    metrics = model.val()

    # 3. Export for Web / Real-time Deployment
    print("\n[Exporting] Exporting model to ONNX for real-time web deployment...")
    onnx_path = model.export(format="onnx")
    print(f"[Success] Fine-tuned YOLO11s ONNX model saved to: {onnx_path}")

    return model


if __name__ == "__main__":
    print("--- SIGN SPEAK: YOLO11s ASL Model Pipeline ---")
    
    # Example usage:
    # 1. Extract frames from raw videos (if raw_videos directory exists)
    if RAW_VIDEOS_DIR.exists():
        extract_frames_from_videos(RAW_VIDEOS_DIR, PROCESSED_IMAGES_DIR, frame_stride=5)

    # 2. Check if dataset configuration exists or prepare it
    yaml_file = DATASET_DIR / "data.yaml"
    if not yaml_file.exists() and (PROCESSED_IMAGES_DIR.exists()):
        yaml_file = prepare_yolo_dataset(DATASET_DIR)

    # 3. Launch training if data.yaml is ready
    if yaml_file and yaml_file.exists():
        train_yolo11s(yaml_file, epochs=50, imgsz=640, batch_size=16)
    else:
        print("\n[Setup Instructions]")
        print("Place your raw videos in: dataset/raw_videos/")
        print("Place your labeled images/annotations in: dataset/images/ and dataset/labels/")
        print("Then run this script to process and train YOLO11s automatically.")
