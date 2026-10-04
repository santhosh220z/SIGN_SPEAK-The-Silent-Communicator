# SIGN SPEAK - Real-Time Sign Language Translation

Real-time Transformer-based sign language translation using MediaPipe landmarks for hand, face, and pose detection.

## Requirements

- Python 3.10+
- Webcam
- MediaPipe task bundles (`face_landmarker.task`, `hand_landmarker.task`, `pose_landmarker.task`) in the repo root
- YOLO hand-detector weights (`yolo11s.pt`) for the inference pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Model

Trained PyTorch checkpoints live in `MODEL/checkpoints/<run_name>/`:

- `best_model.pt` - best validation checkpoint
- `latest_model.pt` - most recent checkpoint

Each checkpoint stores `{'model': state_dict, 'args': {...}, 'classes': [...]}`.
Load it with `torch.load(...)` + `SignTransformer` (see `app_streamlit.py` / `inference_pipeline.py`).

## Usage

```bash
streamlit run app_streamlit.py
```

## Controls

- **▶️ Start Webcam** - Begin real-time recognition
- **⏹️ Stop Webcam** - Stop the webcam feed
- **🗑️ Clear History** - Clear detected gesture history
- **Confidence Threshold** - Adjust detection sensitivity (0.1-0.9)

## Architecture

- **MediaPipe Tasks**: Face, Pose, and Hand landmark detection (543 landmarks)
- **Preprocessing**: Nose-tip reference normalization, velocity/acceleration features (6 channels per landmark)
- **Model**: Two-stream 1D-CNN + Transformer (192 dim, 4 heads, 2 blocks), motion-gated attention
- **Output**: 100-class classification (NSLT-100) with softmax probabilities
- **TTS**: pyttsx3 for spoken translation

## Project Structure

```
SIGN_SPEAK-The-Silent-Communicator/
├── app_streamlit.py            # Main Streamlit application
├── model.py                    # SignTransformer (two-stream + motion gate)
├── dual_head_model.py          # Static + action dual-head variant
├── train.py                    # Training loop (CE + triplet + contrastive)
├── train_dataset.py            # Keypoint dataset / dataloaders
├── inference_pipeline.py       # YOLO + keypoints + dual-head runtime
├── yolo_hand_detector.py       # YOLO hand region detector
├── collect_static_gestures.py  # Static gesture data collection
├── extract_keypoints_*.py      # Keypoint extraction variants
├── requirements.txt
├── MODEL/checkpoints/          # Trained checkpoints
├── dataset/                    # WLASL/NSLT metadata + extracted keypoints
└── *.task                      # MediaPipe model bundles
```

## Development

```bash
pip install pytest
python -m pytest tests/
```
