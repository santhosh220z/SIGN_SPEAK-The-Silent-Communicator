# SIGN SPEAK - Real-Time Sign Language Translation

Real-time sign language recognition using MediaPipe landmarks for hand, face, and pose detection with a PyTorch Transformer.

## Requirements

- Python 3.10+
- Webcam
- MediaPipe task bundles (`face_landmarker.task`, `hand_landmarker.task`, `pose_landmarker.task`) in the repo root

## Installation

```bash
pip install -r requirements.txt
```

## Model

- **Architecture**: Two-stream (appearance + motion) 1D-CNN + Transformer
  (192 dim, 4 heads, 2 blocks) with motion-gated attention
- **Current dataset**: ASL static gestures (36 classes: `0-9`, `a-z`)
- **Input**: up to 64 frames × 543 landmarks × 3 (xyz)
- **Preprocessing**: nose-tip reference normalization, velocity/acceleration features (6 channels/landmark)
- **Checkpoint**: `MODEL/checkpoints/asl_dim192/best_model.pt` (Val ~93.6%, Test ~89.6%)

## Training

```powershell
python extract_asl_keypoints.py        # extract MediaPipe keypoints + build labels
python train.py --dataset asl --epochs 50 --batch-size 32 --num-workers 0 --no-oversample
```

Checkpoints are saved to `MODEL/checkpoints/<dataset>_dim<dim>/`.

## Usage

```bash
streamlit run app_streamlit.py
```

### Controls

- **Start Webcam** / **Stop Webcam** — control the live feed
- **Test on Validation Samples** — run inference on held-out clips
- **Confidence Threshold** — adjust detection sensitivity (10–99)

## Testing

```bash
python -m pytest tests/
```

## Project Structure

```
SIGN_SPEAK-The-Silent-Communicator/
├── app_streamlit.py            # Main Streamlit application
├── model.py                    # SignTransformer (two-stream + motion gate)
├── train.py                    # Training loop (CE + triplet + contrastive)
├── train_dataset.py            # Keypoint dataset / dataloaders
├── extract_asl_keypoints.py    # ASL image keypoint extraction
├── extract_nslt100_fast.py     # Fast NSLT-100 video extraction
├── extract_keypoints_image_mode.py
├── yolo_hand_detector.py       # Legacy YOLO hand detector (unused by default)
├── tests/                      # Pytest suite
├── MODEL/checkpoints/          # Trained checkpoints (gitignored)
├── dataset/                    # ASL + WLASL data (gitignored)
└── *.task                      # MediaPipe model bundles (gitignored)
```
