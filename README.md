# SIGN SPEAK — Real-Time Sign Language Recognition

Real-time sign language recognition using MediaPipe hand/face/pose landmarks
and a two-stream (appearance + motion) 1D-CNN + Transformer with motion-gated
attention.

## Results

| Dataset | Classes | Val Acc | Test Acc |
| --- | --- | --- | --- |
| ASL (static gestures) | 36 | 93.6% | 89.6% |

Model: `MODEL/checkpoints/asl_dim192/best_model.pt`

## Architecture

- MediaPipe Tasks API: 543 landmarks (face 468 + hands 42 + pose 33)
- Nose-tip reference normalization + velocity/acceleration features
- Two-stream 1D-CNN + Transformer (192 dim, 4 heads, 2 blocks), motion-gated attention
- Loss: CE + triplet + contrastive

## Datasets

- `asl_dataset/` — 36 classes (0-9, A-Z), static gesture images
- `wlasl/` — WLASL v0.3 + NSLT subsets (dynamic signs)

## Quickstart

```bash
pip install -r requirements.txt
python extract_asl_keypoints.py
python train.py --dataset asl --epochs 50 --batch-size 32 --num-workers 0 --no-oversample
streamlit run app_streamlit.py
```

## Tests

```bash
python -m pytest tests/
```
