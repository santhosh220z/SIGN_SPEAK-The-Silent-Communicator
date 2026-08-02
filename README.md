# SIGN SPEAK - Real-Time Sign Language Translation

Real-time Transformer-based sign language translation using MediaPipe landmarks for hand, face, and pose detection.

## Requirements

- Python 3.10+
- Webcam

## Installation

```bash
pip install -r requirements.txt
```

## Model

Place your pre-trained Transformer model weights and label map in the `MODEL/` directory:
- `MODEL/Transformer.weights.h5` - Model weights
- `MODEL/sign_to_prediction_index_map.json` - Label mapping (250 classes for ASL Citizen dataset)

## Usage

```bash
streamlit run app.py
```

## Controls

- **▶️ Start Webcam** - Begin real-time recognition
- **⏹️ Stop Webcam** - Stop the webcam feed
- **🗑️ Clear History** - Clear detected gesture history
- **Confidence Threshold** - Adjust detection sensitivity (0.1-0.9)

## Architecture

- **MediaPipe Tasks**: Face, Pose, and Hand landmark detection (543 landmarks)
- **Preprocessing**: Normalization, velocity/acceleration features (6 channels per landmark)
- **Model**: 1D-CNN + Transformer architecture (192 dim, 4 heads, 2 blocks)
- **Output**: 250-class classification with softmax probabilities
- **TTS**: pyttsx3 for spoken translation

## Project Structure

```
SIGN_SPEAK-The-Silent-Communicator/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── MODEL/
│   ├── Transformer.weights.h5  # Pre-trained model weights
│   └── sign_to_prediction_index_map.json  # Label map (250 classes)
└── *.task                      # MediaPipe model files
```