# Graph Report - .  (2026-08-04)

## Corpus Check
- Corpus is ~14,253 words - fits in a single context window. You may not need a graph.

## Summary
- 163 nodes · 245 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 91% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 20 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Web Dashboard UI
- Neural Network Architecture
- Inference & Testing
- Windows pywin32 Utilities
- Streamlit App Pipeline
- Browser MediaPipe JS
- Dataset Loading & Sampling
- pywin32 Test Runner
- Keypoint Extraction Script
- Frontend Dependencies
- Log Stream Tee
- OpenCV Dependency

## God Nodes (most connected - your core abstractions)
1. `install()` - 12 edges
2. `SignTransformer` - 11 edges
3. `SignLanguageDataset` - 10 edges
4. `get_dataloaders()` - 9 edges
5. `uninstall()` - 8 edges
6. `main()` - 7 edges
7. `PreprocessLayer` - 7 edges
8. `SIGN SPEAK Project` - 7 edges
9. `MediaPipe Landmark Detection (543 landmarks)` - 7 edges
10. `SIGN SPEAK AI Dashboard UI` - 7 edges

## Surprising Connections (you probably didn't know these)
- `sign_to_prediction_index_map.json Label Map (250 ASL Citizen classes)` --semantically_similar_to--> `NSLT-100 Dataset (748 train / 165 val / 100 test keypoint samples)`  [INFERRED] [semantically similar]
  README.md → train_log.txt
- `Landmark Mesh Overlay Canvas` --conceptually_related_to--> `MediaPipe Landmark Detection (543 landmarks)`  [INFERRED]
  index.html → README.md
- `Show Mesh Overlay Toggle` --conceptually_related_to--> `MediaPipe Landmark Detection (543 landmarks)`  [INFERRED]
  index.html → README.md
- `Transformer Training Run (epoch 0, loss ~4.39, acc ~0.032)` --conceptually_related_to--> `1D-CNN + Transformer Architecture (192 dim, 4 heads, 2 blocks)`  [AMBIGUOUS]
  train_log.txt → README.md
- `Transformer Training Run (epoch 0, loss ~4.39, acc ~0.032)` --references--> `tensorflow`  [INFERRED]
  train_log.txt → requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Real-Time Inference Pipeline (landmarks -> preprocessing -> 1D-CNN+Transformer -> 250-class -> TTS)** — readme_mediapipe_landmarks, readme_preprocessing, readme_arch_1d_cnn_transformer, readme_output_250_classes, readme_tts [EXTRACTED 1.00]
- **Frontend Recognition UI Loop (camera -> landmark overlay -> prediction -> speak)** — index_webcam_video, index_landmark_canvas, index_model_selector, index_prediction_banner, index_speak_button [INFERRED 0.85]
- **Model Training Pipeline (nslt100 dataset on GPU with AMP)** — train_log_nslt100, train_log_training_run, train_log_model_parameters, train_log_gpu_environment [EXTRACTED 1.00]

## Communities (13 total, 2 thin omitted)

### Community 0 - "Web Dashboard UI"
Cohesion: 0.09
Nodes (30): SIGN SPEAK AI Dashboard UI, Auto Text-to-Speech Toggle, Min Confidence Slider (50-99%), Landmark Mesh Overlay Canvas, Show Mesh Overlay Toggle, Model Selection Dropdown (Transformer-250 / MediaPipe-543 / CNN-Lightweight), Recognition Prediction Banner with Confidence, script.js Application Logic (+22 more)

### Community 1 - "Neural Network Architecture"
Cohesion: 0.11
Nodes (9): CausalDWConv1D, Conv1DBlock, ECA, LateDropout, MotionGatedAttention, PreprocessLayer, Attention modulated by per-frame motion magnitude, PyTorch port of the TF Preprocess layer.     Input: (B, T, 543, 3) - keypoints w (+1 more)

### Community 2 - "Inference & Testing"
Cohesion: 0.15
Nodes (16): load_model(), Load model from checkpoint, run_inference(), Two-stream Sign Language Transformer:     - Appearance stream: static handshape/, Get contrastive embedding for a batch, SignTransformer, no_grad, contrastive_loss() (+8 more)

### Community 3 - "Windows pywin32 Utilities"
Cohesion: 0.24
Nodes (19): CopyTo(), create_shortcut(), fixup_dbi(), get_root_hkey(), get_shortcuts_folder(), get_special_folder_path(), get_system_dir(), install() (+11 more)

### Community 4 - "Streamlit App Pipeline"
Cohesion: 0.18
Nodes (15): draw_landmarks_on_frame(), extract_keypoints_from_results(), load_mediapipe_landmarkers(), load_model(), load_test_samples(), main(), preprocess_keypoints(), Load the trained model (+7 more)

### Community 5 - "Browser MediaPipe JS"
Cohesion: 0.21
Nodes (10): DEMO_SIGNS, initUIControls(), lastFpsTime, resetStats(), speakText(), startCamera(), startDetectionLoop(), stopCamera() (+2 more)

### Community 6 - "Dataset Loading & Sampling"
Cohesion: 0.27
Nodes (4): Dataset, Dataset for sign language keypoints (64, 543, 3), Create weighted sampler for balanced training, SignLanguageDataset

### Community 8 - "pywin32 Test Runner"
Cohesion: 0.60
Nodes (4): find_and_run(), main(), A test runner for pywin32, run_test()

### Community 9 - "Keypoint Extraction Script"
Cohesion: 0.50
Nodes (3): extract_keypoints(), Extract MediaPipe keypoints from WLASL videos in dataset/videos/ Maps video_id -, Extract (MAX_FRAMES, 543, 3) keypoints from video

### Community 10 - "Frontend Dependencies"
Cohesion: 0.50
Nodes (3): framer-motion, dependencies, framer-motion

## Ambiguous Edges - Review These
- `1D-CNN + Transformer Architecture (192 dim, 4 heads, 2 blocks)` → `Transformer Training Run (epoch 0, loss ~4.39, acc ~0.032)`  [AMBIGUOUS]
  train_log.txt · relation: conceptually_related_to

## Knowledge Gaps
- **16 isolated node(s):** `framer-motion`, `lastFpsTime`, `DEMO_SIGNS`, `style.css Stylesheet`, `script.js Application Logic` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `1D-CNN + Transformer Architecture (192 dim, 4 heads, 2 blocks)` and `Transformer Training Run (epoch 0, loss ~4.39, acc ~0.032)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `SignTransformer` connect `Inference & Testing` to `Neural Network Architecture`, `Streamlit App Pipeline`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `get_dataloaders()` connect `Inference & Testing` to `Streamlit App Pipeline`, `Dataset Loading & Sampling`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `SignLanguageDataset` connect `Dataset Loading & Sampling` to `Inference & Testing`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **What connects `framer-motion`, `lastFpsTime`, `DEMO_SIGNS` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Web Dashboard UI` be split into smaller, more focused modules?**
  _Cohesion score 0.08735632183908046 - nodes in this community are weakly interconnected._
- **Should `Neural Network Architecture` be split into smaller, more focused modules?**
  _Cohesion score 0.11375661375661375 - nodes in this community are weakly interconnected._