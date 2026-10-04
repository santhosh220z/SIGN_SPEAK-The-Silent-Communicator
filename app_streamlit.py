import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import time
import sys
sys.path.append('.')

from model import SignTransformer

# MediaPipe Tasks API imports
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Page config
st.set_page_config(
    page_title="SIGN SPEAK - ASL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10b981;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #374151;
    }
    .model-info h3 {
        color: #10b981;
        margin-top: 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #10b981;
        text-align: center;
    }
    .prediction-label {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10b981;
        margin-bottom: 0.5rem;
    }
    .confidence-bar {
        background: #064e3b;
        border-radius: 8px;
        height: 24px;
        overflow: hidden;
        margin: 1rem auto;
        max-width: 400px;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #10b981, #34d399);
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    .confidence-text {
        font-size: 1.5rem;
        font-weight: 600;
        color: #34d399;
        margin-top: 0.5rem;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .stat-card {
        background: #1f2937;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #374151;
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10b981;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    .stButton > button {
        width: 100%;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
    }
    .warning-box {
        background: #78350f;
        border: 1px solid #d97706;
        border-radius: 8px;
        padding: 1rem;
        color: #fde047;
        margin: 1rem 0;
    }
    .success-box {
        background: #064e3b;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        color: #34d399;
        margin: 1rem 0;
    }
    .info-box {
        background: #1e3a5f;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        color: #93c5fd;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
_CKPT_CANDIDATES = [
    Path("MODEL/checkpoints/slt100_dim192"),
    Path("MODEL/checkpoints/nslt100_dim192"),
]
MODEL_DIR = next((p for p in _CKPT_CANDIDATES if (p / "best_model.pt").exists()), _CKPT_CANDIDATES[0])
BEST_MODEL = MODEL_DIR / "best_model.pt"
MAX_FRAMES = 64

# Selected landmarks (118 = LIP + LHAND + RHAND + NOSE + REYE + LEYE) - matches model.py
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 
       314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 
       88, 178, 87, 14, 317, 402, 318, 324, 308]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE


@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not BEST_MODEL.exists():
        return None, device, f"Model not found at {BEST_MODEL}"
    
    checkpoint = torch.load(BEST_MODEL, map_location=device)
    args = checkpoint.get('args', {})
    
    model = SignTransformer(
        num_classes=args.get('num_classes', 100),
        dim=args.get('dim', 192),
        max_len=args.get('max_len', 64),
        dropout_step=args.get('dropout_step', 5000),
        use_motion_gate=args.get('use_motion_gate', True),
        contrastive_dim=args.get('contrastive_dim', 128)
    ).to(device)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    class_names = {i: f"Class {i}" for i in range(100)}
    
    info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_acc': checkpoint.get('best_val_acc', 0),
        'params': sum(p.numel() for p in model.parameters()),
        'dim': args.get('dim', 192)
    }
    
    return model, device, info, class_names


@st.cache_resource
def load_mediapipe_landmarkers():
    """Load MediaPipe Tasks API landmarkers"""
    base_options = mp_python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    
    # Face Landmarker (468 landmarks)
    face_options = FaceLandmarkerOptions(
        base_options=base_options(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_landmarker = FaceLandmarker.create_from_options(face_options)
    
    # Hand Landmarker (21 landmarks per hand = 42 total)
    hand_options = HandLandmarkerOptions(
        base_options=base_options(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    
    # Pose Landmarker (33 landmarks for body/arms)
    pose_options = PoseLandmarkerOptions(
        base_options=base_options(model_asset_path='pose_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)
    
    return face_landmarker, hand_landmarker, pose_landmarker


def extract_keypoints_from_results(face_result, hand_result, pose_result):
    """Extract 543 landmarks (468 face + 42 hands + 33 pose) from MediaPipe results"""
    keypoints = np.full((543, 3), np.nan, dtype=np.float32)
    
    # Face landmarks (0-467)
    if face_result and face_result.face_landmarks:
        for i, lm in enumerate(face_result.face_landmarks[0]):
            keypoints[i] = [lm.x, lm.y, lm.z]
    
    # Hand landmarks (468-509 for left, 510-541 for right - but MediaPipe returns 21 each)
    if hand_result and hand_result.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[hand_idx][0].category_name if hand_result.handedness else "Right"
            start_idx = 468 if handedness == "Left" else 522  # MediaPipe uses 468-488 for left, 522-542 for right
            for i, lm in enumerate(hand_landmarks):
                keypoints[start_idx + i] = [lm.x, lm.y, lm.z]
    
    # Pose landmarks (522-554 but we only have 33, mapped to 522-554 in 543 format)
    # Actually in 543 format: pose starts at different indices
    # For 543 format: pose landmarks are typically at specific indices
    if pose_result and pose_result.pose_landmarks:
        for i, lm in enumerate(pose_result.pose_landmarks[0]):
            # Map pose landmarks to 543 format indices
            # In MediaPipe 543 format, pose landmarks are at indices 522-554 (33 landmarks)
            # But we need to check the exact mapping
            if i < 33:
                keypoints[522 + i] = [lm.x, lm.y, lm.z]
    
    return keypoints


def preprocess_keypoints(keypoints, max_len=MAX_FRAMES):
    """Preprocess keypoints to match training preprocessing."""
    if len(keypoints) == 0:
        return None
    
    T = len(keypoints)
    if T < max_len:
        pad = np.full((max_len - T, 543, 3), np.nan, dtype=np.float32)
        keypoints = np.concatenate([keypoints, pad], axis=0)
    else:
        keypoints = keypoints[:max_len]
    
    kp = keypoints[:, POINT_LANDMARKS, :]
    
    nose_idx = POINT_LANDMARKS.index(17) if 17 in POINT_LANDMARKS else 0
    ref_coords = kp[:, nose_idx, :]
    
    mask = np.isnan(ref_coords)
    ref_clean = np.where(mask, 0, ref_coords)
    count = (~mask).sum(axis=(0, 1), keepdims=True)
    ref_mean = np.where(count > 0, ref_clean.sum(axis=(0, 1), keepdims=True) / count, 0.5)
    ref_mean = ref_mean.reshape(1, 1, 1, 3)
    
    mask_kp = np.isnan(kp)
    kp_clean = np.where(mask_kp, 0, kp)
    count_kp = (~mask_kp).sum(axis=(0, 1), keepdims=True)
    kp_std = np.sqrt(np.where(count_kp > 0, 
                               np.sum((kp_clean - ref_mean)**2, axis=(0, 1), keepdims=True) / count_kp, 
                               1.0))
    
    kp_norm = (kp - ref_mean) / (kp_std + 1e-6)
    kp_norm = np.where(np.isnan(kp_norm), 0, kp_norm)
    
    xy = kp_norm[..., :2]
    
    dx = np.zeros_like(xy)
    if max_len > 1:
        dx[:-1] = xy[1:] - xy[:-1]
    
    dx2 = np.zeros_like(xy)
    if max_len > 2:
        dx2[:-2] = xy[2:] - xy[:-2]
    
    app = xy.reshape(max_len, -1)
    mot = np.concatenate([dx, dx2], axis=-1).reshape(max_len, -1)
    
    return {
        'appearance': torch.from_numpy(app).unsqueeze(0).float(),
        'motion': torch.from_numpy(mot).unsqueeze(0).float()
    }


@st.cache_data
def load_test_samples():
    """Load a few test samples from validation set"""
    from train_dataset import get_dataloaders
    _, val_loader, _, _ = get_dataloaders(
        dataset='nslt100',
        batch_size=1,
        max_len=64,
        num_workers=0,
        persistent_workers=False,
        oversample=False,
        min_samples_per_class=2
    )
    
    samples = []
    for x, y in val_loader:
        samples.append((x.numpy()[0], y.item()))  # (64, 543, 3), label
        if len(samples) >= 10:
            break
    return samples


def draw_landmarks_on_frame(frame, face_result, hand_result, pose_result):
    """Draw MediaPipe landmarks on frame for visualization"""
    h, w = frame.shape[:2]
    annotated = frame.copy()
    
    # Draw face mesh
    if face_result and face_result.face_landmarks:
        for lm in face_result.face_landmarks[0]:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (x, y), 1, (0, 255, 0), -1)
    
    # Draw hand landmarks
    if hand_result and hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),  # thumb
                (0,5),(5,6),(6,7),(7,8),  # index
                (5,9),(9,10),(10,11),(11,12),  # middle
                (9,13),(13,14),(14,15),(15,16),  # ring
                (13,17),(17,18),(18,19),(19,20),  # pinky
                (0,17)  # palm
            ]
            for start, end in connections:
                p1 = hand_landmarks[start]
                p2 = hand_landmarks[end]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
            # Draw points
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (x, y), 3, (255, 165, 0), -1)
    
    # Draw pose landmarks (arms focus)
    if pose_result and pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        # Arm connections
        arm_connections = [
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # left arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # right arm
            (11, 12)  # shoulders
        ]
        for start, end in arm_connections:
            if start < len(landmarks) and end < len(landmarks):
                p1 = landmarks[start]
                p2 = landmarks[end]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(annotated, (x1, y1), (x2, y2), (139, 92, 246), 3)
        # Key points
        for i in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            if i < len(landmarks):
                lm = landmarks[i]
                if lm.visibility > 0.5:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 6, (139, 92, 246), -1)
                    cv2.circle(annotated, (x, y), 9, (168, 85, 247), 2)
    
    return annotated


def main():
    st.markdown('<h1 class="main-header">🤟 SIGN SPEAK</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time ASL Recognition with PyTorch Transformer + MediaPipe Tasks API</p>', unsafe_allow_html=True)
    
    # Load model
    model, device, info, class_names = load_model()
    
    if model is None:
        st.error(f"❌ {info}")
        st.info("Run `python train.py` first to train the model.")
        return
    
    # Load MediaPipe landmarkers
    try:
        face_landmarker, hand_landmarker, pose_landmarker = load_mediapipe_landmarkers()
        st.markdown("""
        <div class="success-box">
            <strong>✅ MediaPipe Tasks API Loaded:</strong> Face (468) + Hand (42) + Pose (33) = 543 landmarks
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Failed to load MediaPipe landmarkers: {e}")
        st.info("Ensure face_landmarker.task, hand_landmarker.task, pose_landmarker.task are in project root.")
        return
    
    st.markdown("""
    <div class="success-box">
        <strong>✅ Model Loaded:</strong> SignTransformer v1.0 | Two-stream + Motion-Gated Attention + Triplet Loss | 
        Best Val Acc: <strong>{:.2%}</strong> at epoch <strong>{}</strong> | 
        Parameters: <strong>{:,}</strong> | Device: <strong>{}</strong>
    </div>
    """.format(info['val_acc'], info['epoch'], info['params'], device.type.upper()), unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Info")
        st.markdown(f"""
        <div class="model-info">
            <h3>SignTransformer</h3>
            <p><strong>Architecture:</strong> Two-stream (Appearance + Motion) + Motion-Gated Attention</p>
            <p><strong>Parameters:</strong> {info['params']:,}</p>
            <p><strong>Dimension:</strong> {info['dim']}</p>
            <p><strong>Best Val Acc:</strong> {info['val_acc']:.2%}</p>
            <p><strong>Best Epoch:</strong> {info['epoch']}</p>
            <p><strong>Device:</strong> {device.type.upper()}</p>
            <p><strong>Classes:</strong> 100 (NSLT-100)</p>
            <p><strong>Input:</strong> 64 frames × 543 landmarks × 3 (xyz)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        conf_threshold = st.slider("Confidence Threshold (%)", 10, 99, 50, 5)
        
        st.markdown("---")
        
        st.markdown("### ⚡ Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("FPS", "0")
        with col2:
            st.metric("Latency", "0 ms")
    
    # Main content
    col_video, col_output = st.columns([2, 1])
    
    with col_video:
        st.markdown("### 📹 Camera Feed (Real-time Landmarks)")
        
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            start_camera = st.button("🎬 Start Camera", key="start", type="primary")
        with cam_col2:
            stop_camera = st.button("⏹️ Stop Camera", key="stop")
        
        st.markdown("---")
        
        # Test on validation samples
        st.markdown("### 🧪 Test on Validation Samples")
        test_samples = load_test_samples()
        
        sample_idx = st.selectbox(
            "Select validation sample",
            range(len(test_samples)),
            format_func=lambda i: f"Sample {i+1}: True class = {test_samples[i][1]}"
        )
        
        if st.button("Run Inference on Sample"):
            x_np, true_label = test_samples[sample_idx]
            x = torch.from_numpy(x_np).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                logits, emb = model(x, return_embedding=True)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(1).item()
                confidence = probs[0, pred].item() * 100
                
                st.session_state.last_prediction = class_names.get(pred, f"Class {pred}")
                st.session_state.last_confidence = confidence
                
                top3 = probs[0].topk(3)
                st.session_state.top3_preds = [(idx.item(), p.item() * 100) for idx, p in zip(top3.indices, top3.values)]
                st.session_state.true_label = true_label
        
        st.markdown("---")
        
        video_placeholder = st.empty()
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            fps_metric = st.empty()
        with stat_col2:
            landmarks_metric = st.empty()
        with stat_col3:
            latency_metric = st.empty()
    
    with col_output:
        st.markdown("### 🎯 Prediction Output")
        
        pred_placeholder = st.empty()
        conf_placeholder = st.empty()
        conf_bar_placeholder = st.empty()
        top3_placeholder = st.empty()
        
        # Show true label if available
        if 'true_label' in st.session_state and st.session_state.true_label is not None:
            st.info(f"True label: {st.session_state.true_label}")
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #1f2937; border-radius: 8px; border: 1px solid #374151;">
            <p style="margin: 0; color: #9ca3af; font-size: 0.85rem;">MODEL</p>
            <p style="margin: 0.25rem 0 0 0; color: #10b981; font-weight: 700; font-size: 1.1rem;">SignTransformer v1.0</p>
            <p style="margin: 0.25rem 0 0 0; color: #6b7280; font-size: 0.75rem;">Two-stream + Motion-Gated Attention + Triplet Loss</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'sequence_buffer' not in st.session_state:
        st.session_state.sequence_buffer = []
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = "WAITING..."
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = 0.0
    if 'top3_preds' not in st.session_state:
        st.session_state.top3_preds = []
    if 'true_label' not in st.session_state:
        st.session_state.true_label = None
    if 'last_inference_time' not in st.session_state:
        st.session_state.last_inference_time = 0
    
    if start_camera:
        st.session_state.camera_running = True
        st.session_state.sequence_buffer = []
        st.session_state.frame_count = 0
        st.rerun()
    
    if stop_camera:
        st.session_state.camera_running = False
        st.rerun()
    
    # Camera loop with MediaPipe Tasks API
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Please check permissions.")
            st.session_state.camera_running = False
            st.rerun()
        
        frame_placeholder = video_placeholder.empty()
        
        try:
            while st.session_state.camera_running:
                loop_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                
                # Run landmark detection
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # Extract keypoints (543 landmarks)
                keypoints = extract_keypoints_from_results(face_result, hand_result, pose_result)
                
                # Count valid landmarks
                valid_count = np.sum(~np.isnan(keypoints[:, 0]))
                
                # Add to sequence buffer
                st.session_state.sequence_buffer.append(keypoints)
                if len(st.session_state.sequence_buffer) > MAX_FRAMES:
                    st.session_state.sequence_buffer.pop(0)
                
                st.session_state.frame_count += 1
                
                # Draw landmarks on frame
                annotated_frame = draw_landmarks_on_frame(frame_rgb, face_result, hand_result, pose_result)
                
                # Display frame
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # Run inference every 30 frames (approx 1 second at 30 FPS)
                if len(st.session_state.sequence_buffer) >= 30 and st.session_state.frame_count % 30 == 0:
                    seq_array = np.array(st.session_state.sequence_buffer)
                    preprocessed = preprocess_keypoints(seq_array)
                    
                    if preprocessed:
                        app_feat = preprocessed['appearance'].to(device)
                        mot_feat = preprocessed['motion'].to(device)
                        
                        with torch.no_grad():
                            # Need to combine features for model forward
                            # Model expects (B, T, 543, 3) - we'll reconstruct from preprocessed
                            x = torch.from_numpy(seq_array).unsqueeze(0).float().to(device)
                            logits, emb = model(x, return_embedding=True)
                            probs = F.softmax(logits, dim=1)
                            pred = logits.argmax(1).item()
                            confidence = probs[0, pred].item() * 100
                            
                            if confidence >= conf_threshold:
                                st.session_state.last_prediction = class_names.get(pred, f"Class {pred}")
                                st.session_state.last_confidence = confidence
                                
                                top3 = probs[0].topk(3)
                                st.session_state.top3_preds = [(idx.item(), p.item() * 100) for idx, p in zip(top3.indices, top3.values)]
                
                # Update metrics
                elapsed = time.time() - loop_start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                
                fps_metric.metric("FPS", f"{fps:.1f}")
                landmarks_metric.metric("Landmarks", f"{valid_count}/543")
                latency_metric.metric("Latency", f"{elapsed*1000:.0f} ms")
                
                # Update prediction display
                with pred_placeholder.container():
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-label">{st.session_state.last_prediction}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {st.session_state.last_confidence}%;"></div>
                        </div>
                        <div class="confidence-text">{st.session_state.last_confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.session_state.top3_preds:
                    top3_html = ""
                    for cls, conf in st.session_state.top3_preds:
                        name = class_names.get(cls, f"Class {cls}")
                        top3_html += f"""
                        <div class="stat-card">
                            <div class="stat-value">{name}</div>
                            <div class="stat-label">{conf:.1f}%</div>
                        </div>
                        """
                    top3_placeholder.markdown(f"""
                    <div style="margin-top: 1rem;">
                        <h4 style="color: #9ca3af;">Top-3 Predictions</h4>
                        <div class="stats-grid">{top3_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                time.sleep(0.01)  # Small delay to prevent busy loop
                
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            cap.release()
            face_landmarker.close()
            hand_landmarker.close()
            pose_landmarker.close()
            st.session_state.camera_running = False
            video_placeholder.empty()
    
    # Update prediction display from session state
    with pred_placeholder.container():
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">{st.session_state.last_prediction}</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {st.session_state.last_confidence}%;"></div>
            </div>
            <div class="confidence-text">{st.session_state.last_confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.top3_preds:
        top3_html = ""
        for cls, conf in st.session_state.top3_preds:
            name = class_names.get(cls, f"Class {cls}")
            top3_html += f"""
            <div class="stat-card">
                <div class="stat-value">{name}</div>
                <div class="stat-label">{conf:.1f}%</div>
            </div>
            """
        top3_placeholder.markdown(f"""
        <div style="margin-top: 1rem;">
            <h4 style="color: #9ca3af;">Top-3 Predictions</h4>
            <div class="stats-grid">{top3_html}</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()