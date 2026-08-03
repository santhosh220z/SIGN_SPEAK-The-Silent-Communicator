import os
import cv2
import time
import json
import numpy as np
import threading
import pyttsx3
import streamlit as st
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from pathlib import Path
from collections import deque

# Page Configuration
st.set_page_config(page_title="SIGN SPEAK - Silent Communicator", page_icon="🤟", layout="wide")

# Session state for page navigation
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "MODEL"
WEIGHTS_PATH = MODEL_DIR / "Transformer.weights.h5"
MAP_PATH = MODEL_DIR / "sign_to_prediction_index_map.json"

FACE_TASK = BASE_DIR / "face_landmarker.task"
POSE_TASK = BASE_DIR / "pose_landmarker.task"
HAND_TASK = BASE_DIR / "hand_landmarker.task"

# Thread-safe TTS function
def speak(text):
    try:
        import pythoncom
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

# Model Parameters & Landmark Indices
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES
NUM_CLASSES = 250
PAD = 0.0

def tf_nan_mean(x, axis=0, keepdims=False):
    # Compute mean ignoring NaNs safely.
    mask = tf.math.is_nan(x)
    clean = tf.where(mask, tf.zeros_like(x), x)
    sum_ = tf.reduce_sum(clean, axis=axis, keepdims=keepdims)
    count = tf.reduce_sum(tf.cast(tf.logical_not(mask), x.dtype), axis=axis, keepdims=keepdims)
    return tf.where(count > 0, sum_ / count, tf.zeros_like(sum_))

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=64, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None, ...]
        else:
            x = inputs
        
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        
        x = (x - mean) / std

        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]

        dx = tf.cond(tf.shape(x)[1] > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]), lambda: tf.zeros_like(x))
        dx2 = tf.cond(tf.shape(x)[1] > 2, lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]), lambda: tf.zeros_like(x))

        x = tf.concat([
            tf.reshape(x, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)
        
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        return x

class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn

class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)
      
    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < self.start_step, lambda: inputs, lambda: self.dropout(inputs, training=training))
        if training:
            self._train_counter.assign_add(1)
        return x

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False, depthwise_initializer='glorot_uniform', name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0), name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size, kernel_size, dilation_rate=1, drop_rate=0.0, expand_ratio=2, se_ratio=0.25, activation='swish', name=None):
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio
        skip = inputs

        x = tf.keras.layers.Dense(channels_expand, use_bias=True, activation=activation, name=name + '_expand_conv')(inputs)
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate, use_bias=False, name=name + '_dwconv')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)
        x = ECA()(x)
        x = tf.keras.layers.Dense(channel_size, use_bias=True, name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        seq_len = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]
        qkv = tf.reshape(qkv, (batch_size, seq_len, self.num_heads, 3 * self.dim // self.num_heads))
        qkv = tf.transpose(qkv, perm=[0, 2, 1, 3])
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]
            attn = tf.where(mask, attn, tf.constant(-1e9, attn.dtype))

        attn = tf.keras.layers.Softmax(axis=-1)(attn)
        attn = self.drop1(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, seq_len, self.dim))
        x = self.proj(x)
        return x

def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

def get_model(max_len=64, dropout_step=0, dim=192):
    inp = tf.keras.Input((max_len, CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD, input_shape=(max_len, CHANNELS))(inp)
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.Dense(dim * 2, activation=None, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, name='classifier')(x)
    return tf.keras.Model(inp, x)

@st.cache_resource
def load_transformer_model():
    if WEIGHTS_PATH.exists():
        try:
            model = get_model(max_len=64, dim=192)
            model.load_weights(str(WEIGHTS_PATH))
            preprocess_layer = Preprocess(max_len=64)
            return model, preprocess_layer
        except Exception as e:
            st.sidebar.error(f"Failed to load Transformer weights: {e}")
    return None, None

@st.cache_data
def load_label_map():
    if MAP_PATH.exists():
        try:
            with open(MAP_PATH, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            return {int(v): str(k) for k, v in mapping.items()}
        except Exception as e:
            st.sidebar.error(f"Failed to load class map: {e}")
    return {}

@st.cache_resource
def load_mediapipe_tasks():
    try:
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

        face_landmarker = vision.FaceLandmarker.create_from_options(f_opt)
        pose_landmarker = vision.PoseLandmarker.create_from_options(p_opt)
        hand_landmarker = vision.HandLandmarker.create_from_options(h_opt)
        return face_landmarker, pose_landmarker, hand_landmarker
    except Exception as e:
        st.sidebar.error(f"MediaPipe Tasks load error: {e}")
        return None, None, None

# Sidebar Navigation
st.sidebar.title("🤟 SIGN SPEAK")
page = st.sidebar.radio("Navigate", ["🏠 Landing Page", "🎥 Live Recognition", "ℹ️ About"], index=0 if st.session_state["page"] == "landing" else 1)
st.session_state["page"] = "landing" if page == "🏠 Landing Page" else ("recognize" if page == "🎥 Live Recognition" else "about")

# Load models only when needed
model, preprocess_layer = load_transformer_model()
index_to_label = load_label_map()
face_lm, pose_lm, hand_lm = load_mediapipe_tasks()

if st.session_state["page"] == "landing":
    # ==================== LANDING PAGE ====================
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">🤟 SIGN SPEAK</h1>
        <h2 style="font-weight: 300; color: #666; margin-bottom: 2rem;">The Silent Communicator</h2>
        <p style="font-size: 1.2rem; color: #444; max-width: 800px; margin: 0 auto;">
            Real-time American Sign Language translation using a 1D-CNN Transformer model.
            Speak with your hands — we'll translate to text and speech instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 10px; background: #f0f2f6; text-align: center;">
            <h3 style="margin-bottom: 0.5rem;">🎯 250 ASL Classes</h3>
            <p style="color: #666;">Trained on ASL Citizen dataset covering words, phrases, and finger-spelling</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 10px; background: #f0f2f6; text-align: center;">
            <h3 style="margin-bottom: 0.5rem;">🧠 Transformer Architecture</h3>
            <p style="color: #666;">1D-CNN + Multi-head Attention for sequence-based gesture understanding</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 10px; background: #f0f2f6; text-align: center;">
            <h3 style="margin-bottom: 0.5rem;">🔊 Real-time TTS</h3>
            <p style="color: #666;">Instant text-to-speech output for seamless communication</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## How It Works")
    st.markdown("""
    <div style="display: flex; justify-content: space-between; text-align: center; margin: 2rem 0;">
        <div style="flex: 1; padding: 1rem;">
            <div style="font-size: 2rem;">📹</div>
            <h4>Capture</h4>
            <p style="color: #666;">Webcam captures hand, face & pose landmarks via MediaPipe (543 points)</p>
        </div>
        <div style="flex: 1; padding: 1rem;">
            <div style="font-size: 2rem;">⚡</div>
            <h4>Process</h4>
            <p style="color: #666;">Sequence of 64 frames normalized & fed into Transformer model</p>
        </div>
        <div style="flex: 1; padding: 1rem;">
            <div style="font-size: 2rem;">🎯</div>
            <h4>Predict</h4>
            <p style="color: #666;">250-class classification with confidence scoring</p>
        </div>
        <div style="flex: 1; padding: 1rem;">
            <div style="font-size: 2rem;">🔊</div>
            <h4>Speak</h4>
            <p style="color: #666;">Detected signs spoken aloud & logged in history</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick start button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Start Live Recognition", use_container_width=True, type="primary"):
            st.session_state["page"] = "recognize"
            st.rerun()

    # System status
    st.markdown("## System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if model is not None:
            st.success("✅ Transformer Model Loaded")
        else:
            st.error("❌ Transformer Model Failed")
    with status_col2:
        if face_lm is not None:
            st.success("✅ MediaPipe Tasks Ready")
        else:
            st.error("❌ MediaPipe Tasks Failed")

    st.info(f"**Model Classes:** {len(index_to_label)} | **Input:** 64 frames × 543 landmarks × 6 channels")

elif st.session_state["page"] == "about":
    # ==================== ABOUT PAGE ====================
    st.title("ℹ️ About SIGN SPEAK")
    
    st.markdown("""
    ### Project Overview
    SIGN SPEAK is a real-time sign language translation system that bridges communication between 
    deaf/hard-of-hearing individuals and hearing people using computer vision and deep learning.
    
    ### Technical Stack
    - **Computer Vision:** MediaPipe Tasks (Face, Pose, Hand Landmarkers)
    - **Deep Learning:** TensorFlow/Keras - 1D-CNN + Transformer Architecture
    - **Frontend:** Streamlit for real-time web interface
    - **Audio:** pyttsx3 for text-to-speech output
    
    ### Model Architecture
    - **Input:** 64-frame sequences of 543 landmarks (x, y, z + velocity + acceleration)
    - **Backbone:** 3× Conv1D blocks + 2× Transformer blocks (192 dim, 4 heads)
    - **Output:** 250-class classification (ASL Citizen dataset)
    - **Preprocessing:** NaN-robust normalization, temporal differencing
    
    ### Landmark Coverage (543 total)
    - **Face:** 468 landmarks (lips, eyes, nose, contours)
    - **Hands:** 21 × 2 = 42 landmarks (left + right)
    - **Pose:** 33 landmarks (upper body keypoints)
    
    ### Dataset
    Trained on the **ASL Citizen** dataset — a large-scale American Sign Language dataset 
    containing 250 distinct signs performed by diverse signers.
    """)

elif st.session_state["page"] == "recognize":
    # ==================== LIVE RECOGNITION ====================
    # Sidebar Configuration
    st.sidebar.title("⚙️ System Status")
    if model is not None and face_lm is not None:
        st.sidebar.success("✅ Transformer Model & MediaPipe Tasks Loaded")
    else:
        st.sidebar.error("❌ Component Initialization Failed")

    st.sidebar.markdown(f"**Loaded Sign Classes:** {len(index_to_label)}")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    
    if st.sidebar.button("🏠 Back to Landing Page"):
        st.session_state["page"] = "landing"
        st.session_state["run_webcam"] = False
        st.rerun()

    # Header UI
    st.title("🤟 SIGN SPEAK - Real-Time Transformer Sign Language Translation")
    st.markdown("Translating sequence-based hand & body gestures into spoken audio & text using a **1D-CNN Transformer** model.")

    # Session State Initialization
if "run_webcam" not in st.session_state:
    st.session_state["run_webcam"] = False
if "detected_word" not in st.session_state:
    st.session_state["detected_word"] = "Waiting for gesture..."
if "confidence" not in st.session_state:
    st.session_state["confidence"] = 0.0
if "last_spoken_word" not in st.session_state:
    st.session_state["last_spoken_word"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = []

# Control Buttons
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
with col_ctrl1:
    if st.button("▶️ Start Webcam", use_container_width=True):
        st.session_state["run_webcam"] = True
with col_ctrl2:
    if st.button("⏹️ Stop Webcam", use_container_width=True):
        st.session_state["run_webcam"] = False
with col_ctrl3:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["detected_word"] = "Waiting for gesture..."
        st.session_state["confidence"] = 0.0

# Layout Columns
col_left, col_right = st.columns([3, 2])
with col_left:
    frame_window = st.empty()

with col_right:
    st.markdown("### 🗣️ Real-Time Translation")
    text_display = st.empty()
    conf_display = st.empty()
    st.markdown("---")
    st.markdown("### 📜 Sentence / History Log")
    history_display = st.empty()

if st.session_state["run_webcam"]:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = time.time()
    
    frame_queue = deque(maxlen=64)
    no_hand_counter = 0

    try:
        while st.session_state["run_webcam"]:
            success, frame = cap.read()
            if not success:
                st.warning("Unable to access webcam video feed.")
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            img_output = frame.copy()
            h, w, _ = frame.shape

            face_res = face_lm.detect(mp_image) if face_lm else None
            pose_res = pose_lm.detect(mp_image) if pose_lm else None
            hand_res = hand_lm.detect(mp_image) if hand_lm else None

            xyz = np.full((543, 3), np.nan, dtype=np.float32)

            # 1. Face (0 to 467)
            if face_res and face_res.face_landmarks and len(face_res.face_landmarks) > 0:
                for i, lm in enumerate(face_res.face_landmarks[0]):
                    if i < 468:
                        xyz[i] = [lm.x, lm.y, lm.z]

            # 2. Left (468 to 488) & Right (522 to 542) Hands
            hand_active = False
            if hand_res and hand_res.hand_landmarks and hand_res.handedness:
                hand_active = True
                for landmarks, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
                    label = handedness[0].category_name
                    # Draw green circles on hand joints
                    for lm in landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img_output, (cx, cy), 3, (0, 255, 0), -1)

                    if label == "Left":
                        for i, lm in enumerate(landmarks):
                            xyz[468 + i] = [lm.x, lm.y, lm.z]
                    elif label == "Right":
                        for i, lm in enumerate(landmarks):
                            xyz[522 + i] = [lm.x, lm.y, lm.z]

            # 3. Pose (489 to 521)
            if pose_res and pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                for i, lm in enumerate(pose_res.pose_landmarks[0]):
                    if i < 33:
                        xyz[489 + i] = [lm.x, lm.y, lm.z]

            if hand_active:
                no_hand_counter = 0
                frame_queue.append(xyz)
            else:
                no_hand_counter += 1

            current_pred_text = st.session_state["detected_word"]
            current_conf = st.session_state["confidence"]

            # Perform Sequence Prediction
            if model is not None and preprocess_layer is not None and len(frame_queue) >= 8:
                try:
                    seq_arr = np.array(frame_queue, dtype=np.float32)[None, ...]
                    tensor_in = tf.constant(seq_arr)
                    prep_in = preprocess_layer(tensor_in)
                    logits = model(prep_in, training=False)
                    probs = tf.nn.softmax(logits[0]).numpy()
                    top_idx = int(np.argmax(probs))
                    conf_val = float(probs[top_idx])

                    if conf_val >= confidence_threshold:
                        predicted_sign = index_to_label.get(top_idx, f"Sign {top_idx}")
                        current_pred_text = predicted_sign
                        current_conf = conf_val
                        st.session_state["detected_word"] = predicted_sign
                        st.session_state["confidence"] = conf_val
                except Exception as e:
                    print(f"Prediction error: {e}")

            # If gesture ended (no hands for 12 frames), trigger TTS & sentence log
            if no_hand_counter == 12 and len(frame_queue) > 0:
                if st.session_state["confidence"] >= confidence_threshold and st.session_state["detected_word"] != "Waiting for gesture...":
                    word_to_speak = st.session_state["detected_word"]
                    if word_to_speak != st.session_state["last_spoken_word"]:
                        threading.Thread(target=speak, args=(word_to_speak,), daemon=True).start()
                        st.session_state["last_spoken_word"] = word_to_speak
                        st.session_state["history"].append(word_to_speak)
                frame_queue.clear()
                st.session_state["detected_word"] = "Waiting for gesture..."
                st.session_state["confidence"] = 0.0

            # Update UI Elements
            text_display.markdown(f"## 🎯 **{st.session_state['detected_word']}**")
            conf_display.progress(min(float(st.session_state["confidence"]), 1.0),
                                  text=f"Confidence: {st.session_state['confidence']*100:.1f}%")
            history_str = " ".join(st.session_state["history"][-15:])
            history_display.info(history_str if history_str else "No gestures recorded yet.")

            # FPS Counter
            curr_time = time.time()
            fps = 1 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time
            cv2.putText(img_output, f"FPS: {int(fps)} | Transformer Engine", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame_window.image(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB), use_container_width=True)

    finally:
        if face_lm:
            face_lm.close()
        if pose_lm:
            pose_lm.close()
        if hand_lm:
            hand_lm.close()
        cap.release()
        cv2.destroyAllWindows()
        frame_window.empty()

# End of recognize page
