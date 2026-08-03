"""
SIGN SPEAK - 2-Stage ASL Recognition Training & Export Pipeline

Stage 1: Keypoint Extraction (MediaPipe / YOLO-Pose)
         Extracts (x, y, z) landmarks across T frames for Face, Hands, and Pose.
         
Stage 2: Sequence Classification (1D-CNN + Transformer)
         Processes keypoint trajectories, velocities (dx), and accelerations (dx2)
         to classify static and dynamic motion ASL gestures.
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
KEYPOINT_DATA_DIR = BASE_DIR / "processed_keypoints"
MODEL_OUTPUT_DIR = BASE_DIR / "MODEL"
TFJS_OUTPUT_DIR = BASE_DIR / "tfjs_model"

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYPOINT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Selected Landmark Indices for 2-Stage Model
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES
NUM_CLASSES = 250
MAX_LEN = 64


# ==========================================
# STAGE 1: KEYPOINT EXTRACTION UTILITIES
# ==========================================

def extract_keypoints_from_video(video_path, max_frames=64):
    """
    Extracts 543 MediaPipe landmark coordinates (x, y, z) per frame over a video sequence.
    Returns array of shape (max_frames, 543, 3).
    """
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    
    cap = cv2.VideoCapture(str(video_path))
    sequence = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            frame_landmarks = np.full((543, 3), np.nan, dtype=np.float32)
            
            # Face Landmarks (468 points)
            if results.face_landmarks:
                for idx, lm in enumerate(results.face_landmarks.landmark[:468]):
                    frame_landmarks[idx] = [lm.x, lm.y, lm.z]
                    
            # Left Hand (21 points)
            if results.left_hand_landmarks:
                for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_landmarks[468 + idx] = [lm.x, lm.y, lm.z]
                    
            # Pose (33 points)
            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    frame_landmarks[489 + idx] = [lm.x, lm.y, lm.z]
                    
            # Right Hand (21 points)
            if results.right_hand_landmarks:
                for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_landmarks[522 + idx] = [lm.x, lm.y, lm.z]
                    
            sequence.append(frame_landmarks)
            if len(sequence) >= max_frames:
                break
                
    cap.release()
    
    if len(sequence) == 0:
        return np.full((max_frames, 543, 3), np.nan, dtype=np.float32)
        
    sequence = np.array(sequence, dtype=np.float32)
    
    # Pad sequence to max_frames if shorter
    if len(sequence) < max_frames:
        pad_width = max_frames - len(sequence)
        padding = np.full((pad_width, 543, 3), np.nan, dtype=np.float32)
        sequence = np.concatenate([sequence, padding], axis=0)
        
    return sequence[:max_frames]


# ==========================================
# STAGE 2: TRANSFORMER MODEL DEFINITION
# ==========================================

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
        
        # Center landmarks relative to lips/nose
        mean = tf.reduce_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)
        
        # Normalize
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
        std = tf.where(tf.math.is_nan(std) | (std == 0), tf.constant(1.0, x.dtype), std)
        x = (x - mean) / std

        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]

        # Calculate Velocities (dx) and Accelerations (dx2)
        dx = tf.cond(tf.shape(x)[1] > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]), lambda: tf.zeros_like(x))
        dx2 = tf.cond(tf.shape(x)[1] > 2, lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]), lambda: tf.zeros_like(x))

        x = tf.concat([
            tf.reshape(x, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)
        
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        return x


def build_2stage_transformer(num_classes=250, max_len=64):
    """
    Builds the 1D-CNN + Transformer classifier network.
    """
    inputs = tf.keras.Input(shape=(max_len, 543, 3), dtype=tf.float32, name="keypoints_input")
    
    # Stage 1 Feature Engineering (Coordinates + Velocity + Acceleration)
    x = Preprocess(max_len=max_len)(inputs)
    
    # 1D Conv Blocks for Temporal Features
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Transformer Encoder Block
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Global Pooling & Output Layer
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="prediction_output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="2Stage_ASL_Transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("      2-STAGE ASL PIPELINE (KEYPOINTS + TRANSFORMER)")
    print("=" * 60)
    
    # Build model
    model = build_2stage_transformer(num_classes=NUM_CLASSES, max_len=MAX_LEN)
    model.summary()
    
    # Save weights
    weights_path = MODEL_OUTPUT_DIR / "Transformer.weights.h5"
    model.save_weights(weights_path)
    print(f"\n[Success] Model initialized & weights saved to: {weights_path}")
