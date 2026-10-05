"""
YOLOv11s Hand Detection Module
Detects hands in frames, crops them, and prepares for MediaPipe landmark extraction.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO


class HandDetector:
    """YOLOv11s-based hand detector for gesture recognition pipeline."""
    
    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        max_hands: int = 2,
    ):
        """
        Args:
            model_path: Path to YOLOv11s model (downloads if not found)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: 'cuda' or 'cpu'
            max_hands: Maximum number of hands to detect per frame
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.max_hands = max_hands
        
        # COCO class IDs: 0=person, we'll use a hand-specific model or filter
        # ultralytics YOLOv11 doesn't have hand class by default
        # We'll use a hand-trained model or detect person and use MediaPipe on crops
        
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect hands in frame.
        
        Returns:
            List of (x1, y1, x2, y2, confidence) for each detected hand
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=[0]  # person class - we'll use MediaPipe on person crops
        )
        
        boxes = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    boxes.append((x1, y1, x2, y2, conf))
        
        # Sort by confidence, take top max_hands
        boxes.sort(key=lambda x: x[4], reverse=True)
        return boxes[:self.max_hands]
    
    def detect_and_crop(
        self, 
        frame: np.ndarray, 
        padding: float = 0.2,
        target_size: Tuple[int, int] = (224, 224)
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect hands and return cropped/resized hand regions.
        
        Returns:
            List of (cropped_hand_image, original_bbox)
        """
        h, w = frame.shape[:2]
        detections = self.detect(frame)
        crops = []
        
        for x1, y1, x2, y2, conf in detections:
            # Add padding
            bw, bh = x2 - x1, y2 - y1
            pad_w, pad_h = int(bw * padding), int(bh * padding)
            
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(w, x2 + pad_w)
            y2_pad = min(h, y2 + pad_h)
            
            # Crop
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                continue
                
            # Resize to target size
            crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            crops.append((crop_resized, (x1, y1, x2, y2)))
        
        return crops


class MediaPipeLandmarkExtractor:
    """Extract 21 hand landmarks using MediaPipe Hands."""
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract hand landmarks from image.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            List of landmark arrays, each (21, 3) for x,y,z normalized coords
        """
        # MediaPipe expects RGB
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        results = self.hands.process(image_rgb)
        
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ], dtype=np.float32)  # (21, 3)
                landmarks_list.append(landmarks)
                
        return landmarks_list
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw landmarks on image for visualization."""
        h, w = image.shape[:2]
        vis = image.copy()
        for lm in landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        return vis
    
    def close(self):
        self.hands.close()


def extract_hand_landmarks_from_frame(
    frame: np.ndarray,
    detector: HandDetector,
    landmark_extractor: MediaPipeLandmarkExtractor,
) -> List[dict]:
    """
    Complete pipeline: detect hand -> crop -> extract landmarks.
    
    Returns:
        List of dicts with keys: 'landmarks', 'bbox', 'confidence', 'crop'
    """
    results = []
    detections = detector.detect(frame)
    
    for x1, y1, x2, y2, conf in detections:
        # Crop hand region with padding
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad = int(max(bw, bh) * 0.3)
        
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(w, x2 + pad)
        y2_pad = min(h, y2 + pad)
        
        hand_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if hand_crop.size == 0:
            continue
            
        # Extract landmarks
        landmarks_list = landmark_extractor.extract(hand_crop)
        
        if landmarks_list:
            # Use the first detected hand in crop
            landmarks = landmarks_list[0]
            results.append({
                'landmarks': landmarks,  # (21, 3)
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'crop': hand_crop,
            })
    
    return results