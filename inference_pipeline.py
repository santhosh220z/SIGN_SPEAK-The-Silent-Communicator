"""
Real-time Inference Pipeline: YOLOv11s Hand Detection + MediaPipe + Dual-Head Gesture Model
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from pathlib import Path
import argparse
import time

from yolo_hand_detector import HandDetector, MediaPipeLandmarkExtractor, extract_hand_landmarks_from_frame
from dual_head_model import DualHeadGestureModel, STATIC_GESTURE_CLASSES


class GestureRecognitionPipeline:
    """Complete real-time gesture recognition pipeline."""
    
    def __init__(
        self,
        model_path: str,
        yolo_model: str = "yolo11s.pt",
        device: str = "cuda",
        static_classes: list = None,
        action_classes: list = None,
        sequence_length: int = 64,
        static_conf_threshold: float = 0.7,
        action_conf_threshold: float = 0.5,
    ):
        """
        Args:
            model_path: Path to trained DualHeadGestureModel checkpoint
            yolo_model: YOLOv11s model path
            device: 'cuda' or 'cpu'
            static_classes: List of static gesture class names
            action_classes: List of action gesture class names
            sequence_length: Number of frames for action recognition
            static_conf_threshold: Min confidence for static prediction
            action_conf_threshold: Min confidence for action prediction
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.static_conf_threshold = static_conf_threshold
        self.action_conf_threshold = action_conf_threshold
        
        # Class names
        self.static_classes = static_classes or STATIC_GESTURE_CLASSES
        self.action_classes = action_classes or [f'Action_{i}' for i in range(100)]
        
        # Initialize components
        print("Loading YOLOv11s hand detector...")
        self.detector = HandDetector(
            model_path=yolo_model,
            conf_threshold=0.5,
            device=device,
            max_hands=2,
        )
        
        print("Loading MediaPipe landmark extractor...")
        self.landmark_extractor = MediaPipeLandmarkExtractor(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        print("Loading dual-head model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = DualHeadGestureModel(
            num_static_classes=len(self.static_classes),
            num_action_classes=len(self.action_classes),
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # Sequence buffer for action recognition
        self.sequence_buffer = deque(maxlen=sequence_length)
        self.frame_count = 0
        
        print(f"Pipeline ready on {self.device}")
        
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the complete pipeline.
        
        Returns:
            dict with keys:
                - 'static_results': list of dicts per detected hand
                - 'action_result': dict if sequence ready
                - 'visualization': annotated frame
        """
        h, w = frame.shape[:2]
        vis_frame = frame.copy()
        
        # 1. Detect hands with YOLO
        hand_results = extract_hand_landmarks_from_frame(
            frame, self.detector, self.landmark_extractor
        )
        
        static_results = []
        
        for hand in hand_results:
            landmarks = hand['landmarks']  # (21, 3)
            bbox = hand['bbox']
            det_conf = hand['confidence']
            
            # 2. Static gesture classification
            with torch.no_grad():
                static_input = torch.from_numpy(landmarks).unsqueeze(0).to(self.device)  # (1, 21, 3)
                static_logits = self.model(static_input=static_input, mode='static')['static_logits']
                static_probs = F.softmax(static_logits, dim=-1)
                static_conf, static_idx = static_probs.max(dim=-1)
                
                static_conf = static_conf.item()
                static_idx = static_idx.item()
                static_label = self.static_classes[static_idx] if static_idx < len(self.static_classes) else f'Class_{static_idx}'
            
            # Draw static result
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if static_conf > self.static_conf_threshold else (0, 165, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{static_label}: {static_conf:.2f}"
            cv2.putText(vis_frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw landmarks on hand crop
            if hand['crop'] is not None:
                crop_vis = self.landmark_extractor.draw_landmarks(hand['crop'], landmarks)
                # Could overlay crop_vis on main frame
            
            static_results.append({
                'label': static_label,
                'confidence': static_conf,
                'bbox': bbox,
                'landmarks': landmarks,
            })
            
            # 3. Add to sequence buffer for action recognition
            # Use full-body keypoints format (pad hand landmarks to 543 format)
            # For simplicity, we'll use a simplified approach
            self._add_to_sequence(landmarks)
        
        # 4. Action recognition (when sequence is full)
        action_result = None
        if len(self.sequence_buffer) == self.sequence_length:
            action_result = self._predict_action(vis_frame)
        
        # Draw sequence progress
        progress = len(self.sequence_buffer) / self.sequence_length
        cv2.rectangle(vis_frame, (10, 10), (10 + int(300 * progress), 30), (0, 255, 0), -1)
        cv2.rectangle(vis_frame, (10, 10), (310, 30), (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Action Buffer: {len(self.sequence_buffer)}/{self.sequence_length}', 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.frame_count += 1
        
        return {
            'static_results': static_results,
            'action_result': action_result,
            'visualization': vis_frame,
        }
    
    def _add_to_sequence(self, hand_landmarks: np.ndarray):
        """Add hand landmarks to sequence buffer."""
        # Convert hand landmarks (21, 3) to full body format (543, 3) with NaN padding
        # In practice, you'd use full body MediaPipe landmarks
        # For now, we pad with NaN
        full_landmarks = np.full((543, 3), np.nan, dtype=np.float32)
        # Place hand landmarks at hand indices (468-489 left, 522-543 right)
        # This is a simplified version - real implementation needs proper mapping
        self.sequence_buffer.append(full_landmarks)
    
    def _predict_action(self, vis_frame: np.ndarray) -> dict:
        """Predict action from buffered sequence."""
        # Stack sequence
        sequence = np.stack(list(self.sequence_buffer), axis=0)  # (T, 543, 3)
        sequence = sequence[np.newaxis, ...]  # (1, T, 543, 3)
        
        with torch.no_grad():
            action_input = torch.from_numpy(sequence).to(self.device)
            action_logits = self.model(action_input=action_input, mode='action')['action_logits']
            action_probs = F.softmax(action_logits, dim=-1)
            action_conf, action_idx = action_probs.max(dim=-1)
            
            action_conf = action_conf.item()
            action_idx = action_idx.item()
            action_label = self.action_classes[action_idx] if action_idx < len(self.action_classes) else f'Action_{action_idx}'
        
        # Draw action result
        if action_conf > self.action_conf_threshold:
            cv2.putText(vis_frame, f"ACTION: {action_label} ({action_conf:.2f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Clear buffer after prediction (or keep sliding window)
        self.sequence_buffer.clear()
        
        return {
            'label': action_label,
            'confidence': action_conf,
        }
    
    def run_webcam(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """Run pipeline on webcam feed."""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print("Starting webcam... Press 'q' to quit, 's' to save frame")
        
        fps_counter = deque(maxlen=30)
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            result = self.process_frame(frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            fps_counter.append(fps)
            avg_fps = sum(fps_counter) / len(fps_counter)
            
            # Draw FPS
            cv2.putText(result['visualization'], f'FPS: {avg_fps:.1f}', 
                       (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show
            cv2.imshow('Gesture Recognition', result['visualization'])
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'capture_{int(time.time())}.jpg', result['visualization'])
                print("Frame saved!")
                
        cap.release()
        cv2.destroyAllWindows()
        self.landmark_extractor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--yolo', type=str, default='yolo11s.pt', help='YOLO model path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length for action recognition')
    args = parser.parse_args()
    
    # Load class names from checkpoint if available
    checkpoint = torch.load(args.model, map_location='cpu')
    static_classes = checkpoint.get('static_classes', STATIC_GESTURE_CLASSES)
    action_classes = checkpoint.get('action_classes', None)
    
    pipeline = GestureRecognitionPipeline(
        model_path=args.model,
        yolo_model=args.yolo,
        device=args.device,
        static_classes=static_classes,
        action_classes=action_classes,
        sequence_length=args.seq_len,
    )
    
    pipeline.run_webcam(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
    )


if __name__ == '__main__':
    main()