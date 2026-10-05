"""
Static Gesture Data Collection Script
Collect hand landmark data for static gesture classification (ASL alphabet, numbers, etc.)
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import os


class StaticGestureCollector:
    def __init__(self, data_dir: str = 'dataset/static_gestures'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture classes to collect
        self.gestures = [
            # ASL Alphabet
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            # Numbers
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # Common gestures
            'thumbs_up', 'thumbs_down', 'ok', 'peace', 'fist', 'open_hand',
            'pointing', 'call_me', 'rock_on', 'crossed_fingers', 'heart',
            'wave', 'clap', 'snap',
        ]
        
        # Create directories
        for g in self.gestures:
            (self.data_dir / g).mkdir(exist_ok=True)
            
    def collect(self, gesture: str, num_samples: int = 100, camera_id: int = 0):
        """Collect samples for a specific gesture."""
        if gesture not in self.gestures:
            print(f"Unknown gesture: {gesture}. Available: {self.gestures}")
            return
            
        gesture_dir = self.data_dir / gesture
        existing = len(list(gesture_dir.glob('*.npy')))
        print(f"Collecting {gesture}: {existing} existing, target {num_samples}")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        collected = 0
        collecting = False
        
        print("Press SPACE to start/stop collecting, ESC to exit")
        
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # Mirror
            h, w = frame.shape[:2]
            
            # Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Extract landmarks
                    landmarks = np.array([
                        [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                    ], dtype=np.float32)  # (21, 3)
                    
                    if collecting:
                        # Save
                        fname = gesture_dir / f'{gesture}_{existing + collected:04d}.npy'
                        np.save(fname, landmarks)
                        collected += 1
                        
            # UI
            status = "COLLECTING" if collecting else "PAUSED"
            color = (0, 255, 0) if collecting else (0, 0, 255)
            cv2.putText(frame, f"{gesture}: {collected}/{num_samples} [{status}]", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "SPACE: start/stop  ESC: exit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Static Gesture Collector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                collecting = not collecting
                print(f"Collecting: {collecting}")
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {collected} samples for {gesture}")
        
    def collect_all(self, samples_per_gesture: int = 100, camera_id: int = 0):
        """Collect data for all gestures interactively."""
        for gesture in self.gestures:
            print(f"\n{'='*50}")
            print(f"Next gesture: {gesture}")
            print("Get ready, press SPACE to start collecting")
            input("Press Enter to continue...")
            self.collect(gesture, samples_per_gesture, camera_id)
            
        print("\nAll gestures collected!")
        
    def create_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """Create train/val/test splits."""
        import shutil
        import random
        
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir()
            for g in self.gestures:
                (split_dir / g).mkdir()
                
        for gesture in self.gestures:
            files = list((self.data_dir / gesture).glob('*.npy'))
            random.shuffle(files)
            
            n_train = int(len(files) * train_ratio)
            n_val = int(len(files) * val_ratio)
            
            for i, f in enumerate(files):
                if i < n_train:
                    dst = self.data_dir / 'train' / gesture
                elif i < n_train + n_val:
                    dst = self.data_dir / 'val' / gesture
                else:
                    dst = self.data_dir / 'test' / gesture
                shutil.copy2(f, dst / f.name)
                
            print(f"{gesture}: train={n_train}, val={n_val}, test={len(files)-n_train-n_val}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gesture', type=str, help='Single gesture to collect')
    parser.add_argument('--samples', type=int, default=100, help='Samples per gesture')
    parser.add_argument('--all', action='store_true', help='Collect all gestures')
    parser.add_argument('--split', action='store_true', help='Create train/val/test splits')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default='dataset/static_gestures')
    args = parser.parse_args()
    
    collector = StaticGestureCollector(args.data_dir)
    
    if args.split:
        collector.create_split()
    elif args.all:
        collector.collect_all(args.samples, args.camera)
    elif args.gesture:
        collector.collect(args.gesture, args.samples, args.camera)
    else:
        print("Available gestures:", collector.gestures)
        print("Use --gesture <name> --samples <N> to collect one")
        print("Use --all to collect all")
        print("Use --split to create train/val/test splits")


if __name__ == '__main__':
    main()