"""
SIGN SPEAK - Stage 2 WLASL CUDA PyTorch Transformer Trainer
Trains a 1D-CNN + Spatiotemporal Transformer model on extracted WLASL keypoints using CUDA GPU acceleration.
Supports resuming training seamlessly from existing checkpoints using PyTorch CUDA.
"""

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
KEYPOINTS_DIR = DATASET_DIR / "keypoints"
VIDEOS_DIR = DATASET_DIR / "videos"
JSON_PATH = DATASET_DIR / "WLASL_v0.3.json"
MODEL_SAVE_DIR = BASE_DIR / "MODEL"

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Landmark Indices Selection
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

NUM_NODES = len(POINT_LANDMARKS)
INPUT_CHANNELS = 6 * NUM_NODES  # (x, y coords + dx velocity + dx2 acceleration) = 6 per landmark node

class WLASLKeypointDataset(Dataset):
    def __init__(self, samples, max_len=64):
        self.samples = samples  # List of (video_id, gloss_idx)
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_id, gloss_idx = self.samples[idx]
        npy_path = KEYPOINTS_DIR / f"{v_id}.npy"
        
        if npy_path.exists():
            try:
                arr = np.load(npy_path) # (64, 543, 3)
            except Exception:
                arr = np.full((self.max_len, 543, 3), 0.0, dtype=np.float32)
        else:
            # Check if video extraction can be triggered on demand
            try:
                from extract_wlasl_keypoints import process_single_video
                _, success, _ = process_single_video(v_id)
                if success and npy_path.exists():
                    arr = np.load(npy_path)
                else:
                    arr = np.full((self.max_len, 543, 3), 0.0, dtype=np.float32)
            except Exception:
                arr = np.full((self.max_len, 543, 3), 0.0, dtype=np.float32)

        # Handle NaNs and ensure shape
        arr = np.nan_to_num(arr, nan=0.0)
        if len(arr) < self.max_len:
            pad = np.zeros((self.max_len - len(arr), 543, 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:self.max_len]

        # Select facial, hand, and pose landmarks
        pts = arr[:, POINT_LANDMARKS, :2] # (64, NUM_NODES, 2)

        # Calculate Velocities (dx) and Accelerations (dx2)
        dx = np.zeros_like(pts)
        dx[:-1] = pts[1:] - pts[:-1]

        dx2 = np.zeros_like(pts)
        dx2[:-2] = pts[2:] - pts[:-2]

        # Reshape & Concatenate features: [Coords, Velocity, Acceleration]
        pts_flat = pts.reshape(self.max_len, -1)
        dx_flat = dx.reshape(self.max_len, -1)
        dx2_flat = dx2.reshape(self.max_len, -1)

        features = np.concatenate([pts_flat, dx_flat, dx2_flat], axis=-1) # (64, 6 * NUM_NODES)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(gloss_idx, dtype=torch.long)


class SpatiotemporalASLTransformer(nn.Module):
    def __init__(self, num_classes=2000, input_dim=INPUT_CHANNELS, hidden_dim=256, nhead=8, num_layers=3):
        super().__init__()
        
        # 1D Convolutional Feature Extractor
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 2, 
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Input shape: (Batch, Sequence_Len, Input_Dim)
        x = x.transpose(1, 2)              # (Batch, Input_Dim, Sequence_Len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)              # (Batch, Sequence_Len, Hidden_Dim)

        x = self.transformer(x)           # (Batch, Sequence_Len, Hidden_Dim)
        x = x.transpose(1, 2)              # (Batch, Hidden_Dim, Sequence_Len)
        x = self.pooling(x).squeeze(-1)    # (Batch, Hidden_Dim)

        logits = self.classifier(x)
        return logits


def train_wlasl_cuda(num_epochs=50, batch_size=32, lr=1e-3, resume=True, reset=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("       SIGN SPEAK - WLASL CUDA PyTorch Transformer Training", flush=True)
    print("=" * 70, flush=True)
    print(f"[*] Compute Device        : {device}", flush=True)
    if device.type == "cuda":
        print(f"[*] GPU Name              : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[*] Memory Allocated     : {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB", flush=True)
        torch.backends.cudnn.benchmark = True
    print("=" * 70, flush=True)

    if not JSON_PATH.exists():
        print(f"[Error] {JSON_PATH} not found!")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build Gloss Mapping
    gloss_list = [entry['gloss'] for entry in data]
    gloss_to_idx = {g: i for i, g in enumerate(gloss_list)}

    # Save Label Map for App Inference
    label_map_file = MODEL_SAVE_DIR / "wlasl_label_map.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump({str(i): g for g, i in gloss_to_idx.items()}, f, indent=2)

    train_samples = []
    val_samples = []

    for entry in data:
        g_idx = gloss_to_idx[entry['gloss']]
        for inst in entry['instances']:
            v_id = inst['video_id']
            split = inst.get('split', 'train')
            if (KEYPOINTS_DIR / f"{v_id}.npy").exists() or (VIDEOS_DIR / f"{v_id}.mp4").exists():
                if split in ('val', 'test'):
                    val_samples.append((v_id, g_idx))
                else:
                    train_samples.append((v_id, g_idx))

    print(f"[*] Dataset Split: {len(train_samples)} Train Samples | {len(val_samples)} Validation Samples | {len(gloss_to_idx)} Classes")

    train_loader = DataLoader(
        WLASLKeypointDataset(train_samples), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        WLASLKeypointDataset(val_samples), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=(device.type == "cuda")
    )

    model = SpatiotemporalASLTransformer(num_classes=len(gloss_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # PyTorch AMP Scaler for CUDA
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    checkpoint_file = MODEL_SAVE_DIR / "wlasl_cuda_transformer_checkpoint.pth"
    best_model_file = MODEL_SAVE_DIR / "wlasl_cuda_transformer.pth"

    start_epoch = 1
    best_acc = 0.0

    # Resume Training from Checkpoint if present
    if resume and not reset and checkpoint_file.exists():
        print(f"\n[*] Found existing checkpoint: {checkpoint_file}")
        try:
            checkpoint = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f"[+] Successfully loaded checkpoint! Resuming from Epoch {start_epoch} (Best Val Acc so far: {best_acc:.2f}%)")
        except Exception as e:
            print(f"[!] Warning: Failed to load checkpoint ({e}). Starting fresh training.")

    if start_epoch > num_epochs:
        print(f"\n[Info] Training already completed for {num_epochs} epochs. Requesting additional epochs to continue.")
        num_epochs = start_epoch + 10
        print(f"[*] Extended total epochs to: {num_epochs}")

    print(f"\n[*] Beginning CUDA Training from Epoch {start_epoch} to {num_epochs}...")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = (correct / total) * 100 if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(X_val)
                val_correct += (out.argmax(dim=1) == y_val).sum().item()
                val_total += y_val.size(0)

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0
        epoch_duration = time.time() - epoch_start

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_duration:.1f}s) | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)

        # Save Checkpoint after every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_acc': max(best_acc, val_acc),
            'loss': avg_loss
        }
        torch.save(checkpoint, checkpoint_file)

        # Save Best Model Weights
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_file)
            print(f"   ---> Improved Best Val Acc! Saved model weights to {best_model_file.name}", flush=True)

    print(f"\n[+] [Training Complete] Best Validation Accuracy: {best_acc:.2f}%")
    print(f"[+] Saved PyTorch CUDA Checkpoint to : {checkpoint_file}")
    print(f"[+] Saved Best Model Weights to      : {best_model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIGN SPEAK - WLASL CUDA Transformer Trainer")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start fresh")

    args = parser.parse_args()

    train_wlasl_cuda(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=(not args.no_resume),
        reset=args.reset
    )
