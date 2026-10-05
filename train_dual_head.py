"""
Training script for Dual-Head Gesture Recognition Model
Trains both static and action heads with appropriate datasets.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
from collections import Counter
import json


# Import our models
from dual_head_model import DualHeadGestureModel, STATIC_GESTURE_CLASSES


class StaticGestureDataset(Dataset):
    """Dataset for static hand gesture classification (single frame landmarks)."""
    
    def __init__(
        self, 
        data_dir: str,
        split: str = 'train',
        max_samples_per_class: Optional[int] = None,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and split == 'train'
        
        # Expected structure: data_dir/split/class_name/*.npy
        # Each .npy is (21, 3) landmarks
        self.samples = []
        self.class_to_idx = {}
        
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
            
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            files = list(class_dir.glob('*.npy'))
            if max_samples_per_class and len(files) > max_samples_per_class:
                files = files[:max_samples_per_class]
            for f in files:
                self.samples.append((f, idx))
                
        self.num_classes = len(self.class_to_idx)
        print(f"[StaticGestureDataset] {split}: {len(self.samples)} samples, {self.num_classes} classes")
        
        # Class weights for imbalanced sampling
        if split == 'train':
            labels = [s[1] for s in self.samples]
            counts = Counter(labels)
            self.class_weights = {cls: 1.0 / cnt for cls, cnt in counts.items()}
            self.sample_weights = torch.DoubleTensor([self.class_weights[l] for l in labels])
        else:
            self.sample_weights = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        landmarks = np.load(fpath).astype(np.float32)  # (21, 3)
        
        # Augmentation
        if self.augment:
            # Add noise
            landmarks += np.random.normal(0, 0.01, landmarks.shape).astype(np.float32)
            # Random rotation in xy plane
            angle = np.random.uniform(-0.1, 0.1)
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]], dtype=np.float32)
            landmarks = landmarks @ rot.T
            
        x = torch.from_numpy(landmarks)  # (21, 3)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class ActionGestureDataset(Dataset):
    """Dataset for action gesture classification (temporal sequences)."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_len: int = 64,
        oversample: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_len = max_len
        self.oversample = oversample and split == 'train'
        
        # Expected: data_dir/split/*.npy where each is (T, 543, 3) or (T, 21, 3)
        # with corresponding labels in a JSON or CSV
        
        # For now, use the existing NSLT keypoints
        # We'll reuse the SignLanguageDataset logic
        from train_dataset import SignLanguageDataset
        
        self.dataset = SignLanguageDataset(
            split=split,
            max_len=max_len,
            dataset='nslt100',  # configurable
            oversample=self.oversample,
        )
        self.num_classes = self.dataset.num_classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # x is (64, 543, 3) - full body keypoints
        return x, y


class CombinedGestureDataset(Dataset):
    """Combined dataset that provides both static and action data."""
    
    def __init__(
        self,
        static_data_dir: str,
        action_data_dir: str,
        split: str = 'train',
        max_len: int = 64,
    ):
        self.static_ds = StaticGestureDataset(static_data_dir, split)
        self.action_ds = ActionGestureDataset(action_data_dir, split, max_len)
        
        # Use the longer dataset length
        self.length = max(len(self.static_ds), len(self.action_ds))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Sample from each dataset with modulo
        static_idx = idx % len(self.static_ds)
        action_idx = idx % len(self.action_ds)
        
        static_x, static_y = self.static_ds[static_idx]
        action_x, action_y = self.action_ds[action_idx]
        
        return {
            'static': (static_x, static_y),
            'action': (action_x, action_y),
        }


def train_static_head(
    model: DualHeadGestureModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    output_dir: Path = Path('MODEL/checkpoints/dual_head'),
):
    """Train only the static head."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / 'logs_static')
    
    # Freeze action head
    for p in model.action_head.parameters():
        p.requires_grad = False
    for p in model.static_head.parameters():
        p.requires_grad = True
        
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    best_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Static Epoch {epoch}', leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(static_input=x, mode='static')
                loss = criterion(out['static_logits'], y)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * y.size(0)
            pred = out['static_logits'].argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
            
        avg_loss = total_loss / total
        acc = correct / total
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/Acc', acc, epoch)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(static_input=x, mode='static')
                loss = criterion(out['static_logits'], y)
                val_loss += loss.item() * y.size(0)
                pred = out['static_logits'].argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
        val_loss /= total
        val_acc = correct / total
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)
        
        print(f'Epoch {epoch}: Train Loss={avg_loss:.4f} Acc={acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_acc': best_acc,
            }, output_dir / 'best_static_model.pt')
            print(f'  -> New best static model! Val Acc: {val_acc:.4f}')
            
    writer.close()
    return best_acc


def train_action_head(
    model: DualHeadGestureModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    contrastive_weight: float = 0.1,
    triplet_margin: float = 0.3,
    output_dir: Path = Path('MODEL/checkpoints/dual_head'),
):
    """Train only the action head (reuse train.py logic)."""
    
    from train import train_epoch, evaluate, triplet_loss
    
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / 'logs_action')
    
    # Freeze static head
    for p in model.static_head.parameters():
        p.requires_grad = False
    for p in model.action_head.parameters():
        p.requires_grad = True
        
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    # Cosine annealing with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor(
            (epoch - warmup_epochs) / (epochs - warmup_epochs) * 3.14159)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model.action_head.transformer, train_loader, ce_criterion, optimizer, 
            scaler, device, epoch, writer,
            contrastive_weight=contrastive_weight, margin=triplet_margin
        )
        
        # Validate
        val_loss, val_acc, _ = evaluate(
            model.action_head.transformer, val_loader, ce_criterion, 
            device, epoch, writer, 'Val'
        )
        
        scheduler.step()
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_acc': best_val_acc,
            }, output_dir / 'best_action_model.pt')
            print(f'  -> New best action model! Val Acc: {val_acc:.4f}')
            
    writer.close()
    return best_val_acc


def train_combined(
    model: DualHeadGestureModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    static_weight: float = 1.0,
    action_weight: float = 1.0,
    contrastive_weight: float = 0.1,
    triplet_margin: float = 0.3,
    output_dir: Path = Path('MODEL/checkpoints/dual_head'),
):
    """Train both heads jointly."""
    
    from train import triplet_loss
    
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / 'logs_combined')
    
    # Unfreeze both
    for p in model.parameters():
        p.requires_grad = True
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor(
            (epoch - warmup_epochs) / (epochs - warmup_epochs) * 3.14159)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        static_correct = 0
        action_correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Combined Epoch {epoch}', leave=False)
        for batch in pbar:
            static_x, static_y = batch['static']
            action_x, action_y = batch['action']
            
            static_x = static_x.to(device, non_blocking=True)
            static_y = static_y.to(device, non_blocking=True)
            action_x = action_x.to(device, non_blocking=True)
            action_y = action_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                out = model(static_input=static_x, action_input=action_x, mode='both')
                
                static_loss = ce_criterion(out['static_logits'], static_y)
                action_ce = ce_criterion(out['action_logits'], action_y)
                
                # Contrastive loss for action head
                _, action_emb = model.action_head.transformer(action_x, return_embedding=True)
                ctr_loss = triplet_loss(action_emb, action_y, margin=triplet_margin)
                
                loss = (static_weight * static_loss + 
                       action_weight * (action_ce + contrastive_weight * ctr_loss))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * static_y.size(0)
            static_correct += (out['static_logits'].argmax(1) == static_y).sum().item()
            action_correct += (out['action_logits'].argmax(1) == action_y).sum().item()
            total += static_y.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'static_acc': f'{static_correct/total:.4f}',
                'action_acc': f'{action_correct/total:.4f}',
            })
            
        avg_loss = total_loss / total
        static_acc = static_correct / total
        action_acc = action_correct / total
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/Static_Acc', static_acc, epoch)
        writer.add_scalar('Train/Action_Acc', action_acc, epoch)
        
        # Validate (static only for simplicity)
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                static_x, static_y = batch['static']
                static_x = static_x.to(device)
                static_y = static_y.to(device)
                
                out = model(static_input=static_x, mode='static')
                loss = ce_criterion(out['static_logits'], static_y)
                val_loss += loss.item() * static_y.size(0)
                correct += (out['static_logits'].argmax(1) == static_y).sum().item()
                total += static_y.size(0)
                
        val_loss /= total
        val_acc = correct / total
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)
        
        print(f'Epoch {epoch}: Train Loss={avg_loss:.4f} Static={static_acc:.4f} Action={action_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_acc': best_val_acc,
            }, output_dir / 'best_combined_model.pt')
            print(f'  -> New best combined model! Val Acc: {val_acc:.4f}')
            
        scheduler.step()
        
    writer.close()
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='combined',
                        choices=['static', 'action', 'combined'],
                        help='Which head(s) to train')
    parser.add_argument('--static-data', type=str, default='dataset/static_gestures',
                        help='Directory for static gesture data')
    parser.add_argument('--action-data', type=str, default='dataset',
                        help='Directory for action gesture data (NSLT)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--static-weight', type=float, default=1.0)
    parser.add_argument('--action-weight', type=float, default=1.0)
    parser.add_argument('--contrastive-weight', type=float, default=0.1)
    parser.add_argument('--triplet-margin', type=float, default=0.3)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='MODEL/checkpoints/dual_head')
    parser.add_argument('--num-static-classes', type=int, default=50)
    parser.add_argument('--num-action-classes', type=int, default=100)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Create model
    model = DualHeadGestureModel(
        num_static_classes=args.num_static_classes,
        num_action_classes=args.num_action_classes,
    ).to(device)
    
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create dataloaders
    if args.mode == 'static':
        train_ds = StaticGestureDataset(args.static_data, 'train')
        val_ds = StaticGestureDataset(args.static_data, 'val')
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                                  sampler=WeightedRandomSampler(train_ds.sample_weights, len(train_ds), replacement=True),
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        train_static_head(model, train_loader, val_loader, device, 
                         epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                         output_dir=Path(args.output_dir))
        
    elif args.mode == 'action':
        train_ds = ActionGestureDataset(args.action_data, 'train', max_len=args.max_len)
        val_ds = ActionGestureDataset(args.action_data, 'val', max_len=args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=WeightedRandomSampler(train_ds.dataset.sample_weights, len(train_ds), replacement=True),
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        train_action_head(model, train_loader, val_loader, device,
                         epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                         contrastive_weight=args.contrastive_weight, triplet_margin=args.triplet_margin,
                         output_dir=Path(args.output_dir))
        
    else:  # combined
        train_ds = CombinedGestureDataset(args.static_data, args.action_data, 'train', max_len=args.max_len)
        val_ds = CombinedGestureDataset(args.static_data, args.action_data, 'val', max_len=args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        train_combined(model, train_loader, val_loader, device,
                      epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                      static_weight=args.static_weight, action_weight=args.action_weight,
                      contrastive_weight=args.contrastive_weight, triplet_margin=args.triplet_margin,
                      output_dir=Path(args.output_dir))


if __name__ == '__main__':
    main()