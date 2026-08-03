import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path

from train_dataset import get_dataloaders
from model import SignTransformer


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, writer=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}', leave=False)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    avg_loss = total_loss / total
    acc = correct / total
    
    if writer:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/Acc', acc, epoch)
    
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, writer=None, prefix='Val'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    for x, y in tqdm(loader, desc=f'{prefix} Epoch {epoch}', leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
    
    avg_loss = total_loss / total
    acc = correct / total
    
    if writer:
        writer.add_scalar(f'{prefix}/Loss', avg_loss, epoch)
        writer.add_scalar(f'{prefix}/Acc', acc, epoch)
    
    # Per-class accuracy
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    per_class_acc = {}
    for c in range(model.classifier.out_features):
        mask = all_labels == c
        if mask.any():
            per_class_acc[c] = (all_preds[mask] == c).float().mean().item()
    
    return avg_loss, acc, per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nslt100', 
                        choices=['nslt100', 'nslt300', 'nslt1000', 'nslt2000', 'wlasl'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--min-samples', type=int, default=2)
    parser.add_argument('--no-oversample', action='store_true')
    parser.add_argument('--dropout-step', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='MODEL/checkpoints')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # Data
    print(f'\nLoading {args.dataset} dataset...')
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=4,
        oversample=not args.no_oversample,
        min_samples_per_class=args.min_samples
    )
    print(f'Classes: {num_classes}')
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}')
    
    # Model
    model = SignTransformer(
        num_classes=num_classes,
        dim=args.dim,
        max_len=args.max_len,
        dropout_step=args.dropout_step
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler: cosine annealing with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor(
            (epoch - warmup_epochs) / (args.epochs - warmup_epochs) * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Tensorboard
    output_dir = Path(args.output_dir) / f'{args.dataset}_dim{args.dim}'
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / 'logs')
    
    # Resume
    start_epoch = 0
    best_val_acc = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f'Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}')
    
    # Training loop
    print('\nStarting training...')
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                             optimizer, scaler, device, epoch, writer)
        
        # Validate
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, 
                                                      device, epoch, writer, 'Val')
        
        # Test (less frequent)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            test_loss, test_acc, _ = evaluate(model, test_loader, criterion, 
                                               device, epoch, writer, 'Test')
        else:
            test_acc = 0
        
        scheduler.step()
        
        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | '
              f'Val Loss={val_loss:.4f} Acc={val_acc:.4f} | '
              f'Test Acc={test_acc:.4f} | LR={lr:.2e}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_acc': best_val_acc,
                'args': vars(args),
            }, output_dir / 'best_model.pt')
            print(f'  -> New best model saved! Val Acc: {val_acc:.4f}')
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'args': vars(args),
        }, output_dir / 'latest_model.pt')
    
    writer.close()
    print(f'\nTraining complete! Best Val Acc: {best_val_acc:.4f}')
    print(f'Models saved to: {output_dir}')


if __name__ == '__main__':
    main()