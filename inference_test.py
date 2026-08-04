#!/usr/bin/env python
"""
Quick inference test for trained SignTransformer model.
Loads best checkpoint and runs on validation set samples.
"""
import torch
import torch.nn.functional as F
from pathlib import Path

from model import SignTransformer
from train_dataset import get_dataloaders


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
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
    return model, args


def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load best model
    ckpt_dir = Path("MODEL/checkpoints/nslt100_dim192")
    best_path = ckpt_dir / "best_model.pt"
    latest_path = ckpt_dir / "latest_model.pt"
    
    if best_path.exists():
        print(f"Loading best model: {best_path}")
        model, args = load_model(best_path, device)
    elif latest_path.exists():
        print(f"Loading latest model: {latest_path}")
        model, args = load_model(latest_path, device)
    else:
        print("No checkpoint found!")
        return
    
    print(f"Model config: {args}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset='nslt100',
        batch_size=16,
        max_len=64,
        num_workers=0,
        persistent_workers=False,
        oversample=False,
        min_samples_per_class=2
    )
    
    # Test on a few validation batches
    model.eval()
    correct = 0
    total = 0
    
    print("\nRunning inference on validation set...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            if batch_idx >= 5:  # Test first 5 batches
                break
            
            x, y = x.to(device), y.to(device)
            
            logits, emb = model(x, return_embedding=True)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(1)
            
            batch_correct = (pred == y).sum().item()
            batch_total = y.size(0)
            correct += batch_correct
            total += batch_total
            
            print(f"\nBatch {batch_idx}: Acc = {batch_correct}/{batch_total} = {batch_correct/batch_total:.2%}")
            
            # Show top-3 predictions for first few samples
            for i in range(min(3, batch_total)):
                top3 = probs[i].topk(3)
                print(f"  Sample {i}: True={y[i].item()}, Pred={pred[i].item()} ({probs[i, pred[i]].item():.2%})")
                print(f"    Top-3: {[(idx.item(), f'{p.item():.2%}') for idx, p in zip(top3.indices, top3.values)]}")
    
    print(f"\nOverall validation accuracy (first 5 batches): {correct}/{total} = {correct/total:.2%}")
    
    # Test embedding quality
    print("\n--- Embedding Analysis ---")
    with torch.no_grad():
        all_embs = []
        all_labels = []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, emb = model(x, return_embedding=True)
            all_embs.append(emb.cpu())
            all_labels.append(y.cpu())
            if len(all_embs) >= 3:
                break
        
        all_embs = torch.cat(all_embs)
        all_labels = torch.cat(all_labels)
        
        # Intra-class distance
        for c in all_labels.unique():
            mask = all_labels == c
            emb_c = all_embs[mask]
            if len(emb_c) > 1:
                intra = torch.cdist(emb_c, emb_c).triu(diagonal=1).mean()
                print(f"Class {c.item()}: intra-class dist = {intra.item():.4f}, samples = {len(emb_c)}")


if __name__ == '__main__':
    run_inference()