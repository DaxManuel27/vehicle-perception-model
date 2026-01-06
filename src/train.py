"""
Training script for CenterPoint-style LiDAR object detection.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from datetime import datetime

from dataset import WaymoDataset, collate_fn
from preprocess import batch_preprocess, CONFIG
from model import CenterPointModel
from loss import CenterPointLoss


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_hm_loss = 0
    total_box_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Preprocess batch
        processed = batch_preprocess(batch, CONFIG, device)
        
        # Forward pass
        heatmap_pred, box_pred = model(
            processed['pillar_features'],
            processed['pillar_coords']
        )
        
        # Compute loss
        loss, hm_loss, box_loss = loss_fn(
            heatmap_pred,
            box_pred,
            processed['heatmaps'],
            processed['box_targets'],
            processed['masks']
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_hm_loss += hm_loss.item()
        total_box_loss += box_loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"HM: {hm_loss.item():.4f} | "
                  f"Box: {box_loss.item():.4f}")
    
    return {
        'loss': total_loss / num_batches,
        'hm_loss': total_hm_loss / num_batches,
        'box_loss': total_box_loss / num_batches
    }


def validate(model, dataloader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_hm_loss = 0
    total_box_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            processed = batch_preprocess(batch, CONFIG, device)
            
            heatmap_pred, box_pred = model(
                processed['pillar_features'],
                processed['pillar_coords']
            )
            
            loss, hm_loss, box_loss = loss_fn(
                heatmap_pred,
                box_pred,
                processed['heatmaps'],
                processed['box_targets'],
                processed['masks']
            )
            
            total_loss += loss.item()
            total_hm_loss += hm_loss.item()
            total_box_loss += box_loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'hm_loss': total_hm_loss / max(num_batches, 1),
        'box_loss': total_box_loss / max(num_batches, 1)
    }


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': CONFIG
    }, path)
    print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train CenterPoint on Waymo LiDAR')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
                        # ... existing code ...
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    # ADD THESE TWO LINES:
    parser.add_argument('--gcs', action='store_true',
                        help='Stream data from GCS instead of local files')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Limit number of files to use')
    args = parser.parse_args()
# ... existing code ...
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        # ... existing code ...
    # Dataset
    print("\nLoading dataset...")
    # CHANGE THIS LINE:
    train_dataset = WaymoDataset(
        data_dir, 
        split='training',
        use_gcs=args.gcs,
        max_files=args.max_files
    )
# ... existing code ...
    print("Using CPU")
    
    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    checkpoint_dir = os.path.join(script_dir, '..', 'checkpoints')
    
    # Dataset
    print("\nLoading dataset...")
    train_dataset = WaymoDataset(data_dir, split='training')  # Using validation for now
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        collate_fn=collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Model
    print("\nInitializing model...")
    model = CenterPointModel(CONFIG).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = CenterPointLoss(heatmap_weight=1.0, box_weight=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"Grid size: {CONFIG['grid_w']}x{CONFIG['grid_h']}")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        print(f"Train - Loss: {train_metrics['loss']:.4f} | "
              f"HM: {train_metrics['hm_loss']:.4f} | "
              f"Box: {train_metrics['box_loss']:.4f}")
        
        # Update LR
        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'],
                          os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'],
                          os.path.join(checkpoint_dir, f'epoch_{epoch+1:03d}.pth'))
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs - 1, train_metrics['loss'],
                   os.path.join(checkpoint_dir, 'final_model.pth'))
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

