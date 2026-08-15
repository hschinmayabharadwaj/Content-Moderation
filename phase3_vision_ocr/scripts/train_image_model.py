#!/usr/bin/env python3
"""
Train Image Classifier for NSFW, Hate Symbols, or Violence Detection
"""

import sys
from pathlib import Path
import argparse
import logging
import time
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_classifier import (
    ImageClassifier,
    ImageClassificationDataset,
    get_transforms,
    compute_class_weights
)


class Trainer:
    """Training manager for image classifiers"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: Path
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.logger = logging.getLogger(__name__)
        
        # Setup criterion
        if config.get('use_class_weights', True):
            class_weights = compute_class_weights(train_loader.dataset)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 20)
        )
        
        # Training state
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config,
            'backbone': self.model.backbone_name,
            'categories': self.train_loader.dataset.categories
        }
        
        # Save latest
        latest_path = self.output_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"✅ Saved best model: {best_path}")
    
    def train(self, num_epochs: int):
        """Full training loop"""
        self.logger.info("🚀 Starting training")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Epochs: {num_epochs}")
        self.logger.info(f"   Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"   Val samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, is_best)
        
        total_time = time.time() - start_time
        
        self.logger.info(f"\n✅ Training complete!")
        self.logger.info(f"   Best val accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"   Total time: {total_time/60:.1f} minutes")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train image classifier")
    
    parser.add_argument('--dataset', required=True,
                       choices=['nsfw', 'hate_symbols', 'violence'],
                       help='Dataset to train on')
    parser.add_argument('--data-dir', default='../data',
                       help='Data directory')
    parser.add_argument('--output-dir', default='../models',
                       help='Output directory for models')
    
    # Model arguments
    parser.add_argument('--backbone', default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'vit_b_16'],
                       help='Model backbone')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Data arguments
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--use-class-weights', action='store_true', default=True)
    parser.add_argument('--no-class-weights', dest='use_class_weights', action='store_false')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Print configuration
    print("\n" + "="*80)
    print("🎯 IMAGE CLASSIFIER TRAINING")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80 + "\n")
    
    # Setup paths
    data_dir = Path(args.data_dir) / args.dataset
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        logger.info("Run: python scripts/download_datasets.py --dummy")
        return
    
    # Create datasets
    logger.info("Loading datasets...")
    
    train_transform = get_transforms('train', args.image_size)
    val_transform = get_transforms('test', args.image_size)
    
    try:
        train_dataset = ImageClassificationDataset(
            data_dir, 'train', train_transform
        )
        val_dataset = ImageClassificationDataset(
            data_dir, 'test', val_transform,
            categories=train_dataset.categories
        )
    except ValueError as e:
        logger.error(f"❌ {e}")
        return
    
    logger.info(f"✅ Train dataset: {len(train_dataset)} images")
    logger.info(f"✅ Val dataset: {len(val_dataset)} images")
    logger.info(f"   Categories: {train_dataset.categories}")
    
    # Print class distribution
    train_counts = train_dataset.get_class_counts()
    logger.info("\n   Class distribution:")
    for cat, count in train_counts.items():
        logger.info(f"      {cat}: {count}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info(f"\nCreating model: {args.backbone}")
    model = ImageClassifier(
        backbone=args.backbone,
        num_classes=len(train_dataset.categories),
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parameters: {num_params:,}")
    
    # Create config
    config = {
        'dataset': args.dataset,
        'backbone': args.backbone,
        'num_classes': len(train_dataset.categories),
        'categories': train_dataset.categories,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'image_size': args.image_size,
        'pretrained': args.pretrained,
        'use_class_weights': args.use_class_weights,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save config
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Saved config: {config_path}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir
    )
    
    # Train
    print("\n" + "="*80)
    trainer.train(args.epochs)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
