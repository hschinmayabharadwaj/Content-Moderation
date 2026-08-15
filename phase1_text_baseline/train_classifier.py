"""
Phase 1.1: Train Baseline Multi-Label Text Classifier

This script fine-tunes a pre-trained transformer model (BERT/RoBERTa) on the
Jigsaw Toxic Comment Classification dataset for multi-label classification.
"""

import os
import yaml
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report
)

from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ToxicCommentDataset(Dataset):
    """Dataset class for toxic comment classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.FloatTensor(labels)
        }


class ToxicCommentClassifier(nn.Module):
    """
    Multi-label toxic comment classifier based on pre-trained transformers.
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        hidden_size: int = 256,
        use_intermediate_layer: bool = True
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        
        if use_intermediate_layer:
            self.intermediate = nn.Linear(self.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.classifier = nn.Linear(hidden_size, num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, num_labels)
            self.intermediate = None
        
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification head
        if self.intermediate is not None:
            hidden = self.relu(self.intermediate(pooled_output))
            hidden = self.dropout(hidden)
            logits = self.classifier(hidden)
        else:
            logits = self.classifier(pooled_output)
        
        outputs = {'logits': logits}
        
        if labels is not None:
            # Calculate loss
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            outputs['loss'] = loss
        
        return outputs


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        return focal_loss.mean()


def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    
    data_dir = Path(config['training']['train_file']).parent
    
    # Load training data
    if config['training'].get('use_sample', False):
        train_file = data_dir / 'train_sample.csv'
        logger.info(f"Loading sample training data from {train_file}")
    else:
        train_file = data_dir / 'train.csv'
        logger.info(f"Loading full training data from {train_file}")
    
    train_df = pd.read_csv(train_file)
    
    # Load validation data
    val_file = Path(config['training']['validation_file'])
    if val_file.exists():
        val_df = pd.read_csv(val_file)
        logger.info(f"Loaded validation data from {val_file}")
    else:
        # Split training data if validation doesn't exist
        logger.info("Validation file not found. Splitting training data...")
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=config['seed']
        )
    
    # Load test data if available
    test_file = Path(config['training']['test_file'])
    test_df = None
    if test_file.exists():
        test_df = pd.read_csv(test_file)
        logger.info(f"Loaded test data from {test_file}")
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    if test_df is not None:
        logger.info(f"Test size: {len(test_df)}")
    
    return train_df, val_df, test_df


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculate class weights for handling imbalanced data."""
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    weights = neg_counts / (pos_counts + 1e-5)
    return torch.FloatTensor(weights)


def binarize_labels(
    labels: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Convert soft/crowd-sourced label scores to hard 0/1 targets."""
    labels = np.asarray(labels, dtype=np.float32)
    if labels.size == 0:
        return labels

    unique_values = np.unique(labels[~np.isnan(labels)])
    if len(unique_values) > 2 or np.any(
        (unique_values != 0) & (unique_values != 1)
    ):
        logger.warning(
            "Detected non-binary label values (e.g. crowd proportions). "
            f"Binarizing with threshold={threshold}."
        )

    return (labels >= threshold).astype(np.float32)


def prepare_data(
    df: pd.DataFrame,
    config: Dict,
    tokenizer
) -> Dataset:
    """Prepare dataset from dataframe."""
    
    text_col = config['labels']['text_column']
    label_cols = config['labels']['names']
    label_threshold = config.get('metrics', {}).get('threshold', 0.5)
    
    texts = df[text_col].fillna('').tolist()
    labels = binarize_labels(
        df[label_cols].values,
        threshold=label_threshold
    )
    
    dataset = ToxicCommentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    
    return dataset


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    scaler,
    use_amp: bool = True,
    use_focal_loss: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
):
    """Train for one epoch."""
    
    model.train()
    total_loss = 0
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if use_focal_loss:
                    loss = criterion(outputs['logits'], labels)
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(outputs['logits'], labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if use_focal_loss:
                loss = criterion(outputs['logits'], labels)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(outputs['logits'], labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    label_names: List[str]
) -> Dict:
    """Evaluate model on validation/test set."""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs['logits']
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_logits.append(probs.cpu().numpy())
    
    all_preds = np.vstack(all_preds).astype(np.int64)
    all_labels = np.vstack(all_labels)
    all_logits = np.vstack(all_logits)
    all_labels_bin = binarize_labels(all_labels).astype(np.int64)
    
    # Calculate metrics
    metrics = {}
    
    # Per-label metrics
    for i, label_name in enumerate(label_names):
        label_preds = all_preds[:, i]
        label_labels = all_labels_bin[:, i]
        label_logits = all_logits[:, i]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_labels, label_preds, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(label_labels, label_logits)
        except ValueError:
            auc = 0.0
        
        metrics[f'{label_name}_precision'] = precision
        metrics[f'{label_name}_recall'] = recall
        metrics[f'{label_name}_f1'] = f1
        metrics[f'{label_name}_auc'] = auc
    
    # Overall metrics (micro-average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_bin.ravel(), all_preds.ravel(), average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels_bin, all_logits, average='macro')
    except ValueError:
        auc = 0.0
    
    metrics['overall_precision'] = precision
    metrics['overall_recall'] = recall
    metrics['overall_f1'] = f1
    metrics['overall_auc'] = auc
    
    # Save predictions for calibration (Step 1.2)
    metrics['predictions'] = all_logits
    metrics['labels'] = all_labels_bin
    
    return metrics


def main(config_path: str):
    """Main training function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Model: {config['model']['name']}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Set device
    if config['training']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif config['training']['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Load data
    train_df, val_df, test_df = load_data(config)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    logger.info(f"Tokenizer loaded: {config['model']['name']}")
    
    # Prepare datasets
    train_dataset = prepare_data(train_df, config, tokenizer)
    val_dataset = prepare_data(val_df, config, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    model = ToxicCommentClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        dropout=config['model']['dropout'],
        hidden_size=config['model']['hidden_size'],
        use_intermediate_layer=config['model']['use_intermediate_layer']
    )
    
    model.to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    
    if config['training']['scheduler'] == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
    elif config['training']['scheduler'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=config['training']['mixed_precision'],
            use_focal_loss=config['training']['focal_loss'],
            focal_alpha=config['training']['focal_alpha'],
            focal_gamma=config['training']['focal_gamma']
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            label_names=config['labels']['names']
        )
        
        logger.info(f"Val F1: {val_metrics['overall_f1']:.4f}, "
                   f"Val AUC: {val_metrics['overall_auc']:.4f}")
        
        # Per-label metrics
        for label in config['labels']['names']:
            logger.info(f"  {label:15s} - F1: {val_metrics[f'{label}_f1']:.4f}, "
                       f"AUC: {val_metrics[f'{label}_auc']:.4f}")
        
        # Save best model
        if val_metrics['overall_f1'] > best_f1:
            best_f1 = val_metrics['overall_f1']
            patience_counter = 0
            
            # Save model
            model_save_path = save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': val_metrics
            }, model_save_path)
            
            # Save predictions for calibration
            np.save(save_dir / 'val_predictions.npy', val_metrics['predictions'])
            np.save(save_dir / 'val_labels.npy', val_metrics['labels'])
            
            logger.info(f"✓ Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if config['training']['early_stopping'] and \
           patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training completed! Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to {save_dir / 'best_model.pt'}")
    logger.info(f"Validation predictions saved for calibration (Step 1.2)")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline text classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="phase1_text_baseline/configs/baseline.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args.config)
