"""
Phase 2.2: Train XLM-RoBERTa for Multilingual Content Moderation

Fine-tune XLM-RoBERTa on multilingual and code-mixed datasets.
Supports English, Hindi, Tamil, Telugu, Kannada, and code-mixed text.
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
    XLMRobertaTokenizer,
    XLMRobertaModel,
    XLMRobertaConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix
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


class MultilingualDataset(Dataset):
    """Dataset class for multilingual text classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        languages: List[str],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        language = self.languages[idx]
        
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
            'labels': torch.FloatTensor([label]),
            'language': language
        }


class MultilingualClassifier(nn.Module):
    """
    XLM-RoBERTa based multilingual classifier.
    
    Supports 100+ languages with shared embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_labels: int = 1,  # Binary classification
        dropout: float = 0.1,
        hidden_size: int = 256,
        use_language_adapter: bool = False
    ):
        super().__init__()
        
        self.config = XLMRobertaConfig.from_pretrained(model_name)
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.intermediate = nn.Linear(self.config.hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Optional: Language-specific adapter layers
        self.use_language_adapter = use_language_adapter
        if use_language_adapter:
            self.language_adapters = nn.ModuleDict({
                'english': nn.Linear(self.config.hidden_size, self.config.hidden_size),
                'hindi': nn.Linear(self.config.hidden_size, self.config.hidden_size),
                'hinglish': nn.Linear(self.config.hidden_size, self.config.hidden_size),
                'tamil': nn.Linear(self.config.hidden_size, self.config.hidden_size),
                'default': nn.Linear(self.config.hidden_size, self.config.hidden_size)
            })
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        languages=None
    ):
        # Get XLM-RoBERTa outputs
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Optional: Apply language-specific adapters
        if self.use_language_adapter and languages is not None:
            batch_adapted = []
            for i, lang in enumerate(languages):
                adapter = self.language_adapters.get(lang, self.language_adapters['default'])
                adapted = adapter(pooled_output[i:i+1])
                batch_adapted.append(adapted)
            pooled_output = torch.cat(batch_adapted, dim=0)
        
        # Classification head
        pooled_output = self.dropout(pooled_output)
        hidden = self.relu(self.intermediate(pooled_output))
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        
        result = {'logits': logits}
        
        if labels is not None:
            # Binary classification
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
            result['loss'] = loss
        
        return result


def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load multilingual training and validation datasets."""
    
    data_dir = Path(config['training']['data_dir'])
    
    # Load training data
    train_file = data_dir / 'multilingual_train_split.csv'
    if not train_file.exists():
        train_file = data_dir / 'multilingual_train.csv'
    
    if not train_file.exists():
        logger.error(f"Training file not found at {train_file}")
        logger.info("Please run: python prepare_datasets.py")
        raise FileNotFoundError(f"No training data at {train_file}")
    
    train_df = pd.read_csv(train_file)
    logger.info(f"Loaded training data: {len(train_df)} examples")
    
    # Load validation data
    val_file = data_dir / 'multilingual_val_split.csv'
    if val_file.exists():
        val_df = pd.read_csv(val_file)
        logger.info(f"Loaded validation data: {len(val_df)} examples")
    else:
        # Split if no separate validation file
        from sklearn.model_selection import train_test_split
        logger.info("No validation file found, splitting training data...")
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=config['seed'], stratify=train_df['label']
        )
    
    # Display language distribution
    logger.info(f"\nLanguage distribution (train):")
    for lang, count in train_df['language'].value_counts().items():
        pct = (count / len(train_df)) * 100
        logger.info(f"  {lang}: {count} ({pct:.1f}%)")
    
    return train_df, val_df


def prepare_data(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 512
) -> Dataset:
    """Prepare dataset from dataframe."""
    
    texts = df['text'].fillna('').tolist()
    labels = df['label'].values.astype(np.float32)
    languages = df['language'].fillna('unknown').tolist()
    
    dataset = MultilingualDataset(
        texts=texts,
        labels=labels,
        languages=languages,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return dataset


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    scaler,
    use_amp: bool = True
):
    """Train for one epoch."""
    
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        languages = batch['language']
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    languages=languages
                )
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                languages=languages
            )
            loss = outputs['loss']
            
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
    return_predictions: bool = False
) -> Dict:
    """Evaluate model on validation/test set."""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    all_languages = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        languages = batch['language']
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            languages=languages
        )
        
        logits = outputs['logits']
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_logits.append(probs.cpu().numpy())
        all_languages.extend(languages)
    
    all_preds = np.vstack(all_preds).squeeze()
    all_labels = np.vstack(all_labels).squeeze()
    all_logits = np.vstack(all_logits).squeeze()
    
    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_logits)
    except:
        auc = 0.0
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc)
    }
    
    # Per-language metrics
    unique_languages = list(set(all_languages))
    per_language_metrics = {}
    
    for lang in unique_languages:
        lang_mask = np.array([l == lang for l in all_languages])
        if lang_mask.sum() > 0:
            lang_preds = all_preds[lang_mask]
            lang_labels = all_labels[lang_mask]
            
            lang_precision, lang_recall, lang_f1, _ = precision_recall_fscore_support(
                lang_labels, lang_preds, average='binary', zero_division=0
            )
            
            per_language_metrics[lang] = {
                'count': int(lang_mask.sum()),
                'precision': float(lang_precision),
                'recall': float(lang_recall),
                'f1': float(lang_f1)
            }
    
    metrics['per_language'] = per_language_metrics
    
    if return_predictions:
        metrics['predictions'] = all_logits
        metrics['labels'] = all_labels
        metrics['languages'] = all_languages
    
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
    train_df, val_df = load_data(config)
    
    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(config['model']['name'])
    logger.info(f"Tokenizer loaded: {config['model']['name']}")
    
    # Prepare datasets
    train_dataset = prepare_data(train_df, tokenizer, config['model']['max_length'])
    val_dataset = prepare_data(val_df, tokenizer, config['model']['max_length'])
    
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
    model = MultilingualClassifier(
        model_name=config['model']['name'],
        num_labels=1,
        dropout=config['model']['dropout'],
        hidden_size=config['model']['hidden_size'],
        use_language_adapter=config['model'].get('use_language_adapter', False)
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
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Initialize gradient scaler
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
            use_amp=config['training']['mixed_precision']
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            return_predictions=True
        )
        
        logger.info(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        
        # Per-language metrics
        logger.info(f"\nPer-language performance:")
        for lang, lang_metrics in val_metrics['per_language'].items():
            logger.info(f"  {lang:15s} - F1: {lang_metrics['f1']:.4f}, "
                       f"Count: {lang_metrics['count']}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
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
            
            # Save predictions
            np.save(save_dir / 'val_predictions.npy', val_metrics['predictions'])
            np.save(save_dir / 'val_labels.npy', val_metrics['labels'])
            
            # Save per-language metrics
            import json
            with open(save_dir / 'per_language_metrics.json', 'w') as f:
                json.dump(val_metrics['per_language'], f, indent=2)
            
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
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multilingual classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/xlm_roberta.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args.config)
