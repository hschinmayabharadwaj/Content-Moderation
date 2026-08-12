"""Shared data utilities for the content moderation system."""

import re
import string
import numpy as np
from typing import List, Dict
import pandas as pd


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Clean and preprocess text.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    return text


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_punctuation: Whether to keep punctuation marks
    
    Returns:
        Text with special characters removed
    """
    if keep_punctuation:
        # Keep letters, numbers, spaces, and punctuation
        pattern = r'[^a-zA-Z0-9\s' + re.escape(string.punctuation) + ']'
    else:
        # Keep only letters, numbers, and spaces
        pattern = r'[^a-zA-Z0-9\s]'
    
    return re.sub(pattern, '', text)


def compute_class_weights(labels: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        labels: Binary label matrix of shape (n_samples, n_classes)
        method: Method to compute weights ('balanced', 'inverse_freq')
    
    Returns:
        Array of class weights
    """
    n_samples, n_classes = labels.shape
    
    if method == 'balanced':
        # Balanced weighting
        pos_counts = labels.sum(axis=0)
        neg_counts = n_samples - pos_counts
        weights = neg_counts / (pos_counts + 1e-8)
    elif method == 'inverse_freq':
        # Inverse frequency
        pos_counts = labels.sum(axis=0)
        weights = n_samples / (pos_counts + 1e-8)
    else:
        weights = np.ones(n_classes)
    
    # Normalize
    weights = weights / weights.sum() * n_classes
    
    return weights


def create_train_val_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    stratify_column: str = None,
    random_state: int = 42
) -> tuple:
    """
    Create train/validation split.
    
    Args:
        df: Input dataframe
        val_size: Proportion of validation set
        stratify_column: Column to stratify by
        random_state: Random seed
    
    Returns:
        train_df, val_df
    """
    from sklearn.model_selection import train_test_split
    
    if stratify_column and stratify_column in df.columns:
        stratify = df[stratify_column]
    else:
        stratify = None
    
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify
    )
    
    return train_df, val_df


def augment_text_with_synonyms(text: str, aug_prob: float = 0.1) -> str:
    """
    Augment text by replacing words with synonyms.
    
    This is a placeholder implementation. For production use,
    consider using libraries like nlpaug or textaugment.
    
    Args:
        text: Input text
        aug_prob: Probability of replacing each word
    
    Returns:
        Augmented text
    """
    # Simple word-level augmentation
    # In practice, use WordNet or contextual embeddings for better synonyms
    words = text.split()
    
    # Common toxic word variants (for demonstration)
    synonym_map = {
        'hate': ['dislike', 'despise', 'loathe'],
        'stupid': ['dumb', 'foolish', 'idiotic'],
        'ugly': ['unattractive', 'hideous'],
    }
    
    augmented_words = []
    for word in words:
        if np.random.random() < aug_prob and word.lower() in synonym_map:
            synonyms = synonym_map[word.lower()]
            augmented_words.append(np.random.choice(synonyms))
        else:
            augmented_words.append(word)
    
    return ' '.join(augmented_words)


def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """
    Batch texts for processing.
    
    Args:
        texts: List of texts
        batch_size: Batch size
    
    Returns:
        List of batches
    """
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


def truncate_text(text: str, max_length: int = 512, tokenizer=None) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length (in tokens if tokenizer provided, else characters)
        tokenizer: Optional tokenizer
    
    Returns:
        Truncated text
    """
    if tokenizer:
        # Truncate by tokens
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            text = tokenizer.convert_tokens_to_string(tokens)
    else:
        # Truncate by characters
        text = text[:max_length]
    
    return text


def analyze_label_distribution(df: pd.DataFrame, label_columns: List[str]) -> Dict:
    """
    Analyze label distribution in dataset.
    
    Args:
        df: Dataframe with label columns
        label_columns: List of label column names
    
    Returns:
        Dictionary with distribution statistics
    """
    stats = {}
    
    for label in label_columns:
        if label in df.columns:
            pos_count = df[label].sum()
            total_count = len(df)
            
            stats[label] = {
                'positive': int(pos_count),
                'negative': int(total_count - pos_count),
                'positive_ratio': float(pos_count / total_count),
                'imbalance_ratio': float((total_count - pos_count) / (pos_count + 1e-8))
            }
    
    # Multi-label statistics
    label_matrix = df[label_columns].values
    num_labels_per_sample = label_matrix.sum(axis=1)
    
    stats['multi_label'] = {
        'mean_labels_per_sample': float(num_labels_per_sample.mean()),
        'max_labels_per_sample': int(num_labels_per_sample.max()),
        'samples_with_no_labels': int((num_labels_per_sample == 0).sum()),
        'samples_with_multiple_labels': int((num_labels_per_sample > 1).sum())
    }
    
    return stats
