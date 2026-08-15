"""
Download and prepare the Jigsaw Toxic Comment Classification dataset.
"""

import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_jigsaw_dataset(output_dir: str = "./data"):
    """
    Download the Jigsaw Toxic Comment Classification dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading Jigsaw Toxic Comment dataset from HuggingFace mirror...")
    
    # Load a public mirrored copy of the Jigsaw toxic comment corpus.
    try:
        dataset = load_dataset('darkcleopas/jigsaw-toxic-comment-multi-binary')
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Available splits: {list(dataset.keys())}")
        
        # Convert to pandas for easier inspection
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test']) if 'test' in dataset else None
        validation_df = pd.DataFrame(dataset['validation']) if 'validation' in dataset else None
        
        logger.info(f"\nTrain set shape: {train_df.shape}")
        if validation_df is not None:
            logger.info(f"Validation set shape: {validation_df.shape}")
        if test_df is not None:
            logger.info(f"Test set shape: {test_df.shape}")
        
        # Display column names and first few rows
        logger.info(f"\nColumns: {train_df.columns.tolist()}")
        logger.info(f"\nFirst few rows:\n{train_df.head()}")
        
        # Binarize soft crowd-sourced scores (e.g. 0.33, 0.5) to 0/1 labels
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        existing_labels = [col for col in label_cols if col in train_df.columns]
        for label_col in existing_labels:
            train_df[label_col] = (train_df[label_col] >= 0.5).astype(int)
        if validation_df is not None:
            for label_col in existing_labels:
                if label_col in validation_df.columns:
                    validation_df[label_col] = (validation_df[label_col] >= 0.5).astype(int)
        if test_df is not None:
            for label_col in existing_labels:
                if label_col in test_df.columns:
                    test_df[label_col] = (test_df[label_col] >= 0.5).astype(int)

        # Save to CSV for easy inspection
        train_df.to_csv(output_path / "train.csv", index=False)
        logger.info(f"Saved training data to {output_path / 'train.csv'}")
        
        if validation_df is not None:
            validation_df.to_csv(output_path / "validation.csv", index=False)
            logger.info(f"Saved validation data to {output_path / 'validation.csv'}")
        
        if test_df is not None:
            test_df.to_csv(output_path / "test.csv", index=False)
            logger.info(f"Saved test data to {output_path / 'test.csv'}")
        
        # Display label distribution
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Check which label columns exist
        existing_labels = [col for col in label_cols if col in train_df.columns]
        
        if existing_labels:
            logger.info("\n" + "="*50)
            logger.info("Label Distribution in Training Set")
            logger.info("="*50)
            for label in existing_labels:
                count = int(train_df[label].sum())
                percentage = (count / len(train_df)) * 100
                logger.info(f"{label:20s}: {count:6d} ({percentage:5.2f}%)")
        
        # Calculate class imbalance
        if existing_labels:
            logger.info("\n" + "="*50)
            logger.info("Class Imbalance Analysis")
            logger.info("="*50)
            clean_count = int(len(train_df[train_df[existing_labels].sum(axis=1) == 0]))
            toxic_count = int(len(train_df[train_df[existing_labels].sum(axis=1) > 0]))
            logger.info(f"Clean comments:  {clean_count:6d} ({(clean_count/len(train_df)*100):5.2f}%)")
            logger.info(f"Toxic comments:  {toxic_count:6d} ({(toxic_count/len(train_df)*100):5.2f}%)")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset from HuggingFace: {e}")
        logger.info("\nTrying alternative: downloading from Kaggle...")
        logger.info("Note: You'll need Kaggle API credentials (~/.kaggle/kaggle.json)")
        logger.info("To get credentials: https://www.kaggle.com/docs/api")
        
        try:
            import kaggle
            
            # Download from Kaggle
            kaggle.api.competition_download_files(
                'jigsaw-toxic-comment-classification-challenge',
                path=output_path
            )
            
            logger.info("Dataset downloaded from Kaggle successfully!")
            logger.info("Please unzip the files manually or use the following command:")
            logger.info(f"unzip {output_path}/*.zip -d {output_path}")
            
        except Exception as kaggle_error:
            logger.error(f"Kaggle download also failed: {kaggle_error}")
            logger.error("\nPlease download manually from:")
            logger.error("https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
            raise


def create_sample_dataset(output_dir: str = "./data", sample_size: int = 10000):
    """
    Create a smaller sample dataset for quick experimentation.
    
    Args:
        output_dir: Directory containing the full dataset
        sample_size: Number of samples to extract
    """
    output_path = Path(output_dir)
    
    logger.info(f"Creating sample dataset with {sample_size} examples...")
    
    train_df = pd.read_csv(output_path / "train.csv")
    
    # Stratified sampling to maintain label distribution
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    existing_labels = [col for col in label_cols if col in train_df.columns]
    
    if existing_labels:
        work_df = train_df.copy()
        work_df['is_toxic'] = work_df[existing_labels].sum(axis=1) > 0
        per_group = max(1, sample_size // 2)
        samples = []
        for _, group in work_df.groupby('is_toxic'):
            samples.append(group.sample(min(len(group), per_group), random_state=42))
        sample_df = pd.concat(samples, ignore_index=True)
        sample_df = sample_df.drop(columns='is_toxic', errors='ignore')
    else:
        sample_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    
    sample_df.to_csv(output_path / "train_sample.csv", index=False)
    logger.info(f"Sample dataset saved to {output_path / 'train_sample.csv'}")
    logger.info(f"Sample size: {len(sample_df)}")


def analyze_dataset(output_dir: str = "./data"):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        output_dir: Directory containing the dataset
    """
    output_path = Path(output_dir)
    train_df = pd.read_csv(output_path / "train.csv")
    
    logger.info("\n" + "="*50)
    logger.info("Dataset Analysis")
    logger.info("="*50)
    
    # Text length statistics
    train_df['text_length'] = train_df['comment_text'].str.len() if 'comment_text' in train_df.columns else 0
    
    logger.info(f"\nText Length Statistics:")
    logger.info(train_df['text_length'].describe())
    
    # Check for missing values
    logger.info(f"\nMissing Values:")
    logger.info(train_df.isnull().sum())
    
    # Multi-label statistics
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    existing_labels = [col for col in label_cols if col in train_df.columns]
    
    if existing_labels:
        train_df['num_labels'] = train_df[existing_labels].sum(axis=1)
        logger.info(f"\nNumber of Labels per Comment:")
        logger.info(train_df['num_labels'].value_counts().sort_index())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare Jigsaw dataset")
    parser.add_argument("--output-dir", type=str, default="./data", 
                        help="Output directory for the dataset")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create a smaller sample dataset")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Size of sample dataset")
    parser.add_argument("--analyze", action="store_true",
                        help="Perform dataset analysis")
    
    args = parser.parse_args()
    
    # Download dataset
    download_jigsaw_dataset(args.output_dir)
    
    # Create sample if requested
    if args.create_sample:
        create_sample_dataset(args.output_dir, args.sample_size)
    
    # Analyze if requested
    if args.analyze:
        analyze_dataset(args.output_dir)
