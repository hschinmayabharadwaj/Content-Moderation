"""
Phase 2: Prepare Multilingual and Code-Mix Datasets

This script downloads and prepares datasets for training:
- HASOC (Hate Speech and Offensive Content) - Hindi-English
- TRAC (Trolling, Aggression and Cyberbullying) - Hindi, Tamil, Bengali
- Custom code-mixed datasets
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import json
import requests
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualDatasetPreparer:
    """Prepare multilingual and code-mixed datasets for training."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets_info = {
            'hasoc2019': {
                'languages': ['hindi', 'english', 'hinglish'],
                'labels': ['hate', 'offensive', 'profane'],
                'url': None,  # Requires manual download from competition
                'description': 'HASOC 2019 - Hate Speech and Offensive Content Identification'
            },
            'trac2020': {
                'languages': ['hindi', 'tamil', 'bengali'],
                'labels': ['aggression', 'overtly_aggressive', 'covertly_aggressive'],
                'url': None,
                'description': 'TRAC 2020 - Trolling, Aggression and Cyberbullying'
            }
        }
    
    def create_sample_code_mix_dataset(self, size: int = 1000) -> pd.DataFrame:
        """
        Create a synthetic code-mixed dataset for testing.
        
        This is for demonstration purposes. In production, use real datasets.
        """
        logger.info(f"Creating sample code-mixed dataset ({size} examples)")
        
        # Sample templates with code-mixing patterns
        templates = {
            'toxic': [
                ("You are such a {hindi_insult} yaar", 1),
                ("Stop being {hindi_insult}, it's annoying", 1),
                ("What {hindi_bad} are you doing?", 1),
                ("Tu bada {english_insult} hai boss", 1),
                ("Kya {english_bad} bakwas hai ye", 1),
            ],
            'neutral': [
                ("Aaj main bahut {hindi_good} feel kar raha hoon", 0),
                ("This movie was {hindi_adj} good yaar", 0),
                ("Main kal {english_place} ja raha hoon", 0),
                ("Yeh {english_thing} bohot accha hai", 0),
                ("Aap kaise {english_verb} ho?", 0),
            ]
        }
        
        # Word banks
        hindi_words = {
            'insult': ['bewakoof', 'pagal', 'badtameez', 'nalayak'],
            'bad': ['ganda', 'bura', 'kharab', 'bekaar'],
            'good': ['khush', 'accha', 'mast', 'badiya'],
            'adj': ['zabardast', 'kamaal', 'shandar', 'badhiya']
        }
        
        english_words = {
            'insult': ['idiot', 'fool', 'jerk', 'stupid'],
            'bad': ['terrible', 'awful', 'horrible', 'bad'],
            'place': ['school', 'office', 'market', 'home'],
            'thing': ['phone', 'laptop', 'car', 'book'],
            'verb': ['working', 'doing', 'going', 'eating']
        }
        
        data = []
        np.random.seed(42)
        
        for i in range(size):
            # Choose toxic or neutral
            category = np.random.choice(['toxic', 'neutral'], p=[0.3, 0.7])
            template, label = np.random.choice(templates[category])
            
            # Fill in template
            text = template
            for key in ['hindi_insult', 'hindi_bad', 'hindi_good', 'hindi_adj']:
                if key in text:
                    word_type = key.split('_')[1]
                    if word_type in hindi_words:
                        text = text.replace(f'{{{key}}}', np.random.choice(hindi_words[word_type]))
            
            for key in ['english_insult', 'english_bad', 'english_place', 'english_thing', 'english_verb']:
                if key in text:
                    word_type = key.split('_')[1]
                    if word_type in english_words:
                        text = text.replace(f'{{{key}}}', np.random.choice(english_words[word_type]))
            
            data.append({
                'text': text,
                'label': label,
                'language': 'hinglish',
                'is_code_mixed': True
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Created {len(df)} samples ({df['label'].sum()} toxic)")
        
        return df
    
    def download_hasoc_instructions(self) -> str:
        """Provide instructions for downloading HASOC dataset."""
        instructions = """
        HASOC Dataset Download Instructions:
        ====================================
        
        The HASOC dataset requires registration and manual download.
        
        Option 1: HASOC 2019 (Hindi-English)
        -------------------------------------
        1. Visit: https://hasocfire.github.io/hasoc/2019/dataset.html
        2. Fill the form and request access
        3. Download the datasets
        4. Place files in: {output_dir}/hasoc2019/
           - hindi_train.csv
           - hindi_test.csv
           - english_train.csv
           - english_test.csv
        
        Option 2: HASOC 2020 (Hindi, Bengali, German)
        ----------------------------------------------
        1. Visit: https://hasocfire.github.io/hasoc/2020/dataset.html
        2. Similar process as above
        3. Place in: {output_dir}/hasoc2020/
        
        Option 3: Use our sample dataset for testing
        ---------------------------------------------
        We'll create a sample code-mixed dataset automatically.
        """
        
        return instructions.format(output_dir=self.output_dir)
    
    def download_trac_instructions(self) -> str:
        """Provide instructions for downloading TRAC dataset."""
        instructions = """
        TRAC Dataset Download Instructions:
        ===================================
        
        TRAC (Trolling, Aggression and Cyberbullying) datasets.
        
        Option 1: TRAC-2020
        -------------------
        1. Visit: https://sites.google.com/view/trac2/shared-task
        2. Download Hindi, Bengali, or Tamil datasets
        3. Place in: {output_dir}/trac2020/
        
        Option 2: Download from research papers
        ----------------------------------------
        Search for: "TRAC Shared Task dataset" on Google Scholar
        Many papers provide links to the datasets.
        
        Dataset Format Expected:
        -----------------------
        CSV with columns: text, label
        Labels: NAG (Non-aggressive), CAG (Covertly aggressive), OAG (Overtly aggressive)
        """
        
        return instructions.format(output_dir=self.output_dir)
    
    def check_dataset_availability(self) -> Dict:
        """Check which datasets are available locally."""
        availability = {}
        
        # Check HASOC
        hasoc_dir = self.output_dir / 'hasoc2019'
        availability['hasoc2019'] = {
            'available': hasoc_dir.exists() and any(hasoc_dir.glob('*.csv')),
            'path': str(hasoc_dir) if hasoc_dir.exists() else None
        }
        
        # Check TRAC
        trac_dir = self.output_dir / 'trac2020'
        availability['trac2020'] = {
            'available': trac_dir.exists() and any(trac_dir.glob('*.csv')),
            'path': str(trac_dir) if trac_dir.exists() else None
        }
        
        return availability
    
    def load_hasoc_dataset(self, language: str = 'hindi') -> pd.DataFrame:
        """
        Load HASOC dataset if available.
        
        Args:
            language: 'hindi', 'english', or 'hinglish'
        """
        hasoc_dir = self.output_dir / 'hasoc2019'
        
        if not hasoc_dir.exists():
            logger.warning(f"HASOC dataset not found at {hasoc_dir}")
            logger.info(self.download_hasoc_instructions())
            return None
        
        # Try to find relevant files
        train_file = hasoc_dir / f'{language}_train.csv'
        
        if not train_file.exists():
            logger.warning(f"Could not find {train_file}")
            return None
        
        try:
            df = pd.read_csv(train_file)
            logger.info(f"Loaded HASOC {language} dataset: {len(df)} examples")
            return df
        except Exception as e:
            logger.error(f"Error loading HASOC dataset: {e}")
            return None
    
    def prepare_combined_dataset(
        self,
        include_phase1_english: bool = True,
        include_sample_codemix: bool = True,
        sample_size: int = 5000
    ) -> pd.DataFrame:
        """
        Prepare a combined multilingual dataset.
        
        Args:
            include_phase1_english: Include Jigsaw English data
            include_sample_codemix: Include synthetic code-mixed data
            sample_size: Size of sample dataset
        
        Returns:
            Combined DataFrame ready for training
        """
        logger.info("Preparing combined multilingual dataset...")
        
        datasets = []
        
        # 1. Include English data from Phase 1 if requested
        if include_phase1_english:
            phase1_data = Path('../phase1_text_baseline/data/train.csv')
            if phase1_data.exists():
                logger.info("Loading Phase 1 English data...")
                df_english = pd.read_csv(phase1_data)
                
                # Sample to balance with other languages
                if len(df_english) > sample_size:
                    df_english = df_english.sample(sample_size, random_state=42)
                
                # Standardize format
                df_english['language'] = 'english'
                df_english['is_code_mixed'] = False
                
                # Create binary toxic label (any toxic category)
                label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                if all(col in df_english.columns for col in label_cols):
                    df_english['label'] = (df_english[label_cols].sum(axis=1) > 0).astype(int)
                    df_english = df_english[['comment_text', 'label', 'language', 'is_code_mixed']]
                    df_english.rename(columns={'comment_text': 'text'}, inplace=True)
                    
                    datasets.append(df_english)
                    logger.info(f"  Added {len(df_english)} English examples")
            else:
                logger.warning(f"Phase 1 data not found at {phase1_data}")
        
        # 2. Include sample code-mixed data
        if include_sample_codemix:
            logger.info("Creating sample code-mixed data...")
            df_codemix = self.create_sample_code_mix_dataset(sample_size)
            datasets.append(df_codemix)
            logger.info(f"  Added {len(df_codemix)} code-mixed examples")
        
        # 3. Try to load real datasets
        availability = self.check_dataset_availability()
        
        if availability['hasoc2019']['available']:
            logger.info("Loading HASOC datasets...")
            for lang in ['hindi', 'hinglish']:
                df = self.load_hasoc_dataset(lang)
                if df is not None:
                    # Standardize format (adjust based on actual HASOC format)
                    # This is a template - adjust column names as needed
                    if 'text' in df.columns and 'task_1' in df.columns:
                        df['label'] = (df['task_1'] == 'HOF').astype(int)  # HOF = Hate and Offensive
                        df['language'] = lang
                        df['is_code_mixed'] = (lang == 'hinglish')
                        df = df[['text', 'label', 'language', 'is_code_mixed']]
                        datasets.append(df)
                        logger.info(f"  Added {len(df)} {lang} examples from HASOC")
        
        # 4. Combine all datasets
        if not datasets:
            logger.error("No datasets available!")
            logger.info("\nTo use real datasets:")
            logger.info(self.download_hasoc_instructions())
            logger.info("\n" + self.download_trac_instructions())
            return None
        
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"\nCombined dataset statistics:")
        logger.info(f"  Total examples: {len(combined_df)}")
        logger.info(f"  Toxic: {combined_df['label'].sum()} ({combined_df['label'].mean()*100:.1f}%)")
        logger.info(f"  Languages: {combined_df['language'].value_counts().to_dict()}")
        logger.info(f"  Code-mixed: {combined_df['is_code_mixed'].sum()}")
        
        # Save
        output_file = self.output_dir / 'multilingual_train.csv'
        combined_df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Saved combined dataset to {output_file}")
        
        return combined_df
    
    def create_train_val_split(
        self,
        df: pd.DataFrame,
        val_size: float = 0.1,
        stratify_by_language: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified train/validation split."""
        from sklearn.model_selection import train_test_split
        
        if stratify_by_language:
            # Stratify by language and label
            df['stratify_key'] = df['language'] + '_' + df['label'].astype(str)
            
            train_df, val_df = train_test_split(
                df,
                test_size=val_size,
                random_state=42,
                stratify=df['stratify_key']
            )
            
            train_df = train_df.drop('stratify_key', axis=1)
            val_df = val_df.drop('stratify_key', axis=1)
        else:
            train_df, val_df = train_test_split(
                df,
                test_size=val_size,
                random_state=42,
                stratify=df['label']
            )
        
        logger.info(f"\nTrain/Val split:")
        logger.info(f"  Train: {len(train_df)} examples")
        logger.info(f"  Val:   {len(val_df)} examples")
        
        # Save splits
        train_df.to_csv(self.output_dir / 'multilingual_train_split.csv', index=False)
        val_df.to_csv(self.output_dir / 'multilingual_val_split.csv', index=False)
        
        return train_df, val_df


def main():
    """Main dataset preparation workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare multilingual datasets")
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for datasets')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='Sample size for synthetic data')
    parser.add_argument('--include-phase1', action='store_true',
                       help='Include Phase 1 English data')
    parser.add_argument('--check-availability', action='store_true',
                       help='Check which datasets are available')
    
    args = parser.parse_args()
    
    preparer = MultilingualDatasetPreparer(args.output_dir)
    
    if args.check_availability:
        print("\n" + "="*60)
        print("Dataset Availability Check")
        print("="*60)
        
        availability = preparer.check_dataset_availability()
        for dataset, info in availability.items():
            status = "✓ Available" if info['available'] else "✗ Not found"
            print(f"\n{dataset}: {status}")
            if info['path']:
                print(f"  Path: {info['path']}")
        
        print("\n" + "="*60)
        print("Download Instructions")
        print("="*60)
        print(preparer.download_hasoc_instructions())
        print("\n" + preparer.download_trac_instructions())
        
        return
    
    # Prepare combined dataset
    print("\n" + "="*60)
    print("Preparing Multilingual Dataset")
    print("="*60)
    
    combined_df = preparer.prepare_combined_dataset(
        include_phase1_english=args.include_phase1,
        include_sample_codemix=True,
        sample_size=args.sample_size
    )
    
    if combined_df is not None:
        # Create train/val split
        train_df, val_df = preparer.create_train_val_split(combined_df)
        
        print("\n" + "="*60)
        print("✓ Dataset Preparation Complete")
        print("="*60)
        print(f"\nReady to train Phase 2 model!")
        print(f"Training data: {len(train_df)} examples")
        print(f"Validation data: {len(val_df)} examples")
        print(f"\nNext step: python train_multilingual.py")


if __name__ == "__main__":
    main()
