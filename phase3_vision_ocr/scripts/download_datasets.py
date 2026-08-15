#!/usr/bin/env python3
"""
Dataset Download and Preparation Script for Image Classification
Downloads and organizes datasets for NSFW, hate symbols, and violence detection
"""

import os
import argparse
import requests
from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import zipfile
import tarfile


class DatasetDownloader:
    """Download and organize image classification datasets"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Dataset information
        self.datasets = {
            'nsfw': {
                'name': 'NSFW Dataset',
                'categories': ['safe', 'nsfw'],
                'description': 'Safe vs NSFW image classification',
                'sources': [
                    'https://github.com/alex000kim/nsfw_data_scraper',
                    'https://github.com/GantMan/nsfw_model'
                ]
            },
            'hate_symbols': {
                'name': 'Hateful Memes',
                'categories': ['no_hate', 'hate'],
                'description': 'Hate speech detection in memes and images',
                'sources': [
                    'https://ai.facebook.com/tools/hatefulmemes/',
                    'https://github.com/facebookresearch/fine_grained_hateful_memes'
                ]
            },
            'violence': {
                'name': 'Violence Detection',
                'categories': ['no_violence', 'violence'],
                'description': 'Violent content detection',
                'sources': [
                    'https://gitlab.com/volzotan/violentscenesdataset'
                ]
            }
        }
    
    def download_file(self, url: str, output_path: Path, desc: str = "Downloading"):
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            self.logger.info(f"✅ Downloaded: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Failed to download {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path):
        """Extract zip or tar archive"""
        self.logger.info(f"Extracting {archive_path}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            self.logger.error(f"Unknown archive format: {archive_path}")
            return False
        
        self.logger.info(f"✅ Extracted to: {extract_to}")
        return True
    
    def create_dummy_dataset(self, dataset_name: str, num_samples: int = 100):
        """Create a dummy dataset with synthetic images for testing"""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        dataset_info = self.datasets[dataset_name]
        categories = dataset_info['categories']
        
        dataset_dir = self.output_dir / dataset_name
        
        print(f"\n📦 Creating dummy dataset: {dataset_info['name']}")
        print(f"   Categories: {', '.join(categories)}")
        print(f"   Samples per category: {num_samples}")
        
        for split in ['train', 'test']:
            for category in categories:
                category_dir = dataset_dir / split / category
                category_dir.mkdir(parents=True, exist_ok=True)
                
                samples_this_split = num_samples if split == 'train' else num_samples // 5
                
                for i in range(samples_this_split):
                    # Create synthetic image
                    img = Image.new('RGB', (224, 224), 
                                   color=tuple(np.random.randint(0, 255, 3).tolist()))
                    draw = ImageDraw.Draw(img)
                    
                    # Add text label
                    text = f"{category}\n{i+1}"
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
                    except:
                        font = ImageFont.load_default()
                    
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    x = (224 - text_width) // 2
                    y = (224 - text_height) // 2
                    
                    draw.text((x, y), text, fill='white', font=font)
                    
                    # Save
                    img_path = category_dir / f"{category}_{i:04d}.jpg"
                    img.save(img_path)
                
                print(f"   ✅ Created {samples_this_split} images: {split}/{category}")
        
        # Create metadata
        metadata = {
            'dataset': dataset_info['name'],
            'categories': categories,
            'num_classes': len(categories),
            'splits': {
                'train': {cat: num_samples for cat in categories},
                'test': {cat: num_samples // 5 for cat in categories}
            },
            'type': 'dummy',
            'description': dataset_info['description']
        }
        
        metadata_path = dataset_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved metadata: {metadata_path}")
        print(f"   📁 Dataset location: {dataset_dir}")
    
    def download_nsfw_dataset(self):
        """
        Download NSFW dataset
        Note: This is a placeholder - actual NSFW datasets require special handling
        """
        print("\n" + "="*80)
        print("📦 NSFW Dataset")
        print("="*80)
        print("\n⚠️  NSFW datasets require manual download and verification")
        print("\nRecommended sources:")
        print("  1. NSFW Data Scraper: https://github.com/alex000kim/nsfw_data_scraper")
        print("  2. Open NSFW Model: https://github.com/GantMan/nsfw_model")
        print("  3. Create your own using web scraping with proper filtering")
        
        print("\n💡 For now, creating a dummy dataset for development...")
        self.create_dummy_dataset('nsfw', num_samples=200)
    
    def download_hateful_memes(self):
        """
        Download Hateful Memes dataset from Facebook AI
        """
        print("\n" + "="*80)
        print("📦 Hateful Memes Dataset")
        print("="*80)
        
        print("\n⚠️  Hateful Memes dataset requires registration and agreement")
        print("\nTo download:")
        print("  1. Visit: https://ai.facebook.com/tools/hatefulmemes/")
        print("  2. Register and agree to terms")
        print("  3. Download dataset (includes images + annotations)")
        print("  4. Extract to: data/hate_symbols/")
        
        print("\n💡 For now, creating a dummy dataset for development...")
        self.create_dummy_dataset('hate_symbols', num_samples=200)
    
    def download_violence_dataset(self):
        """
        Download violence detection dataset
        """
        print("\n" + "="*80)
        print("📦 Violence Detection Dataset")
        print("="*80)
        
        print("\n⚠️  Violence datasets often require manual curation")
        print("\nRecommended sources:")
        print("  1. VSD2014: https://gitlab.com/volzotan/violentscenesdataset")
        print("  2. MediaEval: https://multimediaeval.github.io/")
        print("  3. UCFCRIME: https://www.crcv.ucf.edu/projects/real-world/")
        
        print("\n💡 For now, creating a dummy dataset for development...")
        self.create_dummy_dataset('violence', num_samples=200)
    
    def verify_dataset(self, dataset_name: str) -> Dict:
        """Verify dataset integrity and return statistics"""
        dataset_dir = self.output_dir / dataset_name
        
        if not dataset_dir.exists():
            return {'exists': False}
        
        stats = {
            'exists': True,
            'splits': {}
        }
        
        for split in ['train', 'test', 'val']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                split_stats = {}
                for category in split_dir.iterdir():
                    if category.is_dir():
                        images = list(category.glob('*.jpg')) + list(category.glob('*.png'))
                        split_stats[category.name] = len(images)
                
                if split_stats:
                    stats['splits'][split] = split_stats
        
        return stats
    
    def print_dataset_info(self):
        """Print information about all datasets"""
        print("\n" + "="*80)
        print("📊 DATASET STATUS")
        print("="*80)
        
        for dataset_name, info in self.datasets.items():
            print(f"\n📦 {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Categories: {', '.join(info['categories'])}")
            
            stats = self.verify_dataset(dataset_name)
            
            if stats['exists']:
                print(f"   Status: ✅ Dataset found")
                
                for split, categories in stats['splits'].items():
                    print(f"\n   {split.capitalize()} Split:")
                    for category, count in categories.items():
                        print(f"      • {category}: {count} images")
            else:
                print(f"   Status: ❌ Not found")
                print(f"   Location: {self.output_dir / dataset_name}")
    
    def setup_all_datasets(self, dummy: bool = True):
        """Setup all datasets"""
        print("\n" + "="*80)
        print("🚀 IMAGE CLASSIFICATION DATASETS SETUP")
        print("="*80)
        
        if dummy:
            print("\n💡 Creating dummy datasets for development...")
            self.create_dummy_dataset('nsfw', num_samples=200)
            self.create_dummy_dataset('hate_symbols', num_samples=200)
            self.create_dummy_dataset('violence', num_samples=200)
        else:
            self.download_nsfw_dataset()
            self.download_hateful_memes()
            self.download_violence_dataset()
        
        print("\n" + "="*80)
        self.print_dataset_info()
        print("="*80 + "\n")
        
        print("✅ Dataset setup complete!")
        print("\n📝 Next steps:")
        print("   1. Review dataset statistics above")
        print("   2. For production, replace dummy data with real datasets")
        print("   3. Run training: python scripts/train_image_model.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare image classification datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dummy datasets for testing
  python download_datasets.py --dummy
  
  # Show dataset information
  python download_datasets.py --info
  
  # Download specific dataset
  python download_datasets.py --dataset nsfw
  
  # Download all datasets (real)
  python download_datasets.py --all

Note: Many datasets require manual download due to:
  - Terms of service agreements
  - Content sensitivity (NSFW)
  - Research licenses
  - Large file sizes

For production use, manually download datasets and place in data/ directory.
        """
    )
    
    parser.add_argument('--output-dir', default='../data',
                       help='Output directory for datasets')
    parser.add_argument('--dataset', choices=['nsfw', 'hate_symbols', 'violence'],
                       help='Download specific dataset')
    parser.add_argument('--all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--dummy', action='store_true',
                       help='Create dummy datasets for testing')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset information')
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples per category for dummy datasets')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Initialize downloader
    downloader = DatasetDownloader(args.output_dir)
    
    # Execute commands
    if args.info:
        downloader.print_dataset_info()
    elif args.dummy:
        downloader.setup_all_datasets(dummy=True)
    elif args.all:
        downloader.setup_all_datasets(dummy=False)
    elif args.dataset:
        if args.dataset == 'nsfw':
            downloader.download_nsfw_dataset()
        elif args.dataset == 'hate_symbols':
            downloader.download_hateful_memes()
        elif args.dataset == 'violence':
            downloader.download_violence_dataset()
        
        downloader.print_dataset_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
