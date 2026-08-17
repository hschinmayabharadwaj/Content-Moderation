"""
Simple script to help diagnose and fix Colab training issues
Run this in Colab to see exactly what's happening
"""

import os
import sys
from pathlib import Path

print("\n" + "="*80)
print("🔍 COLAB TRAINING DIAGNOSTICS")
print("="*80)

# Check environment
print("\n1️⃣ ENVIRONMENT CHECK:")
print(f"  Current dir: {os.getcwd()}")
print(f"  Repository exists: {Path('/content/Content-Moderation').exists()}")

if Path('/content/Content-Moderation').exists():
    os.chdir('/content/Content-Moderation')
    print(f"  ✅ Changed to: {os.getcwd()}")
else:
    print("  ❌ Repository not found!")
    print("     Run: !git clone https://github.com/hschinmayabharadwaj/Content-Moderation.git")
    sys.exit(1)

# Check Phase 1
print("\n2️⃣ PHASE 1 STATUS:")
phase1_dir = Path('phase1_text_baseline')
if phase1_dir.exists():
    config_file = phase1_dir / 'configs' / 'baseline.yaml'
    model_file = phase1_dir / 'models' / 'best_model.pt'
    
    print(f"  Directory exists: ✅")
    print(f"  Config file exists: {'✅' if config_file.exists() else '❌'}")
    print(f"  Model file exists: {'✅' if model_file.exists() else '❌'}")
    
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024*1024)
        print(f"  Model size: {size_mb:.1f} MB")
    else:
        print(f"  ⚠️  Model not found. Need to train.")
        print(f"     Command: python phase1_text_baseline/train_classifier.py --config phase1_text_baseline/configs/baseline.yaml")
else:
    print("  ❌ Phase 1 directory not found!")

# Check Phase 2
print("\n3️⃣ PHASE 2 STATUS:")
phase2_dir = Path('phase2_multilingual')
if phase2_dir.exists():
    model_file = phase2_dir / 'models' / 'best_model.pt'
    
    print(f"  Directory exists: ✅")
    print(f"  Model file exists: {'✅' if model_file.exists() else '❌'}")
    
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024*1024)
        print(f"  Model size: {size_mb:.1f} MB")
else:
    print("  ❌ Phase 2 directory not found!")

# Check Phase 3
print("\n4️⃣ PHASE 3 STATUS:")
phase3_dir = Path('phase3_vision_ocr')
if phase3_dir.exists():
    models_dir = phase3_dir / 'models'
    datasets_dir = phase3_dir / 'datasets'
    
    print(f"  Directory exists: ✅")
    
    if models_dir.exists():
        nsfw_model = models_dir / 'nsfw' / 'best_model.pt'
        hate_model = models_dir / 'hate_symbols' / 'best_model.pt'
        violence_model = models_dir / 'violence' / 'best_model.pt'
        
        print(f"  NSFW model exists: {'✅' if nsfw_model.exists() else '❌'}")
        print(f"  Hate symbols model exists: {'✅' if hate_model.exists() else '❌'}")
        print(f"  Violence model exists: {'✅' if violence_model.exists() else '❌'}")
    
    if datasets_dir.exists():
        train_count = len(list(datasets_dir.glob('*/train/*/*.jpg')))
        print(f"  Training images: {train_count}")
else:
    print("  ❌ Phase 3 directory not found!")

# Check Google Drive
print("\n5️⃣ GOOGLE DRIVE STATUS:")
if Path('/content/drive/My Drive').exists():
    backup_dir = Path('/content/drive/My Drive/ContentModeration_TrainedModels')
    print(f"  Google Drive mounted: ✅")
    print(f"  Backup directory exists: {'✅' if backup_dir.exists() else '❌'}")
    
    if backup_dir.exists():
        items = list(backup_dir.iterdir())
        print(f"  Items in backup: {len(items)}")
        for item in items:
            print(f"    - {item.name}")
else:
    print(f"  Google Drive mounted: ❌")
    print(f"  Run this to mount: !python -c \"from google.colab import drive; drive.mount('/content/drive')\"")

print("\n" + "="*80)
print("\n✅ NEXT STEPS:")
print("  1. If models are missing, run training scripts individually:")
print("     python phase1_text_baseline/train_classifier.py --config phase1_text_baseline/configs/baseline.yaml")
print("  2. After training, test inference:")
print("     python test_paths.py")
print("\n" + "="*80 + "\n")
