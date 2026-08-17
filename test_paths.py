"""Debug script to check paths and model files."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("🔍 DEBUGGING PATH DETECTION")
print("="*70)

# Test repo detection
from text_moderator import TextModerator

repo_dir = TextModerator._find_repo_dir()
print(f"\n✅ Detected repo directory: {repo_dir}")

repo_path = Path(repo_dir)
print(f"\n📂 Checking files in repo:")
print(f"  phase1_text_baseline exists: {(repo_path / 'phase1_text_baseline').exists()}")
print(f"  phase2_multilingual exists: {(repo_path / 'phase2_multilingual').exists()}")
print(f"  phase3_vision_ocr exists: {(repo_path / 'phase3_vision_ocr').exists()}")

# Check Phase 1 files
phase1_dir = repo_path / 'phase1_text_baseline'
print(f"\n📂 Phase 1 directory contents:")
if phase1_dir.exists():
    for item in phase1_dir.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
            if item.name == 'models':
                for model_file in item.iterdir():
                    size = model_file.stat().st_size / (1024*1024)  # Convert to MB
                    print(f"     📄 {model_file.name} ({size:.1f} MB)")
            elif item.name == 'configs':
                for config_file in item.iterdir():
                    print(f"     📄 {config_file.name}")
        else:
            print(f"  📄 {item.name}")

# Check config file
config_path = phase1_dir / 'configs' / 'baseline.yaml'
print(f"\n✅ Config file exists: {config_path.exists()}")
if config_path.exists():
    print(f"   Path: {config_path}")

# Check model file
model_path = phase1_dir / 'models' / 'best_model.pt'
print(f"\n✅ Model file exists: {model_path.exists()}")
if model_path.exists():
    size = model_path.stat().st_size / (1024*1024)  # Convert to MB
    print(f"   Path: {model_path}")
    print(f"   Size: {size:.1f} MB")
else:
    print(f"   ⚠️  Model file not found")
    print(f"   Expected: {model_path}")
    print(f"   Run: python phase1_text_baseline/train_classifier.py to train")

print("\n" + "="*70 + "\n")
