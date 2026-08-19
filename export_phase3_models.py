"""
Export Phase 3 Models - Copy this cell into Colab and run it
"""

# ============================================================================
# COLAB CELL: Export Phase 3 Models to Google Drive
# ============================================================================

import os
import shutil
from pathlib import Path

print("\n" + "="*80)
print("📥 EXPORTING PHASE 3 MODELS")
print("="*80 + "\n")

# Source directory
phase3_models_dir = Path('/content/Content-Moderation/phase3_vision_ocr/models')

# Destination on Google Drive
drive_export_dir = Path('/content/drive/My Drive/Phase3_Models_Export')
drive_export_dir.mkdir(exist_ok=True, parents=True)

print(f"Source: {phase3_models_dir}")
print(f"Destination: {drive_export_dir}\n")

# Models to export
models_to_export = ['nsfw', 'hate_symbols', 'violence']

for model_name in models_to_export:
    src = phase3_models_dir / model_name
    dst = drive_export_dir / model_name
    
    if src.exists():
        print(f"📦 Copying {model_name}...")
        
        # Remove destination if it exists
        if dst.exists():
            shutil.rmtree(dst)
        
        # Copy directory
        shutil.copytree(src, dst)
        
        # Show size
        total_size = sum(f.stat().st_size for f in dst.rglob('*') if f.is_file())
        size_mb = total_size / (1024*1024)
        
        print(f"  ✅ Copied ({size_mb:.1f} MB)")
        
        # Show contents
        for file in dst.rglob('*'):
            if file.is_file():
                file_size = file.stat().st_size / (1024*1024)
                print(f"     - {file.name} ({file_size:.1f} MB)")
    else:
        print(f"⚠️  {model_name} model not found: {src}")

print("\n" + "="*80)
print("✅ EXPORT COMPLETE!")
print("="*80)
print(f"\n📁 Models exported to Google Drive:")
print(f"   Location: {drive_export_dir}")
print(f"\n📝 NEXT STEPS:")
print(f"   1. Go to: https://drive.google.com/drive/my-drive")
print(f"   2. Find: Phase3_Models_Export")
print(f"   3. Download the folder")
print(f"   4. Extract on your local machine")
print(f"\n💻 LOCAL USAGE:")
print(f"""
from image_moderator import ImageModerator

# Point to downloaded models
moderator = ImageModerator(
    image_models_dir='/path/to/Phase3_Models_Export'
)

# Moderate an image
result = moderator.moderate('image.jpg')
print(result['explanation'])
""")
print("="*80 + "\n")

# ============================================================================
# END OF COLAB CELL
# ============================================================================
