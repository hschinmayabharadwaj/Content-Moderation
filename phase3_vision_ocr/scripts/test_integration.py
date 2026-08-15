#!/usr/bin/env python3
"""
Test integrated image text moderation on all test images
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_text_moderator import ImageTextModerator


def test_all_images():
    """Test moderation on all images in test_images/"""
    test_images_dir = Path(__file__).parent.parent / "test_images"
    
    print("\n" + "="*80)
    print("🧪 INTEGRATED IMAGE TEXT MODERATION TEST")
    print("="*80)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = sorted([
        f for f in test_images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"❌ No images found in {test_images_dir}")
        return
    
    print(f"\n📁 Found {len(image_files)} test images")
    
    # Initialize moderator
    print("\n🚀 Initializing moderator...")
    moderator = ImageTextModerator(
        models_dir='../../content_moderation_trained',
        ocr_engine='easyocr'
    )
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")
    
    results = []
    
    for image_file in image_files:
        print(f"📸 {image_file.name}")
        print("   ", end="")
        
        try:
            result = moderator.moderate_image(image_file)
            results.append((image_file.name, result))
            
            # Short summary
            verdict_emoji = {
                "DECENT": "✅",
                "BORDERLINE": "⚠️",
                "TOXIC": "🚫",
                "HIGHLY TOXIC": "❌"
            }
            
            text_preview = result.ocr_text[:40] + "..." if len(result.ocr_text) > 40 else result.ocr_text
            text_preview = text_preview.replace("\n", " ")
            
            print(f"{verdict_emoji[result.verdict]} {result.verdict} | "
                  f"Score: {result.overall_score:.0%} | "
                  f"OCR: {result.ocr_confidence:.0%} | "
                  f"\"{text_preview}\"")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((image_file.name, None))
    
    # Overall summary
    print("\n" + "="*80)
    print("📊 OVERALL STATISTICS")
    print("="*80)
    
    successful = sum(1 for _, r in results if r is not None)
    decent = sum(1 for _, r in results if r and r.verdict == "DECENT")
    borderline = sum(1 for _, r in results if r and r.verdict == "BORDERLINE")
    toxic = sum(1 for _, r in results if r and r.verdict in ["TOXIC", "HIGHLY TOXIC"])
    
    print(f"\n  Total Processed:    {successful}/{len(results)}")
    print(f"  ✅ Decent:          {decent} ({decent/successful*100:.1f}%)")
    print(f"  ⚠️  Borderline:      {borderline} ({borderline/successful*100:.1f}%)")
    print(f"  🚫 Toxic:           {toxic} ({toxic/successful*100:.1f}%)")
    
    if successful > 0:
        avg_ocr_conf = sum(r.ocr_confidence for _, r in results if r) / successful
        avg_time = sum(r.total_processing_time for _, r in results if r) / successful
        
        print(f"\n  Avg OCR Confidence: {avg_ocr_conf:.1%}")
        print(f"  Avg Processing:     {avg_time:.2f}s")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_all_images()
