#!/usr/bin/env python3
"""
Test multimodal moderator on batch of images
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multimodal_moderator import MultimodalModerator


def test_batch():
    """Test on all test images"""
    test_images_dir = Path(__file__).parent.parent / "test_images"
    
    print("\n" + "="*80)
    print("🧪 MULTIMODAL MODERATION BATCH TEST")
    print("="*80)
    
    # Find images
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
    print("\n🚀 Initializing multimodal moderator...")
    moderator = MultimodalModerator(
        models_dir='../../content_moderation_trained',
        image_models_dir='../models',
        fusion_strategy='weighted',
        text_weight=0.6,
        image_weight=0.4
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
            
            flags_str = ""
            if result.text_flagged or result.image_flags:
                all_flags = result.text_flagged + result.image_flags
                flags_str = f" | Flags: {', '.join(all_flags[:2])}"
            
            print(f"{verdict_emoji[result.verdict]} {result.verdict} | "
                  f"Score: {result.final_score:.0%} | "
                  f"OCR: {result.ocr_confidence:.0%}{flags_str}")
            print(f"      Text: \"{text_preview}\"")
        
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
        avg_final_score = sum(r.final_score for _, r in results if r) / successful
        avg_time = sum(r.total_processing_time for _, r in results if r) / successful
        
        print(f"\n  Avg OCR Confidence: {avg_ocr_conf:.1%}")
        print(f"  Avg Final Score:    {avg_final_score:.1%}")
        print(f"  Avg Processing:     {avg_time:.2f}s")
        
        # Count image classifiers used
        if results and results[0][1]:
            num_classifiers = len(results[0][1].metadata.get('image_classifier_names', []))
            classifier_names = results[0][1].metadata.get('image_classifier_names', [])
            print(f"\n  Image Classifiers:  {num_classifiers} loaded")
            if classifier_names:
                print(f"  Classifiers:        {', '.join(classifier_names)}")
            else:
                print(f"  Note:               Running in text-only mode (no image models trained)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_batch()
