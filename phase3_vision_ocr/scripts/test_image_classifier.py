#!/usr/bin/env python3
"""
Test trained image classifier
"""

import sys
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_classifier import ImageClassifierInference


def test_single_image(model_path: Path, image_path: Path):
    """Test on single image"""
    print("\n" + "="*80)
    print("🔍 IMAGE CLASSIFICATION TEST")
    print("="*80)
    
    print(f"\n📁 Image: {image_path.name}")
    print(f"📦 Model: {model_path.name}")
    
    # Load model
    print("\n🤖 Loading model...")
    classifier = ImageClassifierInference(model_path)
    
    # Predict
    print("⚙️  Running inference...")
    result = classifier.predict(image_path)
    
    # Display results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"\n🎯 Predicted Class: {result.predicted_class}")
    print(f"📈 Confidence: {result.confidence:.2%}")
    print(f"⏱️  Processing Time: {result.processing_time*1000:.1f}ms")
    
    print("\n📋 All Predictions:")
    for class_name, prob in sorted(result.predictions.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        flag = "✅" if class_name == result.predicted_class else "  "
        print(f"  {flag} {class_name:<20} {prob:>6.2%} |{bar}|")
    
    print("\n" + "="*80 + "\n")


def test_batch(model_path: Path, image_dir: Path):
    """Test on directory of images"""
    print("\n" + "="*80)
    print("🔍 BATCH IMAGE CLASSIFICATION TEST")
    print("="*80)
    
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"\n📁 Directory: {image_dir}")
    print(f"   Found {len(image_files)} images")
    
    # Load model
    print("\n🤖 Loading model...")
    classifier = ImageClassifierInference(model_path)
    
    # Process images
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    results = []
    for image_file in image_files:
        result = classifier.predict(image_file)
        results.append((image_file.name, result))
        
        print(f"📸 {image_file.name}")
        print(f"   🎯 {result.predicted_class} ({result.confidence:.1%}) | ⏱️ {result.processing_time*1000:.0f}ms")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    class_counts = {}
    for _, result in results:
        class_counts[result.predicted_class] = class_counts.get(result.predicted_class, 0) + 1
    
    print(f"\n  Total Images: {len(results)}")
    print(f"  Class Distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    • {class_name}: {count} ({count/len(results)*100:.1f}%)")
    
    avg_conf = sum(r.confidence for _, r in results) / len(results)
    avg_time = sum(r.processing_time for _, r in results) / len(results)
    
    print(f"\n  Avg Confidence: {avg_conf:.1%}")
    print(f"  Avg Time: {avg_time*1000:.0f}ms")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test trained image classifier")
    
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--batch', help='Directory with images')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        parser.error("Either --image or --batch is required")
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING
    )
    
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if args.image:
        test_single_image(model_path, Path(args.image))
    else:
        test_batch(model_path, Path(args.batch))


if __name__ == "__main__":
    main()
