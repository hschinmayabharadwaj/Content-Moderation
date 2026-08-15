#!/usr/bin/env python3
"""
Test OCR Worker - Verify OCR extraction on sample images
"""

import sys
import argparse
from pathlib import Path
import logging
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_worker import OCRWorker, OCRResult


def print_separator(char="━", length=80):
    """Print a separator line"""
    print(char * length)


def format_time(seconds: float) -> str:
    """Format time in readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def test_single_image(image_path: Path, args):
    """Test OCR on a single image"""
    print("\n" + "="*80)
    print("🔍 OCR EXTRACTION TEST")
    print("="*80)
    
    print(f"\n📁 Image: {image_path.name}")
    print(f"   Path: {image_path}")
    
    # Check if file exists
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    # Initialize worker
    print(f"\n🤖 Initializing OCR Worker...")
    print(f"   Engine: {args.engine}")
    print(f"   Languages: {args.languages}")
    print(f"   GPU: {args.gpu}")
    
    start_init = time.time()
    worker = OCRWorker(
        primary_engine=args.engine,
        fallback_engine=args.fallback,
        languages=args.languages,
        gpu=args.gpu,
        config={
            'preprocessing': {
                'grayscale': args.grayscale,
                'contrast_enhancement': args.contrast,
                'denoise': args.denoise,
                'deskew': args.deskew,
                'resize_max': args.resize_max
            },
            'remove_special_chars': args.remove_special
        }
    )
    init_time = time.time() - start_init
    print(f"   ✅ Initialized in {format_time(init_time)}")
    
    # Extract text
    print(f"\n⚙️  Processing Image...")
    print(f"   Preprocessing: {args.preprocess}")
    print(f"   Min Confidence: {args.min_confidence}")
    
    result = worker.extract_text(
        image_path,
        preprocess=args.preprocess,
        min_confidence=args.min_confidence,
        min_text_length=args.min_length
    )
    
    # Display results
    print("\n" + "="*80)
    print("📄 EXTRACTED TEXT")
    print("="*80)
    
    if result.text:
        print(f"\n{result.text}\n")
    else:
        print("\n⚠️  No text detected or confidence too low\n")
    
    print_separator()
    print("📊 METADATA")
    print_separator()
    print(f"  Engine Used:        {result.engine.upper()}")
    print(f"  Overall Confidence: {result.confidence:.2%}")
    print(f"  Processing Time:    {format_time(result.processing_time)}")
    print(f"  Text Detections:    {result.metadata.get('num_detections', 0)}")
    print(f"  Text Length:        {len(result.text)} characters")
    print(f"  Bounding Boxes:     {len(result.bounding_boxes)}")
    
    # Confidence distribution
    if 'confidences' in result.metadata and result.metadata['confidences']:
        confidences = result.metadata['confidences']
        print(f"\n  Confidence Distribution:")
        print(f"    Min:    {min(confidences):.2%}")
        print(f"    Max:    {max(confidences):.2%}")
        print(f"    Mean:   {sum(confidences)/len(confidences):.2%}")
    
    # Visualize if requested
    if args.visualize:
        print(f"\n🎨 Creating visualization...")
        image = worker.load_image(image_path)
        output_path = Path(args.output_dir) / f"{image_path.stem}_ocr_viz.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        vis_image = worker.visualize_detections(image, result, output_path)
        print(f"   ✅ Saved to: {output_path}")
    
    print("\n" + "="*80 + "\n")
    
    return result


def test_batch(image_dir: Path, args):
    """Test OCR on multiple images"""
    print("\n" + "="*80)
    print("🔍 BATCH OCR EXTRACTION TEST")
    print("="*80)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"\n📁 Directory: {image_dir}")
    print(f"   Found {len(image_files)} images")
    
    # Initialize worker once
    print(f"\n🤖 Initializing OCR Worker...")
    worker = OCRWorker(
        primary_engine=args.engine,
        fallback_engine=args.fallback,
        languages=args.languages,
        gpu=args.gpu
    )
    
    # Process each image
    results = []
    total_time = 0
    
    print("\n" + "="*80)
    print("PROCESSING IMAGES")
    print("="*80 + "\n")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {image_path.name}...", end=" ")
        
        try:
            result = worker.extract_text(
                image_path,
                preprocess=args.preprocess,
                min_confidence=args.min_confidence
            )
            
            results.append((image_path.name, result))
            total_time += result.processing_time
            
            text_preview = result.text[:50] + "..." if len(result.text) > 50 else result.text
            text_preview = text_preview.replace("\n", " ")
            
            print(f"✅ {result.confidence:.0%} | {format_time(result.processing_time)} | \"{text_preview}\"")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((image_path.name, None))
    
    # Summary
    print("\n" + "="*80)
    print("📊 BATCH SUMMARY")
    print("="*80)
    
    successful = sum(1 for _, r in results if r and r.text)
    failed = len(results) - successful
    
    print(f"\n  Total Images:       {len(results)}")
    print(f"  Successful:         {successful} ({successful/len(results)*100:.1f}%)")
    print(f"  Failed/No Text:     {failed}")
    print(f"  Total Time:         {format_time(total_time)}")
    print(f"  Average Time:       {format_time(total_time/len(results))}")
    
    if successful > 0:
        avg_confidence = sum(r.confidence for _, r in results if r) / successful
        print(f"  Average Confidence: {avg_confidence:.2%}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test OCR extraction on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python test_ocr.py --image test_images/meme.jpg
  
  # Test with visualization
  python test_ocr.py --image test.jpg --visualize
  
  # Test batch
  python test_ocr.py --batch test_images/
  
  # Use Tesseract instead of EasyOCR
  python test_ocr.py --image test.jpg --engine tesseract
  
  # Adjust preprocessing
  python test_ocr.py --image test.jpg --no-denoise --no-deskew
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Single image path')
    input_group.add_argument('--batch', type=str, help='Directory with images')
    
    # OCR engine options
    parser.add_argument('--engine', default='easyocr', choices=['easyocr', 'tesseract'],
                       help='Primary OCR engine (default: easyocr)')
    parser.add_argument('--fallback', default='tesseract', choices=['easyocr', 'tesseract', 'none'],
                       help='Fallback engine (default: tesseract)')
    parser.add_argument('--languages', nargs='+', default=['en'],
                       help='Languages to detect (default: en)')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU if available (default: True)')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                       help='Disable GPU')
    
    # Preprocessing options
    parser.add_argument('--preprocess', action='store_true', default=True,
                       help='Enable preprocessing (default: True)')
    parser.add_argument('--no-preprocess', dest='preprocess', action='store_false',
                       help='Disable preprocessing')
    parser.add_argument('--grayscale', action='store_true', default=True)
    parser.add_argument('--no-grayscale', dest='grayscale', action='store_false')
    parser.add_argument('--contrast', action='store_true', default=True)
    parser.add_argument('--no-contrast', dest='contrast', action='store_false')
    parser.add_argument('--denoise', action='store_true', default=True)
    parser.add_argument('--no-denoise', dest='denoise', action='store_false')
    parser.add_argument('--deskew', action='store_true', default=True)
    parser.add_argument('--no-deskew', dest='deskew', action='store_false')
    parser.add_argument('--resize-max', type=int, default=1920,
                       help='Max image dimension (default: 1920)')
    
    # Filtering options
    parser.add_argument('--min-confidence', type=float, default=0.3,
                       help='Minimum confidence threshold (default: 0.3)')
    parser.add_argument('--min-length', type=int, default=3,
                       help='Minimum text length (default: 3)')
    parser.add_argument('--remove-special', action='store_true',
                       help='Remove special characters')
    
    # Output options
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization with bounding boxes')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handle fallback=none
    if args.fallback == 'none':
        args.fallback = None
    
    # Run test
    if args.image:
        test_single_image(Path(args.image), args)
    else:
        test_batch(Path(args.batch), args)


if __name__ == "__main__":
    main()
