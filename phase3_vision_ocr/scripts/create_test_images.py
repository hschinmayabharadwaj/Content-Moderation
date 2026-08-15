#!/usr/bin/env python3
"""
Create sample test images with text for OCR testing
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path


def create_simple_text_image(
    text: str,
    output_path: Path,
    size: tuple = (800, 200),
    bg_color: str = "white",
    text_color: str = "black",
    font_size: int = 40
):
    """Create a simple image with text"""
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text size and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill=text_color, font=font)
    img.save(output_path)
    print(f"✅ Created: {output_path}")


def create_meme_style_image(text: str, output_path: Path):
    """Create a meme-style image"""
    size = (600, 600)
    img = Image.new('RGB', size, color='#1a1a1a')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 50)
    except:
        font = ImageFont.load_default()
    
    # Add colored rectangle background
    draw.rectangle([50, 50, 550, 550], fill='#3b3b3b')
    
    # Draw text with shadow
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Shadow
    draw.text((x+2, y+2), text, fill='black', font=font)
    # Main text
    draw.text((x, y), text, fill='white', font=font)
    
    img.save(output_path)
    print(f"✅ Created: {output_path}")


def create_noisy_image(text: str, output_path: Path):
    """Create an image with noise"""
    size = (800, 200)
    
    # Create base image
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    
    # Add noise
    img_array = np.array(img)
    noise = np.random.randint(-30, 30, img_array.shape, dtype=np.int16)
    noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    img_noisy = Image.fromarray(noisy_img)
    img_noisy.save(output_path)
    print(f"✅ Created: {output_path}")


def create_multiline_image(text: str, output_path: Path):
    """Create an image with multiline text"""
    size = (800, 400)
    img = Image.new('RGB', size, color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 35)
    except:
        font = ImageFont.load_default()
    
    lines = text.split('\n')
    y = 50
    
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        draw.text((x, y), line, fill='#333333', font=font)
        y += 80
    
    img.save(output_path)
    print(f"✅ Created: {output_path}")


def main():
    # Create test_images directory
    test_images_dir = Path(__file__).parent.parent / "test_images"
    test_images_dir.mkdir(exist_ok=True)
    
    print("🎨 Creating test images for OCR...")
    print("=" * 60)
    
    # Sample 1: Clean simple text
    create_simple_text_image(
        "This is a clean test image",
        test_images_dir / "clean_text.jpg"
    )
    
    # Sample 2: Meme style
    create_meme_style_image(
        "TOP TEXT\nBOTTOM TEXT",
        test_images_dir / "meme_style.jpg"
    )
    
    # Sample 3: Noisy text
    create_noisy_image(
        "Text with noise and artifacts",
        test_images_dir / "noisy_text.jpg"
    )
    
    # Sample 4: Multiline
    create_multiline_image(
        "First line of text\nSecond line here\nThird line too",
        test_images_dir / "multiline_text.jpg"
    )
    
    # Sample 5: Different colors
    create_simple_text_image(
        "White text on dark background",
        test_images_dir / "dark_bg.jpg",
        bg_color="#2c2c2c",
        text_color="white"
    )
    
    # Sample 6: Potential toxic content (for testing moderation)
    create_simple_text_image(
        "You are amazing and wonderful",
        test_images_dir / "positive_text.jpg",
        bg_color="#e8f5e9",
        text_color="#2e7d32"
    )
    
    print("=" * 60)
    print(f"✅ Created 6 test images in {test_images_dir}/")
    print("\nTest them with:")
    print(f"  python scripts/test_ocr.py --image {test_images_dir}/clean_text.jpg")
    print(f"  python scripts/test_ocr.py --batch {test_images_dir}/")


if __name__ == "__main__":
    main()
