#!/usr/bin/env python3
"""
Create a toxic text image for testing
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_toxic_text_image():
    """Create an image with potentially toxic text"""
    size = (800, 200)
    img = Image.new('RGB', size, color='#ff6b6b')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font = ImageFont.load_default()
    
    text = "You stupid idiot get lost"
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill='white', font=font)
    
    output_path = Path(__file__).parent.parent / "test_images" / "toxic_text.jpg"
    img.save(output_path)
    print(f"✅ Created: {output_path}")


if __name__ == "__main__":
    create_toxic_text_image()
