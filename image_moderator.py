"""
Simple image moderation wrapper for Colab notebooks.
Provides unified interface to Phase 3 (vision + OCR) systems.
"""

from pathlib import Path
import sys

# Add phase3 to path
sys.path.insert(0, str(Path(__file__).parent / 'phase3_vision_ocr' / 'src'))

from multimodal_moderator import MultimodalModerator
from typing import Dict, Optional


class ImageModerator:
    """Image content moderation using Phase 3 (OCR + Vision)."""
    
    def __init__(self, 
                 ocr_device: str = 'auto',
                 image_models_dir: str = None,
                 fusion_strategy: str = 'weighted'):
        """
        Initialize image moderator.
        
        Args:
            ocr_device: Device for OCR ('cpu', 'cuda', or 'auto')
            image_models_dir: Directory with trained image models
            fusion_strategy: How to combine OCR and image signals ('weighted', 'max', 'adaptive')
        """
        if image_models_dir is None:
            image_models_dir = 'phase3_vision_ocr/models'
        
        self.moderator = MultimodalModerator(
            models_dir=Path(image_models_dir),
            ocr_device=ocr_device,
            fusion_strategy=fusion_strategy
        )
        
        print(f"✅ Initialized Image Moderator")
        print(f"   OCR Device: {ocr_device}")
        print(f"   Fusion Strategy: {fusion_strategy}")
    
    def moderate(self, image_path: str, return_details: bool = False) -> Dict:
        """
        Moderate image content.
        
        Args:
            image_path: Path to image file
            return_details: If True, return detailed breakdown of all signals
        
        Returns:
            Dict with keys:
            - 'toxic': Whether image is flagged as toxic
            - 'confidence': Overall confidence score
            - 'recommendation': 'CLEAN', 'REVIEW', or 'BLOCK'
            - 'ocr_text': Extracted text from image
            - 'explanation': Human-readable explanation
            - 'details': (if return_details=True) Breakdown of all signals
        """
        result = self.moderator.moderate(Path(image_path))
        
        # Build simplified result
        output = {
            'image': image_path,
            'toxic': result.is_toxic,
            'confidence': result.confidence,
            'recommendation': result.recommendation,
            'ocr_text': result.extracted_text,
            'explanation': self._build_explanation(result)
        }
        
        if return_details:
            output['details'] = {
                'ocr': {
                    'confidence': result.ocr_confidence,
                    'text': result.extracted_text
                },
                'image_classification': result.image_scores,
                'text_moderation': result.text_scores,
                'fusion_scores': {
                    'weighted': result.weighted_score,
                    'max': result.max_score,
                    'adaptive': result.adaptive_score
                }
            }
        
        return output
    
    def _build_explanation(self, result) -> str:
        """Build human-readable explanation."""
        if result.is_toxic:
            signals = []
            if result.extracted_text and any(s > 0.5 for s in result.text_scores.values()):
                signals.append("toxic text detected")
            if any(s > 0.5 for s in result.image_scores.values()):
                toxic_images = [k for k, v in result.image_scores.items() if v > 0.5]
                signals.append(f"flagged image ({', '.join(toxic_images)})")
            
            signal_str = " + ".join(signals) if signals else "content flagged"
            return f"⚠️ TOXIC: {signal_str} (confidence: {result.confidence:.1%})"
        else:
            return f"✅ CLEAN: Image appears safe (confidence: {result.confidence:.1%})"
    
    def moderate_batch(self, image_paths: list) -> list:
        """Moderate multiple images."""
        return [self.moderate(img_path) for img_path in image_paths]


# Convenience function
def moderate_image(image_path: str, models_dir: str = None) -> Dict:
    """Quick moderation without initializing moderator."""
    moderator = ImageModerator(image_models_dir=models_dir)
    return moderator.moderate(image_path)


if __name__ == '__main__':
    # Example usage
    import glob
    
    moderator = ImageModerator()
    
    # Find test images
    test_images = glob.glob('phase3_vision_ocr/test_images/*.jpg')
    
    if test_images:
        print("🧪 Testing Image Moderation:\n")
        for img_path in test_images[:3]:  # First 3 images
            result = moderator.moderate(img_path, return_details=True)
            print(f"Image: {Path(img_path).name}")
            print(f"  Result: {result['explanation']}")
            if result['ocr_text']:
                print(f"  OCR Text: {result['ocr_text'][:100]}...")
            print()
    else:
        print("⚠️ No test images found. Generate them with:")
        print("  python phase3_vision_ocr/scripts/create_test_images.py")
