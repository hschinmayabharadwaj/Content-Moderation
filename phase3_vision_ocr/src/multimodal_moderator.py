"""
Unified Multimodal Moderator
Combines OCR text analysis + Image classification for comprehensive moderation
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Union, List
import logging
import time
from dataclasses import dataclass
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_worker import OCRWorker, OCRResult
from image_classifier import ImageClassifierInference, ClassificationResult
from image_text_moderator import ImageTextModerator

# Try to import text moderation
try:
    import torch
    TEXT_MODERATION_AVAILABLE = True
except ImportError:
    TEXT_MODERATION_AVAILABLE = False
    logging.warning("Text moderation not available")


@dataclass
class MultimodalModerationResult:
    """Complete multimodal moderation result"""
    # Input
    image_path: str
    
    # OCR Results
    ocr_text: str
    ocr_confidence: float
    ocr_processing_time: float
    
    # Text Moderation (from OCR)
    text_toxic_score: float
    text_categories: Dict[str, float]
    text_flagged: List[str]
    
    # Image Classification
    image_predictions: Dict[str, Dict[str, float]]  # {dataset: {class: prob}}
    image_flags: List[str]  # Flagged image categories
    
    # Fusion & Final Verdict
    fusion_strategy: str
    text_weight: float
    image_weight: float
    final_score: float
    verdict: str
    confidence_level: str
    action: str
    
    # Timing
    total_processing_time: float
    
    # Metadata
    metadata: Dict


class FusionStrategy:
    """Strategies for fusing text and image signals"""
    
    @staticmethod
    def weighted_average(
        text_score: float,
        image_scores: Dict[str, float],
        text_weight: float = 0.6,
        image_weight: float = 0.4
    ) -> float:
        """
        Weighted average of text and image scores
        
        Args:
            text_score: Text toxicity score [0, 1]
            image_scores: Dict of image classifier scores
            text_weight: Weight for text signal
            image_weight: Weight for image signals
        
        Returns:
            Fused score [0, 1]
        """
        # Take max of image scores
        max_image_score = max(image_scores.values()) if image_scores else 0.0
        
        # Weighted combination
        fused = text_weight * text_score + image_weight * max_image_score
        
        return fused
    
    @staticmethod
    def max_score(
        text_score: float,
        image_scores: Dict[str, float]
    ) -> float:
        """
        Take maximum score across all modalities
        
        Args:
            text_score: Text toxicity score
            image_scores: Dict of image classifier scores
        
        Returns:
            Max score
        """
        all_scores = [text_score] + list(image_scores.values())
        return max(all_scores)
    
    @staticmethod
    def adaptive_fusion(
        text_score: float,
        image_scores: Dict[str, float],
        ocr_confidence: float,
        confidence_threshold: float = 0.5
    ) -> float:
        """
        Adaptive fusion based on OCR confidence
        If OCR confidence is high, trust text more
        If OCR confidence is low, trust image more
        
        Args:
            text_score: Text toxicity score
            image_scores: Image classifier scores
            ocr_confidence: OCR confidence [0, 1]
            confidence_threshold: Threshold for switching weights
        
        Returns:
            Fused score
        """
        max_image_score = max(image_scores.values()) if image_scores else 0.0
        
        if ocr_confidence > confidence_threshold:
            # High OCR confidence - trust text more
            text_weight = 0.7
            image_weight = 0.3
        else:
            # Low OCR confidence - trust image more
            text_weight = 0.4
            image_weight = 0.6
        
        return text_weight * text_score + image_weight * max_image_score


class MultimodalModerator:
    """
    Unified multimodal content moderator
    Combines OCR + text moderation + image classification
    """
    
    def __init__(
        self,
        models_dir: str = '../../../content_moderation_trained',
        image_models_dir: str = '../models',
        ocr_engine: str = 'easyocr',
        fusion_strategy: str = 'weighted',
        text_weight: float = 0.6,
        image_weight: float = 0.4,
        config: Optional[Dict] = None
    ):
        """
        Initialize multimodal moderator
        
        Args:
            models_dir: Path to text moderation models
            image_models_dir: Path to image classification models
            ocr_engine: OCR engine to use
            fusion_strategy: 'weighted', 'max', or 'adaptive'
            text_weight: Weight for text signal (weighted fusion)
            image_weight: Weight for image signal (weighted fusion)
            config: Optional configuration dict
        """
        self.logger = logging.getLogger(__name__)
        self.fusion_strategy_name = fusion_strategy
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.config = config or {}
        
        # Initialize OCR worker
        self.logger.info("Initializing OCR worker...")
        self.ocr_worker = OCRWorker(
            primary_engine=ocr_engine,
            languages=['en'],
            gpu=True
        )
        
        # Initialize text moderator (for OCR text)
        self.text_moderator = None
        if TEXT_MODERATION_AVAILABLE:
            try:
                self.logger.info("Initializing text moderator...")
                from image_text_moderator import ImageTextModerator
                self.text_moderator = ImageTextModerator(
                    models_dir=models_dir,
                    ocr_engine=ocr_engine
                )
                self.logger.info("✅ Text moderator initialized")
            except Exception as e:
                self.logger.warning(f"Text moderator not available: {e}")
        
        # Initialize image classifiers
        self.image_classifiers = {}
        self._load_image_classifiers(image_models_dir)
        
        # Select fusion strategy
        self.fusion_fn = self._get_fusion_function(fusion_strategy)
    
    def _load_image_classifiers(self, models_dir: Path):
        """Load available image classification models"""
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            self.logger.warning(f"Image models directory not found: {models_dir}")
            return
        
        # Try to load each classifier
        for dataset_name in ['nsfw', 'hate_symbols', 'violence']:
            model_path = models_dir / dataset_name / 'best_model.pt'
            
            if model_path.exists():
                try:
                    self.logger.info(f"Loading {dataset_name} classifier...")
                    classifier = ImageClassifierInference(model_path)
                    self.image_classifiers[dataset_name] = classifier
                    self.logger.info(f"✅ Loaded {dataset_name} classifier")
                except Exception as e:
                    self.logger.warning(f"Failed to load {dataset_name}: {e}")
        
        if not self.image_classifiers:
            self.logger.warning("No image classifiers loaded - text-only mode")
    
    def _get_fusion_function(self, strategy: str):
        """Get fusion function by name"""
        if strategy == 'weighted':
            return FusionStrategy.weighted_average
        elif strategy == 'max':
            return FusionStrategy.max_score
        elif strategy == 'adaptive':
            return FusionStrategy.adaptive_fusion
        else:
            self.logger.warning(f"Unknown fusion strategy: {strategy}, using weighted")
            return FusionStrategy.weighted_average
    
    def moderate_image(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.65,
        min_ocr_confidence: float = 0.3
    ) -> MultimodalModerationResult:
        """
        Complete multimodal moderation pipeline
        
        Args:
            image_path: Path to image
            threshold: Decision threshold
            min_ocr_confidence: Minimum OCR confidence
        
        Returns:
            MultimodalModerationResult
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        # Step 1: OCR Extraction
        self.logger.info(f"Processing image: {image_path}")
        ocr_result = self.ocr_worker.extract_text(
            image_path,
            preprocess=True,
            min_confidence=min_ocr_confidence
        )
        
        # Step 2: Text Moderation (if text found)
        text_toxic_score = 0.0
        text_categories = {}
        text_flagged = []
        
        if ocr_result.text and self.text_moderator:
            text_mod = self.text_moderator.moderate_text(ocr_result.text, threshold)
            
            if text_mod.get('available', False):
                text_toxic_score = text_mod.get('phase2_toxic_score', 0.0)
                text_categories = text_mod.get('phase1_predictions', {})
                text_flagged = text_mod.get('phase1_flagged', [])
        
        # Step 3: Image Classification
        image_predictions = {}
        image_flags = []
        image_scores = {}
        
        for dataset_name, classifier in self.image_classifiers.items():
            try:
                result = classifier.predict(image_path)
                image_predictions[dataset_name] = result.predictions
                
                # Check if flagged (assuming first category is negative, second is positive)
                categories = list(result.predictions.keys())
                if len(categories) == 2:
                    # Get the "positive" class (unsafe/hate/violence)
                    positive_class = categories[1]  # Assumes [safe/no_X, unsafe/X] order
                    positive_score = result.predictions[positive_class]
                    
                    image_scores[dataset_name] = positive_score
                    
                    if positive_score > threshold:
                        image_flags.append(f"{dataset_name}:{positive_class}")
            
            except Exception as e:
                self.logger.error(f"Image classification failed for {dataset_name}: {e}")
        
        # Step 4: Fusion
        if self.fusion_strategy_name == 'adaptive':
            final_score = self.fusion_fn(
                text_toxic_score,
                image_scores,
                ocr_result.confidence
            )
        elif self.fusion_strategy_name == 'weighted':
            final_score = self.fusion_fn(
                text_toxic_score,
                image_scores,
                self.text_weight,
                self.image_weight
            )
        else:  # max
            final_score = self.fusion_fn(text_toxic_score, image_scores)
        
        # Step 5: Determine verdict
        is_flagged = bool(text_flagged or image_flags)
        
        if not is_flagged and final_score < 0.5:
            verdict = "DECENT"
            confidence_level = "High"
            action = "AUTO-APPROVE"
        elif final_score < 0.5:
            verdict = "BORDERLINE"
            confidence_level = "Medium"
            action = "HUMAN-REVIEW"
        elif final_score < 0.7:
            verdict = "TOXIC"
            confidence_level = "High"
            action = "HUMAN-REVIEW"
        else:
            verdict = "HIGHLY TOXIC"
            confidence_level = "Very High"
            action = "AUTO-REMOVE"
        
        total_time = time.time() - start_time
        
        return MultimodalModerationResult(
            # Input
            image_path=str(image_path),
            
            # OCR
            ocr_text=ocr_result.text,
            ocr_confidence=ocr_result.confidence,
            ocr_processing_time=ocr_result.processing_time,
            
            # Text moderation
            text_toxic_score=text_toxic_score,
            text_categories=text_categories,
            text_flagged=text_flagged,
            
            # Image classification
            image_predictions=image_predictions,
            image_flags=image_flags,
            
            # Fusion
            fusion_strategy=self.fusion_strategy_name,
            text_weight=self.text_weight,
            image_weight=self.image_weight,
            final_score=final_score,
            verdict=verdict,
            confidence_level=confidence_level,
            action=action,
            
            # Timing
            total_processing_time=total_time,
            
            # Metadata
            metadata={
                'ocr_detections': ocr_result.metadata.get('num_detections', 0),
                'text_length': len(ocr_result.text),
                'num_image_classifiers': len(self.image_classifiers),
                'image_classifier_names': list(self.image_classifiers.keys())
            }
        )
    
    def display_result(self, result: MultimodalModerationResult):
        """Display moderation result"""
        print("\n" + "="*80)
        print("🛡️  MULTIMODAL CONTENT MODERATION REPORT")
        print("="*80)
        
        print(f"\n📁 Image: {Path(result.image_path).name}")
        
        # OCR Section
        print("\n" + "━"*80)
        print("📸 OCR TEXT EXTRACTION")
        print("━"*80)
        print(f"  Confidence: {result.ocr_confidence:.2%}")
        print(f"  Time: {result.ocr_processing_time:.2f}s")
        print(f"  Text: \"{result.ocr_text}\" " if result.ocr_text else "  (No text detected)")
        
        # Text Moderation Section
        if result.text_categories or result.text_toxic_score > 0:
            print("\n" + "━"*80)
            print("📝 TEXT ANALYSIS (from OCR)")
            print("━"*80)
            print(f"  Toxic Score: {result.text_toxic_score:.2%}")
            
            if result.text_categories:
                print(f"\n  Categories:")
                for cat, score in result.text_categories.items():
                    flag = "🚫" if cat in result.text_flagged else "✅"
                    bar_length = int(score * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    print(f"    {flag} {cat:<18} {score:>6.2%} |{bar}|")
        
        # Image Classification Section
        if result.image_predictions:
            print("\n" + "━"*80)
            print("🖼️  IMAGE CLASSIFICATION")
            print("━"*80)
            
            for dataset, preds in result.image_predictions.items():
                print(f"\n  {dataset.upper()}:")
                for class_name, prob in preds.items():
                    flag = "🚫" if prob > 0.65 else "✅"
                    bar_length = int(prob * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    print(f"    {flag} {class_name:<18} {prob:>6.2%} |{bar}|")
            
            if result.image_flags:
                print(f"\n  ⚠️  Image Flags: {', '.join(result.image_flags)}")
        
        # Fusion Section
        print("\n" + "━"*80)
        print("🔀 FUSION & FINAL VERDICT")
        print("━"*80)
        print(f"  Strategy: {result.fusion_strategy}")
        print(f"  Text Weight: {result.text_weight:.1%}")
        print(f"  Image Weight: {result.image_weight:.1%}")
        
        emoji_map = {
            "DECENT": "✅",
            "BORDERLINE": "⚠️",
            "TOXIC": "🚫",
            "HIGHLY TOXIC": "❌"
        }
        
        print(f"\n  {emoji_map.get(result.verdict, '❓')} Verdict: {result.verdict}")
        print(f"  📊 Final Score: {result.final_score:.2%}")
        print(f"  🎯 Confidence: {result.confidence_level}")
        print(f"  🎬 Action: {result.action}")
        print(f"  ⏱️  Total Time: {result.total_processing_time:.2f}s")
        
        # Recommendation
        print("\n" + "━"*80)
        print("💡 RECOMMENDATION")
        print("━"*80)
        
        if result.action == "AUTO-APPROVE":
            print("\n  ✅ Content is appropriate - safe to display")
        elif result.action == "HUMAN-REVIEW":
            print("\n  ⚠️  Requires human review before publishing")
            if result.text_flagged:
                print(f"     Text flags: {', '.join(result.text_flagged)}")
            if result.image_flags:
                print(f"     Image flags: {', '.join(result.image_flags)}")
        else:  # AUTO-REMOVE
            print("\n  ❌ Content violates policy - should be removed immediately")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multimodal image content moderation"
    )
    parser.add_argument('--image', required=True, help='Image path')
    parser.add_argument('--models-dir', default='../../../content_moderation_trained',
                       help='Text models directory')
    parser.add_argument('--image-models', default='../models',
                       help='Image models directory')
    parser.add_argument('--fusion', default='weighted',
                       choices=['weighted', 'max', 'adaptive'],
                       help='Fusion strategy')
    parser.add_argument('--text-weight', type=float, default=0.6,
                       help='Text weight (weighted fusion)')
    parser.add_argument('--image-weight', type=float, default=0.4,
                       help='Image weight (weighted fusion)')
    parser.add_argument('--threshold', type=float, default=0.65,
                       help='Decision threshold')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Initializing Multimodal Moderator...")
    moderator = MultimodalModerator(
        models_dir=args.models_dir,
        image_models_dir=args.image_models,
        fusion_strategy=args.fusion,
        text_weight=args.text_weight,
        image_weight=args.image_weight
    )
    
    result = moderator.moderate_image(
        args.image,
        threshold=args.threshold
    )
    
    moderator.display_result(result)
