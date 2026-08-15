"""
Image Text Moderator - Integrate OCR with text moderation pipeline
Extracts text from images and runs through Phase 1/2 text classifiers
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import time
from dataclasses import dataclass
import yaml

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ocr_worker import OCRWorker, OCRResult

# Import text moderation components
try:
    from custom_models import (
        CustomDistilBertForSequenceClassification,
        CustomXLMRobertaForSequenceClassification,
        load_phase2_checkpoint
    )
    from transformers import AutoTokenizer, AutoConfig
    import torch
    TEXT_MODERATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Text moderation not available: {e}")
    TEXT_MODERATION_AVAILABLE = False


@dataclass
class ImageModerationResult:
    """Combined result from OCR + text moderation"""
    # OCR results
    ocr_text: str
    ocr_confidence: float
    ocr_engine: str
    ocr_processing_time: float
    
    # Text moderation results (Phase 1)
    phase1_predictions: Dict[str, float]
    phase1_flagged: list
    phase1_max_score: float
    
    # Text moderation results (Phase 2)
    phase2_toxic_score: float
    phase2_clean_score: float
    
    # Overall verdict
    verdict: str
    is_toxic: bool
    overall_score: float
    confidence_level: str
    action: str  # AUTO-APPROVE, HUMAN-REVIEW, AUTO-REMOVE
    
    # Timing
    total_processing_time: float
    
    # Metadata
    metadata: Dict


class ImageTextModerator:
    """
    Integrated image text moderation system
    Image → OCR → Text Classification → Verdict
    """
    
    def __init__(
        self,
        models_dir: str = '../../../content_moderation_trained',
        ocr_engine: str = 'easyocr',
        ocr_languages: list = ['en'],
        ocr_gpu: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Initialize integrated moderator
        
        Args:
            models_dir: Path to trained models directory
            ocr_engine: OCR engine to use
            ocr_languages: Languages for OCR
            ocr_gpu: Use GPU for OCR
            config_path: Optional config file path
        """
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path(models_dir)
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Use defaults from parent config
            config_file = Path(__file__).parent.parent / 'configs' / 'default.yaml'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}
        
        # Initialize OCR worker
        self.logger.info("Initializing OCR worker...")
        self.ocr_worker = OCRWorker(
            primary_engine=ocr_engine,
            fallback_engine='tesseract',
            languages=ocr_languages,
            gpu=ocr_gpu,
            config=self.config.get('ocr', {})
        )
        self.logger.info("✅ OCR worker initialized")
        
        # Initialize text moderation models
        if TEXT_MODERATION_AVAILABLE:
            self._load_text_moderation_models()
        else:
            self.logger.warning("Text moderation not available - OCR only mode")
            self.text_moderation_enabled = False
    
    def _load_text_moderation_models(self):
        """Load Phase 1 and Phase 2 text moderation models"""
        self.logger.info("Loading text moderation models...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Phase 1 setup
        try:
            phase1_path = self.models_dir / 'phase1_text_baseline'
            
            # Load config
            with open(phase1_path / 'configs/colab_run.yaml', 'r') as f:
                self.phase1_config = yaml.safe_load(f)
            
            self.phase1_labels = self.phase1_config['labels']['names']
            
            # Load tokenizer
            self.phase1_tokenizer = AutoTokenizer.from_pretrained(
                self.phase1_config['model']['name']
            )
            
            # Load model
            config = AutoConfig.from_pretrained(self.phase1_config['model']['name'])
            self.phase1_model = CustomDistilBertForSequenceClassification(
                config,
                num_labels=self.phase1_config['model']['num_labels'],
                hidden_size=self.phase1_config['model']['hidden_size'],
                dropout=self.phase1_config['model']['dropout']
            )
            
            # Load weights
            checkpoint = torch.load(
                phase1_path / 'models/best_model.pt',
                map_location=self.device,
                weights_only=False
            )
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                weights = checkpoint['model_state_dict']
            else:
                weights = checkpoint
            
            self.phase1_model.load_state_dict(weights)
            self.phase1_model.to(self.device)
            self.phase1_model.eval()
            
            self.logger.info("✅ Phase 1 model loaded")
        
        except Exception as e:
            self.logger.error(f"Failed to load Phase 1 model: {e}")
            self.phase1_model = None
        
        # Phase 2 setup
        try:
            phase2_path = self.models_dir / 'phase2_multilingual'
            
            # Load config
            with open(phase2_path / 'configs/colab_run.yaml', 'r') as f:
                self.phase2_config = yaml.safe_load(f)
            
            # Load tokenizer
            self.phase2_tokenizer = AutoTokenizer.from_pretrained(
                self.phase2_config['model']['name']
            )
            
            # Load model
            config = AutoConfig.from_pretrained(self.phase2_config['model']['name'])
            self.phase2_model = CustomXLMRobertaForSequenceClassification(
                config,
                num_labels=1,
                hidden_size=self.phase2_config['model']['hidden_size'],
                dropout=self.phase2_config['model']['dropout'],
                use_single_output=True
            )
            
            # Load checkpoint
            self.phase2_model = load_phase2_checkpoint(
                self.phase2_model,
                phase2_path / 'models/best_model.pt',
                self.device
            )
            self.phase2_model.to(self.device)
            self.phase2_model.eval()
            
            self.logger.info("✅ Phase 2 model loaded")
        
        except Exception as e:
            self.logger.error(f"Failed to load Phase 2 model: {e}")
            self.phase2_model = None
        
        self.text_moderation_enabled = (
            self.phase1_model is not None and
            self.phase2_model is not None
        )
    
    def moderate_text(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Run text through Phase 1 and Phase 2 moderation
        
        Args:
            text: Text to moderate
            threshold: Classification threshold
        
        Returns:
            Moderation results dictionary
        """
        if not self.text_moderation_enabled:
            return {
                'available': False,
                'error': 'Text moderation models not loaded'
            }
        
        if not text or len(text.strip()) < 3:
            return {
                'available': True,
                'phase1_predictions': {},
                'phase1_flagged': [],
                'phase1_max_score': 0.0,
                'phase2_toxic_score': 0.0,
                'phase2_clean_score': 1.0,
                'note': 'Text too short for moderation'
            }
        
        # Apply bias correction from text_moderator.py
        bias_margin = 0.25
        adjusted_threshold = max(threshold, 0.65)
        
        # Phase 1 predictions
        inputs1 = self.phase1_tokenizer(
            text,
            max_length=self.phase1_config['model']['max_length'],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs1 = self.phase1_model(**inputs1)
            pred1 = torch.sigmoid(outputs1.logits).cpu().numpy()[0]
        
        # Phase 2 predictions
        inputs2 = self.phase2_tokenizer(
            text,
            max_length=self.phase2_config['model']['max_length'],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs2 = self.phase2_model(**inputs2)
            pred2_toxic = torch.sigmoid(outputs2.logits).cpu().numpy()[0][0]
        
        # Apply bias correction
        phase1_predictions = {}
        phase1_flagged = []
        max_phase1_score = 0.0
        
        for label, score in zip(self.phase1_labels, pred1):
            corrected_score = max(0, float(score) - bias_margin)
            phase1_predictions[label] = corrected_score
            
            if corrected_score > adjusted_threshold:
                phase1_flagged.append(label)
            
            max_phase1_score = max(max_phase1_score, corrected_score)
        
        # Phase 2 with bias correction
        phase2_toxic_score = max(0, float(pred2_toxic) - bias_margin)
        phase2_clean_score = 1.0 - phase2_toxic_score
        
        return {
            'available': True,
            'phase1_predictions': phase1_predictions,
            'phase1_flagged': phase1_flagged,
            'phase1_max_score': max_phase1_score,
            'phase2_toxic_score': phase2_toxic_score,
            'phase2_clean_score': phase2_clean_score,
            'threshold': adjusted_threshold
        }
    
    def moderate_image(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.5,
        min_ocr_confidence: float = 0.3
    ) -> ImageModerationResult:
        """
        Complete moderation pipeline: Image → OCR → Text Moderation → Verdict
        
        Args:
            image_path: Path to image file
            threshold: Classification threshold
            min_ocr_confidence: Minimum OCR confidence
        
        Returns:
            ImageModerationResult with complete analysis
        """
        start_time = time.time()
        
        # Step 1: OCR extraction
        self.logger.info(f"Extracting text from image: {image_path}")
        ocr_result = self.ocr_worker.extract_text(
            image_path,
            preprocess=True,
            min_confidence=min_ocr_confidence
        )
        
        # Step 2: Text moderation
        if ocr_result.text and self.text_moderation_enabled:
            self.logger.info(f"Moderating extracted text ({len(ocr_result.text)} chars)")
            text_mod_result = self.moderate_text(ocr_result.text, threshold)
        else:
            if not ocr_result.text:
                self.logger.warning("No text extracted from image")
            text_mod_result = {
                'available': False,
                'phase1_predictions': {},
                'phase1_flagged': [],
                'phase1_max_score': 0.0,
                'phase2_toxic_score': 0.0,
                'phase2_clean_score': 1.0,
            }
        
        # Step 3: Compute overall verdict
        if text_mod_result.get('available', False):
            overall_score = max(
                text_mod_result['phase1_max_score'],
                text_mod_result['phase2_toxic_score']
            )
            is_toxic = bool(text_mod_result['phase1_flagged'])
        else:
            overall_score = 0.0
            is_toxic = False
        
        # Determine verdict and action
        if not is_toxic:
            verdict = "DECENT"
            confidence_level = "High"
            action = "AUTO-APPROVE"
        elif overall_score < 0.5:
            verdict = "BORDERLINE"
            confidence_level = "Medium"
            action = "HUMAN-REVIEW"
        elif overall_score < 0.7:
            verdict = "TOXIC"
            confidence_level = "High"
            action = "HUMAN-REVIEW"
        else:
            verdict = "HIGHLY TOXIC"
            confidence_level = "Very High"
            action = "AUTO-REMOVE"
        
        total_time = time.time() - start_time
        
        return ImageModerationResult(
            # OCR
            ocr_text=ocr_result.text,
            ocr_confidence=ocr_result.confidence,
            ocr_engine=ocr_result.engine,
            ocr_processing_time=ocr_result.processing_time,
            
            # Phase 1
            phase1_predictions=text_mod_result.get('phase1_predictions', {}),
            phase1_flagged=text_mod_result.get('phase1_flagged', []),
            phase1_max_score=text_mod_result.get('phase1_max_score', 0.0),
            
            # Phase 2
            phase2_toxic_score=text_mod_result.get('phase2_toxic_score', 0.0),
            phase2_clean_score=text_mod_result.get('phase2_clean_score', 1.0),
            
            # Verdict
            verdict=verdict,
            is_toxic=is_toxic,
            overall_score=overall_score,
            confidence_level=confidence_level,
            action=action,
            
            # Timing
            total_processing_time=total_time,
            
            # Metadata
            metadata={
                'ocr_detections': ocr_result.metadata.get('num_detections', 0),
                'text_length': len(ocr_result.text),
                'text_moderation_available': text_mod_result.get('available', False),
                'threshold': text_mod_result.get('threshold', threshold)
            }
        )
    
    def display_result(self, result: ImageModerationResult):
        """Display moderation result in readable format"""
        print("\n" + "="*80)
        print("🛡️  IMAGE TEXT MODERATION REPORT")
        print("="*80)
        
        # OCR Section
        print("\n" + "━"*80)
        print("📸 OCR EXTRACTION")
        print("━"*80)
        print(f"  Engine:      {result.ocr_engine.upper()}")
        print(f"  Confidence:  {result.ocr_confidence:.2%}")
        print(f"  Time:        {result.ocr_processing_time:.2f}s")
        print(f"  Detections:  {result.metadata['ocr_detections']}")
        
        print(f"\n  📄 Extracted Text:")
        if result.ocr_text:
            print(f"     \"{result.ocr_text}\"")
        else:
            print(f"     (No text detected)")
        
        # Text Moderation Section
        if result.metadata['text_moderation_available']:
            print("\n" + "━"*80)
            print("🔍 TEXT MODERATION ANALYSIS")
            print("━"*80)
            
            # Phase 2 (Multilingual)
            print(f"\n  🌍 Multilingual Model (Phase 2):")
            print(f"     Clean:  {result.phase2_clean_score:.2%} {'✅' if result.phase2_clean_score > 0.5 else '⚠️'}")
            print(f"     Toxic:  {result.phase2_toxic_score:.2%} {'🚫' if result.phase2_toxic_score > 0.5 else '✅'}")
            
            # Phase 1 (Detailed)
            print(f"\n  📋 Category Breakdown (Phase 1):")
            for category, score in result.phase1_predictions.items():
                flag = "🚫" if category in result.phase1_flagged else "✅"
                bar_length = int(score * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"     {flag} {category:<18} {score:>6.2%} |{bar}|")
            
            if result.phase1_flagged:
                print(f"\n  ⚠️  Flagged Categories:")
                for cat in result.phase1_flagged:
                    print(f"     • {cat}")
        else:
            print("\n  ⚠️  Text moderation not available (OCR only mode)")
        
        # Final Verdict
        print("\n" + "━"*80)
        print("⚖️  FINAL VERDICT")
        print("━"*80)
        
        emoji_map = {
            "DECENT": "✅",
            "BORDERLINE": "⚠️",
            "TOXIC": "🚫",
            "HIGHLY TOXIC": "❌"
        }
        
        print(f"\n  {emoji_map.get(result.verdict, '❓')} Verdict:    {result.verdict}")
        print(f"  📊 Score:      {result.overall_score:.2%}")
        print(f"  🎯 Confidence: {result.confidence_level}")
        print(f"  🎬 Action:     {result.action}")
        print(f"  ⏱️  Total Time: {result.total_processing_time:.2f}s")
        
        # Recommendation
        print("\n" + "━"*80)
        print("💡 RECOMMENDATION")
        print("━"*80)
        
        if result.action == "AUTO-APPROVE":
            print("\n  ✅ Content is appropriate - safe to display")
        elif result.action == "HUMAN-REVIEW":
            print("\n  ⚠️  Requires human review before publishing")
        else:  # AUTO-REMOVE
            print("\n  ❌ Content violates policy - should be removed")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Moderate images by extracting and analyzing embedded text"
    )
    parser.add_argument('--image', required=True, help='Image path')
    parser.add_argument('--models-dir', default='../../../content_moderation_trained',
                       help='Path to trained models')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--ocr-engine', default='easyocr',
                       choices=['easyocr', 'tesseract'],
                       help='OCR engine')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize moderator
    print("🚀 Initializing Image Text Moderator...")
    moderator = ImageTextModerator(
        models_dir=args.models_dir,
        ocr_engine=args.ocr_engine
    )
    
    # Moderate image
    result = moderator.moderate_image(
        args.image,
        threshold=args.threshold
    )
    
    # Display result
    moderator.display_result(result)
