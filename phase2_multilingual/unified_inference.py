"""
Phase 2: Unified Inference Pipeline

Combines Phase 1 (English) and Phase 2 (Multilingual) models with
intelligent language-based routing.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging
import json

from transformers import AutoTokenizer, XLMRobertaTokenizer

# Import Phase 1 model
import sys
sys.path.append('../phase1_text_baseline')
from train_classifier import ToxicCommentClassifier

# Import Phase 2 components
from language_identifier import LanguageIdentifier, LanguageRouter
from train_multilingual import MultilingualClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedContentModerator:
    """
    Unified content moderation system with automatic language routing.
    
    Features:
    - Automatic language detection
    - Route to appropriate model (Phase 1 or Phase 2)
    - Apply calibrated thresholds
    - Return enforcement tier decisions
    """
    
    def __init__(
        self,
        phase1_model_path: str,
        phase2_model_path: str,
        language_model_path: Optional[str] = None,
        phase1_calibration_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize unified moderator.
        
        Args:
            phase1_model_path: Path to Phase 1 model checkpoint
            phase2_model_path: Path to Phase 2 model checkpoint
            language_model_path: Path to FastText language model
            phase1_calibration_path: Path to Phase 1 calibration results
            device: Device for inference
        """
        self.device = torch.device(device)
        logger.info(f"Initializing UnifiedContentModerator on {device}")
        
        # Load Phase 1 (English) model
        self.phase1_model, self.phase1_config, self.phase1_tokenizer = self._load_phase1(
            phase1_model_path
        )
        
        # Load Phase 2 (Multilingual) model
        self.phase2_model, self.phase2_config, self.phase2_tokenizer = self._load_phase2(
            phase2_model_path
        )
        
        # Load calibration thresholds
        self.calibration = None
        if phase1_calibration_path:
            self.calibration = self._load_calibration(phase1_calibration_path)
        
        # Initialize language routing
        self.language_identifier = LanguageIdentifier(
            fasttext_model_path=language_model_path,
            confidence_threshold=0.5,
            use_fallback=True
        )
        
        self.router = LanguageRouter(self.language_identifier)
        
        logger.info("✓ UnifiedContentModerator initialized successfully")
    
    def _load_phase1(self, model_path: str):
        """Load Phase 1 model (English BERT/RoBERTa)."""
        logger.info(f"Loading Phase 1 model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Initialize model
        model = ToxicCommentClassifier(
            model_name=config['model']['name'],
            num_labels=config['model']['num_labels'],
            dropout=config['model']['dropout'],
            hidden_size=config['model']['hidden_size'],
            use_intermediate_layer=config['model']['use_intermediate_layer']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        
        logger.info(f"✓ Phase 1 model loaded: {config['model']['name']}")
        
        return model, config, tokenizer
    
    def _load_phase2(self, model_path: str):
        """Load Phase 2 model (XLM-RoBERTa)."""
        logger.info(f"Loading Phase 2 model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Initialize model
        model = MultilingualClassifier(
            model_name=config['model']['name'],
            num_labels=1,
            dropout=config['model']['dropout'],
            hidden_size=config['model']['hidden_size'],
            use_language_adapter=config['model'].get('use_language_adapter', False)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained(config['model']['name'])
        
        logger.info(f"✓ Phase 2 model loaded: {config['model']['name']}")
        
        return model, config, tokenizer
    
    def _load_calibration(self, calibration_path: str):
        """Load calibration thresholds from Phase 1."""
        logger.info(f"Loading calibration from {calibration_path}")
        
        with open(calibration_path, 'r') as f:
            calibration = json.load(f)
        
        logger.info("✓ Calibration thresholds loaded")
        return calibration
    
    @torch.no_grad()
    def predict_phase1(self, text: str) -> Dict:
        """Predict using Phase 1 model (English multi-label)."""
        # Tokenize
        encoding = self.phase1_tokenizer(
            text,
            max_length=self.phase1_config['model']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        outputs = self.phase1_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs['logits']
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Format predictions
        label_names = self.phase1_config['labels']['names']
        predictions = {}
        
        for i, label in enumerate(label_names):
            predictions[label] = float(probs[i])
        
        return predictions
    
    @torch.no_grad()
    def predict_phase2(self, text: str, language: str = None) -> Dict:
        """Predict using Phase 2 model (Multilingual binary)."""
        # Tokenize
        encoding = self.phase2_tokenizer(
            text,
            max_length=self.phase2_config['model']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        outputs = self.phase2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            languages=[language] if language else None
        )
        
        logits = outputs['logits']
        prob = torch.sigmoid(logits).cpu().numpy()[0][0]
        
        return {'toxic': float(prob)}
    
    def apply_enforcement_tiers(
        self,
        predictions: Dict,
        label: str = 'toxic'
    ) -> str:
        """
        Apply enforcement tier thresholds.
        
        Returns:
            'auto_remove', 'human_review', or 'auto_approve'
        """
        if not self.calibration or label not in self.calibration.get('labels', {}):
            # Fallback to simple thresholds
            prob = predictions.get(label, 0.0)
            if prob > 0.9:
                return 'auto_remove'
            elif prob > 0.5:
                return 'human_review'
            else:
                return 'auto_approve'
        
        # Use calibrated thresholds
        label_calib = self.calibration['labels'][label]
        tier_thresholds = label_calib['tier_thresholds']
        
        prob = predictions.get(label, 0.0)
        
        if prob >= tier_thresholds['t_high']:
            return 'auto_remove'
        elif prob >= tier_thresholds['t_low']:
            return 'human_review'
        else:
            return 'auto_approve'
    
    def moderate(
        self,
        text: str,
        return_details: bool = True
    ) -> Dict:
        """
        Main moderation method.
        
        Args:
            text: Input text to moderate
            return_details: Include detailed information
        
        Returns:
            Dictionary with:
            - action: 'auto_remove', 'human_review', or 'auto_approve'
            - predictions: Model predictions
            - language_info: Language detection results
            - model_used: Which model was used
            - reasoning: Why this action was taken
        """
        if not text or not text.strip():
            return {
                'action': 'auto_approve',
                'predictions': {},
                'reasoning': 'Empty text'
            }
        
        # Route to appropriate model
        route_info = self.router.route(text)
        
        model_used = route_info['model']
        language_info = route_info['language_info']
        
        # Get predictions
        if model_used == 'phase1':
            predictions = self.predict_phase1(text)
            # Use most toxic label for action
            max_label = max(predictions.keys(), key=lambda k: predictions[k])
            action = self.apply_enforcement_tiers(predictions, max_label)
            reasoning = f"English model detected {max_label} with confidence {predictions[max_label]:.3f}"
        else:
            predictions = self.predict_phase2(
                text,
                language=language_info['language']
            )
            action = self.apply_enforcement_tiers(predictions, 'toxic')
            reasoning = f"Multilingual model detected toxic content with confidence {predictions['toxic']:.3f}"
        
        result = {
            'action': action,
            'predictions': predictions,
            'reasoning': reasoning
        }
        
        if return_details:
            result['language_info'] = language_info
            result['model_used'] = model_used
            result['route_reasoning'] = route_info['reasoning']
        
        return result
    
    def batch_moderate(
        self,
        texts: List[str],
        return_details: bool = False
    ) -> List[Dict]:
        """Moderate a batch of texts."""
        return [self.moderate(text, return_details) for text in texts]
    
    def get_statistics(
        self,
        texts: List[str]
    ) -> Dict:
        """Get moderation statistics for a dataset."""
        results = self.batch_moderate(texts, return_details=True)
        
        stats = {
            'total': len(results),
            'auto_remove': sum(1 for r in results if r['action'] == 'auto_remove'),
            'human_review': sum(1 for r in results if r['action'] == 'human_review'),
            'auto_approve': sum(1 for r in results if r['action'] == 'auto_approve'),
            'phase1_used': sum(1 for r in results if r['model_used'] == 'phase1'),
            'phase2_used': sum(1 for r in results if r['model_used'] == 'phase2'),
            'languages': {},
            'code_mixed': sum(1 for r in results if r['language_info']['is_code_mixed'])
        }
        
        # Language distribution
        for result in results:
            lang = result['language_info']['language']
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
        
        # Percentages
        stats['auto_remove_pct'] = (stats['auto_remove'] / stats['total']) * 100
        stats['human_review_pct'] = (stats['human_review'] / stats['total']) * 100
        stats['auto_approve_pct'] = (stats['auto_approve'] / stats['total']) * 100
        
        return stats


def main():
    """Demo of unified inference pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified content moderation")
    parser.add_argument('--phase1-model', type=str, required=True,
                       help='Path to Phase 1 model')
    parser.add_argument('--phase2-model', type=str, required=True,
                       help='Path to Phase 2 model')
    parser.add_argument('--language-model', type=str, default=None,
                       help='Path to FastText language model')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Path to calibration results')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to moderate')
    
    args = parser.parse_args()
    
    # Initialize moderator
    moderator = UnifiedContentModerator(
        phase1_model_path=args.phase1_model,
        phase2_model_path=args.phase2_model,
        language_model_path=args.language_model,
        phase1_calibration_path=args.calibration
    )
    
    # Test texts
    test_texts = [
        "This is a toxic comment and you are an idiot!",  # English
        "Aaj tu bahut pagal hai yaar",  # Hinglish
        "Great article, very informative!",  # English positive
        "Kya bakwas hai ye, stop this nonsense",  # Hinglish mixed
    ]
    
    if args.text:
        test_texts = [args.text]
    
    print("\n" + "="*80)
    print("Unified Content Moderation Demo")
    print("="*80)
    
    for text in test_texts:
        result = moderator.moderate(text, return_details=True)
        
        print(f"\nText: {text}")
        print(f"  Language: {result['language_info']['language']} " +
              f"(confidence: {result['language_info']['confidence']:.3f})")
        if result['language_info']['is_code_mixed']:
            print(f"  Code-mixed: {result['language_info']['code_mix_type']}")
        print(f"  Model used: {result['model_used'].upper()}")
        print(f"  Predictions: {result['predictions']}")
        print(f"  ⚡ ACTION: {result['action'].upper().replace('_', ' ')}")
        print(f"  Reasoning: {result['reasoning']}")
    
    # Statistics
    print("\n" + "="*80)
    print("Moderation Statistics")
    print("="*80)
    
    stats = moderator.get_statistics(test_texts)
    print(f"Total: {stats['total']}")
    print(f"Auto-remove: {stats['auto_remove']} ({stats['auto_remove_pct']:.1f}%)")
    print(f"Human review: {stats['human_review']} ({stats['human_review_pct']:.1f}%)")
    print(f"Auto-approve: {stats['auto_approve']} ({stats['auto_approve_pct']:.1f}%)")
    print(f"\nModel usage:")
    print(f"  Phase 1: {stats['phase1_used']}")
    print(f"  Phase 2: {stats['phase2_used']}")


if __name__ == "__main__":
    main()
