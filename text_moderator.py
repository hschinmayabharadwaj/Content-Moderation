"""
Simple text moderation wrapper for Colab notebooks.
Provides unified interface to Phase 1 (text) classifier.
"""

import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple


class TextModerator:
    """Text content moderation using Phase 1 baseline classifier."""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize text moderator.
        
        Args:
            model_path: Path to trained model (auto-detects if None)
            config_path: Path to config file (auto-detects if None)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Auto-detect paths
        if model_path is None:
            model_path = 'phase1_text_baseline/models/best_model.pt'
        if config_path is None:
            config_path = 'phase1_text_baseline/configs/baseline.yaml'
        
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.label_names = self.config['labels']['names']
        self.bias_correction_margin = self.config.get('bias_correction', {}).get('margin', 0.25)
        self.bias_correction_threshold = self.config.get('bias_correction', {}).get('threshold', 0.65)
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load trained model and tokenizer."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize tokenizer
        model_name = self.config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize base model
        base_model = AutoModel.from_pretrained(model_name)
        
        # Build classifier
        from phase1_text_baseline.train_classifier import TextClassifier
        self.model = TextClassifier(
            model_name=model_name,
            num_labels=len(self.label_names),
            hidden_dim=self.config['model']['hidden_dim']
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded model from {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Labels: {', '.join(self.label_names)}")
    
    def moderate(self, text: str, return_all_scores: bool = False) -> Dict:
        """
        Moderate text content.
        
        Args:
            text: Text to moderate
            return_all_scores: If True, return scores for all categories
        
        Returns:
            Dict with keys:
            - 'text': Input text
            - 'toxic': Whether text is toxic (bool)
            - 'scores': Dict of category scores
            - 'confidence': Confidence in prediction
            - 'explanation': Human-readable explanation
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        # Convert to probabilities
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply bias correction
        corrected_probs = self._apply_bias_correction(probs)
        
        # Build result
        scores = {label: float(prob) for label, prob in zip(self.label_names, corrected_probs)}
        
        # Determine if toxic (any category > threshold)
        is_toxic = any(prob > self.bias_correction_threshold for prob in corrected_probs)
        
        # Find most likely toxicity
        max_score = max(corrected_probs)
        max_label = self.label_names[np.argmax(corrected_probs)]
        
        # Build explanation
        toxic_categories = [
            label for label, score in zip(self.label_names, corrected_probs)
            if score > self.bias_correction_threshold
        ]
        
        if is_toxic:
            explanation = f"⚠️ TOXIC: Detected {', '.join(toxic_categories)} (confidence: {max_score:.1%})"
        else:
            explanation = f"✅ CLEAN: {max_label} detected ({max_score:.1%} confidence)"
        
        result = {
            'text': text,
            'toxic': is_toxic,
            'confidence': float(max_score),
            'top_category': max_label,
            'scores': scores if return_all_scores else None,
            'explanation': explanation
        }
        
        return result
    
    def _apply_bias_correction(self, probs: np.ndarray) -> np.ndarray:
        """Apply bias correction to probabilities."""
        corrected = probs.copy()
        
        # Apply margin correction
        corrected = np.maximum(corrected - self.bias_correction_margin, 0)
        
        # Clip to [0, 1]
        corrected = np.clip(corrected, 0, 1)
        
        return corrected
    
    def moderate_batch(self, texts: List[str]) -> List[Dict]:
        """Moderate multiple texts."""
        return [self.moderate(text) for text in texts]
    
    def get_label_names(self) -> List[str]:
        """Get list of toxicity categories."""
        return self.label_names


# Convenience function
def moderate_text(text: str, model_path: str = None) -> Dict:
    """Quick moderation without initializing moderator."""
    moderator = TextModerator(model_path=model_path)
    return moderator.moderate(text)


if __name__ == '__main__':
    # Example usage
    moderator = TextModerator()
    
    test_texts = [
        "This is a friendly message",
        "I hate this",
        "Have a great day!",
    ]
    
    print("🧪 Testing Text Moderation:\n")
    for text in test_texts:
        result = moderator.moderate(text, return_all_scores=True)
        print(f"Text: {text}")
        print(f"  Result: {result['explanation']}")
        print(f"  Scores: {result['scores']}")
        print()
