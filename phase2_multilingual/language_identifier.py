"""
Phase 2.1: Language Identification & Routing

This module implements language detection using FastText and handles
code-mixed text (Hinglish, Kanglish, Tanglish, etc.)
"""

import re
import string
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageIdentifier:
    """
    Multi-level language identification system.
    
    Features:
    - FastText lid.176.bin model for 176 languages
    - Code-mix detection (script mixing, n-gram analysis)
    - Confidence thresholding
    - Fallback mechanisms
    """
    
    def __init__(
        self,
        fasttext_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        use_fallback: bool = True
    ):
        """
        Initialize language identifier.
        
        Args:
            fasttext_model_path: Path to FastText lid.176.bin model
            confidence_threshold: Minimum confidence for language prediction
            use_fallback: Use langdetect as fallback
        """
        self.confidence_threshold = confidence_threshold
        self.use_fallback = use_fallback
        
        # Try to load FastText
        self.fasttext_model = None
        if fasttext_model_path:
            self.fasttext_model = self._load_fasttext(fasttext_model_path)
        
        # Fallback detector
        self.fallback_detector = None
        if use_fallback:
            try:
                from langdetect import detect_langs
                self.fallback_detector = detect_langs
                logger.info("Loaded langdetect as fallback")
            except ImportError:
                logger.warning("langdetect not available, no fallback")
        
        # Code-mix patterns
        self._init_code_mix_patterns()
    
    def _load_fasttext(self, model_path: str):
        """Load FastText model."""
        try:
            import fasttext
            model = fasttext.load_model(model_path)
            logger.info(f"Loaded FastText model from {model_path}")
            return model
        except ImportError:
            logger.error("fasttext-wheel not installed. Install with: pip install fasttext-wheel")
            return None
        except Exception as e:
            logger.error(f"Failed to load FastText model: {e}")
            logger.info("Download with: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
            return None
    
    def _init_code_mix_patterns(self):
        """Initialize patterns for code-mix detection."""
        
        # Script character ranges
        self.script_ranges = {
            'latin': r'[a-zA-Z]',
            'devanagari': r'[\u0900-\u097F]',  # Hindi, Sanskrit
            'tamil': r'[\u0B80-\u0BFF]',
            'telugu': r'[\u0C00-\u0C7F]',
            'kannada': r'[\u0C80-\u0CFF]',
            'malayalam': r'[\u0D00-\u0D7F]',
            'bengali': r'[\u0980-\u09FF]',
            'gujarati': r'[\u0A80-\u0AFF]',
            'punjabi': r'[\u0A00-\u0A7F]',
        }
        
        # Common code-mix indicators
        self.code_mix_markers = {
            'hinglish': [
                r'\b(kar|hai|tha|ho|kya|nahi|haan|aur|ki|ke|ko)\b',  # Common Hindi words in Roman
                r'\b(mein|aap|tum|main|yaar|bhai|dude)\b',
            ],
            'kanglish': [
                r'\b(alla|ide|aitu|enu|yaar|da|guru)\b',  # Kannada in Roman
            ],
            'tanglish': [
                r'\b(illa|iruku|poda|dai|machaan|ya)\b',  # Tamil in Roman
            ],
            'tenglish': [
                r'\b(ledhu|undi|ra|babu|bro)\b',  # Telugu in Roman
            ]
        }
    
    def detect_script_mix(self, text: str) -> Dict[str, float]:
        """
        Detect script mixing in text.
        
        Returns:
            Dictionary with script proportions
        """
        # Remove punctuation and spaces
        clean_text = ''.join(c for c in text if c not in string.punctuation and c != ' ')
        
        if not clean_text:
            return {'latin': 1.0}
        
        script_counts = {}
        total_chars = len(clean_text)
        
        for script, pattern in self.script_ranges.items():
            matches = re.findall(pattern, clean_text)
            script_counts[script] = len(matches) / total_chars if total_chars > 0 else 0
        
        return script_counts
    
    def detect_code_mix_type(self, text: str) -> Optional[str]:
        """
        Detect specific type of code-mixing.
        
        Returns:
            Code-mix type (e.g., 'hinglish') or None
        """
        text_lower = text.lower()
        
        # Check for script mixing first
        scripts = self.detect_script_mix(text)
        
        # If both Latin and Indic scripts present, it's code-mixed
        has_latin = scripts.get('latin', 0) > 0.3
        has_indic = any(scripts.get(s, 0) > 0.1 for s in ['devanagari', 'tamil', 'telugu', 'kannada'])
        
        if has_latin and has_indic:
            # Identify which Indic script
            for script in ['devanagari', 'tamil', 'telugu', 'kannada']:
                if scripts.get(script, 0) > 0.1:
                    if script == 'devanagari':
                        return 'hinglish'
                    elif script == 'tamil':
                        return 'tanglish'
                    elif script == 'telugu':
                        return 'tenglish'
                    elif script == 'kannada':
                        return 'kanglish'
        
        # Check for romanized code-mix patterns
        if has_latin and not has_indic:
            for mix_type, patterns in self.code_mix_markers.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return mix_type
        
        return None
    
    def detect_language_fasttext(self, text: str) -> Tuple[str, float]:
        """
        Detect language using FastText.
        
        Returns:
            (language_code, confidence)
        """
        if not self.fasttext_model:
            return ('unknown', 0.0)
        
        try:
            # FastText expects single line, no newlines
            text_clean = text.replace('\n', ' ').strip()
            
            if not text_clean:
                return ('unknown', 0.0)
            
            predictions = self.fasttext_model.predict(text_clean, k=1)
            
            # predictions is tuple: (labels, probabilities)
            language = predictions[0][0].replace('__label__', '')
            confidence = float(predictions[1][0])
            
            return (language, confidence)
        
        except Exception as e:
            logger.error(f"FastText prediction failed: {e}")
            return ('unknown', 0.0)
    
    def detect_language_fallback(self, text: str) -> Tuple[str, float]:
        """
        Detect language using langdetect (fallback).
        
        Returns:
            (language_code, confidence)
        """
        if not self.fallback_detector:
            return ('unknown', 0.0)
        
        try:
            results = self.fallback_detector(text)
            if results:
                return (results[0].lang, results[0].prob)
            return ('unknown', 0.0)
        except Exception as e:
            logger.debug(f"Fallback detection failed: {e}")
            return ('unknown', 0.0)
    
    def identify(self, text: str, return_all: bool = False) -> Dict:
        """
        Main language identification method.
        
        Args:
            text: Input text
            return_all: Return detailed information
        
        Returns:
            Dictionary with:
            - language: Detected language code
            - confidence: Confidence score
            - is_code_mixed: Boolean
            - code_mix_type: Type of code-mixing (if applicable)
            - script_distribution: Script proportions (if return_all)
        """
        if not text or not text.strip():
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'is_code_mixed': False,
                'code_mix_type': None
            }
        
        # 1. Check for code-mixing
        code_mix_type = self.detect_code_mix_type(text)
        is_code_mixed = code_mix_type is not None
        
        # 2. Detect primary language
        language, confidence = self.detect_language_fasttext(text)
        
        # 3. Use fallback if confidence is low
        if confidence < self.confidence_threshold and self.use_fallback:
            lang_fallback, conf_fallback = self.detect_language_fallback(text)
            if conf_fallback > confidence:
                language, confidence = lang_fallback, conf_fallback
        
        result = {
            'language': language,
            'confidence': float(confidence),
            'is_code_mixed': is_code_mixed,
            'code_mix_type': code_mix_type
        }
        
        if return_all:
            result['script_distribution'] = self.detect_script_mix(text)
        
        return result
    
    def batch_identify(self, texts: List[str]) -> List[Dict]:
        """Identify languages for a batch of texts."""
        return [self.identify(text) for text in texts]


class LanguageRouter:
    """
    Routes text to appropriate model based on language.
    
    Routing Logic:
    - English → Phase 1 (BERT/RoBERTa)
    - Code-mixed (Hinglish, etc.) → Phase 2 (XLM-RoBERTa)
    - Regional languages → Phase 2 (XLM-RoBERTa)
    - Unknown → Phase 2 (XLM-RoBERTa as fallback)
    """
    
    def __init__(
        self,
        language_identifier: LanguageIdentifier,
        phase1_languages: List[str] = None,
        phase2_languages: List[str] = None
    ):
        """
        Initialize router.
        
        Args:
            language_identifier: LanguageIdentifier instance
            phase1_languages: Languages supported by Phase 1 model
            phase2_languages: Languages supported by Phase 2 model
        """
        self.identifier = language_identifier
        
        # Default language routing
        self.phase1_languages = phase1_languages or ['en', 'english']
        self.phase2_languages = phase2_languages or [
            'hi', 'hindi', 'ta', 'tamil', 'te', 'telugu', 
            'kn', 'kannada', 'ml', 'malayalam', 'bn', 'bengali'
        ]
    
    def route(self, text: str) -> Dict:
        """
        Route text to appropriate model.
        
        Returns:
            Dictionary with:
            - model: 'phase1' or 'phase2'
            - language_info: Language detection results
            - reasoning: Why this route was chosen
        """
        # Identify language
        lang_info = self.identifier.identify(text, return_all=True)
        
        # Routing decision
        language = lang_info['language']
        is_code_mixed = lang_info['is_code_mixed']
        confidence = lang_info['confidence']
        
        # Rule 1: Code-mixed text always goes to Phase 2
        if is_code_mixed:
            return {
                'model': 'phase2',
                'language_info': lang_info,
                'reasoning': f"Code-mixed text ({lang_info['code_mix_type']})"
            }
        
        # Rule 2: English goes to Phase 1
        if language in self.phase1_languages and confidence > 0.7:
            return {
                'model': 'phase1',
                'language_info': lang_info,
                'reasoning': f"English text (confidence: {confidence:.2f})"
            }
        
        # Rule 3: Regional languages go to Phase 2
        if language in self.phase2_languages:
            return {
                'model': 'phase2',
                'language_info': lang_info,
                'reasoning': f"Regional language: {language}"
            }
        
        # Rule 4: Unknown or low confidence → Phase 2 (more robust)
        return {
            'model': 'phase2',
            'language_info': lang_info,
            'reasoning': f"Fallback to multilingual model (lang: {language}, conf: {confidence:.2f})"
        }
    
    def batch_route(self, texts: List[str]) -> List[Dict]:
        """Route a batch of texts."""
        return [self.route(text) for text in texts]
    
    def get_routing_statistics(self, texts: List[str]) -> Dict:
        """
        Analyze routing distribution for a dataset.
        
        Returns statistics on how texts would be routed.
        """
        routes = self.batch_route(texts)
        
        stats = {
            'total': len(routes),
            'phase1_count': sum(1 for r in routes if r['model'] == 'phase1'),
            'phase2_count': sum(1 for r in routes if r['model'] == 'phase2'),
            'code_mixed_count': sum(1 for r in routes if r['language_info']['is_code_mixed']),
            'languages': {},
            'code_mix_types': {}
        }
        
        # Language distribution
        for route in routes:
            lang = route['language_info']['language']
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
            
            if route['language_info']['code_mix_type']:
                mix_type = route['language_info']['code_mix_type']
                stats['code_mix_types'][mix_type] = stats['code_mix_types'].get(mix_type, 0) + 1
        
        # Percentages
        stats['phase1_percentage'] = (stats['phase1_count'] / stats['total']) * 100
        stats['phase2_percentage'] = (stats['phase2_count'] / stats['total']) * 100
        stats['code_mixed_percentage'] = (stats['code_mixed_count'] / stats['total']) * 100
        
        return stats


def download_fasttext_model(output_dir: str = "./models") -> str:
    """
    Download FastText language identification model.
    
    Returns:
        Path to downloaded model
    """
    import urllib.request
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "lid.176.bin"
    
    if model_path.exists():
        logger.info(f"FastText model already exists at {model_path}")
        return str(model_path)
    
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    
    logger.info(f"Downloading FastText model from {url}")
    logger.info("This may take a few minutes (131 MB)...")
    
    try:
        urllib.request.urlretrieve(url, model_path)
        logger.info(f"✓ Downloaded FastText model to {model_path}")
        return str(model_path)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info(f"Please download manually from {url}")
        raise


if __name__ == "__main__":
    # Example usage and testing
    
    print("="*60)
    print("Language Identification Demo")
    print("="*60)
    
    # Download model if needed
    try:
        model_path = download_fasttext_model("models")
    except:
        print("\nNote: Run this to download the model manually:")
        print("wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
        model_path = None
    
    # Initialize identifier
    identifier = LanguageIdentifier(
        fasttext_model_path=model_path,
        confidence_threshold=0.5,
        use_fallback=True
    )
    
    # Test cases
    test_texts = [
        "This is a toxic comment and you should be ashamed!",  # English
        "यह एक अच्छा उदाहरण है",  # Hindi (Devanagari)
        "Aaj main bahut khush hoon yaar",  # Hinglish
        "Nee yenu maadhthidya guru? Super aythu!",  # Kanglish
        "Enna da pandra? Romba nalladhaa iruku!",  # Tanglish
        "మీరు ఎలా ఉన్నారు?",  # Telugu
        "This is mixed Hindi English text kya kar rahe ho?",  # Hinglish with English
    ]
    
    print("\n" + "="*60)
    print("Language Detection Results")
    print("="*60)
    
    for text in test_texts:
        result = identifier.identify(text, return_all=True)
        print(f"\nText: {text[:50]}...")
        print(f"  Language: {result['language']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Code-mixed: {result['is_code_mixed']}")
        if result['code_mix_type']:
            print(f"  Code-mix type: {result['code_mix_type']}")
        if 'script_distribution' in result:
            scripts = {k: f"{v:.2f}" for k, v in result['script_distribution'].items() if v > 0}
            print(f"  Scripts: {scripts}")
    
    # Test routing
    print("\n" + "="*60)
    print("Routing Demo")
    print("="*60)
    
    router = LanguageRouter(identifier)
    
    for text in test_texts:
        route = router.route(text)
        print(f"\nText: {text[:50]}...")
        print(f"  → Route to: {route['model'].upper()}")
        print(f"  Reasoning: {route['reasoning']}")
    
    # Statistics
    print("\n" + "="*60)
    print("Routing Statistics")
    print("="*60)
    
    stats = router.get_routing_statistics(test_texts)
    print(f"\nTotal texts: {stats['total']}")
    print(f"Phase 1 (English): {stats['phase1_count']} ({stats['phase1_percentage']:.1f}%)")
    print(f"Phase 2 (Multilingual): {stats['phase2_count']} ({stats['phase2_percentage']:.1f}%)")
    print(f"Code-mixed: {stats['code_mixed_count']} ({stats['code_mixed_percentage']:.1f}%)")
    
    print(f"\nLanguage distribution:")
    for lang, count in stats['languages'].items():
        print(f"  {lang}: {count}")
    
    if stats['code_mix_types']:
        print(f"\nCode-mix types:")
        for mix_type, count in stats['code_mix_types'].items():
            print(f"  {mix_type}: {count}")
