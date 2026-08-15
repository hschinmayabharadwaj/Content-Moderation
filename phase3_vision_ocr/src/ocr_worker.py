"""
OCR Worker - Extract text from images using multiple OCR engines
Supports EasyOCR (primary) and Tesseract (fallback)
"""

import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
from dataclasses import dataclass

# OCR engines
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install with: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available. Install with: pip install pytesseract")


@dataclass
class OCRResult:
    """OCR extraction result"""
    text: str
    confidence: float
    engine: str
    processing_time: float
    metadata: Dict
    bounding_boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)


class ImagePreprocessor:
    """Image preprocessing utilities for improved OCR"""
    
    @staticmethod
    def resize_image(image: np.ndarray, max_dimension: int = 1920) -> np.ndarray:
        """Resize image if too large"""
        height, width = image.shape[:2]
        
        if max(height, width) <= max_dimension:
            return image
        
        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Remove noise"""
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return denoised
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Deskew image using detected text orientation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is None:
            return image
        
        # Calculate average angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:  # Only consider reasonable angles
                angles.append(angle)
        
        if not angles:
            return image
        
        median_angle = np.median(angles)
        
        # Rotate image
        if abs(median_angle) > 0.5:  # Only rotate if significant
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    @staticmethod
    def binarize(image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return binary


class OCRWorker:
    """
    Main OCR worker supporting multiple engines
    """
    
    def __init__(
        self,
        primary_engine: str = "easyocr",
        fallback_engine: str = "tesseract",
        languages: List[str] = ["en"],
        gpu: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Initialize OCR worker
        
        Args:
            primary_engine: Primary OCR engine ("easyocr" or "tesseract")
            fallback_engine: Fallback engine if primary fails
            languages: List of language codes (e.g., ["en", "hi", "ta"])
            gpu: Use GPU if available
            config: Additional configuration
        """
        self.logger = logging.getLogger(__name__)
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.languages = languages
        self.gpu = gpu
        self.config = config or {}
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Initialize engines
        self.easyocr_reader = None
        self.tesseract_available = TESSERACT_AVAILABLE
        
        if primary_engine == "easyocr" and EASYOCR_AVAILABLE:
            self.logger.info(f"Initializing EasyOCR for languages: {languages}")
            self.easyocr_reader = easyocr.Reader(
                languages,
                gpu=gpu,
                verbose=False
            )
            self.logger.info("✅ EasyOCR initialized")
        elif primary_engine == "tesseract" and not TESSERACT_AVAILABLE:
            self.logger.warning("Tesseract not available, falling back to EasyOCR")
            if EASYOCR_AVAILABLE:
                self.easyocr_reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with OpenCV
        image = cv2.imread(str(image_path))
        
        if image is None:
            # Try with PIL as fallback
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    
    def preprocess_image(
        self,
        image: np.ndarray,
        grayscale: bool = True,
        enhance_contrast: bool = True,
        denoise: bool = True,
        deskew: bool = True,
        resize_max: int = 1920
    ) -> np.ndarray:
        """
        Apply preprocessing pipeline
        
        Args:
            image: Input image
            grayscale: Convert to grayscale
            enhance_contrast: Enhance contrast with CLAHE
            denoise: Remove noise
            deskew: Correct skew
            resize_max: Maximum dimension
        
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Resize if too large
        processed = self.preprocessor.resize_image(processed, resize_max)
        
        # Denoise (before grayscale for better results)
        if denoise:
            processed = self.preprocessor.denoise(processed)
        
        # Deskew (before grayscale)
        if deskew:
            try:
                processed = self.preprocessor.deskew(processed)
            except Exception as e:
                self.logger.warning(f"Deskew failed: {e}")
        
        # Convert to grayscale
        if grayscale:
            processed = self.preprocessor.to_grayscale(processed)
        
        # Enhance contrast
        if enhance_contrast:
            processed = self.preprocessor.enhance_contrast(processed)
        
        return processed
    
    def extract_with_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR"""
        if not self.easyocr_reader:
            raise RuntimeError("EasyOCR not initialized")
        
        start_time = time.time()
        
        # EasyOCR returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
        results = self.easyocr_reader.readtext(image)
        
        processing_time = time.time() - start_time
        
        # Extract text and metadata
        texts = []
        confidences = []
        bounding_boxes = []
        
        for bbox, text, confidence in results:
            texts.append(text)
            confidences.append(confidence)
            
            # Convert bbox to (x, y, w, h) format
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x, y = int(min(x_coords)), int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)
            bounding_boxes.append((x, y, w, h))
        
        # Combine text
        full_text = "\n".join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=float(avg_confidence),
            engine="easyocr",
            processing_time=processing_time,
            metadata={
                "num_detections": len(results),
                "confidences": [float(c) for c in confidences]
            },
            bounding_boxes=bounding_boxes
        )
    
    def extract_with_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract"""
        if not self.tesseract_available:
            raise RuntimeError("Tesseract not available")
        
        start_time = time.time()
        
        # Get detailed output with confidence
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            lang="+".join(self.languages)
        )
        
        # Extract text
        text = pytesseract.image_to_string(image, lang="+".join(self.languages))
        
        processing_time = time.time() - start_time
        
        # Calculate average confidence
        confidences = [
            float(conf) / 100.0
            for conf in data['conf']
            if conf != -1
        ]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Extract bounding boxes for words with decent confidence
        bounding_boxes = []
        for i, conf in enumerate(data['conf']):
            if conf > 0:  # Valid detection
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bounding_boxes.append((x, y, w, h))
        
        return OCRResult(
            text=text.strip(),
            confidence=float(avg_confidence),
            engine="tesseract",
            processing_time=processing_time,
            metadata={
                "num_detections": len([c for c in data['conf'] if c > 0]),
                "confidences": confidences
            },
            bounding_boxes=bounding_boxes
        )
    
    def post_process_text(
        self,
        text: str,
        min_length: int = 3,
        remove_special_chars: bool = False
    ) -> str:
        """
        Post-process extracted text
        
        Args:
            text: Raw OCR text
            min_length: Minimum text length to keep
            remove_special_chars: Remove special characters
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters if requested
        if remove_special_chars:
            import re
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Check minimum length
        if len(text.strip()) < min_length:
            return ""
        
        return text.strip()
    
    def extract_text(
        self,
        image_path: Union[str, Path, np.ndarray],
        preprocess: bool = True,
        min_confidence: float = 0.3,
        min_text_length: int = 3,
        engine: Optional[str] = None
    ) -> OCRResult:
        """
        Extract text from image
        
        Args:
            image_path: Path to image or image array
            preprocess: Apply preprocessing
            min_confidence: Minimum confidence threshold
            min_text_length: Minimum text length
            engine: Force specific engine ("easyocr" or "tesseract")
        
        Returns:
            OCRResult object
        """
        # Load image
        if isinstance(image_path, np.ndarray):
            image = image_path
        else:
            image = self.load_image(image_path)
        
        # Preprocess
        if preprocess:
            preprocessed = self.preprocess_image(
                image,
                grayscale=self.config.get('preprocessing', {}).get('grayscale', True),
                enhance_contrast=self.config.get('preprocessing', {}).get('contrast_enhancement', True),
                denoise=self.config.get('preprocessing', {}).get('denoise', True),
                deskew=self.config.get('preprocessing', {}).get('deskew', True),
                resize_max=self.config.get('preprocessing', {}).get('resize_max', 1920)
            )
        else:
            preprocessed = image
        
        # Select engine
        use_engine = engine or self.primary_engine
        
        # Try extraction
        result = None
        try:
            if use_engine == "easyocr" and self.easyocr_reader:
                result = self.extract_with_easyocr(preprocessed)
            elif use_engine == "tesseract" and self.tesseract_available:
                result = self.extract_with_tesseract(preprocessed)
            else:
                raise RuntimeError(f"Engine {use_engine} not available")
        except Exception as e:
            self.logger.error(f"OCR failed with {use_engine}: {e}")
            
            # Try fallback
            if self.fallback_engine and self.fallback_engine != use_engine:
                self.logger.info(f"Trying fallback engine: {self.fallback_engine}")
                try:
                    if self.fallback_engine == "easyocr" and self.easyocr_reader:
                        result = self.extract_with_easyocr(preprocessed)
                    elif self.fallback_engine == "tesseract" and self.tesseract_available:
                        result = self.extract_with_tesseract(preprocessed)
                except Exception as e2:
                    self.logger.error(f"Fallback OCR also failed: {e2}")
        
        if result is None:
            return OCRResult(
                text="",
                confidence=0.0,
                engine="none",
                processing_time=0.0,
                metadata={"error": "All OCR engines failed"},
                bounding_boxes=[]
            )
        
        # Post-process text
        result.text = self.post_process_text(
            result.text,
            min_length=min_text_length,
            remove_special_chars=self.config.get('remove_special_chars', False)
        )
        
        # Filter by confidence
        if result.confidence < min_confidence:
            self.logger.warning(
                f"OCR confidence {result.confidence:.2f} below threshold {min_confidence}"
            )
        
        return result
    
    def visualize_detections(
        self,
        image: np.ndarray,
        result: OCRResult,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Visualize OCR detections on image
        
        Args:
            image: Original image
            result: OCR result
            output_path: Optional path to save visualization
        
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        # Draw bounding boxes
        for bbox in result.bounding_boxes:
            x, y, w, h = bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text annotation
        cv2.putText(
            vis_image,
            f"{result.engine.upper()} | Conf: {result.confidence:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
        
        return vis_image


# Convenience function
def extract_text_from_image(
    image_path: Union[str, Path],
    engine: str = "easyocr",
    languages: List[str] = ["en"],
    gpu: bool = True,
    min_confidence: float = 0.3
) -> str:
    """
    Convenience function to extract text from image
    
    Args:
        image_path: Path to image
        engine: OCR engine to use
        languages: List of languages
        gpu: Use GPU
        min_confidence: Minimum confidence
    
    Returns:
        Extracted text
    """
    worker = OCRWorker(
        primary_engine=engine,
        languages=languages,
        gpu=gpu
    )
    
    result = worker.extract_text(image_path, min_confidence=min_confidence)
    return result.text


if __name__ == "__main__":
    # Test OCR worker
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_worker.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("🔍 Testing OCR Worker")
    print("=" * 50)
    
    worker = OCRWorker(primary_engine="easyocr", languages=["en"], gpu=True)
    result = worker.extract_text(image_path)
    
    print(f"\n📄 Extracted Text:")
    print(f"{result.text}")
    print(f"\n📊 Metadata:")
    print(f"  Engine: {result.engine}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Processing Time: {result.processing_time:.2f}s")
    print(f"  Detections: {result.metadata.get('num_detections', 0)}")
    print("=" * 50)
