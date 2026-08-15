"""
Image Classifier - Train and infer on image classification tasks
Supports NSFW, hate symbols, and violence detection
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
import time
import json


@dataclass
class ClassificationResult:
    """Image classification result"""
    predictions: Dict[str, float]
    predicted_class: str
    confidence: float
    processing_time: float


class ImageClassificationDataset(Dataset):
    """Dataset for image classification"""
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        categories: Optional[List[str]] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Root directory containing train/test folders
            split: 'train', 'test', or 'val'
            transform: Image transformations
            categories: List of category names (auto-detected if None)
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        
        # Auto-detect categories if not provided
        if categories is None:
            self.categories = sorted([
                d.name for d in self.data_dir.iterdir() 
                if d.is_dir()
            ])
        else:
            self.categories = categories
        
        self.class_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_class = {idx: cat for cat, idx in self.class_to_idx.items()}
        
        # Load image paths
        self.samples = []
        for category in self.categories:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                continue
            
            for img_path in category_dir.glob('*.jpg'):
                self.samples.append((img_path, self.class_to_idx[category]))
            for img_path in category_dir.glob('*.png'):
                self.samples.append((img_path, self.class_to_idx[category]))
        
        if not self.samples:
            raise ValueError(f"No images found in {self.data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class"""
        counts = {cat: 0 for cat in self.categories}
        for _, label in self.samples:
            counts[self.idx_to_class[label]] += 1
        return counts


class ImageClassifier(nn.Module):
    """
    Image classifier based on pretrained models
    Supports ResNet, EfficientNet, ViT
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize classifier
        
        Args:
            backbone: Model architecture (resnet50, efficientnet_b0, vit_b_16)
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        elif backbone == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_transforms(split: str = 'train', image_size: int = 224) -> transforms.Compose:
    """
    Get image transformations
    
    Args:
        split: 'train' or 'test'
        image_size: Target image size
    
    Returns:
        Composed transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class ImageClassifierInference:
    """
    Inference wrapper for trained image classifiers
    """
    
    def __init__(
        self,
        model_path: Path,
        device: Optional[torch.device] = None
    ):
        """
        Initialize inference
        
        Args:
            model_path: Path to saved model checkpoint
            device: Torch device (auto-detected if None)
        """
        self.logger = logging.getLogger(__name__)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract metadata
        self.categories = checkpoint['categories']
        self.num_classes = len(self.categories)
        self.idx_to_class = {i: cat for i, cat in enumerate(self.categories)}
        
        # Create model
        self.model = ImageClassifier(
            backbone=checkpoint.get('backbone', 'resnet50'),
            num_classes=self.num_classes,
            pretrained=False
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transforms('test', image_size=224)
        
        self.logger.info(f"✅ Loaded model from {model_path}")
        self.logger.info(f"   Categories: {self.categories}")
    
    def predict(
        self,
        image_path: Path,
        return_probabilities: bool = True
    ) -> ClassificationResult:
        """
        Predict on single image
        
        Args:
            image_path: Path to image
            return_probabilities: Return class probabilities
        
        Returns:
            ClassificationResult
        """
        start_time = time.time()
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            
            if return_probabilities:
                probs = torch.softmax(logits, dim=1)[0]
            else:
                probs = logits[0]
        
        # Get predictions
        probs_np = probs.cpu().numpy()
        predicted_idx = int(np.argmax(probs_np))
        predicted_class = self.idx_to_class[predicted_idx]
        confidence = float(probs_np[predicted_idx])
        
        # Create predictions dict
        predictions = {
            self.idx_to_class[i]: float(probs_np[i])
            for i in range(self.num_classes)
        }
        
        processing_time = time.time() - start_time
        
        return ClassificationResult(
            predictions=predictions,
            predicted_class=predicted_class,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def predict_batch(
        self,
        image_paths: List[Path],
        batch_size: int = 32
    ) -> List[ClassificationResult]:
        """
        Predict on batch of images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size
        
        Returns:
            List of ClassificationResults
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load and transform batch
            images = []
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                image_tensor = self.transform(image)
                images.append(image_tensor)
            
            batch_tensor = torch.stack(images).to(self.device)
            
            # Predict
            start_time = time.time()
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
            
            processing_time = time.time() - start_time
            
            # Convert to results
            probs_np = probs.cpu().numpy()
            for j, probs_single in enumerate(probs_np):
                predicted_idx = int(np.argmax(probs_single))
                predicted_class = self.idx_to_class[predicted_idx]
                confidence = float(probs_single[predicted_idx])
                
                predictions = {
                    self.idx_to_class[k]: float(probs_single[k])
                    for k in range(self.num_classes)
                }
                
                results.append(ClassificationResult(
                    predictions=predictions,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    processing_time=processing_time / len(batch_paths)
                ))
        
        return results


def compute_class_weights(dataset: ImageClassificationDataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        dataset: Dataset
    
    Returns:
        Tensor of class weights
    """
    class_counts = dataset.get_class_counts()
    total = sum(class_counts.values())
    
    weights = []
    for cat in dataset.categories:
        count = class_counts[cat]
        weight = total / (len(class_counts) * count) if count > 0 else 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test classifier
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Image Classifier")
    print("="*60)
    
    # Create dummy model
    model = ImageClassifier(backbone='resnet18', num_classes=2)
    print(f"✅ Created model: {model.backbone_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"✅ Forward pass: {output.shape}")
    
    print("="*60)
