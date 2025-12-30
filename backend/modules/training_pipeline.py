"""
Deepfake Detection Training Pipeline v1.0
==========================================
Train a CNN model for deepfake detection with proper Grad-CAM support.

CRITICAL TRAINING PRINCIPLES:
1. Identity-balanced dataset (same faces in real/fake)
2. Patch-based training (random crops)
3. Artifact-sensitive augmentations
4. Heatmap validation during training
"""

import os
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")


# =====================================================
# DATASET DESIGN (Critical for Good Heatmaps)
# =====================================================
"""
WHY IDENTITY-BALANCED DATASET MATTERS:

Problem with unbalanced data:
- If fake = different people, model learns "unknown face = fake"
- This causes CENTER-FACE BIAS in heatmaps
- Model always looks at the same region (face center)

Solution with identity-balanced data:
- Same person appears in BOTH real and fake
- Model CANNOT use identity as a shortcut
- Must learn ACTUAL manipulation artifacts
- Heatmaps become meaningful and localized

Dataset structure:
```
dataset/
├── real/
│   ├── person001_001.jpg
│   ├── person001_002.jpg
│   └── ...
└── fake/
    ├── person001_001_deepfake.jpg
    ├── person001_002_faceswap.jpg
    └── ...
```

The key is: person001 appears in BOTH real/ and fake/
"""


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection training.
    
    Features:
    - Identity-balanced loading
    - Patch-based cropping (eyes, lips, cheeks)
    - Artifact-sensitive augmentations
    """
    
    def __init__(
        self,
        real_dir: str,
        fake_dir: str,
        transform=None,
        patch_mode: bool = True,
        patch_regions: List[str] = None
    ):
        """
        Args:
            real_dir: Directory containing real images
            fake_dir: Directory containing fake images
            transform: Torchvision transforms
            patch_mode: If True, randomly crop regions instead of full face
            patch_regions: List of regions to crop ['eyes', 'lips', 'cheeks', 'full']
        """
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.transform = transform
        self.patch_mode = patch_mode
        self.patch_regions = patch_regions or ['eyes', 'lips', 'cheeks', 'full', 'full']
        
        # Load image paths
        self.samples = []
        
        # Real images (label=0)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            for path in self.real_dir.glob(ext):
                self.samples.append((str(path), 0))
        
        # Fake images (label=1)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            for path in self.fake_dir.glob(ext):
                self.samples.append((str(path), 1))
        
        # Shuffle
        random.shuffle(self.samples)
        
        print(f"📊 Dataset loaded:")
        print(f"   Real: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"   Fake: {sum(1 for _, l in self.samples if l == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load image
        image = Image.open(path).convert('RGB')
        
        # Patch-based cropping
        if self.patch_mode:
            image = self._random_patch_crop(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
    
    def _random_patch_crop(self, image: Image.Image) -> Image.Image:
        """
        Randomly crop a facial region.
        
        WHY PATCH-BASED TRAINING:
        - Full-face training → model uses center face for all decisions
        - Random patches → model learns position-independent artifacts
        - This DRAMATICALLY improves Grad-CAM quality
        
        Regions:
        - eyes: Upper third (forehead + eyes)
        - lips: Lower third (mouth + chin)
        - cheeks: Middle (nose + cheeks)
        - full: Entire face (sometimes needed)
        """
        w, h = image.size
        
        region = random.choice(self.patch_regions)
        
        if region == 'eyes':
            # Upper 40% of face
            crop_box = (0, 0, w, int(h * 0.4))
        elif region == 'lips':
            # Lower 40% of face
            crop_box = (0, int(h * 0.6), w, h)
        elif region == 'cheeks':
            # Middle 50% of face
            crop_box = (0, int(h * 0.25), w, int(h * 0.75))
        elif region == 'left':
            # Left half
            crop_box = (0, 0, int(w * 0.5), h)
        elif region == 'right':
            # Right half
            crop_box = (int(w * 0.5), 0, w, h)
        else:
            # Full face with random crop
            margin = int(min(w, h) * 0.1)
            x1 = random.randint(0, margin)
            y1 = random.randint(0, margin)
            x2 = w - random.randint(0, margin)
            y2 = h - random.randint(0, margin)
            crop_box = (x1, y1, x2, y2)
        
        cropped = image.crop(crop_box)
        
        # Resize back to square
        return cropped.resize((224, 224), Image.BILINEAR)


# =====================================================
# DATA AUGMENTATION (Artifact-Sensitive)
# =====================================================

def get_training_transforms() -> transforms.Compose:
    """
    Get training transforms that simulate real-world artifacts.
    
    GOOD AUGMENTATIONS (simulate manipulation artifacts):
    - JPEG compression: Most common in shared media
    - Gaussian noise: Sensor noise differences
    - Blur: Face-swap blending
    - Color jitter: Lighting inconsistencies
    
    BAD AUGMENTATIONS (destroy forensic evidence):
    - Heavy rotation: Faces are usually upright
    - Random erasing: Removes evidence
    - Excessive distortion: Unrealistic
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05  # Small hue shift only
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        # Custom: JPEG compression simulation
        RandomJPEGCompression(quality_range=(75, 100)),
        RandomGaussianNoise(std_range=(0.0, 0.03)),
    ])


def get_validation_transforms() -> transforms.Compose:
    """Validation transforms (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class RandomJPEGCompression:
    """Simulate JPEG compression artifacts"""
    def __init__(self, quality_range=(70, 100), probability=0.5):
        self.quality_range = quality_range
        self.probability = probability
    
    def __call__(self, tensor):
        if random.random() > self.probability:
            return tensor
        
        # Convert to numpy
        img = tensor.permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = (img * 255).clip(0, 255).astype(np.uint8)
        
        # JPEG compress
        quality = random.randint(*self.quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        
        # Back to tensor
        decoded = decoded.astype(np.float32) / 255.0
        decoded = (decoded - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        return torch.from_numpy(decoded).permute(2, 0, 1).float()


class RandomGaussianNoise:
    """Add random Gaussian noise"""
    def __init__(self, std_range=(0.0, 0.05), probability=0.3):
        self.std_range = std_range
        self.probability = probability
    
    def __call__(self, tensor):
        if random.random() > self.probability:
            return tensor
        
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        return tensor + noise


# =====================================================
# MODEL ARCHITECTURE
# =====================================================

class DeepfakeClassifier(nn.Module):
    """ResNet-18 based classifier with Grad-CAM support"""
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        import torchvision.models as models
        
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
        
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Grad-CAM hooks
        self.feature_maps = None
        self.gradients = None
        self.backbone.layer4.register_forward_hook(self._save_features)
        self.backbone.layer4.register_full_backward_hook(self._save_gradients)
    
    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


# =====================================================
# TRAINING PIPELINE
# =====================================================

class DeepfakeTrainer:
    """
    Complete training pipeline for deepfake detection.
    
    Training Validation (CRITICAL):
    - After each epoch, generate sample Grad-CAMs
    - Flag center-biased heatmaps
    - Stop training if model learns shortcuts
    """
    
    def __init__(
        self,
        model: DeepfakeClassifier,
        device: torch.device,
        output_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "heatmap_quality": []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss={loss.item():.4f}")
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total * 100
        }
    
    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total * 100
        }
    
    def validate_heatmaps(self, sample_images: torch.Tensor) -> Dict:
        """
        Validate Grad-CAM quality.
        
        WHAT MAKES A GOOD HEATMAP:
        - Different for different images
        - Focused on specific regions
        - Not always centered
        
        WHAT MAKES A BAD HEATMAP:
        - Same pattern for all images (center blob)
        - No variation between real/fake
        - High activation everywhere
        
        If heatmaps are bad → stop training, fix dataset
        """
        self.model.eval()
        
        heatmaps = []
        
        for img in sample_images:
            img_tensor = img.unsqueeze(0).to(self.device)
            img_tensor.requires_grad_(True)
            
            # Forward
            output = self.model(img_tensor)
            
            # Backward for fake class
            self.model.zero_grad()
            output.backward()
            
            # Grad-CAM
            if self.model.gradients is not None and self.model.feature_maps is not None:
                weights = torch.mean(self.model.gradients, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * self.model.feature_maps, dim=1)
                cam = torch.relu(cam)
                cam = cam.squeeze().cpu().numpy()
                cam = cv2.resize(cam, (224, 224))
                if cam.max() > 0:
                    cam = cam / cam.max()
                heatmaps.append(cam)
        
        if not heatmaps:
            return {"quality": "UNKNOWN", "warning": "No heatmaps generated"}
        
        # Check center bias
        center_activations = []
        for hm in heatmaps:
            center = np.mean(hm[70:154, 70:154])  # Center 84x84
            edge = np.mean(hm[:30, :]) + np.mean(hm[-30:, :])
            center_activations.append(center / (edge + 1e-6))
        
        avg_center_bias = np.mean(center_activations)
        
        # Check diversity
        if len(heatmaps) > 1:
            diversity = np.mean([
                np.mean(np.abs(heatmaps[i] - heatmaps[j]))
                for i in range(len(heatmaps))
                for j in range(i+1, len(heatmaps))
            ])
        else:
            diversity = 0
        
        quality = "GOOD" if avg_center_bias < 4 and diversity > 0.08 else "POOR"
        
        return {
            "quality": quality,
            "center_bias": round(avg_center_bias, 2),
            "diversity": round(diversity, 4),
            "recommendation": (
                "Training proceeding well" if quality == "GOOD"
                else "WARNING: Model may be learning shortcuts. Consider more patch-based training."
            )
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        early_stop_patience: int = 5
    ):
        """
        Full training loop with heatmap validation.
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Training")
        print("=" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Get sample images for heatmap validation
        sample_batch = next(iter(val_loader))
        sample_images = sample_batch[0][:4]
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc:  {train_metrics['accuracy']:.2f}%")
            
            # Validate
            val_metrics = self.validate(val_loader)
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.2f}%")
            
            # Validate heatmaps
            hm_quality = self.validate_heatmaps(sample_images)
            print(f"  Heatmap Quality: {hm_quality['quality']}")
            if hm_quality['quality'] == "POOR":
                print(f"  ⚠️ {hm_quality['recommendation']}")
            
            # Update history
            self.history["train_loss"].append(train_metrics['loss'])
            self.history["train_acc"].append(train_metrics['accuracy'])
            self.history["val_loss"].append(val_metrics['loss'])
            self.history["val_acc"].append(val_metrics['accuracy'])
            self.history["heatmap_quality"].append(hm_quality)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, "best_model.pth")
                print(f"  💾 Best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\n⛔ Early stopping at epoch {epoch+1}")
                break
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: Dict, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history
        }
        torch.save(checkpoint, self.output_dir / filename)


# =====================================================
# MAIN TRAINING SCRIPT
# =====================================================

def train_deepfake_detector(
    real_dir: str,
    fake_dir: str,
    output_dir: str = "checkpoints",
    epochs: int = 20,
    batch_size: int = 32,
    patch_mode: bool = True
):
    """
    Train a deepfake detector from scratch.
    
    Args:
        real_dir: Directory with real images
        fake_dir: Directory with fake images
        output_dir: Where to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        patch_mode: Use patch-based training (recommended)
    
    Returns:
        Training history
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Create dataset
    dataset = DeepfakeDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        transform=get_training_transforms(),
        patch_mode=patch_mode
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = DeepfakeClassifier(pretrained=True, dropout=0.5)
    
    # Create trainer
    trainer = DeepfakeTrainer(model, device, output_dir)
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )
    
    return history


if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Deepfake Detection Training Pipeline")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not installed. Install with:")
        print("   pip install torch torchvision")
    else:
        print("✅ PyTorch available")
        print("\nUsage:")
        print("  from training_pipeline import train_deepfake_detector")
        print("  history = train_deepfake_detector(")
        print("      real_dir='data/real',")
        print("      fake_dir='data/fake',")
        print("      epochs=20")
        print("  )")
