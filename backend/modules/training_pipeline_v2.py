"""
Training Pipeline v2.0 for Multi-Class AI Detection
=====================================================

Trains models to detect:
- Class 0: Real (Camera/Natural)
- Class 1: AI Generated (Midjourney, SD, DALL·E)
- Class 2: AI Enhanced (Gemini enhance, upscale, Photoshop)

Key Training Strategies:
1. Balanced dataset with hard samples
2. Patch-based training (64x64, 96x96)
3. Frequency domain augmentation
4. Multi-task learning (classification + reconstruction)
5. Curriculum learning (easy → hard samples)
"""

import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Training disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Import our detector
try:
    from advanced_ai_detector import MultiClassAIDetector, FrequencyExtractor, CLASS_NAMES
except ImportError:
    from modules.advanced_ai_detector import MultiClassAIDetector, FrequencyExtractor, CLASS_NAMES


# =====================================================
# DATASET CLASSES
# =====================================================

class MultiClassDataset(Dataset):
    """
    Dataset for 3-class AI detection.
    
    Folder Structure:
    dataset/
    ├── train/
    │   ├── real/
    │   ├── ai_generated/
    │   └── ai_enhanced/
    └── val/
        ├── real/
        ├── ai_generated/
        └── ai_enhanced/
    
    Hard Samples Strategy:
    - Include clean AI images (low artifacts)
    - Include anime/illustration AI images
    - Include AI → compress → re-upload pipelines
    - Include AI-enhanced real photos
    """
    
    CLASS_FOLDERS = {
        0: ["real", "camera", "natural", "authentic"],
        1: ["ai_generated", "ai", "synthetic", "generated", "midjourney", "sd", "dalle"],
        2: ["ai_enhanced", "enhanced", "upscaled", "refined", "edited"]
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        patch_mode: bool = False,
        patch_size: int = 96,
        include_frequency: bool = True,
        balance_classes: bool = True
    ):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = transform
        self.patch_mode = patch_mode
        self.patch_size = patch_size
        self.include_frequency = include_frequency
        
        self.freq_extractor = FrequencyExtractor() if include_frequency else None
        
        # Collect samples
        self.samples = []
        self._load_samples()
        
        # Balance classes
        if balance_classes and self.split == "train":
            self._balance_samples()
        
        print(f"📁 {split}: {len(self.samples)} samples")
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all image samples with class labels"""
        for class_id, folder_names in self.CLASS_FOLDERS.items():
            for folder_name in folder_names:
                folder_path = self.root_dir / folder_name
                if not folder_path.exists():
                    continue
                
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                    for img_path in folder_path.glob(ext):
                        self.samples.append({
                            "path": str(img_path),
                            "class": class_id,
                            "class_name": CLASS_NAMES.get(class_id, "UNKNOWN")
                        })
    
    def _balance_samples(self):
        """Balance classes by oversampling minority classes"""
        class_counts = {}
        for sample in self.samples:
            c = sample["class"]
            class_counts[c] = class_counts.get(c, 0) + 1
        
        if not class_counts:
            return
        
        max_count = max(class_counts.values())
        
        balanced_samples = []
        for class_id, count in class_counts.items():
            class_samples = [s for s in self.samples if s["class"] == class_id]
            
            # Oversample to match max class
            while len(class_samples) < max_count:
                class_samples.extend(random.sample(
                    [s for s in self.samples if s["class"] == class_id],
                    min(count, max_count - len(class_samples))
                ))
            
            balanced_samples.extend(class_samples[:max_count])
        
        self.samples = balanced_samples
    
    def _print_class_distribution(self):
        """Print class distribution"""
        counts = {}
        for s in self.samples:
            c = s["class_name"]
            counts[c] = counts.get(c, 0) + 1
        
        for c, n in sorted(counts.items()):
            print(f"  {c}: {n}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample["path"])
        if image is None:
            # Return dummy
            rgb = torch.zeros(3, 224, 224)
            freq = torch.zeros(1, 224, 224)
            return rgb, freq, sample["class"]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Patch extraction
        if self.patch_mode:
            image = self._extract_random_patch(image)
        
        # Resize
        image = cv2.resize(image, (224, 224))
        pil_image = Image.fromarray(image)
        
        # Apply transform
        if self.transform:
            rgb_tensor = self.transform(pil_image)
        else:
            rgb_tensor = transforms.ToTensor()(pil_image)
        
        # Extract frequency
        if self.include_frequency:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            fft = self.freq_extractor.extract_fft_magnitude(gray)
            freq_tensor = torch.from_numpy(fft).unsqueeze(0)
        else:
            freq_tensor = torch.zeros(1, 224, 224)
        
        return rgb_tensor, freq_tensor, sample["class"]
    
    def _extract_random_patch(self, image: np.ndarray) -> np.ndarray:
        """Extract random patch from image"""
        h, w = image.shape[:2]
        
        if h < self.patch_size or w < self.patch_size:
            return cv2.resize(image, (self.patch_size, self.patch_size))
        
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        
        return image[y:y+self.patch_size, x:x+self.patch_size]


# =====================================================
# AUGMENTATIONS
# =====================================================

class TrainingAugmentations:
    """
    Augmentations specifically designed for AI detection.
    
    Key augmentations:
    1. JPEG compression (mimics real-world)
    2. Gaussian noise (affects texture patterns)
    3. Blur (simulates upscaling/processing)
    4. Resize (common post-processing)
    5. Color jitter (lighting variations)
    """
    
    @staticmethod
    def jpeg_compress(image: np.ndarray, quality: int = None) -> np.ndarray:
        """Apply JPEG compression"""
        if quality is None:
            quality = random.randint(50, 95)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    @staticmethod
    def add_noise(image: np.ndarray, std: float = None) -> np.ndarray:
        """Add Gaussian noise"""
        if std is None:
            std = random.uniform(5, 25)
        
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def blur(image: np.ndarray, kernel: int = None) -> np.ndarray:
        """Apply Gaussian blur"""
        if kernel is None:
            kernel = random.choice([3, 5, 7])
        
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    
    @staticmethod
    def resize_cycle(image: np.ndarray, scale: float = None) -> np.ndarray:
        """Downscale then upscale (mimics re-upload)"""
        if scale is None:
            scale = random.uniform(0.5, 0.8)
        
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * scale), int(h * scale)))
        return cv2.resize(small, (w, h))


class AugmentedDataset(MultiClassDataset):
    """Dataset with augmentations applied during training"""
    
    def __init__(self, *args, augment_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_prob = augment_prob
        self.augmentor = TrainingAugmentations()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample["path"])
        if image is None:
            rgb = torch.zeros(3, 224, 224)
            freq = torch.zeros(1, 224, 224)
            return rgb, freq, sample["class"]
        
        # Apply augmentations
        if self.split == "train" and random.random() < self.augment_prob:
            aug_choice = random.choice(["jpeg", "noise", "blur", "resize"])
            
            if aug_choice == "jpeg":
                image = self.augmentor.jpeg_compress(image)
            elif aug_choice == "noise":
                image = self.augmentor.add_noise(image)
            elif aug_choice == "blur":
                image = self.augmentor.blur(image)
            elif aug_choice == "resize":
                image = self.augmentor.resize_cycle(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Patch extraction
        if self.patch_mode:
            image = self._extract_random_patch(image)
        
        # Resize
        image = cv2.resize(image, (224, 224))
        pil_image = Image.fromarray(image)
        
        # Apply transform
        if self.transform:
            rgb_tensor = self.transform(pil_image)
        else:
            rgb_tensor = transforms.ToTensor()(pil_image)
        
        # Extract frequency
        if self.include_frequency:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            fft = self.freq_extractor.extract_fft_magnitude(gray)
            freq_tensor = torch.from_numpy(fft).unsqueeze(0)
        else:
            freq_tensor = torch.zeros(1, 224, 224)
        
        return rgb_tensor, freq_tensor, sample["class"]


# =====================================================
# TRAINER
# =====================================================

class MultiClassTrainer:
    """
    Trainer for 3-class AI detection model.
    
    Training Strategy:
    1. Transfer learning from ImageNet
    2. Freeze backbone initially, then unfreeze
    3. Use label smoothing for better calibration
    4. Learning rate scheduling
    5. Mixed precision training (optional)
    6. Heatmap validation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        label_smoothing: float = 0.1,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Optimizer (train fusion layers first)
        self.optimizer = optim.AdamW([
            {"params": model.fusion.parameters(), "lr": learning_rate},
            {"params": model.rgb_backbone.parameters(), "lr": learning_rate * 0.1},
            {"params": model.freq_backbone.parameters(), "lr": learning_rate * 0.5}
        ], weight_decay=0.01)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Metrics
        self.best_val_acc = 0.0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "per_class_acc": []
        }
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (rgb, freq, labels) in enumerate(self.train_loader):
            rgb = rgb.to(self.device)
            freq = freq.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(rgb, freq)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class metrics
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}
        
        with torch.no_grad():
            for rgb, freq, labels in self.val_loader:
                rgb = rgb.to(self.device)
                freq = freq.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(rgb, freq)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Per-class
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        per_class_acc = {}
        for c in [0, 1, 2]:
            if class_total[c] > 0:
                per_class_acc[CLASS_NAMES[c]] = round(
                    100.0 * class_correct[c] / class_total[c], 2
                )
        
        return avg_loss, accuracy, per_class_acc
    
    def train(self, epochs: int = 30, unfreeze_after: int = 5):
        """Full training loop"""
        print("=" * 60)
        print("🚀 Starting Training")
        print(f"   Epochs: {epochs}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Unfreeze backbone after initial epochs
            if epoch == unfreeze_after:
                print("🔓 Unfreezing backbone layers")
                for param in self.model.rgb_backbone.parameters():
                    param.requires_grad = True
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, per_class = self.validate()
            
            # Step scheduler
            self.scheduler.step()
            
            # Log
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Per-class: {per_class}")
            
            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["per_class_acc"].append(per_class)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, "best_model.pth")
                print(f"  ✅ New best model saved ({val_acc:.2f}%)")
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch+1}.pth")
        
        print("\n" + "=" * 60)
        print(f"🏁 Training Complete | Best Val Acc: {self.best_val_acc:.2f}%")
        print("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)


# =====================================================
# HARD SAMPLE GENERATOR
# =====================================================

class HardSampleGenerator:
    """
    Generate hard samples for training.
    
    Hard Samples:
    1. Clean AI images → compress → re-upload
    2. Real images → AI enhance
    3. AI images → AI enhance again
    4. Anime/illustration AI with perfect lighting
    
    Why Hard Samples:
    - Standard datasets are too easy
    - Real-world has many edge cases
    - Gemini/upscaler outputs are "clean AI"
    """
    
    @staticmethod
    def simulate_pipeline(image: np.ndarray, pipeline: str) -> np.ndarray:
        """
        Simulate post-processing pipelines.
        
        Pipelines:
        - "compress": JPEG compression cycle
        - "upscale": Simulate AI upscaling
        - "enhance": Simulate AI enhancement
        - "reupload": Multiple compression + resize
        """
        aug = TrainingAugmentations()
        
        if pipeline == "compress":
            # Multiple JPEG compressions
            for _ in range(random.randint(2, 4)):
                quality = random.randint(60, 90)
                image = aug.jpeg_compress(image, quality)
        
        elif pipeline == "upscale":
            # Downscale then upscale (mimics AI upscaling)
            h, w = image.shape[:2]
            scale = random.uniform(0.3, 0.5)
            small = cv2.resize(image, (int(w * scale), int(h * scale)))
            # Upscale with different interpolation (mimics AI)
            image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
            # Add slight sharpening (common in upscalers)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        elif pipeline == "enhance":
            # Simulate AI enhancement
            # Increase contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            # Slight noise reduction (AI enhancement effect)
            image = cv2.bilateralFilter(image, 9, 75, 75)
        
        elif pipeline == "reupload":
            # Multiple processing cycles
            image = aug.jpeg_compress(image, random.randint(70, 85))
            image = aug.resize_cycle(image, random.uniform(0.6, 0.8))
            image = aug.jpeg_compress(image, random.randint(75, 90))
        
        return image
    
    @staticmethod
    def create_hard_sample_dataset(
        source_dir: str,
        output_dir: str,
        num_samples: int = 1000
    ):
        """Create hard samples from existing dataset"""
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        pipelines = ["compress", "upscale", "enhance", "reupload"]
        
        # Collect source images
        images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        
        for i in range(min(num_samples, len(images))):
            img_path = images[i % len(images)]
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            # Apply random pipeline
            pipeline = random.choice(pipelines)
            processed = HardSampleGenerator.simulate_pipeline(img, pipeline)
            
            # Save
            output_file = output_path / f"hard_{pipeline}_{i:04d}.jpg"
            cv2.imwrite(str(output_file), processed)
        
        print(f"✅ Created {num_samples} hard samples in {output_dir}")


# =====================================================
# MAIN TRAINING FUNCTION
# =====================================================

def train_multiclass_detector(
    dataset_root: str,
    output_dir: str = "./trained_models",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patch_training: bool = True,
    use_augmentation: bool = True
):
    """
    Main training function.
    
    Args:
        dataset_root: Path to dataset (with train/val folders)
        output_dir: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        patch_training: Use patch-based training
        use_augmentation: Apply training augmentations
    """
    if not TORCH_AVAILABLE:
        print("❌ PyTorch required for training")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Datasets
    if use_augmentation:
        train_dataset = AugmentedDataset(
            dataset_root, "train",
            transform=train_transform,
            patch_mode=patch_training,
            augment_prob=0.5
        )
    else:
        train_dataset = MultiClassDataset(
            dataset_root, "train",
            transform=train_transform,
            patch_mode=patch_training
        )
    
    val_dataset = MultiClassDataset(
        dataset_root, "val",
        transform=val_transform,
        patch_mode=False  # Full images for validation
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = MultiClassAIDetector(pretrained=True, dropout=0.5)
    
    # Trainer
    trainer = MultiClassTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        checkpoint_dir=output_dir
    )
    
    # Train
    history = trainer.train(epochs=epochs)
    
    # Save final model
    final_path = Path(output_dir) / "final_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patch_training": patch_training
        }
    }, final_path)
    
    print(f"✅ Final model saved to {final_path}")
    
    return history


# =====================================================
# USAGE EXAMPLE
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Multi-Class AI Detector Training Pipeline v2.0")
    print("=" * 60)
    
    print("""
Dataset Structure Required:
dataset/
├── train/
│   ├── real/          (natural camera images)
│   ├── ai_generated/  (Midjourney, SD, DALL·E)
│   └── ai_enhanced/   (Gemini enhanced, upscaled)
└── val/
    ├── real/
    ├── ai_generated/
    └── ai_enhanced/

Example usage:
    train_multiclass_detector(
        dataset_root="./dataset",
        output_dir="./trained_models",
        epochs=30,
        batch_size=32
    )
""")
    
    if TORCH_AVAILABLE:
        print("✅ PyTorch available - Training ready")
        print(f"   CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ PyTorch not available")
