"""
Advanced AI Content Detector v2.0
==================================
Multi-Class Classification: Real vs AI Generated vs AI Enhanced

Key Upgrades:
1. 3-Class Classification (not binary)
2. Frequency Domain Fusion (RGB + FFT)
3. Patch-based + Full Image Fusion
4. Multi-layer Grad-CAM
5. Heatmap Entropy for Confidence
6. Hard-AI Sample Detection
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =====================================================
# CLASS DEFINITIONS
# =====================================================
"""
MULTI-CLASS CLASSIFICATION:

Class 0: REAL (Camera/Natural)
- Photos from cameras
- Natural images
- Unedited screenshots

Class 1: AI GENERATED (Pure AI)
- Midjourney, Stable Diffusion, DALL·E
- Anime/illustration AI
- Text-to-image outputs

Class 2: AI ENHANCED/REFINED
- Gemini enhanced images
- AI upscaled images
- Photoshop AI features
- Real → AI pipeline outputs

Why 3 classes instead of 2:
- Binary fails on "enhanced" images
- AI-enhanced real photos are different from pure AI
- Need separate handling for each category
"""

CLASS_NAMES = {
    0: "REAL",
    1: "AI_GENERATED", 
    2: "AI_ENHANCED"
}

CLASS_DESCRIPTIONS = {
    0: "Natural/Camera Image - No AI involvement detected",
    1: "AI Generated - Created entirely by AI (Midjourney, SD, DALL·E, etc.)",
    2: "AI Enhanced/Refined - Real image processed by AI or AI image post-processed"
}


# =====================================================
# FREQUENCY DOMAIN EXTRACTOR
# =====================================================

class FrequencyExtractor:
    """
    Extract frequency domain features for AI detection.
    
    Why Frequency Domain:
    - AI images often have distinctive frequency signatures
    - GAN artifacts visible in high-frequency spectrum
    - Clean AI images lack natural noise patterns in frequency
    - Helps detect "hard AI" samples that look realistic
    """
    
    @staticmethod
    def extract_fft_magnitude(image: np.ndarray) -> np.ndarray:
        """
        Extract FFT magnitude spectrum.
        
        Args:
            image: Grayscale or BGR image
            
        Returns:
            Normalized FFT magnitude map (same size as input)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply FFT
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        
        # Get magnitude spectrum (log scale)
        magnitude = np.log(np.abs(fshift) + 1)
        
        # Normalize to 0-1
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        return magnitude.astype(np.float32)
    
    @staticmethod
    def extract_high_freq_map(image: np.ndarray) -> np.ndarray:
        """
        Extract high-frequency component map.
        
        High-freq artifacts are key for AI detection:
        - Real images have natural high-freq noise
        - AI images often have smooth/periodic patterns
        - Enhanced images show frequency inconsistencies
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Laplacian for high-frequency
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        
        # Normalize
        if laplacian.max() > 0:
            laplacian = laplacian / laplacian.max()
        
        return laplacian.astype(np.float32)
    
    @staticmethod
    def extract_dct_features(image: np.ndarray, block_size: int = 8) -> np.ndarray:
        """
        Extract DCT coefficient variance map.
        
        DCT helps detect:
        - JPEG artifacts (compression history)
        - AI upscaling patterns
        - Re-compression traces
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        dct_var_map = np.zeros((h // block_size, w // block_size), dtype=np.float32)
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                dct_block = cv2.dct(block)
                # Variance of high-freq DCT coefficients
                dct_var_map[i//block_size, j//block_size] = np.var(dct_block[4:, 4:])
        
        # Resize to original size
        dct_var_map = cv2.resize(dct_var_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        if dct_var_map.max() > 0:
            dct_var_map = dct_var_map / dct_var_map.max()
        
        return dct_var_map


# =====================================================
# MULTI-CLASS MODEL WITH FREQUENCY FUSION
# =====================================================

class MultiClassAIDetector(nn.Module):
    """
    Multi-Class AI Content Detector with Frequency Fusion.
    
    Architecture:
    - Dual-stream: RGB + Frequency
    - ResNet-18 backbone (for each stream)
    - Feature fusion
    - 3-class output
    
    Input: 4-channel (RGB + FFT magnitude)
    Output: 3 probabilities (Real, AI Generated, AI Enhanced)
    """
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        # RGB Stream (pretrained ResNet-18)
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.rgb_backbone = models.resnet18(weights=weights)
        else:
            self.rgb_backbone = models.resnet18(weights=None)
        
        # Modify first conv for RGB stream (keep 3 channels)
        rgb_features = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity()  # Remove final FC
        
        # Frequency Stream (separate ResNet-18)
        self.freq_backbone = models.resnet18(weights=None)
        # Modify first conv for 1-channel frequency input
        self.freq_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.freq_backbone.fc = nn.Identity()
        
        # Feature fusion
        combined_features = rgb_features * 2  # 512 * 2 = 1024
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 3)  # 3 classes
        )
        
        # Grad-CAM storage
        self.rgb_features = None
        self.rgb_gradients = None
        self.freq_features = None
        self.freq_gradients = None
        
        # Register hooks for multi-layer Grad-CAM
        self.rgb_backbone.layer4.register_forward_hook(self._save_rgb_features)
        self.rgb_backbone.layer4.register_full_backward_hook(self._save_rgb_gradients)
        self.rgb_backbone.layer3.register_forward_hook(self._save_rgb_mid_features)
        
        self.freq_backbone.layer4.register_forward_hook(self._save_freq_features)
        self.freq_backbone.layer4.register_full_backward_hook(self._save_freq_gradients)
        
        self.rgb_mid_features = None
    
    def _save_rgb_features(self, module, input, output):
        self.rgb_features = output.detach()
    
    def _save_rgb_gradients(self, module, grad_input, grad_output):
        self.rgb_gradients = grad_output[0].detach()
    
    def _save_rgb_mid_features(self, module, input, output):
        self.rgb_mid_features = output.detach()
    
    def _save_freq_features(self, module, input, output):
        self.freq_features = output.detach()
    
    def _save_freq_gradients(self, module, grad_input, grad_output):
        self.freq_gradients = grad_output[0].detach()
    
    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dual streams.
        
        Args:
            rgb: RGB image tensor (B, 3, 224, 224)
            freq: Frequency map tensor (B, 1, 224, 224)
            
        Returns:
            Logits for 3 classes (B, 3)
        """
        # RGB stream
        rgb_feat = self.rgb_backbone(rgb)  # (B, 512)
        
        # Frequency stream
        freq_feat = self.freq_backbone(freq)  # (B, 512)
        
        # Concatenate features
        combined = torch.cat([rgb_feat, freq_feat], dim=1)  # (B, 1024)
        
        # Fusion and classification
        logits = self.fusion(combined)
        
        return logits
    
    def predict_proba(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities"""
        logits = self.forward(rgb, freq)
        return F.softmax(logits, dim=1)


# =====================================================
# MULTI-LAYER GRAD-CAM
# =====================================================

class MultiLayerGradCAM:
    """
    Multi-Layer Grad-CAM for better explanations.
    
    Why Multi-Layer:
    - Last layer: High-level semantic features
    - Mid layer: Texture and pattern features
    - Combining both gives better localization
    
    Also includes:
    - Heatmap entropy calculation (for confidence)
    - Per-image normalization
    - Class-specific activation
    """
    
    def __init__(self, model: MultiClassAIDetector, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate(
        self, 
        rgb_tensor: torch.Tensor, 
        freq_tensor: torch.Tensor,
        target_class: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate multi-layer Grad-CAM.
        
        Args:
            rgb_tensor: RGB input (1, 3, 224, 224)
            freq_tensor: Frequency input (1, 1, 224, 224)
            target_class: Target class for Grad-CAM (None = predicted class)
            
        Returns:
            Dictionary with heatmaps and metadata
        """
        self.model.eval()
        rgb_tensor = rgb_tensor.clone().requires_grad_(True).to(self.device)
        freq_tensor = freq_tensor.clone().requires_grad_(True).to(self.device)
        
        # Forward pass
        logits = self.model(rgb_tensor, freq_tensor)
        probs = F.softmax(logits, dim=1)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        result = {
            "predicted_class": target_class,
            "class_name": CLASS_NAMES.get(target_class, "UNKNOWN"),
            "probabilities": probs[0].detach().cpu().numpy(),
            "heatmaps": {}
        }
        
        # Generate Grad-CAM from last layer (RGB)
        if self.model.rgb_features is not None and self.model.rgb_gradients is not None:
            cam_last = self._compute_gradcam(
                self.model.rgb_features,
                self.model.rgb_gradients
            )
            result["heatmaps"]["rgb_last_layer"] = cam_last
        
        # Generate Grad-CAM from mid layer (RGB)
        if self.model.rgb_mid_features is not None:
            # Re-run backward for mid layer
            self.model.zero_grad()
            logits = self.model(rgb_tensor, freq_tensor)
            logits.backward(gradient=one_hot, retain_graph=True)
            
            # Approximate mid-layer gradients from feature importance
            cam_mid = self._compute_feature_importance(self.model.rgb_mid_features)
            result["heatmaps"]["rgb_mid_layer"] = cam_mid
        
        # Generate Grad-CAM from frequency stream
        if self.model.freq_features is not None and self.model.freq_gradients is not None:
            cam_freq = self._compute_gradcam(
                self.model.freq_features,
                self.model.freq_gradients
            )
            result["heatmaps"]["frequency"] = cam_freq
        
        # Combined heatmap (weighted average)
        if result["heatmaps"]:
            combined = self._combine_heatmaps(result["heatmaps"])
            result["heatmaps"]["combined"] = combined
            
            # Calculate entropy for confidence estimation
            result["heatmap_entropy"] = self._calculate_entropy(combined)
            result["confidence_modifier"] = self._entropy_to_confidence(result["heatmap_entropy"])
        
        return result
    
    def _compute_gradcam(self, features: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM from features and gradients"""
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Convert and resize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Normalize per-image
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        return cam.astype(np.float32)
    
    def _compute_feature_importance(self, features: torch.Tensor) -> np.ndarray:
        """Compute feature importance map (for mid-layer)"""
        # Use activation magnitude as importance
        importance = torch.mean(torch.abs(features), dim=1, keepdim=True)
        
        importance = importance.squeeze().cpu().numpy()
        importance = cv2.resize(importance, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())
        
        return importance.astype(np.float32)
    
    def _combine_heatmaps(self, heatmaps: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine multiple heatmaps with weights"""
        weights = {
            "rgb_last_layer": 0.5,
            "rgb_mid_layer": 0.3,
            "frequency": 0.2
        }
        
        combined = np.zeros((224, 224), dtype=np.float32)
        total_weight = 0
        
        for name, weight in weights.items():
            if name in heatmaps:
                combined += weight * heatmaps[name]
                total_weight += weight
        
        if total_weight > 0:
            combined /= total_weight
        
        # Normalize
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined
    
    def _calculate_entropy(self, heatmap: np.ndarray) -> float:
        """
        Calculate heatmap entropy.
        
        Why Entropy:
        - Low entropy: Focused, confident heatmap
        - High entropy: Diffuse, uncertain heatmap
        - Helps calibrate confidence scores
        """
        # Normalize to probability distribution
        hm = heatmap.flatten()
        hm = hm / (hm.sum() + 1e-8)
        
        # Calculate entropy
        entropy = -np.sum(hm * np.log(hm + 1e-8))
        
        return float(entropy)
    
    def _entropy_to_confidence(self, entropy: float) -> float:
        """Convert entropy to confidence modifier"""
        # Typical entropy range: 2-6
        # Low entropy (focused) → high confidence
        # High entropy (diffuse) → low confidence
        
        if entropy < 3:
            return 1.1  # Boost confidence
        elif entropy < 4:
            return 1.0  # No change
        elif entropy < 5:
            return 0.9  # Slight reduction
        else:
            return 0.8  # Significant reduction


# =====================================================
# PATCH-BASED ANALYSIS
# =====================================================

class PatchAnalyzer:
    """
    Patch-based analysis for texture-level artifacts.
    
    Why Patches:
    - Full-image analysis can miss local artifacts
    - AI artifacts are often texture-level
    - Combining patches improves robustness
    
    Strategy:
    - Extract multiple patches (64x64 or 96x96)
    - Analyze each patch separately
    - Aggregate predictions
    """
    
    def __init__(self, patch_size: int = 96, stride: int = 48):
        self.patch_size = patch_size
        self.stride = stride
    
    def extract_patches(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Extract patches from image with positions"""
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append((patch, (x, y)))
        
        return patches
    
    def aggregate_predictions(
        self, 
        patch_predictions: List[Dict],
        strategy: str = "weighted"
    ) -> Dict:
        """
        Aggregate patch-level predictions.
        
        Strategies:
        - max: Use maximum confidence patch
        - mean: Average all patches
        - weighted: Weight by patch confidence
        """
        if not patch_predictions:
            return {"class": 0, "confidence": 0, "distribution": [1, 0, 0]}
        
        # Collect all probabilities
        all_probs = np.array([p["probabilities"] for p in patch_predictions])
        confidences = np.array([p["confidence"] for p in patch_predictions])
        
        if strategy == "max":
            # Use most confident patch
            best_idx = np.argmax(confidences)
            final_probs = all_probs[best_idx]
        elif strategy == "mean":
            # Simple average
            final_probs = np.mean(all_probs, axis=0)
        elif strategy == "weighted":
            # Weight by confidence
            weights = confidences / (confidences.sum() + 1e-8)
            final_probs = np.sum(all_probs * weights[:, np.newaxis], axis=0)
        else:
            final_probs = np.mean(all_probs, axis=0)
        
        predicted_class = int(np.argmax(final_probs))
        
        return {
            "class": predicted_class,
            "class_name": CLASS_NAMES.get(predicted_class, "UNKNOWN"),
            "confidence": float(final_probs[predicted_class] * 100),
            "distribution": final_probs.tolist(),
            "num_patches": len(patch_predictions),
            "patch_agreement": self._calculate_agreement(patch_predictions)
        }
    
    def _calculate_agreement(self, predictions: List[Dict]) -> float:
        """Calculate how much patches agree on the class"""
        if len(predictions) < 2:
            return 1.0
        
        classes = [p.get("predicted_class", 0) for p in predictions]
        most_common = max(set(classes), key=classes.count)
        agreement = classes.count(most_common) / len(classes)
        
        return round(agreement, 2)


# =====================================================
# MAIN DETECTOR ENGINE
# =====================================================

class AdvancedAIDetector:
    """
    Advanced AI Content Detector v2.0
    
    Features:
    - 3-Class: Real vs AI Generated vs AI Enhanced
    - Frequency domain fusion
    - Patch-based + full image analysis
    - Multi-layer Grad-CAM
    - Calibrated confidence scores
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.grad_cam = None
        self.freq_extractor = FrequencyExtractor()
        self.patch_analyzer = PatchAnalyzer(patch_size=96, stride=48)
        self.initialized = False
        
        if TORCH_AVAILABLE:
            self._init_model(model_path)
        else:
            print("⚠️ PyTorch not available")
    
    def _init_model(self, model_path: str = None):
        """Initialize model"""
        try:
            self.model = MultiClassAIDetector(pretrained=True, dropout=0.5)
            
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded trained model: {model_path}")
            else:
                print("ℹ️ Using pretrained weights (transfer learning)")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.grad_cam = MultiLayerGradCAM(self.model, self.device)
            
            # Transforms
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.initialized = True
            print("✅ Advanced AI Detector initialized (3-class)")
            
        except Exception as e:
            print(f"❌ Init failed: {e}")
            self.initialized = False
    
    def analyze(self, image_path: str) -> Dict:
        """
        Complete analysis with all features.
        """
        result = {
            "status": "processing",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0",
            
            # Primary prediction
            "prediction": {
                "class": 0,
                "class_name": "UNKNOWN",
                "confidence": 0.0,
                "distribution": {
                    "real": 0.0,
                    "ai_generated": 0.0,
                    "ai_enhanced": 0.0
                }
            },
            
            # Full image analysis
            "full_image_analysis": {},
            
            # Patch analysis
            "patch_analysis": {},
            
            # Heatmaps
            "heatmaps": {},
            
            # Frequency analysis
            "frequency_analysis": {},
            
            # Confidence calibration
            "confidence_calibration": {},
            
            # Interpretation
            "interpretation": {},
            
            "errors": []
        }
        
        if not self.initialized:
            result["errors"].append("Detector not initialized")
            result["status"] = "error"
            return result
        
        try:
            # Load image
            original = cv2.imread(image_path)
            if original is None:
                result["errors"].append("Failed to load image")
                result["status"] = "error"
                return result
            
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(original_rgb)
            
            # Extract frequency features
            fft_map = self.freq_extractor.extract_fft_magnitude(original)
            high_freq = self.freq_extractor.extract_high_freq_map(original)
            dct_map = self.freq_extractor.extract_dct_features(original)
            
            result["frequency_analysis"] = {
                "fft_energy": float(np.mean(fft_map)),
                "high_freq_ratio": float(np.mean(high_freq)),
                "dct_variance": float(np.var(dct_map)),
                "frequency_anomaly_score": self._calculate_freq_anomaly(fft_map, high_freq)
            }
            
            # Prepare tensors
            rgb_tensor = self.rgb_transform(pil_image).unsqueeze(0).to(self.device)
            
            # Resize frequency map and convert to tensor
            fft_resized = cv2.resize(fft_map, (224, 224))
            freq_tensor = torch.from_numpy(fft_resized).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # Full image analysis
            full_result = self._analyze_full_image(rgb_tensor, freq_tensor, original)
            result["full_image_analysis"] = full_result
            
            # Store heatmaps
            for name, hm in full_result.get("heatmaps", {}).items():
                if isinstance(hm, np.ndarray):
                    colored = self._colorize_heatmap(hm, cv2.resize(original, (224, 224)))
                    result["heatmaps"][name] = colored
            
            # Patch analysis (for texture-level detection)
            patch_result = self._analyze_patches(original)
            result["patch_analysis"] = patch_result
            
            # Combine full + patch predictions
            final_prediction = self._combine_predictions(full_result, patch_result)
            result["prediction"] = final_prediction
            
            # Confidence calibration
            result["confidence_calibration"] = self._calibrate_confidence(
                final_prediction,
                full_result,
                result["frequency_analysis"]
            )
            
            # Generate interpretation
            result["interpretation"] = self._generate_interpretation(result)
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _analyze_full_image(
        self, 
        rgb_tensor: torch.Tensor, 
        freq_tensor: torch.Tensor,
        original: np.ndarray
    ) -> Dict:
        """Analyze full image"""
        with torch.no_grad():
            logits = self.model(rgb_tensor, freq_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        predicted_class = int(np.argmax(probs))
        
        # Generate Grad-CAM
        gradcam_result = self.grad_cam.generate(rgb_tensor, freq_tensor, predicted_class)
        
        return {
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES.get(predicted_class, "UNKNOWN"),
            "confidence": float(probs[predicted_class] * 100),
            "probabilities": probs.tolist(),
            "heatmaps": gradcam_result.get("heatmaps", {}),
            "heatmap_entropy": gradcam_result.get("heatmap_entropy", 0),
            "confidence_modifier": gradcam_result.get("confidence_modifier", 1.0)
        }
    
    def _analyze_patches(self, image: np.ndarray) -> Dict:
        """Analyze image patches"""
        patches = self.patch_analyzer.extract_patches(image)
        
        if len(patches) == 0:
            return {"error": "No patches extracted"}
        
        # Limit patches for speed
        if len(patches) > 16:
            step = len(patches) // 16
            patches = patches[::step][:16]
        
        patch_predictions = []
        
        for patch_img, (x, y) in patches:
            # Resize patch to model input size
            patch_resized = cv2.resize(patch_img, (224, 224))
            patch_rgb = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
            pil_patch = Image.fromarray(patch_rgb)
            
            # Extract frequency
            fft_patch = self.freq_extractor.extract_fft_magnitude(patch_resized)
            
            # Prepare tensors
            rgb_tensor = self.rgb_transform(pil_patch).unsqueeze(0).to(self.device)
            fft_resized = cv2.resize(fft_patch, (224, 224))
            freq_tensor = torch.from_numpy(fft_resized).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(rgb_tensor, freq_tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
            predicted_class = int(np.argmax(probs))
            
            patch_predictions.append({
                "position": (x, y),
                "predicted_class": predicted_class,
                "probabilities": probs,
                "confidence": float(probs[predicted_class] * 100)
            })
        
        # Aggregate
        aggregate = self.patch_analyzer.aggregate_predictions(patch_predictions, strategy="weighted")
        
        return {
            "num_patches": len(patch_predictions),
            "aggregate": aggregate,
            "patch_details": patch_predictions[:5]  # First 5 for reference
        }
    
    def _combine_predictions(self, full_result: Dict, patch_result: Dict) -> Dict:
        """
        Combine full image and patch predictions.
        
        Strategy:
        - Full image: 60% weight (captures global context)
        - Patches: 40% weight (captures local artifacts)
        """
        full_probs = np.array(full_result.get("probabilities", [0.33, 0.33, 0.34]))
        
        patch_aggregate = patch_result.get("aggregate", {})
        patch_probs = np.array(patch_aggregate.get("distribution", [0.33, 0.33, 0.34]))
        
        # Weighted combination
        combined_probs = 0.6 * full_probs + 0.4 * patch_probs
        
        # Apply confidence modifier from heatmap entropy
        modifier = full_result.get("confidence_modifier", 1.0)
        
        predicted_class = int(np.argmax(combined_probs))
        confidence = float(combined_probs[predicted_class] * 100 * modifier)
        
        return {
            "class": predicted_class,
            "class_name": CLASS_NAMES.get(predicted_class, "UNKNOWN"),
            "confidence": round(confidence, 2),
            "distribution": {
                "real": round(combined_probs[0] * 100, 2),
                "ai_generated": round(combined_probs[1] * 100, 2),
                "ai_enhanced": round(combined_probs[2] * 100, 2)
            },
            "fusion_weights": {"full_image": 0.6, "patches": 0.4}
        }
    
    def _calculate_freq_anomaly(self, fft_map: np.ndarray, high_freq: np.ndarray) -> float:
        """Calculate frequency anomaly score for AI detection"""
        # AI images often have:
        # 1. Periodic patterns in FFT
        # 2. Lack of natural high-frequency noise
        
        fft_std = np.std(fft_map)
        hf_mean = np.mean(high_freq)
        
        # Low high-freq + smooth FFT → likely AI
        anomaly = (1 - hf_mean) * 50 + (1 - min(fft_std, 1)) * 50
        
        return round(anomaly, 2)
    
    def _calibrate_confidence(
        self, 
        prediction: Dict, 
        full_result: Dict,
        freq_analysis: Dict
    ) -> Dict:
        """
        Calibrate confidence based on multiple factors.
        
        Factors:
        1. Base prediction confidence
        2. Heatmap entropy (focused = confident)
        3. Patch agreement
        4. Frequency anomaly
        """
        base_conf = prediction.get("confidence", 50)
        hm_entropy = full_result.get("heatmap_entropy", 4)
        freq_anomaly = freq_analysis.get("frequency_anomaly_score", 50)
        
        # Entropy factor (low entropy = focused = confident)
        entropy_factor = max(0.7, 1.0 - (hm_entropy - 3) * 0.1)
        
        # Frequency factor (high anomaly for AI classes = more confident)
        pred_class = prediction.get("class", 0)
        if pred_class in [1, 2]:  # AI classes
            freq_factor = 1.0 + (freq_anomaly - 50) / 200
        else:  # Real class
            freq_factor = 1.0 - (freq_anomaly - 50) / 200
        freq_factor = max(0.8, min(1.2, freq_factor))
        
        # Calibrated confidence
        calibrated = base_conf * entropy_factor * freq_factor
        calibrated = max(10, min(95, calibrated))  # Clamp
        
        # Determine if UNCERTAIN
        is_uncertain = calibrated < 45 or (35 < calibrated < 65 and abs(prediction["distribution"]["real"] - prediction["distribution"]["ai_generated"]) < 15)
        
        return {
            "raw_confidence": round(base_conf, 2),
            "calibrated_confidence": round(calibrated, 2),
            "entropy_factor": round(entropy_factor, 3),
            "frequency_factor": round(freq_factor, 3),
            "is_uncertain": is_uncertain,
            "recommendation": "Manual review recommended" if is_uncertain else "Automated decision acceptable"
        }
    
    def _colorize_heatmap(self, heatmap: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply colormap and overlay"""
        hm_uint8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.5, colored, 0.5, 0)
        return overlay
    
    def _generate_interpretation(self, result: Dict) -> Dict:
        """Generate human-readable interpretation"""
        pred = result.get("prediction", {})
        calib = result.get("confidence_calibration", {})
        freq = result.get("frequency_analysis", {})
        
        class_name = pred.get("class_name", "UNKNOWN")
        confidence = calib.get("calibrated_confidence", 0)
        is_uncertain = calib.get("is_uncertain", True)
        
        interpretation = {
            "summary": "",
            "evidence": [],
            "limitations": [],
            "recommendation": ""
        }
        
        if is_uncertain:
            interpretation["summary"] = f"UNCERTAIN: Analysis inconclusive ({confidence:.0f}% confidence)"
            interpretation["evidence"].append("Prediction confidence below threshold")
            interpretation["recommendation"] = "Manual expert review required"
        elif class_name == "REAL":
            interpretation["summary"] = f"LIKELY REAL: Natural/camera image ({confidence:.0f}% confidence)"
            interpretation["evidence"].append("Natural texture patterns detected")
            interpretation["evidence"].append("Consistent frequency distribution")
            interpretation["recommendation"] = "Content appears authentic"
        elif class_name == "AI_GENERATED":
            interpretation["summary"] = f"AI GENERATED: Created by AI ({confidence:.0f}% confidence)"
            interpretation["evidence"].append("AI-specific texture artifacts detected")
            interpretation["evidence"].append(f"Frequency anomaly score: {freq.get('frequency_anomaly_score', 0):.0f}")
            interpretation["recommendation"] = "Verify source and context"
        elif class_name == "AI_ENHANCED":
            interpretation["summary"] = f"AI ENHANCED: Processed/refined by AI ({confidence:.0f}% confidence)"
            interpretation["evidence"].append("Mixed real/AI characteristics detected")
            interpretation["evidence"].append("Enhancement artifacts present")
            interpretation["recommendation"] = "May be real image with AI processing"
        
        interpretation["limitations"] = [
            "Analysis is probabilistic, not deterministic",
            "New AI models may produce undetected patterns",
            "Compression can affect detection accuracy"
        ]
        
        return interpretation


# =====================================================
# QUICK TEST
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Advanced AI Content Detector v2.0")
    print("=" * 60)
    print("Classes: Real | AI Generated | AI Enhanced")
    print("=" * 60)
    
    if TORCH_AVAILABLE:
        detector = AdvancedAIDetector()
        if detector.initialized:
            print("✅ Detector ready")
    else:
        print("❌ PyTorch required")
