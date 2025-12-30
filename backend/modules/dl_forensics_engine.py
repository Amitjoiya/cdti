"""
Deep Learning Forensics Engine v2.0
====================================
PRIMARY Decision Maker with True Grad-CAM

Why Classical Methods Fail on Deepfakes:
----------------------------------------
1. ELA: Designed for JPEG re-save detection, not AI artifacts
2. Noise: AI models produce consistent noise patterns
3. Edge: Deepfakes have smooth, natural-looking edges
4. Frequency: GANs learn to match real frequency distributions

Solution: CNN trained on ACTUAL deepfake/real pairs
with Grad-CAM for decision-based explanations.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tempfile

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =====================================================
# WHY GRAD-CAM REQUIRES A TRAINED CNN
# =====================================================
"""
Grad-CAM (Gradient-weighted Class Activation Mapping):
- Uses GRADIENTS of the target class score
- Weights feature maps by how much they CONTRIBUTED to the decision
- Shows WHICH regions made the model predict "fake"

Saliency Maps (What we DON'T use):
- Just show input sensitivity (where changes affect output)
- Produce uniform, edge-like patterns
- NOT decision-based, just gradient magnitude

Why CNN is Required:
- Grad-CAM needs CONVOLUTIONAL feature maps
- Dense layers lose spatial information
- Last conv layer retains "where" + "what" information
"""


# =====================================================
# MODEL ARCHITECTURE (ResNet-18 based)
# =====================================================

class DeepfakeClassifier(nn.Module):
    """
    ResNet-18 based deepfake classifier.
    
    Why ResNet-18:
    - Lightweight for real-time inference
    - Good feature extraction from ImageNet pretraining
    - Last conv layer (layer4) perfect for Grad-CAM
    - Residual connections help gradient flow
    
    Architecture:
    - Conv layers: Extract spatial features
    - Global Average Pooling: Aggregate features
    - FC layers: Binary classification
    """
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        # Load pretrained ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Get feature dimension
        num_features = self.backbone.fc.in_features  # 512 for ResNet-18
        
        # Replace classifier for binary classification
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Storage for Grad-CAM
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks on layer4 (last conv layer)
        self.backbone.layer4.register_forward_hook(self._save_features)
        self.backbone.layer4.register_full_backward_hook(self._save_gradients)
    
    def _save_features(self, module, input, output):
        """Hook to save feature maps"""
        self.feature_maps = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0].detach()
    
    def forward(self, x):
        """Forward pass"""
        logits = self.backbone(x)
        return torch.sigmoid(logits)
    
    def get_cam_weights(self):
        """Get Grad-CAM weights from gradients"""
        if self.gradients is None:
            return None
        # Global average pooling of gradients
        return torch.mean(self.gradients, dim=[2, 3], keepdim=True)


# =====================================================
# TRUE GRAD-CAM IMPLEMENTATION
# =====================================================

class TrueGradCAM:
    """
    True Grad-CAM Implementation (Decision-Based)
    
    Why Grad-CAM is Decision-Based:
    1. Backpropagate from the TARGET CLASS score
    2. Gradients show which features INCREASED the score
    3. Weight feature maps by gradient importance
    4. ReLU keeps only POSITIVE contributions
    
    Result: Heatmap shows WHERE the model saw evidence of manipulation
    
    Why Saliency Maps are Misleading:
    1. Just gradient magnitude (not direction)
    2. Shows input sensitivity, not decision reasoning
    3. Produces edge-like patterns for all images
    4. No class-specific information
    """
    
    def __init__(self, model: DeepfakeClassifier, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed image (1, 3, 224, 224)
            target_class: 1 = fake/manipulated, 0 = real
            
        Returns:
            Heatmap as numpy array (224, 224), normalized 0-1
        """
        # Ensure model is in eval mode but gradients enabled
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True).to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        if target_class == 1:
            # Gradient w.r.t. fake probability
            output.backward(retain_graph=True)
        else:
            # Gradient w.r.t. real probability (1 - fake)
            (1 - output).backward(retain_graph=True)
        
        # Get Grad-CAM components
        weights = self.model.get_cam_weights()  # (1, 512, 1, 1)
        features = self.model.feature_maps       # (1, 512, 7, 7)
        
        if weights is None or features is None:
            return np.zeros((224, 224), dtype=np.float32)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * features, dim=1, keepdim=True)  # (1, 1, 7, 7)
        
        # ReLU - only positive contributions
        cam = F.relu(cam)
        
        # Convert to numpy and resize
        cam = cam.squeeze().cpu().numpy()  # (7, 7)
        
        # Bicubic interpolation to original size
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        return cam.astype(np.float32)
    
    def generate_comparison(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Generate comparison heatmaps for both classes.
        
        This shows:
        - fake_cam: Where model sees FAKE evidence
        - real_cam: Where model sees REAL evidence
        - decision_map: Difference (which regions were decisive)
        """
        fake_cam = self.generate(input_tensor.clone(), target_class=1)
        real_cam = self.generate(input_tensor.clone(), target_class=0)
        
        # Decision map: Where fake evidence outweighs real
        decision_map = fake_cam - real_cam
        decision_map = np.clip(decision_map, 0, 1)
        
        return {
            "fake_evidence": fake_cam,
            "real_evidence": real_cam,
            "decision_map": decision_map
        }
    
    def generate_regional(self, input_tensor: torch.Tensor) -> Dict[str, Dict]:
        """
        Generate region-specific analysis.
        
        Why Regional Analysis:
        - Deepfakes have artifacts in specific areas (lips, eyes, boundaries)
        - Full-face analysis can miss localized manipulation
        - Different regions have different artifact patterns
        """
        base_cam = self.generate(input_tensor, target_class=1)
        
        h, w = 224, 224
        regions = {
            "upper_face": {"mask": self._region_mask(h, w, 0, 0.35), "desc": "Forehead, Eyes"},
            "mid_face": {"mask": self._region_mask(h, w, 0.30, 0.65), "desc": "Nose, Cheeks"},
            "lower_face": {"mask": self._region_mask(h, w, 0.60, 1.0), "desc": "Mouth, Chin"},
            "left_boundary": {"mask": self._side_mask(h, w, "left", 0.15), "desc": "Left Edge"},
            "right_boundary": {"mask": self._side_mask(h, w, "right", 0.15), "desc": "Right Edge"},
        }
        
        results = {}
        for name, data in regions.items():
            mask = data["mask"]
            region_cam = base_cam * mask
            activation = float(np.mean(region_cam[mask > 0])) if np.sum(mask) > 0 else 0
            
            results[name] = {
                "heatmap": region_cam,
                "activation": round(activation * 100, 2),
                "description": data["desc"],
                "suspicious": activation > 0.3,
                "interpretation": self._interpret_region(name, activation)
            }
        
        return results
    
    def _region_mask(self, h, w, start_ratio, end_ratio):
        """Create horizontal region mask"""
        mask = np.zeros((h, w), dtype=np.float32)
        start_y = int(h * start_ratio)
        end_y = int(h * end_ratio)
        mask[start_y:end_y, :] = 1.0
        return mask
    
    def _side_mask(self, h, w, side, width_ratio):
        """Create side boundary mask"""
        mask = np.zeros((h, w), dtype=np.float32)
        width = int(w * width_ratio)
        if side == "left":
            mask[:, :width] = 1.0
        else:
            mask[:, -width:] = 1.0
        return mask
    
    def _interpret_region(self, region: str, activation: float) -> str:
        """Generate forensic interpretation"""
        level = "high" if activation > 0.4 else "medium" if activation > 0.2 else "low"
        
        interpretations = {
            "upper_face": {
                "high": "Strong artifacts in eye/forehead region - possible blink inconsistency or forehead blending",
                "medium": "Moderate upper face anomalies detected",
                "low": "Upper face region appears natural"
            },
            "mid_face": {
                "high": "Significant cheek/nose artifacts - common in face-swap blending zones",
                "medium": "Some mid-face inconsistencies present",
                "low": "Mid-face region appears authentic"
            },
            "lower_face": {
                "high": "Lip-sync or mouth artifacts detected - common in audio-driven deepfakes",
                "medium": "Minor mouth region anomalies",
                "low": "Lower face appears natural"
            },
            "left_boundary": {
                "high": "Left face boundary artifacts - possible mask edge or blending seam",
                "medium": "Some left boundary inconsistencies",
                "low": "Left boundary appears clean"
            },
            "right_boundary": {
                "high": "Right face boundary artifacts - possible mask edge or blending seam",
                "medium": "Some right boundary inconsistencies",
                "low": "Right boundary appears clean"
            }
        }
        
        return interpretations.get(region, {}).get(level, "No specific interpretation")


# =====================================================
# MAIN FORENSICS ENGINE
# =====================================================

class DLForensicsEngine:
    """
    Deep Learning Forensics Engine
    
    PRIMARY Decision Maker:
    - CNN-based deepfake detection
    - True Grad-CAM heatmaps
    - Region-specific analysis
    
    This replaces classical methods as the main detector.
    Classical methods become SUPPORTING evidence only.
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.grad_cam = None
        self.transform = None
        self.initialized = False
        
        if TORCH_AVAILABLE:
            self._init_model(model_path)
        else:
            print("⚠️ PyTorch not available - DL Engine disabled")
    
    def _init_model(self, model_path: str = None):
        """Initialize model and Grad-CAM"""
        try:
            self.model = DeepfakeClassifier(pretrained=True, dropout=0.5)
            
            # Load custom weights if available
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded trained model: {model_path}")
            else:
                print("ℹ️ Using ImageNet pretrained weights (transfer learning mode)")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize Grad-CAM
            self.grad_cam = TrueGradCAM(self.model, self.device)
            
            # Preprocessing transform
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.initialized = True
            print("✅ DL Forensics Engine initialized")
            
        except Exception as e:
            print(f"❌ DL Engine init failed: {e}")
            self.initialized = False
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze single image with DL + Grad-CAM.
        
        Returns comprehensive forensic analysis.
        """
        result = {
            "status": "processing",
            "type": "image",
            "timestamp": datetime.now().isoformat(),
            
            # PRIMARY EVIDENCE (DL-based)
            "primary": {
                "prediction": 0,  # 0=real, 1=fake
                "confidence": 0.0,
                "label": "UNKNOWN",
                "model": "ResNet-18 CNN",
                "method": "Grad-CAM"
            },
            
            # Heatmaps (decision-based)
            "heatmaps": {},
            
            # Regional analysis
            "regions": {},
            
            # Forensic interpretation
            "interpretation": {},
            
            # Quality metrics
            "quality": {},
            
            "errors": []
        }
        
        if not self.initialized:
            result["errors"].append("DL Engine not initialized")
            result["status"] = "error"
            return result
        
        try:
            # Load and preprocess image
            original = cv2.imread(image_path)
            if original is None:
                result["errors"].append("Failed to load image")
                result["status"] = "error"
                return result
            
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(original_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                prob = output.item()
            
            result["primary"]["confidence"] = round(prob * 100, 2)
            result["primary"]["prediction"] = 1 if prob > 0.5 else 0
            
            if prob > 0.7:
                result["primary"]["label"] = "LIKELY MANIPULATED"
            elif prob > 0.5:
                result["primary"]["label"] = "POSSIBLY MANIPULATED"
            elif prob > 0.3:
                result["primary"]["label"] = "UNCERTAIN"
            else:
                result["primary"]["label"] = "LIKELY AUTHENTIC"
            
            # Generate Grad-CAM heatmaps
            comparison = self.grad_cam.generate_comparison(input_tensor)
            
            # Resize original for overlay
            original_resized = cv2.resize(original, (224, 224))
            
            # Create colored heatmaps
            for name, cam in comparison.items():
                colored = self._colorize_cam(cam, original_resized)
                result["heatmaps"][name] = colored
            
            # Regional analysis
            regions = self.grad_cam.generate_regional(input_tensor)
            for name, data in regions.items():
                colored_region = self._colorize_cam(data["heatmap"], original_resized)
                result["regions"][name] = {
                    "heatmap": colored_region,
                    "activation": data["activation"],
                    "suspicious": data["suspicious"],
                    "interpretation": data["interpretation"]
                }
            
            # Generate forensic interpretation
            result["interpretation"] = self._generate_interpretation(result, prob)
            
            # Heatmap quality check
            result["quality"] = self._check_heatmap_quality(comparison["fake_evidence"])
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def analyze_video(self, video_path: str, sample_rate: int = 10, max_frames: int = 30) -> Dict:
        """
        Analyze video with frame-by-frame Grad-CAM.
        
        Why Frame-by-Frame (not separate video model):
        1. Deepfake artifacts are TEMPORALLY INCONSISTENT
        2. Same model ensures consistent detection
        3. Frame sampling catches flickering artifacts
        4. Temporal aggregation shows manipulation patterns
        
        Why Deepfake Artifacts are Temporally Inconsistent:
        - Face-swap quality varies frame-to-frame
        - Blink patterns are often unnatural
        - Lip-sync timing mismatches
        - Boundary flickering at edges
        """
        result = {
            "status": "processing",
            "type": "video",
            "timestamp": datetime.now().isoformat(),
            
            # PRIMARY EVIDENCE
            "primary": {
                "prediction": 0,
                "confidence": 0.0,
                "label": "UNKNOWN",
                "model": "ResNet-18 CNN (Frame-wise)",
                "method": "Temporal Grad-CAM"
            },
            
            # Frame analysis
            "frames": {
                "total": 0,
                "analyzed": 0,
                "sample_rate": sample_rate
            },
            
            # Temporal data
            "temporal": {
                "scores": [],
                "timeline": [],
                "suspicious_segments": [],
                "consistency": {}
            },
            
            # Aggregated heatmaps
            "heatmaps": {},
            
            # Keyframes (high confidence frames)
            "keyframes": {},
            
            "errors": []
        }
        
        if not self.initialized:
            result["errors"].append("DL Engine not initialized")
            result["status"] = "error"
            return result
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            result["frames"]["total"] = total_frames
            
            frame_scores = []
            frame_cams = []
            timeline = []
            keyframes = {}
            
            frame_idx = 0
            analyzed = 0
            
            while analyzed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Analyze frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    input_tensor = self.transform(pil_frame).unsqueeze(0).to(self.device)
                    
                    # Prediction
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        prob = output.item()
                    
                    # Grad-CAM
                    cam = self.grad_cam.generate(input_tensor, target_class=1)
                    
                    frame_scores.append(prob * 100)
                    frame_cams.append(cam)
                    
                    # Timeline entry
                    time_sec = frame_idx / fps if fps > 0 else 0
                    timeline.append({
                        "frame": frame_idx,
                        "time_sec": round(time_sec, 2),
                        "confidence": round(prob * 100, 2),
                        "label": "FAKE" if prob > 0.5 else "REAL"
                    })
                    
                    # Store keyframes (high confidence fake)
                    if prob > 0.6:
                        frame_resized = cv2.resize(frame, (224, 224))
                        keyframes[f"frame_{frame_idx}"] = {
                            "heatmap": self._colorize_cam(cam, frame_resized),
                            "confidence": round(prob * 100, 2),
                            "time_sec": round(time_sec, 2)
                        }
                    
                    analyzed += 1
                
                frame_idx += 1
            
            cap.release()
            
            result["frames"]["analyzed"] = analyzed
            
            if frame_scores:
                # Aggregate results
                avg_score = np.mean(frame_scores)
                max_score = np.max(frame_scores)
                
                result["primary"]["confidence"] = round(avg_score, 2)
                result["primary"]["prediction"] = 1 if avg_score > 50 else 0
                
                if avg_score > 70:
                    result["primary"]["label"] = "LIKELY MANIPULATED"
                elif avg_score > 50:
                    result["primary"]["label"] = "POSSIBLY MANIPULATED"
                elif avg_score > 30:
                    result["primary"]["label"] = "UNCERTAIN"
                else:
                    result["primary"]["label"] = "LIKELY AUTHENTIC"
                
                # Temporal data
                result["temporal"]["scores"] = [round(s, 2) for s in frame_scores]
                result["temporal"]["timeline"] = timeline
                result["temporal"]["suspicious_segments"] = self._find_suspicious_segments(timeline)
                result["temporal"]["consistency"] = self._check_temporal_consistency(frame_scores)
                
                # Aggregate heatmap
                if frame_cams:
                    aggregate_cam = np.mean(frame_cams, axis=0)
                    aggregate_cam = aggregate_cam / (aggregate_cam.max() + 1e-8)
                    
                    # Get a frame for overlay
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                    ret, mid_frame = cap.read()
                    cap.release()
                    
                    if ret:
                        mid_frame = cv2.resize(mid_frame, (224, 224))
                        result["heatmaps"]["aggregate"] = self._colorize_cam(aggregate_cam, mid_frame)
                
                result["keyframes"] = keyframes
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _colorize_cam(self, cam: np.ndarray, original: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Apply colormap and overlay on original"""
        # Apply JET colormap
        cam_uint8 = (cam * 255).astype(np.uint8)
        colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # Overlay on original
        overlay = cv2.addWeighted(original, 1 - alpha, colored, alpha, 0)
        
        return overlay
    
    def _generate_interpretation(self, result: Dict, probability: float) -> Dict:
        """Generate forensic interpretation"""
        interpretation = {
            "summary": "",
            "key_findings": [],
            "confidence_explanation": "",
            "recommendations": []
        }
        
        # Summary based on probability
        if probability > 0.7:
            interpretation["summary"] = "HIGH confidence of digital manipulation detected"
            interpretation["key_findings"].append("Strong manipulation artifacts identified by CNN")
        elif probability > 0.5:
            interpretation["summary"] = "MODERATE signs of possible manipulation"
            interpretation["key_findings"].append("Some suspicious patterns detected")
        elif probability > 0.3:
            interpretation["summary"] = "INCONCLUSIVE - manual review recommended"
            interpretation["key_findings"].append("Weak signals present, but not definitive")
        else:
            interpretation["summary"] = "Content appears AUTHENTIC"
            interpretation["key_findings"].append("No significant manipulation artifacts detected")
        
        # Add regional findings
        suspicious_regions = [
            name for name, data in result.get("regions", {}).items()
            if data.get("suspicious", False)
        ]
        
        if suspicious_regions:
            interpretation["key_findings"].append(
                f"Suspicious regions: {', '.join(suspicious_regions)}"
            )
        
        # Confidence explanation
        interpretation["confidence_explanation"] = (
            f"The CNN model analyzed visual patterns and detected "
            f"{'significant' if probability > 0.5 else 'minimal'} manipulation artifacts. "
            f"Grad-CAM heatmaps highlight the specific regions that influenced this decision."
        )
        
        # Recommendations
        if probability > 0.5:
            interpretation["recommendations"] = [
                "Cross-verify with original source if available",
                "Check metadata for editing software signatures",
                "Examine keyframes for temporal inconsistencies"
            ]
        else:
            interpretation["recommendations"] = [
                "Content appears genuine based on visual analysis",
                "Consider additional verification if context requires"
            ]
        
        return interpretation
    
    def _check_heatmap_quality(self, cam: np.ndarray) -> Dict:
        """
        Check if heatmap is meaningful (not center-biased).
        
        Common Failure Modes:
        1. Center Bias: Model always looks at center (bad training)
        2. Uniform Activation: Same heatmap for all images
        3. Edge Detection: Model learned edges, not artifacts
        
        Good Heatmap:
        - Varies per image
        - Focuses on specific regions
        - Not always centered
        """
        h, w = cam.shape
        
        # Center region activation
        center_mask = np.zeros((h, w))
        center_mask[h//4:3*h//4, w//4:3*w//4] = 1
        center_activation = np.mean(cam * center_mask)
        
        # Edge region activation
        edge_mask = 1 - center_mask
        edge_activation = np.mean(cam * edge_mask)
        
        # Center bias ratio
        center_bias = center_activation / (edge_activation + 1e-6)
        
        # Activation spread (entropy-like measure)
        cam_flat = cam.flatten()
        cam_normalized = cam_flat / (cam_flat.sum() + 1e-8)
        entropy = -np.sum(cam_normalized * np.log(cam_normalized + 1e-8))
        
        quality = "GOOD" if center_bias < 4 and entropy > 2 else "POOR"
        
        return {
            "quality": quality,
            "center_bias": round(center_bias, 2),
            "activation_spread": round(entropy, 2),
            "warning": None if quality == "GOOD" else "Possible model shortcut detected - heatmap may be unreliable"
        }
    
    def _check_temporal_consistency(self, scores: List[float]) -> Dict:
        """Check temporal consistency of predictions"""
        if len(scores) < 2:
            return {"consistent": True, "variance": 0}
        
        variance = np.var(scores)
        max_diff = max(scores) - min(scores)
        
        # High variance indicates flickering (common in deepfakes)
        is_consistent = variance < 200  # Threshold for consistency
        
        return {
            "consistent": is_consistent,
            "variance": round(variance, 2),
            "max_difference": round(max_diff, 2),
            "interpretation": (
                "High temporal variance detected - possible frame-by-frame manipulation artifacts"
                if not is_consistent else
                "Consistent predictions across frames"
            )
        }
    
    def _find_suspicious_segments(self, timeline: List[Dict]) -> List[Dict]:
        """Find time segments with high manipulation scores"""
        segments = []
        current_segment = None
        
        for entry in timeline:
            if entry["confidence"] > 50:
                if current_segment is None:
                    current_segment = {
                        "start_time": entry["time_sec"],
                        "start_frame": entry["frame"],
                        "max_confidence": entry["confidence"]
                    }
                else:
                    current_segment["max_confidence"] = max(
                        current_segment["max_confidence"],
                        entry["confidence"]
                    )
            else:
                if current_segment is not None:
                    current_segment["end_time"] = entry["time_sec"]
                    current_segment["end_frame"] = entry["frame"]
                    segments.append(current_segment)
                    current_segment = None
        
        if current_segment and timeline:
            current_segment["end_time"] = timeline[-1]["time_sec"]
            current_segment["end_frame"] = timeline[-1]["frame"]
            segments.append(current_segment)
        
        return segments


# =====================================================
# TRAINING PIPELINE (FOR REFERENCE)
# =====================================================
"""
TRAINING STRATEGY (Critical for Good Heatmaps)
==============================================

1. DATASET DESIGN (Identity-Balanced):
   - Same faces in BOTH real and fake samples
   - Prevents model from learning "known face = real"
   - Forces model to learn ARTIFACTS, not identities
   
   Why this prevents center-face bias:
   - If fake = different person, model learns "unknown = fake"
   - With same identities, only artifacts distinguish classes
   - Model must find LOCAL patterns, not global face features

2. PATCH-BASED TRAINING:
   - Random crop regions (eyes, lips, cheeks)
   - Don't always feed full face
   - Forces model to detect LOCAL artifacts
   
   Why this improves Grad-CAM:
   - Full-face training → model uses center face for all decisions
   - Patch training → model learns position-independent features
   - Heatmaps become localized and meaningful

3. AUGMENTATION STRATEGY:
   Good (simulate real-world artifacts):
   - JPEG compression (70-100 quality)
   - Gaussian noise (0-5%)
   - Blur (simulate compression)
   - Color jitter (lighting variations)
   
   Bad (destroys forensic evidence):
   - Heavy rotation (faces are upright)
   - Random erasing (removes evidence)
   - Heavy cropping (loses context)

4. TRAINING VALIDATION:
   After each epoch:
   - Check accuracy
   - Generate sample Grad-CAMs
   - Flag center-biased heatmaps
   - Stop if model learns shortcuts

Training code available in training_pipeline.py
"""


# =====================================================
# QUICK TEST
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Deep Learning Forensics Engine v2.0")
    print("=" * 60)
    
    if TORCH_AVAILABLE:
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        engine = DLForensicsEngine()
        
        if engine.initialized:
            print("✅ Engine ready for inference")
        else:
            print("❌ Engine initialization failed")
    else:
        print("❌ PyTorch not installed")
