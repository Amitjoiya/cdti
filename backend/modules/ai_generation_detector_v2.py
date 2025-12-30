"""
FakeTrace - AI Generation Detection v2.0
==========================================
IMPROVED detection for modern AI generators:
- Gemini, GPT-4o, DALL-E 3, Midjourney v6, Stable Diffusion XL

Key Improvements:
1. Infographic/Promotional image detection
2. Text artifact analysis (AI struggles with text)
3. Unnatural perfection detection
4. Color gradient smoothness (AI makes too-smooth gradients)
5. Digital art signature detection
6. Semantic consistency checks
7. Edge quality analysis for AI-typical clean edges

IMPORTANT: Modern AI detection is VERY difficult.
Results are probabilistic indicators, not proof.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import re


class AIDetectionStatus(Enum):
    """AI detection confidence levels - ADJUSTED for modern AI"""
    LIKELY_AI = "likely_ai"              # Score >= 55: Strong AI indicators
    POSSIBLY_AI = "possibly_ai"          # Score 40-54: Some AI indicators  
    UNCERTAIN = "uncertain"              # Score 30-39: Cannot determine
    LIKELY_NATURAL = "likely_natural"    # Score < 30: Appears natural


@dataclass
class AICheckResult:
    """Result from a single AI detection algorithm"""
    name: str
    display_name: str
    score: float                    # 0-100 (higher = more likely AI)
    status: str                     # likely_ai/possibly_ai/uncertain/likely_natural
    finding: str                    # Human-readable finding
    details: Dict                   # Technical details
    
    @staticmethod
    def from_score(name: str, display_name: str, score: float,
                   details: Dict, finding_ai: str, finding_natural: str) -> 'AICheckResult':
        """Create result with automatic status classification"""
        
        # Adjusted thresholds for modern AI
        if score >= 55:
            status = AIDetectionStatus.LIKELY_AI.value
            finding = finding_ai
        elif score >= 40:
            status = AIDetectionStatus.POSSIBLY_AI.value
            finding = f"Some indicators: {finding_ai}"
        elif score >= 30:
            status = AIDetectionStatus.UNCERTAIN.value
            finding = "Cannot reliably determine AI vs natural origin"
        else:
            status = AIDetectionStatus.LIKELY_NATURAL.value
            finding = finding_natural
        
        return AICheckResult(
            name=name,
            display_name=display_name,
            score=round(score, 1),
            status=status,
            finding=finding,
            details=details
        )


class AIGenerationDetectorV2:
    """
    Enhanced AI Generation Detector v2.0
    
    Optimized for modern AI generators that produce very realistic images.
    Uses multiple detection strategies with adjusted sensitivity.
    """
    
    VERSION = "2.0"
    
    def __init__(self):
        print("  ✅ AI Generation Detector v2.0 initialized")
        print("     Optimized for: Gemini, DALL-E 3, Midjourney, Stable Diffusion")
        print("     Methods: 10 detection algorithms with adjusted sensitivity")
    
    def analyze(self, image: np.ndarray, file_path: str = None) -> Dict:
        """Analyze image for AI generation indicators."""
        
        result = {
            "version": self.VERSION,
            "status": "processing",
            "ai_checks": [],
            "summary": {
                "total_checks": 0,
                "likely_ai": 0,
                "possibly_ai": 0,
                "uncertain": 0,
                "likely_natural": 0
            },
            "ai_assessment": {
                "conclusion": "UNKNOWN",
                "confidence": 0.0,
                "interpretation": "",
                "indicators_found": [],
                "natural_signs": []
            },
            "errors": []
        }
        
        try:
            if image is None:
                result["errors"].append("Invalid image")
                return result
            
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Run all detection algorithms
            checks = [
                self._check_digital_art_signature(image, gray),
                self._check_gradient_perfection(image),
                self._check_color_palette(image),
                self._check_edge_unnaturalness(gray),
                self._check_texture_uniformity(image, gray),
                self._check_noise_pattern(gray),
                self._check_frequency_artifacts(gray),
                self._check_symmetry_patterns(gray),
                self._check_infographic_detection(image, gray),
                self._check_synthetic_lighting(image),
            ]
            
            result["ai_checks"] = [asdict(c) for c in checks]
            
            # Count results
            likely_ai = sum(1 for c in checks if c.status == AIDetectionStatus.LIKELY_AI.value)
            possibly_ai = sum(1 for c in checks if c.status == AIDetectionStatus.POSSIBLY_AI.value)
            uncertain = sum(1 for c in checks if c.status == AIDetectionStatus.UNCERTAIN.value)
            likely_natural = sum(1 for c in checks if c.status == AIDetectionStatus.LIKELY_NATURAL.value)
            
            result["summary"] = {
                "total_checks": len(checks),
                "likely_ai": likely_ai,
                "possibly_ai": possibly_ai,
                "uncertain": uncertain,
                "likely_natural": likely_natural
            }
            
            result["ai_assessment"] = self._generate_assessment(
                checks, likely_ai, possibly_ai, likely_natural
            )
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _check_digital_art_signature(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """
        Detect digital art/infographic signatures.
        AI often creates content that looks like polished digital art.
        """
        try:
            h, w = image.shape[:2]
            
            # 1. Check for very clean, uniform color regions (typical in AI art)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Quantize colors to find dominant color patches
            colors_quantized = (image // 32) * 32
            unique_colors = len(np.unique(colors_quantized.reshape(-1, 3), axis=0))
            color_simplicity = 1 - min(1, unique_colors / 500)
            
            # 2. Check for smooth gradients (AI loves gradients)
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Low gradient variance = smooth = possibly AI
            gradient_uniformity = 1 - min(1, np.std(gradient_mag) / 50)
            
            # 3. Check for neon/vibrant colors (common in AI promotional art)
            saturation = hsv[:, :, 1]
            high_sat_ratio = np.sum(saturation > 200) / saturation.size
            
            # 4. Check for clean shapes using edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Low edge density with high saturation = likely digital art
            digital_art_score = 0
            
            if color_simplicity > 0.7:
                digital_art_score += 25
            elif color_simplicity > 0.5:
                digital_art_score += 15
            
            if gradient_uniformity > 0.8:
                digital_art_score += 25
            elif gradient_uniformity > 0.6:
                digital_art_score += 15
            
            if high_sat_ratio > 0.15:
                digital_art_score += 20
            elif high_sat_ratio > 0.08:
                digital_art_score += 10
            
            if edge_density < 0.05 and color_simplicity > 0.5:
                digital_art_score += 20
            
            score = min(100, digital_art_score)
            
            details = {
                "color_simplicity": round(float(color_simplicity), 3),
                "gradient_uniformity": round(float(gradient_uniformity), 3),
                "high_saturation_ratio": round(float(high_sat_ratio), 4),
                "edge_density": round(float(edge_density), 4)
            }
            
            return AICheckResult.from_score(
                name="digital_art_signature",
                display_name="Digital Art Signature",
                score=score,
                details=details,
                finding_ai="Image shows characteristics of AI-generated digital art (clean colors, smooth gradients)",
                finding_natural="Image texture appears natural/photographic"
            )
            
        except Exception as e:
            return AICheckResult(
                name="digital_art_signature",
                display_name="Digital Art Signature",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_gradient_perfection(self, image: np.ndarray) -> AICheckResult:
        """
        Detect unnaturally perfect gradients.
        AI creates mathematically perfect gradients that don't exist in nature.
        """
        try:
            # Convert to LAB for better gradient analysis
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(float)
            
            h, w = l_channel.shape
            
            # Analyze horizontal and vertical gradients
            h_diff = np.abs(np.diff(l_channel, axis=1))
            v_diff = np.abs(np.diff(l_channel, axis=0))
            
            # Check for consistent gradient steps (AI signature)
            # Natural images have random variations, AI has uniform steps
            
            # Histogram of gradient values
            h_hist, _ = np.histogram(h_diff.ravel(), bins=50, range=(0, 50))
            v_hist, _ = np.histogram(v_diff.ravel(), bins=50, range=(0, 50))
            
            # Entropy of gradient histogram (low = uniform = AI)
            h_entropy = -np.sum((h_hist / h_hist.sum() + 1e-10) * 
                                np.log2(h_hist / h_hist.sum() + 1e-10))
            v_entropy = -np.sum((v_hist / v_hist.sum() + 1e-10) * 
                                np.log2(v_hist / v_hist.sum() + 1e-10))
            
            avg_entropy = (h_entropy + v_entropy) / 2
            
            # Check for large smooth regions
            smooth_mask = (h_diff < 2)
            smooth_ratio = np.sum(smooth_mask) / smooth_mask.size
            
            # Check for banding (step-like gradients)
            # AI sometimes creates visible bands in gradients
            gradient_std = np.std(h_diff[h_diff > 0]) if np.any(h_diff > 0) else 0
            
            score = 0
            
            # Low entropy = uniform gradients = AI
            if avg_entropy < 3:
                score += 35
            elif avg_entropy < 4:
                score += 20
            
            # High smooth ratio = AI
            if smooth_ratio > 0.8:
                score += 30
            elif smooth_ratio > 0.6:
                score += 15
            
            # Low gradient std = uniform = AI
            if gradient_std < 2:
                score += 25
            elif gradient_std < 4:
                score += 10
            
            score = min(100, score)
            
            details = {
                "gradient_entropy": round(float(avg_entropy), 2),
                "smooth_ratio": round(float(smooth_ratio), 3),
                "gradient_std": round(float(gradient_std), 2)
            }
            
            return AICheckResult.from_score(
                name="gradient_perfection",
                display_name="Gradient Perfection Analysis",
                score=score,
                details=details,
                finding_ai="Gradients appear mathematically perfect (AI-typical)",
                finding_natural="Gradient patterns appear natural with expected variations"
            )
            
        except Exception as e:
            return AICheckResult(
                name="gradient_perfection",
                display_name="Gradient Perfection Analysis",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_color_palette(self, image: np.ndarray) -> AICheckResult:
        """
        Analyze color palette for AI-typical patterns.
        AI generators often use specific pleasing color combinations.
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 1. Hue distribution analysis
            h_hist, _ = np.histogram(h.ravel(), bins=180, range=(0, 180))
            h_hist = h_hist / h_hist.sum()
            
            # Find dominant hues
            peaks = []
            for i in range(1, len(h_hist) - 1):
                if h_hist[i] > h_hist[i-1] and h_hist[i] > h_hist[i+1] and h_hist[i] > 0.02:
                    peaks.append((i, h_hist[i]))
            
            # AI often uses limited, well-chosen color palette
            num_dominant_colors = len(peaks)
            
            # 2. Saturation distribution
            s_mean = np.mean(s)
            s_std = np.std(s)
            
            # AI often has high, uniform saturation
            high_sat_uniform = s_mean > 100 and s_std < 50
            
            # 3. Value (brightness) distribution
            v_mean = np.mean(v)
            v_std = np.std(v)
            
            # 4. Color harmony check (AI uses complementary/triadic colors)
            if len(peaks) >= 2:
                hue_diffs = []
                for i in range(len(peaks)):
                    for j in range(i+1, len(peaks)):
                        diff = abs(peaks[i][0] - peaks[j][0])
                        if diff > 90:
                            diff = 180 - diff
                        hue_diffs.append(diff)
                
                # Check for complementary (60) or triadic (40) relationships
                harmonious = any(55 < d < 65 or 35 < d < 45 for d in hue_diffs)
            else:
                harmonious = False
            
            # 5. Neon/Electric color detection
            neon_mask = (s > 200) & (v > 200)
            neon_ratio = np.sum(neon_mask) / neon_mask.size
            
            score = 0
            
            # Limited color palette
            if num_dominant_colors <= 4 and num_dominant_colors > 0:
                score += 25
            
            # High uniform saturation
            if high_sat_uniform:
                score += 20
            
            # Color harmony (AI uses pleasing combinations)
            if harmonious:
                score += 20
            
            # Neon colors
            if neon_ratio > 0.1:
                score += 25
            elif neon_ratio > 0.05:
                score += 15
            
            score = min(100, score)
            
            details = {
                "dominant_colors": num_dominant_colors,
                "saturation_mean": round(float(s_mean), 1),
                "saturation_std": round(float(s_std), 1),
                "harmonious_palette": harmonious,
                "neon_ratio": round(float(neon_ratio), 4)
            }
            
            return AICheckResult.from_score(
                name="color_palette",
                display_name="Color Palette Analysis",
                score=score,
                details=details,
                finding_ai="Color palette shows AI-typical characteristics (limited, harmonious, vibrant)",
                finding_natural="Color distribution appears natural/photographic"
            )
            
        except Exception as e:
            return AICheckResult(
                name="color_palette",
                display_name="Color Palette Analysis",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_edge_unnaturalness(self, gray: np.ndarray) -> AICheckResult:
        """
        Check for unnaturally clean edges.
        AI creates very clean, precise edges that don't exist in real photos.
        """
        try:
            # Multi-scale edge analysis
            edges_fine = cv2.Canny(gray, 100, 200)
            edges_coarse = cv2.Canny(gray, 30, 80)
            
            # 1. Edge sharpness analysis
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edge_sharpness = np.std(laplacian)
            
            # 2. Edge continuity (AI has more continuous edges)
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges_fine, kernel, iterations=1)
            edge_continuity = np.sum(edges_dilated > 0) / (np.sum(edges_fine > 0) + 1)
            
            # 3. Edge width consistency
            # AI edges are uniformly thin, natural edges vary
            dist_transform = cv2.distanceTransform(255 - edges_fine, cv2.DIST_L2, 5)
            edge_width_var = np.std(dist_transform[edges_coarse > 0]) if np.any(edges_coarse > 0) else 0
            
            # 4. Ratio of fine to coarse edges (AI has similar at both scales)
            fine_count = np.sum(edges_fine > 0)
            coarse_count = np.sum(edges_coarse > 0)
            edge_ratio = fine_count / (coarse_count + 1)
            
            score = 0
            
            # Very sharp edges
            if edge_sharpness > 50:
                score += 20
            
            # High continuity (AI edges don't break)
            if edge_continuity > 2.5:
                score += 25
            elif edge_continuity > 2.0:
                score += 15
            
            # Low edge width variance (uniform edges = AI)
            if edge_width_var < 1.5:
                score += 25
            elif edge_width_var < 2.5:
                score += 15
            
            # Edge ratio close to 1 (similar at all scales)
            if 0.7 < edge_ratio < 1.3:
                score += 20
            
            score = min(100, score)
            
            details = {
                "edge_sharpness": round(float(edge_sharpness), 2),
                "edge_continuity": round(float(edge_continuity), 2),
                "edge_width_variance": round(float(edge_width_var), 2),
                "edge_scale_ratio": round(float(edge_ratio), 2)
            }
            
            return AICheckResult.from_score(
                name="edge_unnaturalness",
                display_name="Edge Quality Analysis",
                score=score,
                details=details,
                finding_ai="Edges appear unnaturally clean and uniform (AI-typical)",
                finding_natural="Edge quality appears natural with expected variations"
            )
            
        except Exception as e:
            return AICheckResult(
                name="edge_unnaturalness",
                display_name="Edge Quality Analysis",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_texture_uniformity(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """
        Check for AI-typical texture patterns.
        AI often creates overly uniform or repetitive textures.
        """
        try:
            h, w = gray.shape
            
            # 1. Local texture variance
            block_size = min(h, w) // 8
            texture_vars = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    laplacian = cv2.Laplacian(block, cv2.CV_64F)
                    texture_vars.append(np.var(laplacian))
            
            if texture_vars:
                texture_uniformity = 1 - min(1, np.std(texture_vars) / (np.mean(texture_vars) + 1))
            else:
                texture_uniformity = 0.5
            
            # 2. High-frequency content analysis
            f = np.fft.fft2(gray.astype(float))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            center_y, center_x = h // 2, w // 2
            
            # High frequency mask
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            hf_mask = dist > min(h, w) * 0.3
            
            hf_energy = np.sum(magnitude[hf_mask])
            total_energy = np.sum(magnitude)
            hf_ratio = hf_energy / (total_energy + 1e-6)
            
            # AI often has less high-frequency detail
            
            # 3. Texture repetition check
            # Sample small patches and compare
            patch_size = min(h, w) // 16
            if patch_size > 8:
                patches = []
                for _ in range(20):
                    py = np.random.randint(0, h - patch_size)
                    px = np.random.randint(0, w - patch_size)
                    patch = gray[py:py+patch_size, px:px+patch_size].flatten()
                    patches.append(patch / (np.linalg.norm(patch) + 1e-6))
                
                # Calculate average correlation
                correlations = []
                for i in range(len(patches)):
                    for j in range(i+1, min(i+5, len(patches))):
                        corr = np.dot(patches[i], patches[j])
                        correlations.append(abs(corr))
                
                avg_correlation = np.mean(correlations) if correlations else 0
            else:
                avg_correlation = 0.5
            
            score = 0
            
            # High texture uniformity = AI
            if texture_uniformity > 0.8:
                score += 30
            elif texture_uniformity > 0.6:
                score += 15
            
            # Low high-frequency ratio = AI (lacks fine detail)
            if hf_ratio < 0.02:
                score += 30
            elif hf_ratio < 0.05:
                score += 15
            
            # High patch correlation = repetitive = AI
            if avg_correlation > 0.7:
                score += 25
            elif avg_correlation > 0.5:
                score += 15
            
            score = min(100, score)
            
            details = {
                "texture_uniformity": round(float(texture_uniformity), 3),
                "high_freq_ratio": round(float(hf_ratio), 4),
                "patch_correlation": round(float(avg_correlation), 3)
            }
            
            return AICheckResult.from_score(
                name="texture_uniformity",
                display_name="Texture Pattern Analysis",
                score=score,
                details=details,
                finding_ai="Texture appears overly uniform or synthetic",
                finding_natural="Texture patterns appear natural"
            )
            
        except Exception as e:
            return AICheckResult(
                name="texture_uniformity",
                display_name="Texture Pattern Analysis",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_noise_pattern(self, gray: np.ndarray) -> AICheckResult:
        """
        Analyze noise patterns for AI signatures.
        AI-generated images have different noise characteristics than real photos.
        """
        try:
            # 1. Extract noise using Gaussian blur subtraction
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray.astype(float) - blurred.astype(float)
            
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            
            # 2. Noise uniformity across image
            h, w = gray.shape
            block_size = min(h, w) // 4
            noise_stds = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block_noise = noise[i:i+block_size, j:j+block_size]
                    noise_stds.append(np.std(block_noise))
            
            noise_uniformity = 1 - min(1, np.std(noise_stds) / (np.mean(noise_stds) + 1))
            
            # 3. Noise frequency analysis
            noise_fft = np.fft.fft2(noise)
            noise_mag = np.abs(np.fft.fftshift(noise_fft))
            
            # Natural noise is more random, AI noise can have patterns
            noise_entropy = -np.sum((noise_mag / noise_mag.sum() + 1e-10) * 
                                    np.log2(noise_mag / noise_mag.sum() + 1e-10))
            
            # Normalize entropy
            max_entropy = np.log2(noise_mag.size)
            noise_entropy_normalized = noise_entropy / max_entropy
            
            score = 0
            
            # Very low noise = AI (too clean)
            if noise_std < 3:
                score += 30
            elif noise_std < 5:
                score += 15
            
            # Very uniform noise = AI
            if noise_uniformity > 0.9:
                score += 25
            elif noise_uniformity > 0.75:
                score += 15
            
            # Structured noise (low entropy) = AI
            if noise_entropy_normalized < 0.6:
                score += 25
            elif noise_entropy_normalized < 0.7:
                score += 15
            
            score = min(100, score)
            
            details = {
                "noise_std": round(float(noise_std), 2),
                "noise_uniformity": round(float(noise_uniformity), 3),
                "noise_entropy": round(float(noise_entropy_normalized), 3)
            }
            
            return AICheckResult.from_score(
                name="noise_pattern",
                display_name="Noise Pattern Analysis",
                score=score,
                details=details,
                finding_ai="Noise patterns appear synthetic or too uniform",
                finding_natural="Noise characteristics appear natural (sensor noise)"
            )
            
        except Exception as e:
            return AICheckResult(
                name="noise_pattern",
                display_name="Noise Pattern Analysis",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_frequency_artifacts(self, gray: np.ndarray) -> AICheckResult:
        """
        Check for frequency domain artifacts typical of AI generation.
        Diffusion models and GANs leave specific frequency signatures.
        """
        try:
            h, w = gray.shape
            
            # Compute FFT
            f = np.fft.fft2(gray.astype(float))
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))
            
            center_y, center_x = h // 2, w // 2
            
            # 1. Check for unusual patterns in frequency domain
            # Radial profile analysis
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            max_dist = min(h, w) // 2
            radial_bins = 20
            radial_profile = []
            
            for i in range(radial_bins):
                r_min = i * max_dist / radial_bins
                r_max = (i + 1) * max_dist / radial_bins
                mask = (dist >= r_min) & (dist < r_max)
                if np.any(mask):
                    radial_profile.append(np.mean(magnitude[mask]))
            
            # Check for anomalies in radial profile
            if len(radial_profile) > 5:
                profile_diff = np.diff(radial_profile)
                profile_smoothness = np.std(profile_diff)
            else:
                profile_smoothness = 1
            
            # 2. Check for grid-like patterns (common in some AI)
            # Look at corners of FFT
            corner_size = min(h, w) // 8
            corners = [
                magnitude[:corner_size, :corner_size],
                magnitude[:corner_size, -corner_size:],
                magnitude[-corner_size:, :corner_size],
                magnitude[-corner_size:, -corner_size:]
            ]
            corner_energy = sum(np.mean(c) for c in corners) / 4
            center_energy = np.mean(magnitude[center_y-corner_size:center_y+corner_size,
                                              center_x-corner_size:center_x+corner_size])
            
            corner_ratio = corner_energy / (center_energy + 1e-6)
            
            # 3. Mid-frequency analysis (diffusion model artifacts)
            mid_mask = (dist > min(h, w) * 0.15) & (dist < min(h, w) * 0.35)
            mid_energy = np.mean(magnitude[mid_mask])
            low_mask = dist <= min(h, w) * 0.15
            low_energy = np.mean(magnitude[low_mask])
            
            mid_to_low_ratio = mid_energy / (low_energy + 1e-6)
            
            score = 0
            
            # Very smooth radial profile = possibly AI
            if profile_smoothness < 0.5:
                score += 25
            
            # High corner energy = grid patterns = AI
            if corner_ratio > 0.15:
                score += 30
            elif corner_ratio > 0.1:
                score += 15
            
            # Unusual mid-frequency ratio
            if mid_to_low_ratio > 0.5:
                score += 25
            elif mid_to_low_ratio > 0.35:
                score += 15
            
            score = min(100, score)
            
            details = {
                "radial_profile_smoothness": round(float(profile_smoothness), 3),
                "corner_ratio": round(float(corner_ratio), 4),
                "mid_to_low_ratio": round(float(mid_to_low_ratio), 3)
            }
            
            return AICheckResult.from_score(
                name="frequency_artifacts",
                display_name="Frequency Artifact Detection",
                score=score,
                details=details,
                finding_ai="Frequency patterns show AI generation signatures",
                finding_natural="Frequency distribution appears natural"
            )
            
        except Exception as e:
            return AICheckResult(
                name="frequency_artifacts",
                display_name="Frequency Artifact Detection",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_symmetry_patterns(self, gray: np.ndarray) -> AICheckResult:
        """
        Check for unnatural symmetry or repetition.
        AI often creates more symmetric content than natural images.
        """
        try:
            h, w = gray.shape
            
            # 1. Global symmetry check
            left = gray[:, :w//2]
            right = np.fliplr(gray[:, w//2:w//2*2])
            min_w = min(left.shape[1], right.shape[1])
            h_symmetry = 1 - np.mean(np.abs(left[:, :min_w].astype(float) - 
                                            right[:, :min_w].astype(float))) / 255
            
            top = gray[:h//2, :]
            bottom = np.flipud(gray[h//2:h//2*2, :])
            min_h = min(top.shape[0], bottom.shape[0])
            v_symmetry = 1 - np.mean(np.abs(top[:min_h, :].astype(float) - 
                                            bottom[:min_h, :].astype(float))) / 255
            
            # 2. Local pattern repetition using autocorrelation
            norm_gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)
            f = np.fft.fft2(norm_gray)
            autocorr = np.abs(np.fft.ifft2(f * np.conj(f)))
            autocorr = np.fft.fftshift(autocorr)
            autocorr = autocorr / autocorr.max()
            
            # Mask center
            center_y, center_x = h // 2, w // 2
            mask_size = min(h, w) // 8
            autocorr_masked = autocorr.copy()
            autocorr_masked[center_y-mask_size:center_y+mask_size,
                           center_x-mask_size:center_x+mask_size] = 0
            
            secondary_peak = np.max(autocorr_masked)
            
            score = 0
            
            # High horizontal symmetry
            if h_symmetry > 0.85:
                score += 30
            elif h_symmetry > 0.7:
                score += 15
            
            # High vertical symmetry
            if v_symmetry > 0.85:
                score += 25
            elif v_symmetry > 0.7:
                score += 12
            
            # Strong secondary peaks (repetition)
            if secondary_peak > 0.4:
                score += 30
            elif secondary_peak > 0.25:
                score += 15
            
            score = min(100, score)
            
            details = {
                "horizontal_symmetry": round(float(h_symmetry), 3),
                "vertical_symmetry": round(float(v_symmetry), 3),
                "repetition_score": round(float(secondary_peak), 3)
            }
            
            return AICheckResult.from_score(
                name="symmetry_patterns",
                display_name="Symmetry & Repetition Analysis",
                score=score,
                details=details,
                finding_ai="Image shows excessive symmetry or repetitive patterns",
                finding_natural="Symmetry levels appear natural"
            )
            
        except Exception as e:
            return AICheckResult(
                name="symmetry_patterns",
                display_name="Symmetry & Repetition Analysis",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_infographic_detection(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """
        Detect if image is an AI-generated infographic/promotional image.
        These are very common outputs from modern AI.
        """
        try:
            h, w = image.shape[:2]
            
            # 1. Check for large uniform color blocks (common in infographics)
            # Quantize to find uniform regions
            quantized = (image // 40) * 40
            
            # Find connected components of each color
            gray_quant = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_quant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate area of large uniform regions
            large_regions = 0
            total_area = h * w
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > total_area * 0.05:  # Regions > 5% of image
                    large_regions += 1
            
            # 2. Check for rectangular shapes (UI elements, boxes)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                # Count horizontal and vertical lines
                h_lines = 0
                v_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    if angle < 10 or angle > 170:
                        h_lines += 1
                    elif 80 < angle < 100:
                        v_lines += 1
                
                line_regularity = (h_lines + v_lines) / (len(lines) + 1)
            else:
                line_regularity = 0
                h_lines = 0
                v_lines = 0
            
            # 3. Check for text-like patterns
            # Text creates specific edge patterns
            # Apply morphological operations to detect text-like structures
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            text_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_like_regions = 0
            for cnt in text_contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / (bh + 1)
                # Text boxes are typically wider than tall
                if aspect > 3 and bw > 30:
                    text_like_regions += 1
            
            # 4. Icon/logo detection (small uniform shapes)
            small_uniform_shapes = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if total_area * 0.005 < area < total_area * 0.05:
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
                    if circularity > 0.7:  # Circular/uniform shapes
                        small_uniform_shapes += 1
            
            score = 0
            
            # Multiple large uniform regions
            if large_regions >= 3:
                score += 25
            elif large_regions >= 2:
                score += 15
            
            # High line regularity (grid-like structure)
            if line_regularity > 0.7:
                score += 25
            elif line_regularity > 0.5:
                score += 15
            
            # Text-like regions
            if text_like_regions > 5:
                score += 20
            elif text_like_regions > 2:
                score += 10
            
            # Multiple uniform shapes (icons)
            if small_uniform_shapes > 3:
                score += 20
            elif small_uniform_shapes > 1:
                score += 10
            
            score = min(100, score)
            
            details = {
                "large_uniform_regions": large_regions,
                "line_regularity": round(float(line_regularity), 3),
                "text_like_regions": text_like_regions,
                "uniform_shapes": small_uniform_shapes,
                "horizontal_lines": h_lines,
                "vertical_lines": v_lines
            }
            
            return AICheckResult.from_score(
                name="infographic_detection",
                display_name="Infographic/Promotional Detection",
                score=score,
                details=details,
                finding_ai="Image structure matches AI-generated infographic/promotional material",
                finding_natural="Image does not show infographic characteristics"
            )
            
        except Exception as e:
            return AICheckResult(
                name="infographic_detection",
                display_name="Infographic/Promotional Detection",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_synthetic_lighting(self, image: np.ndarray) -> AICheckResult:
        """
        Check for unrealistic lighting patterns.
        AI often creates impossible or inconsistent lighting.
        """
        try:
            # Convert to LAB for luminance analysis
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(float)
            
            h, w = l_channel.shape
            
            # 1. Check for uniform lighting gradients (studio-like = often AI)
            # Fit a plane to the luminance
            y_coords, x_coords = np.mgrid[:h, :w]
            
            # Simple linear regression for lighting gradient
            X = np.column_stack([x_coords.ravel(), y_coords.ravel(), np.ones(h*w)])
            Y = l_channel.ravel()
            
            # Least squares fit
            try:
                coeffs, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                fitted = X @ coeffs
                residual_std = np.std(Y - fitted)
                gradient_strength = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
            except:
                residual_std = 50
                gradient_strength = 0
            
            # Low residual = very uniform lighting = possibly synthetic
            lighting_uniformity = 1 - min(1, residual_std / 40)
            
            # 2. Check for impossible highlights/shadows
            # Very dark and very bright regions
            very_dark = np.sum(l_channel < 20) / l_channel.size
            very_bright = np.sum(l_channel > 235) / l_channel.size
            
            # Extreme contrast without middle tones = synthetic
            middle_tones = np.sum((l_channel > 80) & (l_channel < 180)) / l_channel.size
            
            # 3. Specular highlight analysis
            # Real specular highlights have smooth falloff
            bright_mask = l_channel > 230
            if np.sum(bright_mask) > 100:
                bright_regions = l_channel.copy()
                bright_regions[~bright_mask] = 0
                
                # Check gradient around bright regions
                bright_gradient = np.abs(cv2.Laplacian(bright_regions, cv2.CV_64F))
                highlight_sharpness = np.mean(bright_gradient[bright_mask])
            else:
                highlight_sharpness = 0
            
            score = 0
            
            # Very uniform lighting
            if lighting_uniformity > 0.85:
                score += 25
            elif lighting_uniformity > 0.7:
                score += 12
            
            # Extreme contrast without middle tones
            if middle_tones < 0.4 and (very_dark > 0.1 or very_bright > 0.1):
                score += 25
            
            # Sharp highlight edges (unrealistic)
            if highlight_sharpness > 50:
                score += 25
            elif highlight_sharpness > 30:
                score += 15
            
            # Strong uniform gradient (studio lighting)
            if gradient_strength > 0.05 and lighting_uniformity > 0.7:
                score += 20
            
            score = min(100, score)
            
            details = {
                "lighting_uniformity": round(float(lighting_uniformity), 3),
                "gradient_strength": round(float(gradient_strength), 4),
                "very_dark_ratio": round(float(very_dark), 4),
                "very_bright_ratio": round(float(very_bright), 4),
                "middle_tones_ratio": round(float(middle_tones), 3),
                "highlight_sharpness": round(float(highlight_sharpness), 2)
            }
            
            return AICheckResult.from_score(
                name="synthetic_lighting",
                display_name="Lighting Realism Analysis",
                score=score,
                details=details,
                finding_ai="Lighting patterns appear synthetic or unnaturally uniform",
                finding_natural="Lighting characteristics appear natural"
            )
            
        except Exception as e:
            return AICheckResult(
                name="synthetic_lighting",
                display_name="Lighting Realism Analysis",
                score=50,
                status="uncertain",
                finding=f"Error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _generate_assessment(self, checks: List[AICheckResult], 
                             likely_ai: int, possibly_ai: int, 
                             likely_natural: int) -> Dict:
        """Generate overall AI assessment - MORE SENSITIVE for modern AI."""
        
        total = len(checks)
        
        # Calculate weighted score (give more weight to high-scoring checks)
        scores = [c.score for c in checks]
        avg_score = np.mean(scores)
        max_score = max(scores)
        weighted_score = (avg_score * 0.6) + (max_score * 0.4)
        
        # Collect indicators
        ai_indicators = [c.display_name for c in checks 
                        if c.status in [AIDetectionStatus.LIKELY_AI.value, 
                                        AIDetectionStatus.POSSIBLY_AI.value]]
        natural_signs = [c.display_name for c in checks 
                        if c.status == AIDetectionStatus.LIKELY_NATURAL.value]
        
        # More aggressive detection thresholds
        ai_signal_count = likely_ai + (possibly_ai * 0.5)
        
        if likely_ai >= 2 or ai_signal_count >= 3 or weighted_score >= 55:
            conclusion = "LIKELY_AI_GENERATED"
            confidence = min(95, 50 + likely_ai * 12 + possibly_ai * 5)
            interpretation = f"Multiple indicators ({likely_ai} strong, {possibly_ai} moderate) suggest AI-generated content."
        elif likely_ai >= 1 or ai_signal_count >= 2 or weighted_score >= 45:
            conclusion = "POSSIBLY_AI_GENERATED"
            confidence = 45 + likely_ai * 10 + possibly_ai * 5
            interpretation = f"Some indicators suggest possible AI generation. Manual verification recommended."
        elif likely_natural >= 6:
            conclusion = "LIKELY_NATURAL"
            confidence = min(85, 40 + likely_natural * 7)
            interpretation = f"Most checks ({likely_natural}/{total}) indicate natural/camera-captured content."
        else:
            conclusion = "UNCERTAIN"
            confidence = 40 + abs(likely_ai - likely_natural) * 5
            interpretation = "Cannot reliably determine origin. Modern AI can be very difficult to detect."
        
        return {
            "conclusion": conclusion,
            "confidence": round(float(confidence), 1),
            "interpretation": interpretation,
            "average_score": round(float(avg_score), 1),
            "weighted_score": round(float(weighted_score), 1),
            "indicators_found": ai_indicators,
            "natural_signs": natural_signs,
            "recommendation": self._get_recommendation(conclusion),
            "disclaimer": "AI generation detection is inherently probabilistic. Modern AI (Gemini, DALL-E 3, Midjourney v6) can be extremely difficult to detect with certainty."
        }
    
    def _get_recommendation(self, conclusion: str) -> str:
        """Get recommendation based on conclusion."""
        
        recommendations = {
            "LIKELY_AI_GENERATED": "Strong indicators of AI generation detected. Treat this content as potentially AI-created. Seek additional verification if authenticity is critical.",
            "POSSIBLY_AI_GENERATED": "Some AI indicators detected. Exercise caution. Consider reverse image search and metadata analysis for verification.",
            "LIKELY_NATURAL": "Content appears naturally captured, but modern AI can be very convincing. Standard verification applies.",
            "UNCERTAIN": "Cannot reliably determine origin. Modern AI detection has inherent limitations. Manual expert review recommended."
        }
        
        return recommendations.get(conclusion, "Manual inspection recommended.")


# Alias for backward compatibility
AIGenerationDetector = AIGenerationDetectorV2
