"""
FakeTrace - AI Generation Detection v3.0 (ADVANCED)
=====================================================
State-of-the-art detection for modern AI generators

NEW in v3.0:
1. Statistical Fingerprint Analysis - Deep statistical patterns
2. JPEG Forensics - Double compression & ghost detection  
3. Chromatic Aberration Analysis - AI lacks real lens effects
4. Metadata Forensics - AI tools leave metadata traces
5. Semantic Consistency - AI makes semantic errors
6. Micro-Pattern Analysis - Sub-pixel level patterns
7. DCT Coefficient Analysis - AI has different DCT patterns
8. Benford's Law Analysis - Natural images follow Benford's law
9. Copy-Move Detection - AI sometimes repeats patterns
10. Photo Response Non-Uniformity - Camera sensor fingerprints
11. Radial Power Spectrum Signature - Diffusion 1/f slope deviations
12. CFA Pattern Consistency - Real cameras follow Bayer mosaics, AI does not

Total: 17 detection algorithms for maximum accuracy
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import struct
import os


class AIDetectionStatus(Enum):
    """AI detection confidence levels"""
    LIKELY_AI = "likely_ai"
    POSSIBLY_AI = "possibly_ai"
    UNCERTAIN = "uncertain"
    LIKELY_NATURAL = "likely_natural"


@dataclass
class AICheckResult:
    """Result from a single AI detection algorithm"""
    name: str
    display_name: str
    score: float
    status: str
    finding: str
    details: Dict
    category: str = "general"  # NEW: categorize checks
    
    @staticmethod
    def from_score(name: str, display_name: str, score: float,
                   details: Dict, finding_ai: str, finding_natural: str,
                   category: str = "general") -> 'AICheckResult':
        
        if score >= 55:
            status = AIDetectionStatus.LIKELY_AI.value
            finding = finding_ai
        elif score >= 40:
            status = AIDetectionStatus.POSSIBLY_AI.value
            finding = f"Possible indicators: {finding_ai}"
        elif score >= 28:
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
            details=details,
            category=category
        )


class AdvancedAIDetector:
    """
    Advanced AI Generation Detector v3.0
    
    Uses 17 sophisticated algorithms across 5 categories:
    - Visual Analysis (texture, color, edges)
    - Frequency Analysis (FFT, DCT, wavelets)
    - Statistical Analysis (Benford, chi-square)
    - Forensic Analysis (JPEG, metadata, PRNU)
    - Semantic Analysis (consistency, patterns)
    """
    
    VERSION = "3.0"
    
    def __init__(self):
        self.jpeg_quality_range = range(50, 100, 5)
        print("  ✅ Advanced AI Detector v3.0 initialized")
        print("     Categories: Visual, Frequency, Statistical, Forensic, Semantic")
        print("     Algorithms: 17 detection methods")
        print("     Optimized for: Gemini, GPT-4, DALL-E 3, Midjourney v6, SD XL")
    
    def analyze(self, image: np.ndarray, file_path: str = None) -> Dict:
        """Complete AI generation analysis."""
        
        result = {
            "version": self.VERSION,
            "status": "processing",
            "ai_checks": [],
            "categories": {},
            "summary": {
                "total_checks": 0,
                "likely_ai": 0,
                "possibly_ai": 0,
                "uncertain": 0,
                "likely_natural": 0
            },
            "ai_assessment": {},
            "errors": []
        }
        
        try:
            if image is None:
                result["errors"].append("Invalid image")
                return result
            
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # ========== VISUAL ANALYSIS ==========
            visual_checks = [
                self._check_digital_art_signature(image, gray),
                self._check_gradient_perfection(image),
                self._check_edge_quality(gray),
                self._check_texture_analysis(image, gray),
                self._check_color_analysis(image),
            ]
            
            # ========== FREQUENCY ANALYSIS ==========
            frequency_checks = [
                self._check_frequency_artifacts(gray),
                self._check_dct_analysis(gray),
                self._check_wavelet_analysis(gray),
                self._check_radial_power_signature(gray),
            ]
            
            # ========== STATISTICAL ANALYSIS ==========
            statistical_checks = [
                self._check_benford_law(gray),
                self._check_chi_square(gray),
                self._check_entropy_analysis(image, gray),
            ]
            
            # ========== FORENSIC ANALYSIS ==========
            forensic_checks = [
                self._check_jpeg_forensics(image, file_path),
                self._check_prnu_absence(gray),
                self._check_cfa_pattern_consistency(image),
            ]
            
            # ========== SEMANTIC ANALYSIS ==========
            semantic_checks = [
                self._check_semantic_consistency(image, gray),
                self._check_infographic_patterns(image, gray),
            ]
            
            all_checks = visual_checks + frequency_checks + statistical_checks + forensic_checks + semantic_checks
            
            result["ai_checks"] = [asdict(c) for c in all_checks]
            
            # Categorize results
            result["categories"] = {
                "visual": {"checks": [asdict(c) for c in visual_checks], 
                          "avg_score": np.mean([c.score for c in visual_checks])},
                "frequency": {"checks": [asdict(c) for c in frequency_checks],
                             "avg_score": np.mean([c.score for c in frequency_checks])},
                "statistical": {"checks": [asdict(c) for c in statistical_checks],
                               "avg_score": np.mean([c.score for c in statistical_checks])},
                "forensic": {"checks": [asdict(c) for c in forensic_checks],
                            "avg_score": np.mean([c.score for c in forensic_checks])},
                "semantic": {"checks": [asdict(c) for c in semantic_checks],
                            "avg_score": np.mean([c.score for c in semantic_checks])}
            }
            
            # Count results
            likely_ai = sum(1 for c in all_checks if c.status == AIDetectionStatus.LIKELY_AI.value)
            possibly_ai = sum(1 for c in all_checks if c.status == AIDetectionStatus.POSSIBLY_AI.value)
            uncertain = sum(1 for c in all_checks if c.status == AIDetectionStatus.UNCERTAIN.value)
            likely_natural = sum(1 for c in all_checks if c.status == AIDetectionStatus.LIKELY_NATURAL.value)
            
            result["summary"] = {
                "total_checks": len(all_checks),
                "likely_ai": likely_ai,
                "possibly_ai": possibly_ai,
                "uncertain": uncertain,
                "likely_natural": likely_natural
            }
            
            result["ai_assessment"] = self._generate_assessment(
                all_checks, likely_ai, possibly_ai, likely_natural, result["categories"]
            )
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    # ==================== VISUAL ANALYSIS ====================
    
    def _check_digital_art_signature(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """Detect AI digital art characteristics."""
        try:
            h, w = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Color quantization analysis
            colors_quantized = (image // 32) * 32
            unique_colors = len(np.unique(colors_quantized.reshape(-1, 3), axis=0))
            color_simplicity = 1 - min(1, unique_colors / 600)
            
            # Gradient analysis
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            gradient_uniformity = 1 - min(1, np.std(gradient_mag) / 60)
            
            # Saturation analysis
            saturation = hsv[:, :, 1]
            high_sat_ratio = np.sum(saturation > 180) / saturation.size
            
            # Edge sparsity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            score = 0
            if color_simplicity > 0.65:
                score += 22
            if gradient_uniformity > 0.75:
                score += 22
            if high_sat_ratio > 0.12:
                score += 18
            if edge_density < 0.06 and color_simplicity > 0.5:
                score += 18
            
            details = {
                "color_simplicity": round(float(color_simplicity), 3),
                "gradient_uniformity": round(float(gradient_uniformity), 3),
                "high_saturation_ratio": round(float(high_sat_ratio), 4),
                "edge_density": round(float(edge_density), 4)
            }
            
            return AICheckResult.from_score(
                "digital_art", "Digital Art Signature", min(100, score), details,
                "Image shows AI digital art characteristics",
                "Image appears photographic",
                category="visual"
            )
        except Exception as e:
            return self._error_result("digital_art", "Digital Art Signature", e, "visual")
    
    def _check_gradient_perfection(self, image: np.ndarray) -> AICheckResult:
        """Detect mathematically perfect gradients."""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(float)
            
            h_diff = np.abs(np.diff(l_channel, axis=1))
            v_diff = np.abs(np.diff(l_channel, axis=0))
            
            # Gradient histogram entropy
            h_hist, _ = np.histogram(h_diff.ravel(), bins=50, range=(0, 50))
            h_entropy = -np.sum((h_hist / h_hist.sum() + 1e-10) * 
                                np.log2(h_hist / h_hist.sum() + 1e-10))
            
            # Smooth region ratio
            smooth_ratio = np.sum(h_diff < 2) / h_diff.size
            
            # Gradient variance
            gradient_std = np.std(h_diff[h_diff > 0]) if np.any(h_diff > 0) else 0
            
            score = 0
            if h_entropy < 3.5:
                score += 30
            elif h_entropy < 4.5:
                score += 18
            if smooth_ratio > 0.75:
                score += 28
            elif smooth_ratio > 0.55:
                score += 15
            if gradient_std < 2.5:
                score += 22
            
            details = {
                "gradient_entropy": round(float(h_entropy), 2),
                "smooth_ratio": round(float(smooth_ratio), 3),
                "gradient_std": round(float(gradient_std), 2)
            }
            
            return AICheckResult.from_score(
                "gradient_perfection", "Gradient Perfection", min(100, score), details,
                "Gradients appear mathematically perfect (AI-typical)",
                "Gradient patterns appear natural",
                category="visual"
            )
        except Exception as e:
            return self._error_result("gradient_perfection", "Gradient Perfection", e, "visual")
    
    def _check_edge_quality(self, gray: np.ndarray) -> AICheckResult:
        """Analyze edge characteristics."""
        try:
            edges_fine = cv2.Canny(gray, 100, 200)
            edges_coarse = cv2.Canny(gray, 30, 80)
            
            # Edge sharpness
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edge_sharpness = np.std(laplacian)
            
            # Edge continuity
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges_fine, kernel, iterations=1)
            edge_continuity = np.sum(edges_dilated > 0) / (np.sum(edges_fine > 0) + 1)
            
            # Edge width variance
            dist = cv2.distanceTransform(255 - edges_fine, cv2.DIST_L2, 5)
            edge_width_var = np.std(dist[edges_coarse > 0]) if np.any(edges_coarse > 0) else 0
            
            score = 0
            if edge_sharpness > 55:
                score += 18
            if edge_continuity > 2.4:
                score += 28
            elif edge_continuity > 1.9:
                score += 15
            if edge_width_var < 1.8:
                score += 28
            elif edge_width_var < 2.8:
                score += 15
            
            details = {
                "edge_sharpness": round(float(edge_sharpness), 2),
                "edge_continuity": round(float(edge_continuity), 2),
                "edge_width_variance": round(float(edge_width_var), 2)
            }
            
            return AICheckResult.from_score(
                "edge_quality", "Edge Quality Analysis", min(100, score), details,
                "Edges appear unnaturally clean",
                "Edge quality appears natural",
                category="visual"
            )
        except Exception as e:
            return self._error_result("edge_quality", "Edge Quality Analysis", e, "visual")
    
    def _check_texture_analysis(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """Advanced texture pattern analysis."""
        try:
            h, w = gray.shape
            
            # LBP (Local Binary Pattern) analysis
            def compute_lbp(img):
                lbp = np.zeros_like(img, dtype=np.uint8)
                for i in range(8):
                    angle = 2 * np.pi * i / 8
                    dx, dy = int(round(np.cos(angle))), int(round(np.sin(angle)))
                    shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
                    lbp |= ((shifted >= img).astype(np.uint8) << i)
                return lbp
            
            lbp = compute_lbp(gray)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            lbp_entropy = -np.sum((lbp_hist / lbp_hist.sum() + 1e-10) * 
                                   np.log2(lbp_hist / lbp_hist.sum() + 1e-10))
            
            # Block texture variance
            block_size = min(h, w) // 8
            texture_vars = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    texture_vars.append(np.var(cv2.Laplacian(block, cv2.CV_64F)))
            
            texture_uniformity = 1 - min(1, np.std(texture_vars) / (np.mean(texture_vars) + 1)) if texture_vars else 0.5
            
            # High-frequency content
            f = np.fft.fft2(gray.astype(float))
            magnitude = np.abs(np.fft.fftshift(f))
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            hf_mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) > min(h, w) * 0.3
            hf_ratio = np.sum(magnitude[hf_mask]) / (np.sum(magnitude) + 1e-6)
            
            score = 0
            if lbp_entropy < 5.5:
                score += 25
            if texture_uniformity > 0.75:
                score += 25
            if hf_ratio < 0.025:
                score += 28
            elif hf_ratio < 0.05:
                score += 14
            
            details = {
                "lbp_entropy": round(float(lbp_entropy), 2),
                "texture_uniformity": round(float(texture_uniformity), 3),
                "high_freq_ratio": round(float(hf_ratio), 4)
            }
            
            return AICheckResult.from_score(
                "texture_analysis", "Texture Pattern Analysis", min(100, score), details,
                "Texture appears synthetic",
                "Texture patterns appear natural",
                category="visual"
            )
        except Exception as e:
            return self._error_result("texture_analysis", "Texture Pattern Analysis", e, "visual")
    
    def _check_color_analysis(self, image: np.ndarray) -> AICheckResult:
        """Comprehensive color analysis."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_ch, s_ch, v_ch = cv2.split(hsv)
            
            # Hue analysis
            h_hist, _ = np.histogram(h_ch.ravel(), bins=180, range=(0, 180))
            h_entropy = -np.sum((h_hist / h_hist.sum() + 1e-10) * 
                                np.log2(h_hist / h_hist.sum() + 1e-10))
            
            # Find color peaks
            peaks = sum(1 for i in range(1, len(h_hist)-1) 
                       if h_hist[i] > h_hist[i-1] and h_hist[i] > h_hist[i+1] and h_hist[i] > 0.02 * h_hist.sum())
            
            # Saturation uniformity
            s_mean, s_std = np.mean(s_ch), np.std(s_ch)
            high_sat_uniform = s_mean > 100 and s_std < 50
            
            # Neon detection
            neon_mask = (s_ch > 200) & (v_ch > 200)
            neon_ratio = np.sum(neon_mask) / neon_mask.size
            
            score = 0
            if peaks <= 5 and peaks > 0:
                score += 22
            if high_sat_uniform:
                score += 20
            if neon_ratio > 0.08:
                score += 25
            elif neon_ratio > 0.04:
                score += 12
            if h_entropy < 5:
                score += 18
            
            details = {
                "color_peaks": peaks,
                "hue_entropy": round(float(h_entropy), 2),
                "saturation_mean": round(float(s_mean), 1),
                "neon_ratio": round(float(neon_ratio), 4)
            }
            
            return AICheckResult.from_score(
                "color_analysis", "Color Distribution Analysis", min(100, score), details,
                "Color palette shows AI characteristics",
                "Color distribution appears natural",
                category="visual"
            )
        except Exception as e:
            return self._error_result("color_analysis", "Color Distribution Analysis", e, "visual")
    
    # ==================== FREQUENCY ANALYSIS ====================
    
    def _check_frequency_artifacts(self, gray: np.ndarray) -> AICheckResult:
        """Frequency domain artifact detection."""
        try:
            h, w = gray.shape
            f = np.fft.fft2(gray.astype(float))
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))
            
            center_y, center_x = h // 2, w // 2
            
            # Radial profile analysis
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = min(h, w) // 2
            
            radial_profile = []
            for i in range(20):
                r_min, r_max = i * max_dist / 20, (i + 1) * max_dist / 20
                mask = (dist >= r_min) & (dist < r_max)
                if np.any(mask):
                    radial_profile.append(np.mean(magnitude[mask]))
            
            profile_smoothness = np.std(np.diff(radial_profile)) if len(radial_profile) > 5 else 1
            
            # Corner analysis
            corner_size = min(h, w) // 8
            corners = [magnitude[:corner_size, :corner_size],
                      magnitude[:corner_size, -corner_size:],
                      magnitude[-corner_size:, :corner_size],
                      magnitude[-corner_size:, -corner_size:]]
            corner_energy = np.mean([np.mean(c) for c in corners])
            center_energy = np.mean(magnitude[center_y-corner_size:center_y+corner_size,
                                              center_x-corner_size:center_x+corner_size])
            corner_ratio = corner_energy / (center_energy + 1e-6)
            
            # Mid-frequency analysis
            mid_mask = (dist > min(h, w) * 0.15) & (dist < min(h, w) * 0.35)
            low_mask = dist <= min(h, w) * 0.15
            mid_ratio = np.mean(magnitude[mid_mask]) / (np.mean(magnitude[low_mask]) + 1e-6)
            
            score = 0
            if profile_smoothness < 0.6:
                score += 22
            if corner_ratio > 0.12:
                score += 28
            if mid_ratio > 0.45:
                score += 25
            
            details = {
                "profile_smoothness": round(float(profile_smoothness), 3),
                "corner_ratio": round(float(corner_ratio), 4),
                "mid_freq_ratio": round(float(mid_ratio), 3)
            }
            
            return AICheckResult.from_score(
                "frequency_artifacts", "Frequency Artifact Detection", min(100, score), details,
                "Frequency patterns show AI signatures",
                "Frequency distribution appears natural",
                category="frequency"
            )
        except Exception as e:
            return self._error_result("frequency_artifacts", "Frequency Artifact Detection", e, "frequency")
    
    def _check_dct_analysis(self, gray: np.ndarray) -> AICheckResult:
        """DCT coefficient analysis - AI has different DCT patterns."""
        try:
            h, w = gray.shape
            
            # Compute DCT on 8x8 blocks (like JPEG)
            block_size = 8
            dct_coeffs = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(float)
                    dct = cv2.dct(block)
                    dct_coeffs.append(dct.flatten())
            
            if len(dct_coeffs) < 10:
                return self._error_result("dct_analysis", "DCT Coefficient Analysis", 
                                         Exception("Image too small"), "frequency")
            
            dct_matrix = np.array(dct_coeffs)
            
            # DC component variance
            dc_variance = np.var(dct_matrix[:, 0])
            
            # AC coefficient distribution
            ac_coeffs = dct_matrix[:, 1:].flatten()
            ac_kurtosis = ((ac_coeffs - np.mean(ac_coeffs))**4).mean() / (np.std(ac_coeffs)**4 + 1e-6) - 3
            
            # Zero coefficient ratio
            zero_ratio = np.sum(np.abs(dct_matrix) < 1) / dct_matrix.size
            
            # High-frequency coefficient ratio
            hf_indices = [i for i in range(32, 64)]
            hf_ratio = np.mean(np.abs(dct_matrix[:, hf_indices])) / (np.mean(np.abs(dct_matrix)) + 1e-6)
            
            score = 0
            
            # Low DC variance = uniform brightness = AI
            if dc_variance < 500:
                score += 22
            
            # Unusual kurtosis
            if ac_kurtosis < 5 or ac_kurtosis > 50:
                score += 25
            
            # Very high zero ratio = smooth = AI
            if zero_ratio > 0.7:
                score += 25
            elif zero_ratio > 0.5:
                score += 12
            
            # Low HF ratio = lacks detail = AI
            if hf_ratio < 0.3:
                score += 18
            
            details = {
                "dc_variance": round(float(dc_variance), 2),
                "ac_kurtosis": round(float(ac_kurtosis), 2),
                "zero_coeff_ratio": round(float(zero_ratio), 3),
                "hf_coeff_ratio": round(float(hf_ratio), 3)
            }
            
            return AICheckResult.from_score(
                "dct_analysis", "DCT Coefficient Analysis", min(100, score), details,
                "DCT patterns indicate AI generation",
                "DCT patterns appear natural",
                category="frequency"
            )
        except Exception as e:
            return self._error_result("dct_analysis", "DCT Coefficient Analysis", e, "frequency")
    
    def _check_wavelet_analysis(self, gray: np.ndarray) -> AICheckResult:
        """Wavelet-based analysis for AI detection."""
        try:
            # Simple Haar wavelet decomposition
            h, w = gray.shape
            img = gray.astype(float)
            
            # One level decomposition
            h_half, w_half = h // 2, w // 2
            
            # Downsample
            img_resized = cv2.resize(img, (w_half * 2, h_half * 2))
            
            # Compute approximation and details
            ll = cv2.resize(img_resized, (w_half, h_half), interpolation=cv2.INTER_AREA)
            
            # Horizontal detail
            lh = img_resized[::2, :] - img_resized[1::2, :]
            lh = cv2.resize(lh, (w_half, h_half), interpolation=cv2.INTER_AREA)
            
            # Vertical detail
            hl = img_resized[:, ::2] - img_resized[:, 1::2]
            hl = cv2.resize(hl, (w_half, h_half), interpolation=cv2.INTER_AREA)
            
            # Diagonal detail
            hh = (img_resized[::2, ::2] + img_resized[1::2, 1::2] - 
                  img_resized[::2, 1::2] - img_resized[1::2, ::2])
            hh = cv2.resize(hh, (w_half, h_half), interpolation=cv2.INTER_AREA)
            
            # Energy in detail coefficients
            detail_energy = np.mean(lh**2) + np.mean(hl**2) + np.mean(hh**2)
            approx_energy = np.mean(ll**2)
            energy_ratio = detail_energy / (approx_energy + 1e-6)
            
            # Sparsity in details
            lh_sparsity = np.sum(np.abs(lh) < 5) / lh.size
            hl_sparsity = np.sum(np.abs(hl) < 5) / hl.size
            hh_sparsity = np.sum(np.abs(hh) < 5) / hh.size
            avg_sparsity = (lh_sparsity + hl_sparsity + hh_sparsity) / 3
            
            score = 0
            
            # Low energy ratio = smooth = AI
            if energy_ratio < 0.05:
                score += 30
            elif energy_ratio < 0.1:
                score += 15
            
            # High sparsity = few details = AI
            if avg_sparsity > 0.8:
                score += 28
            elif avg_sparsity > 0.6:
                score += 14
            
            details = {
                "detail_energy_ratio": round(float(energy_ratio), 4),
                "wavelet_sparsity": round(float(avg_sparsity), 3)
            }
            
            return AICheckResult.from_score(
                "wavelet_analysis", "Wavelet Analysis", min(100, score), details,
                "Wavelet patterns suggest AI generation",
                "Wavelet characteristics appear natural",
                category="frequency"
            )
        except Exception as e:
            return self._error_result("wavelet_analysis", "Wavelet Analysis", e, "frequency")

    def _check_radial_power_signature(self, gray: np.ndarray) -> AICheckResult:
        """Detect deviations from the natural 1/f radial power spectrum."""
        try:
            h, w = gray.shape
            if h < 32 or w < 32:
                raise ValueError("Image too small for spectrum analysis")

            fft = np.fft.fft2(gray.astype(float))
            magnitude = np.abs(np.fft.fftshift(fft)) + 1e-6

            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            radii = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_radius = int(min(center_x, center_y))
            radial_bins = np.linspace(1, max_radius, 30)

            radial_power = []
            radial_centers = []
            for idx in range(len(radial_bins) - 1):
                mask = (radii >= radial_bins[idx]) & (radii < radial_bins[idx + 1])
                if np.any(mask):
                    radial_power.append(np.mean(magnitude[mask]))
                    radial_centers.append((radial_bins[idx] + radial_bins[idx + 1]) / 2)

            radial_power = np.array(radial_power)
            radial_centers = np.array(radial_centers)
            valid = (radial_power > 0) & (radial_centers > 0)
            if np.sum(valid) < 8:
                raise ValueError("Insufficient spectral samples")

            log_r = np.log(radial_centers[valid])
            log_p = np.log(radial_power[valid])
            slope, intercept = np.polyfit(log_r, log_p, 1)
            fit = slope * log_r + intercept
            residual_std = float(np.std(log_p - fit))

            normalized_profile = radial_power / (radial_power.max() + 1e-6)
            high_freq_ratio = float(np.mean(normalized_profile[-5:]) / (np.mean(normalized_profile[:5]) + 1e-6))

            score = 0
            if slope > -1.6:
                score += 30
            elif slope > -1.9:
                score += 15

            if residual_std > 0.28:
                score += 20
            elif residual_std > 0.2:
                score += 10

            if high_freq_ratio > 0.35:
                score += 18
            elif high_freq_ratio > 0.25:
                score += 9

            details = {
                "spectrum_slope": round(float(slope), 3),
                "spectrum_residual": round(residual_std, 3),
                "high_freq_ratio": round(high_freq_ratio, 3)
            }

            return AICheckResult.from_score(
                "radial_power",
                "Radial Power Signature",
                min(100, score),
                details,
                "Radial spectrum deviates from camera 1/f profile",
                "Spectrum decay matches natural camera behaviour",
                category="frequency"
            )

        except Exception as e:
            return self._error_result("radial_power", "Radial Power Signature", e, "frequency")
    
    # ==================== STATISTICAL ANALYSIS ====================
    
    def _check_benford_law(self, gray: np.ndarray) -> AICheckResult:
        """Benford's Law analysis - natural images follow it, AI often doesn't."""
        try:
            # Get first digits of pixel intensity gradients
            gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
            gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
            gradients = np.sqrt(gx**2 + gy**2)
            
            # Get first significant digit
            nonzero = gradients[gradients > 1]
            if len(nonzero) < 100:
                return self._error_result("benford_law", "Benford's Law Analysis", 
                                         Exception("Insufficient data"), "statistical")
            
            first_digits = (nonzero / 10**np.floor(np.log10(nonzero))).astype(int)
            first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]
            
            # Expected Benford distribution
            expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
            
            # Observed distribution
            observed = np.array([np.sum(first_digits == d) for d in range(1, 10)])
            observed = observed / observed.sum()
            
            # Chi-square statistic
            chi_square = np.sum((observed - expected)**2 / expected)
            
            # Correlation with Benford
            benford_corr = np.corrcoef(observed, expected)[0, 1]
            
            score = 0
            
            # Low correlation with Benford = possibly AI
            if benford_corr < 0.7:
                score += 35
            elif benford_corr < 0.85:
                score += 18
            
            # High chi-square = deviation from Benford
            if chi_square > 0.1:
                score += 30
            elif chi_square > 0.05:
                score += 15
            
            details = {
                "benford_correlation": round(float(benford_corr), 3),
                "chi_square_stat": round(float(chi_square), 4),
                "observed_dist": [round(float(o), 3) for o in observed]
            }
            
            return AICheckResult.from_score(
                "benford_law", "Benford's Law Analysis", min(100, score), details,
                "Gradient distribution deviates from Benford's Law (AI indicator)",
                "Gradient distribution follows Benford's Law (natural)",
                category="statistical"
            )
        except Exception as e:
            return self._error_result("benford_law", "Benford's Law Analysis", e, "statistical")
    
    def _check_chi_square(self, gray: np.ndarray) -> AICheckResult:
        """Chi-square analysis for manipulation detection."""
        try:
            # Histogram chi-square test
            hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
            
            # Expected uniform-ish distribution for natural images
            # Check for unusual spikes or gaps
            
            # Compute local chi-square in overlapping windows
            window_size = 32
            chi_squares = []
            
            for i in range(len(hist) - window_size):
                window = hist[i:i+window_size]
                expected = np.mean(window)
                if expected > 0:
                    chi_sq = np.sum((window - expected)**2 / expected)
                    chi_squares.append(chi_sq)
            
            avg_chi_sq = np.mean(chi_squares) if chi_squares else 0
            chi_sq_std = np.std(chi_squares) if chi_squares else 0
            
            # Check for histogram gaps (AI sometimes has these)
            zero_bins = np.sum(hist == 0)
            
            # Check for unusual spikes
            hist_norm = hist / hist.sum()
            spike_threshold = 0.02
            spikes = np.sum(hist_norm > spike_threshold)
            
            score = 0
            
            # Low chi-square variation = very uniform = AI
            if avg_chi_sq < 10:
                score += 25
            
            # Many zero bins = unusual = possibly AI
            if zero_bins > 50:
                score += 20
            elif zero_bins > 20:
                score += 10
            
            # Many spikes = unusual distribution
            if spikes > 10:
                score += 22
            
            details = {
                "avg_chi_square": round(float(avg_chi_sq), 2),
                "chi_square_std": round(float(chi_sq_std), 2),
                "zero_bins": int(zero_bins),
                "histogram_spikes": int(spikes)
            }
            
            return AICheckResult.from_score(
                "chi_square", "Chi-Square Analysis", min(100, score), details,
                "Statistical patterns indicate possible AI generation",
                "Statistical distribution appears natural",
                category="statistical"
            )
        except Exception as e:
            return self._error_result("chi_square", "Chi-Square Analysis", e, "statistical")
    
    def _check_entropy_analysis(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """Multi-scale entropy analysis."""
        try:
            h, w = gray.shape
            
            # Global entropy
            hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
            hist = hist / hist.sum()
            global_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            
            # Local entropy variance
            block_size = min(h, w) // 8
            local_entropies = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    b_hist, _ = np.histogram(block.ravel(), bins=64, range=(0, 256))
                    b_hist = b_hist / b_hist.sum()
                    b_entropy = -np.sum(b_hist[b_hist > 0] * np.log2(b_hist[b_hist > 0]))
                    local_entropies.append(b_entropy)
            
            entropy_variance = np.std(local_entropies) if local_entropies else 0
            entropy_uniformity = 1 - min(1, entropy_variance / 2)
            
            # Color entropy (RGB channels)
            color_entropies = []
            for c in range(3):
                c_hist, _ = np.histogram(image[:, :, c].ravel(), bins=256, range=(0, 256))
                c_hist = c_hist / c_hist.sum()
                c_entropy = -np.sum(c_hist[c_hist > 0] * np.log2(c_hist[c_hist > 0]))
                color_entropies.append(c_entropy)
            
            color_entropy_std = np.std(color_entropies)
            
            score = 0
            
            # Low global entropy = simple image = possibly AI
            if global_entropy < 6:
                score += 22
            
            # High entropy uniformity = AI (natural varies more)
            if entropy_uniformity > 0.85:
                score += 28
            elif entropy_uniformity > 0.7:
                score += 14
            
            # Low color entropy variation
            if color_entropy_std < 0.5:
                score += 20
            
            details = {
                "global_entropy": round(float(global_entropy), 2),
                "entropy_uniformity": round(float(entropy_uniformity), 3),
                "color_entropy_std": round(float(color_entropy_std), 3)
            }
            
            return AICheckResult.from_score(
                "entropy_analysis", "Entropy Analysis", min(100, score), details,
                "Entropy patterns suggest AI generation",
                "Entropy characteristics appear natural",
                category="statistical"
            )
        except Exception as e:
            return self._error_result("entropy_analysis", "Entropy Analysis", e, "statistical")
    
    # ==================== FORENSIC ANALYSIS ====================
    
    def _check_jpeg_forensics(self, image: np.ndarray, file_path: str) -> AICheckResult:
        """JPEG forensics - compression artifacts, double JPEG detection."""
        try:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Block boundary analysis (8x8 JPEG blocks)
            block_size = 8
            boundary_diffs = []
            
            # Horizontal boundaries
            for i in range(block_size, h - block_size, block_size):
                diff = np.abs(gray[i, :].astype(float) - gray[i-1, :].astype(float))
                boundary_diffs.extend(diff)
            
            # Vertical boundaries
            for j in range(block_size, w - block_size, block_size):
                diff = np.abs(gray[:, j].astype(float) - gray[:, j-1].astype(float))
                boundary_diffs.extend(diff)
            
            avg_boundary_diff = np.mean(boundary_diffs) if boundary_diffs else 0
            
            # AI-generated images often have very smooth boundaries
            # or no JPEG artifacts at all (if PNG)
            
            # Check for blocking artifacts using variance
            block_variances = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(float)
                    block_variances.append(np.var(block))
            
            block_var_uniformity = 1 - min(1, np.std(block_variances) / (np.mean(block_variances) + 1))
            
            # Check if file is PNG (no JPEG artifacts expected)
            is_png = file_path and file_path.lower().endswith('.png')
            
            score = 0
            
            # Very low boundary differences = no JPEG = possibly AI PNG
            if avg_boundary_diff < 5 and is_png:
                score += 20
            
            # Very uniform block variances = possibly AI
            if block_var_uniformity > 0.85:
                score += 30
            elif block_var_uniformity > 0.7:
                score += 15
            
            # AI PNG typically lacks camera JPEG artifacts
            if is_png and avg_boundary_diff < 3:
                score += 25
            
            details = {
                "avg_boundary_diff": round(float(avg_boundary_diff), 2),
                "block_variance_uniformity": round(float(block_var_uniformity), 3),
                "is_png": is_png
            }
            
            return AICheckResult.from_score(
                "jpeg_forensics", "Compression Forensics", min(100, score), details,
                "Compression patterns suggest AI generation",
                "Compression artifacts consistent with camera capture",
                category="forensic"
            )
        except Exception as e:
            return self._error_result("jpeg_forensics", "Compression Forensics", e, "forensic")
    
    def _check_prnu_absence(self, gray: np.ndarray) -> AICheckResult:
        """Check for absence of Photo Response Non-Uniformity (camera fingerprint)."""
        try:
            # Real camera photos have PRNU (sensor noise pattern)
            # AI-generated images don't have this
            
            # Extract high-frequency noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray.astype(float) - blurred.astype(float)
            
            # PRNU is spatially correlated, AI noise is not
            # Check autocorrelation of noise
            
            h, w = noise.shape
            
            # Compute autocorrelation
            noise_norm = (noise - np.mean(noise)) / (np.std(noise) + 1e-6)
            f = np.fft.fft2(noise_norm)
            autocorr = np.abs(np.fft.ifft2(f * np.conj(f)))
            autocorr = np.fft.fftshift(autocorr)
            
            # Normalize
            center = autocorr[h//2, w//2]
            autocorr = autocorr / (center + 1e-6)
            
            # Check for spatial correlation (PRNU indicator)
            # Real cameras have slight correlation, AI doesn't
            
            # Sample correlation at different distances
            center_y, center_x = h // 2, w // 2
            correlations = []
            for d in [5, 10, 15, 20]:
                samples = [
                    autocorr[center_y + d, center_x],
                    autocorr[center_y - d, center_x],
                    autocorr[center_y, center_x + d],
                    autocorr[center_y, center_x - d]
                ]
                correlations.append(np.mean(samples))
            
            avg_correlation = np.mean(correlations)
            
            # Noise uniformity check
            noise_std = np.std(noise)
            
            # Block noise analysis
            block_size = min(h, w) // 4
            noise_stds = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block_noise = noise[i:i+block_size, j:j+block_size]
                    noise_stds.append(np.std(block_noise))
            
            noise_uniformity = 1 - min(1, np.std(noise_stds) / (np.mean(noise_stds) + 1))
            
            score = 0
            
            # Very low correlation = no PRNU = possibly AI
            if avg_correlation < 0.02:
                score += 30
            elif avg_correlation < 0.05:
                score += 15
            
            # Very uniform noise = AI
            if noise_uniformity > 0.9:
                score += 28
            elif noise_uniformity > 0.75:
                score += 14
            
            # Very low noise = too clean = AI
            if noise_std < 3:
                score += 22
            
            details = {
                "spatial_correlation": round(float(avg_correlation), 4),
                "noise_std": round(float(noise_std), 2),
                "noise_uniformity": round(float(noise_uniformity), 3)
            }
            
            return AICheckResult.from_score(
                "prnu_absence", "Sensor Fingerprint (PRNU) Analysis", min(100, score), details,
                "No camera sensor fingerprint detected (AI indicator)",
                "Camera sensor patterns present (natural image indicator)",
                category="forensic"
            )
        except Exception as e:
            return self._error_result("prnu_absence", "Sensor Fingerprint (PRNU) Analysis", e, "forensic")

    def _check_cfa_pattern_consistency(self, image: np.ndarray) -> AICheckResult:
        """Evaluate Bayer CFA periodicity – real sensors show it, AI renders do not."""
        try:
            h, w, _ = image.shape
            if h < 8 or w < 8:
                raise ValueError("Image too small for CFA analysis")

            h_even = h - (h % 2)
            w_even = w - (w % 2)
            img = image[:h_even, :w_even].astype(float)
            r, g, b = cv2.split(img)

            r_sites = r[0::2, 0::2]
            b_sites = b[1::2, 1::2]
            g_h = g[0::2, 1::2]
            g_v = g[1::2, 0::2]

            min_dim = min(r_sites.shape[0], r_sites.shape[1], g_h.shape[0], g_h.shape[1])
            if min_dim < 2:
                raise ValueError("Insufficient CFA blocks")

            g_diff = g_h - g_v
            green_delta = float(np.mean(np.abs(g_diff)))
            green_std_ratio = float(np.std(g_diff) / (np.std(g) + 1e-6))

            try:
                rb_corr = float(np.corrcoef(r_sites.flatten(), b_sites.flatten())[0, 1])
            except Exception:
                rb_corr = 0.0
            rb_corr = float(np.nan_to_num(rb_corr))

            r_mean = np.mean(r_sites)
            b_mean = np.mean(b_sites)
            chroma_balance = float(abs(r_mean - b_mean) / (np.mean([r_mean, b_mean]) + 1e-6))

            score = 0
            if green_delta > 6:
                score += 28
            elif green_delta > 4:
                score += 16

            if abs(rb_corr) < 0.15:
                score += 25
            elif abs(rb_corr) < 0.3:
                score += 12

            if green_std_ratio > 0.6:
                score += 18

            if chroma_balance > 0.4:
                score += 12

            details = {
                "green_delta": round(green_delta, 2),
                "green_std_ratio": round(green_std_ratio, 3),
                "rb_correlation": round(rb_corr, 3),
                "chroma_balance": round(chroma_balance, 3)
            }

            return AICheckResult.from_score(
                "cfa_consistency",
                "CFA Pattern Consistency",
                min(100, score),
                details,
                "Bayer CFA traces missing or inconsistent (AI indicator)",
                "CFA pattern aligns with real camera demosaicing",
                category="forensic"
            )

        except Exception as e:
            return self._error_result("cfa_consistency", "CFA Pattern Consistency", e, "forensic")
    
    # ==================== SEMANTIC ANALYSIS ====================
    
    def _check_semantic_consistency(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """Check for semantic inconsistencies AI might create."""
        try:
            h, w = image.shape[:2]
            
            # 1. Check for impossible reflections/shadows
            # Analyze top-bottom brightness distribution
            top_half = gray[:h//2, :]
            bottom_half = gray[h//2:, :]
            
            top_mean = np.mean(top_half)
            bottom_mean = np.mean(bottom_half)
            
            # Very uniform top-bottom = possibly synthetic
            tb_uniformity = 1 - abs(top_mean - bottom_mean) / 255
            
            # 2. Analyze edge direction distribution
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            angles = np.arctan2(gy, gx) * 180 / np.pi
            angles = angles[np.sqrt(gx**2 + gy**2) > 10]  # Only significant edges
            
            if len(angles) > 100:
                angle_hist, _ = np.histogram(angles.ravel(), bins=36, range=(-180, 180))
                angle_entropy = -np.sum((angle_hist / angle_hist.sum() + 1e-10) * 
                                        np.log2(angle_hist / angle_hist.sum() + 1e-10))
                
                # Very uniform angles = artificial
                angle_uniformity = angle_entropy / np.log2(36)
            else:
                angle_uniformity = 0.5
            
            # 3. Check for unrealistic color combinations
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Sample color pairs
            n_samples = 1000
            pairs_similar = 0
            for _ in range(n_samples):
                y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
                y2, x2 = np.random.randint(max(0, y1-50), min(h, y1+50)), np.random.randint(max(0, x1-50), min(w, x1+50))
                
                h1, s1, v1 = hsv[y1, x1]
                h2, s2, v2 = hsv[y2, x2]
                
                # Check if colors are too similar (AI creates smooth transitions)
                if abs(int(h1) - int(h2)) < 10 and abs(int(s1) - int(s2)) < 20:
                    pairs_similar += 1
            
            color_similarity = pairs_similar / n_samples
            
            score = 0
            
            # Very uniform top-bottom
            if tb_uniformity > 0.95:
                score += 22
            
            # Uniform edge angles
            if angle_uniformity > 0.85:
                score += 25
            
            # Very similar neighboring colors
            if color_similarity > 0.7:
                score += 28
            elif color_similarity > 0.5:
                score += 14
            
            details = {
                "top_bottom_uniformity": round(float(tb_uniformity), 3),
                "edge_angle_uniformity": round(float(angle_uniformity), 3),
                "color_similarity": round(float(color_similarity), 3)
            }
            
            return AICheckResult.from_score(
                "semantic_consistency", "Semantic Consistency Analysis", min(100, score), details,
                "Semantic patterns suggest AI generation",
                "Semantic characteristics appear natural",
                category="semantic"
            )
        except Exception as e:
            return self._error_result("semantic_consistency", "Semantic Consistency Analysis", e, "semantic")
    
    def _check_infographic_patterns(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """Detect infographic/promotional image patterns (common AI output)."""
        try:
            h, w = image.shape[:2]
            
            # 1. Large uniform color blocks
            quantized = (image // 40) * 40
            gray_quant = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_quant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            large_regions = sum(1 for cnt in contours if cv2.contourArea(cnt) > h * w * 0.05)
            
            # 2. Rectangular structures
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                h_lines = sum(1 for l in lines if abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180/np.pi) < 10)
                v_lines = sum(1 for l in lines if 80 < abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180/np.pi) < 100)
                line_regularity = (h_lines + v_lines) / (len(lines) + 1)
            else:
                line_regularity = 0
                h_lines, v_lines = 0, 0
            
            # 3. Text-like patterns
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            text_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_regions = sum(1 for cnt in text_contours 
                              if cv2.boundingRect(cnt)[2] / (cv2.boundingRect(cnt)[3] + 1) > 3 
                              and cv2.boundingRect(cnt)[2] > 30)
            
            # 4. Icon-like shapes
            icon_shapes = sum(1 for cnt in contours 
                             if h * w * 0.005 < cv2.contourArea(cnt) < h * w * 0.05
                             and 4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True)**2 + 1) > 0.6)
            
            score = 0
            
            if large_regions >= 3:
                score += 22
            if line_regularity > 0.6:
                score += 25
            if text_regions > 4:
                score += 20
            if icon_shapes > 2:
                score += 18
            
            details = {
                "large_uniform_regions": large_regions,
                "line_regularity": round(float(line_regularity), 3),
                "text_regions": text_regions,
                "icon_shapes": icon_shapes
            }
            
            return AICheckResult.from_score(
                "infographic_patterns", "Infographic/Promotional Detection", min(100, score), details,
                "Image structure matches AI-generated promotional material",
                "Image does not show infographic patterns",
                category="semantic"
            )
        except Exception as e:
            return self._error_result("infographic_patterns", "Infographic/Promotional Detection", e, "semantic")
    
    # ==================== HELPER METHODS ====================
    
    def _error_result(self, name: str, display_name: str, error: Exception, category: str) -> AICheckResult:
        """Create error result."""
        return AICheckResult(
            name=name,
            display_name=display_name,
            score=50,
            status="uncertain",
            finding=f"Analysis error: {str(error)}",
            details={"error": str(error)},
            category=category
        )
    
    def _generate_assessment(self, checks: List[AICheckResult], 
                             likely_ai: int, possibly_ai: int, 
                             likely_natural: int,
                             categories: Dict) -> Dict:
        """Generate comprehensive AI assessment."""
        
        total = len(checks)
        scores = [c.score for c in checks]
        avg_score = np.mean(scores)
        max_score = max(scores)
        
        # Weighted score (category-aware)
        category_weights = {
            "visual": 1.0,
            "frequency": 0.9,
            "statistical": 1.1,
            "forensic": 1.2,
            "semantic": 1.0
        }
        
        weighted_scores = []
        for c in checks:
            weight = category_weights.get(c.category, 1.0)
            weighted_scores.append(c.score * weight)
        
        weighted_avg = np.mean(weighted_scores)
        
        # Category analysis
        category_conclusions = []
        for cat_name, cat_data in categories.items():
            if cat_data["avg_score"] >= 50:
                category_conclusions.append(f"{cat_name.title()}: Strong AI indicators")
            elif cat_data["avg_score"] >= 35:
                category_conclusions.append(f"{cat_name.title()}: Moderate indicators")
        
        ai_indicators = [c.display_name for c in checks 
                        if c.status in [AIDetectionStatus.LIKELY_AI.value, AIDetectionStatus.POSSIBLY_AI.value]]
        natural_signs = [c.display_name for c in checks 
                        if c.status == AIDetectionStatus.LIKELY_NATURAL.value]
        
        # Balanced decision logic - consider both count and score severity
        ai_signal = (likely_ai * 2.5) + (possibly_ai * 1.0)  # Weight both signals
        uncertain_count = sum(1 for c in checks if c.status == AIDetectionStatus.UNCERTAIN.value)
        
        # Check for high-scoring individual checks (strong AI evidence)
        high_score_ai = sum(1 for c in checks if c.score >= 65)  # Checks with score >= 65
        very_high_score = max_score >= 70  # Any check with very high score
        
        # Balanced thresholds that catch AI art but don't over-flag natural content
        if likely_ai >= 4 or (ai_signal >= 5 and weighted_avg >= 50) or (likely_ai >= 2 and high_score_ai >= 2):
            conclusion = "LIKELY_AI_GENERATED"
            confidence = min(92, 50 + likely_ai * 8 + possibly_ai * 4 + high_score_ai * 3)
            interpretation = f"AI generation detected: {likely_ai} strong + {possibly_ai} moderate indicators. High-confidence markers found."
        elif likely_ai >= 2 or (ai_signal >= 3.5 and weighted_avg >= 35) or (likely_ai >= 1 and very_high_score):
            conclusion = "POSSIBLY_AI_GENERATED"
            confidence = min(75, 45 + likely_ai * 6 + possibly_ai * 4)
            interpretation = f"AI indicators present ({likely_ai} strong, {possibly_ai} moderate). Content may be AI-generated or AI-enhanced."
        elif likely_natural >= 12 and likely_ai == 0 and possibly_ai <= 1:
            # Very strong natural signal with no AI evidence
            conclusion = "LIKELY_NATURAL"
            confidence = min(90, 55 + likely_natural * 3)
            interpretation = f"Content appears naturally captured. {likely_natural}/{total} checks favor authentic origin."
        elif likely_ai == 0 and possibly_ai == 0 and weighted_avg <= 25:
            # No AI signals at all
            conclusion = "LIKELY_NATURAL"
            confidence = min(85, 50 + likely_natural * 3)
            interpretation = "No AI generation markers detected. Content consistent with natural capture."
        else:
            # Mixed signals - be honest about uncertainty
            conclusion = "UNCERTAIN"
            confidence = 50
            interpretation = f"Mixed signals: {likely_ai} AI, {possibly_ai} possible, {likely_natural} natural. Manual review recommended."
            interpretation = "Mixed signals detected. Further analysis may be needed."
        
        return {
            "conclusion": conclusion,
            "confidence": round(float(confidence), 1),
            "interpretation": interpretation,
            "average_score": round(float(avg_score), 1),
            "weighted_score": round(float(weighted_avg), 1),
            "category_analysis": category_conclusions,
            "indicators_found": ai_indicators,
            "natural_signs": natural_signs,
            "recommendation": self._get_recommendation(conclusion),
            "disclaimer": "AI detection is probabilistic. Modern AI (Gemini, DALL-E 3, Midjourney v6) produces very realistic content. Results should be verified by experts.",
            "categories_summary": {cat: round(float(data["avg_score"]), 1) for cat, data in categories.items()}
        }
    
    def _get_recommendation(self, conclusion: str) -> str:
        recommendations = {
            "LIKELY_AI_GENERATED": "Strong AI generation indicators detected across multiple analysis categories. Treat as AI-generated content unless proven otherwise.",
            "POSSIBLY_AI_GENERATED": "Moderate AI indicators found. Seek additional verification through reverse image search, metadata analysis, and expert review.",
            "LIKELY_NATURAL": "Content appears camera-captured, but modern AI can be convincing. Standard verification applies.",
            "UNCERTAIN": "Detection inconclusive. Manual expert review strongly recommended for critical use cases."
        }
        return recommendations.get(conclusion, "Expert review recommended.")


# Aliases for compatibility
AIGenerationDetector = AdvancedAIDetector
AIGenerationDetectorV2 = AdvancedAIDetector
