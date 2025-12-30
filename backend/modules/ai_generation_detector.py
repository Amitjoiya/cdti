"""
FakeTrace - AI Generation Detection Module
============================================
Detects signs of AI-generated content (Midjourney, DALL-E, Stable Diffusion, etc.)

This module uses CLASSICAL COMPUTER VISION techniques to detect
patterns commonly found in AI-generated images/videos.

Detection Methods:
1. GAN Fingerprint Detection - Periodic patterns in frequency domain
2. Synthetic Texture Analysis - Unnatural texture patterns
3. Color Coherence Check - AI-typical color distributions
4. Symmetry Analysis - Over-symmetry common in AI art
5. Fine Detail Analysis - AI struggles with fine details
6. Background Consistency - AI often has inconsistent backgrounds
7. Face Analysis - AI face artifacts (if faces present)

IMPORTANT: Results are INDICATORS, not definitive proof.
AI detection is inherently probabilistic.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class AIDetectionStatus(Enum):
    """AI detection confidence levels"""
    LIKELY_AI = "likely_ai"              # Score >= 65: Strong AI indicators
    POSSIBLY_AI = "possibly_ai"          # Score 50-64: Some AI indicators
    UNCERTAIN = "uncertain"              # Score 35-49: Cannot determine
    LIKELY_NATURAL = "likely_natural"    # Score < 35: Appears natural/real


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
        
        if score >= 65:
            status = AIDetectionStatus.LIKELY_AI.value
            finding = finding_ai
        elif score >= 50:
            status = AIDetectionStatus.POSSIBLY_AI.value
            finding = f"Some indicators present: {finding_ai}"
        elif score >= 35:
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


class AIGenerationDetector:
    """
    Detects AI-generated content using classical computer vision.
    
    Philosophy:
    - Higher score = More likely AI-generated
    - Multiple weak signals combined = stronger evidence
    - Always provide uncertainty levels
    """
    
    VERSION = "1.0"
    
    def __init__(self):
        print("  ✅ AI Generation Detector v1.0 initialized")
        print("     Methods: GAN Fingerprint, Texture, Color, Symmetry, Details, Face")
    
    def analyze(self, image: np.ndarray, file_path: str = None) -> Dict:
        """
        Analyze image for AI generation indicators.
        
        Returns:
            Dict with AI detection results
        """
        
        result = {
            "version": self.VERSION,
            "status": "processing",
            
            # Individual AI checks
            "ai_checks": [],
            
            # Summary
            "summary": {
                "total_checks": 0,
                "likely_ai": 0,
                "possibly_ai": 0,
                "uncertain": 0,
                "likely_natural": 0
            },
            
            # Overall AI assessment
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
            
            # Run all AI detection algorithms
            checks = [
                self._check_gan_fingerprint(gray),
                self._check_synthetic_texture(image, gray),
                self._check_color_coherence(image),
                self._check_symmetry(gray),
                self._check_fine_details(gray),
                self._check_background_consistency(image),
                self._check_face_artifacts(image, gray)
            ]
            
            # Store results
            result["ai_checks"] = [asdict(c) for c in checks]
            
            # Count by status
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
            
            # Generate overall assessment
            result["ai_assessment"] = self._generate_assessment(checks, likely_ai, possibly_ai, likely_natural)
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _check_gan_fingerprint(self, gray: np.ndarray) -> AICheckResult:
        """
        Detect GAN fingerprints in frequency domain.
        GANs often leave periodic patterns visible in FFT.
        """
        
        try:
            # Compute FFT
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            # Log transform for better visualization
            log_magnitude = np.log1p(magnitude)
            
            h, w = gray.shape
            center_y, center_x = h // 2, w // 2
            
            # Analyze different frequency bands
            # GANs often have unusual patterns in mid-frequencies
            
            # Create radial masks
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Mid-frequency band (where GAN artifacts often appear)
            mid_mask = (dist > min(h, w) * 0.1) & (dist < min(h, w) * 0.4)
            high_mask = dist >= min(h, w) * 0.4
            
            # Check for periodic peaks in mid-frequencies
            mid_freq = log_magnitude[mid_mask]
            high_freq = log_magnitude[high_mask]
            
            # GAN indicator: unusually regular peaks in mid-frequency
            mid_std = np.std(mid_freq)
            mid_mean = np.mean(mid_freq)
            mid_max = np.max(mid_freq)
            
            # Calculate peak ratio (high peaks relative to mean = possible GAN pattern)
            peak_ratio = (mid_max - mid_mean) / (mid_std + 1e-6)
            
            # Check for symmetry in frequency domain (GANs often have this)
            upper_half = log_magnitude[:center_y, :]
            lower_half = np.flipud(log_magnitude[center_y:, :])
            min_h = min(upper_half.shape[0], lower_half.shape[0])
            freq_symmetry = np.mean(np.abs(upper_half[:min_h] - lower_half[:min_h]))
            
            # Score calculation
            # Higher peak_ratio and lower freq_symmetry = more likely GAN
            score = 0
            
            if peak_ratio > 8:
                score += 35
            elif peak_ratio > 5:
                score += 20
            elif peak_ratio > 3:
                score += 10
            
            if freq_symmetry < 0.5:
                score += 30
            elif freq_symmetry < 1.0:
                score += 15
            
            # Additional: check for grid patterns
            # GANs sometimes have checkerboard artifacts
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_fft = np.fft.fft2(laplacian)
            lap_mag = np.abs(np.fft.fftshift(lap_fft))
            
            # Check corners of FFT (grid pattern indicator)
            corner_size = min(h, w) // 8
            corners = [
                lap_mag[:corner_size, :corner_size],
                lap_mag[:corner_size, -corner_size:],
                lap_mag[-corner_size:, :corner_size],
                lap_mag[-corner_size:, -corner_size:]
            ]
            corner_energy = sum(np.mean(c) for c in corners)
            center_energy = np.mean(lap_mag[center_y-corner_size:center_y+corner_size, 
                                            center_x-corner_size:center_x+corner_size])
            
            if corner_energy > center_energy * 0.3:
                score += 20
            
            score = min(100, score)
            
            details = {
                "peak_ratio": round(float(peak_ratio), 2),
                "freq_symmetry": round(float(freq_symmetry), 3),
                "corner_energy_ratio": round(float(corner_energy / (center_energy + 1e-6)), 3)
            }
            
            return AICheckResult.from_score(
                name="gan_fingerprint",
                display_name="GAN Fingerprint Analysis",
                score=score,
                details=details,
                finding_ai="Periodic patterns in frequency domain suggest GAN-based generation",
                finding_natural="No unusual periodic patterns detected in frequency domain"
            )
            
        except Exception as e:
            return AICheckResult(
                name="gan_fingerprint",
                display_name="GAN Fingerprint Analysis",
                score=50,
                status="uncertain",
                finding=f"Analysis error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_synthetic_texture(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """
        Detect synthetic/unnatural texture patterns.
        AI-generated images often have unnaturally smooth or repetitive textures.
        """
        
        try:
            # Multiple texture analysis approaches
            
            # 1. Local Binary Pattern variance
            def compute_lbp(img, radius=1, n_points=8):
                lbp = np.zeros_like(img, dtype=np.uint8)
                for i in range(n_points):
                    angle = 2 * np.pi * i / n_points
                    dx = int(round(radius * np.cos(angle)))
                    dy = int(round(radius * np.sin(angle)))
                    
                    shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
                    lbp |= ((shifted >= img).astype(np.uint8) << i)
                
                return lbp
            
            lbp = compute_lbp(gray)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            lbp_entropy = -np.sum((lbp_hist / lbp_hist.sum() + 1e-10) * 
                                   np.log2(lbp_hist / lbp_hist.sum() + 1e-10))
            
            # AI images often have lower texture entropy
            
            # 2. Gradient magnitude variance
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            gradient_var = np.var(gradient_mag)
            
            # 3. Check for unnaturally smooth regions
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = np.abs(gray.astype(float) - blur.astype(float))
            smooth_ratio = np.sum(diff < 3) / diff.size
            
            # 4. Texture repetition check
            # Divide image into blocks and compare textures
            h, w = gray.shape
            block_size = min(h, w) // 8
            if block_size > 16:
                blocks = []
                for i in range(0, h - block_size, block_size):
                    for j in range(0, w - block_size, block_size):
                        block = gray[i:i+block_size, j:j+block_size]
                        blocks.append(block.flatten())
                
                if len(blocks) > 4:
                    blocks = np.array(blocks[:16])  # Limit for speed
                    correlations = []
                    for i in range(len(blocks)):
                        for j in range(i+1, len(blocks)):
                            corr = np.corrcoef(blocks[i], blocks[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    
                    avg_correlation = np.mean(correlations) if correlations else 0
                else:
                    avg_correlation = 0
            else:
                avg_correlation = 0
            
            # Score calculation
            score = 0
            
            # Low entropy = possibly synthetic
            if lbp_entropy < 5:
                score += 30
            elif lbp_entropy < 6:
                score += 15
            
            # High smooth ratio = possibly AI
            if smooth_ratio > 0.7:
                score += 25
            elif smooth_ratio > 0.5:
                score += 10
            
            # High block correlation = repetitive texture (AI sign)
            if avg_correlation > 0.6:
                score += 30
            elif avg_correlation > 0.4:
                score += 15
            
            score = min(100, score)
            
            details = {
                "lbp_entropy": round(float(lbp_entropy), 2),
                "smooth_ratio": round(float(smooth_ratio), 3),
                "gradient_variance": round(float(gradient_var), 2),
                "texture_correlation": round(float(avg_correlation), 3)
            }
            
            return AICheckResult.from_score(
                name="synthetic_texture",
                display_name="Synthetic Texture Analysis",
                score=score,
                details=details,
                finding_ai="Texture patterns appear synthetic with unnatural smoothness or repetition",
                finding_natural="Texture patterns appear natural with expected variation"
            )
            
        except Exception as e:
            return AICheckResult(
                name="synthetic_texture",
                display_name="Synthetic Texture Analysis",
                score=50,
                status="uncertain",
                finding=f"Analysis error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_color_coherence(self, image: np.ndarray) -> AICheckResult:
        """
        Check for AI-typical color distributions.
        AI images often have unusual color coherence or saturation patterns.
        """
        
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            h, s, v = cv2.split(hsv)
            l, a, b = cv2.split(lab)
            
            # 1. Saturation analysis
            # AI images often have unusual saturation distribution
            s_mean = np.mean(s)
            s_std = np.std(s)
            s_skew = ((s - s_mean) ** 3).mean() / (s_std ** 3 + 1e-6)
            
            # 2. Hue distribution
            # Natural images have more random hue distribution
            h_hist, _ = np.histogram(h.ravel(), bins=180, range=(0, 180))
            h_entropy = -np.sum((h_hist / h_hist.sum() + 1e-10) * 
                                np.log2(h_hist / h_hist.sum() + 1e-10))
            
            # 3. Color gradient smoothness
            # AI often has unnaturally smooth color transitions
            color_gradient = np.sqrt(
                cv2.Sobel(a.astype(float), cv2.CV_64F, 1, 0)**2 +
                cv2.Sobel(b.astype(float), cv2.CV_64F, 0, 1)**2
            )
            gradient_smoothness = 1 - (np.std(color_gradient) / (np.mean(color_gradient) + 1e-6))
            
            # 4. Check for banding (common in AI images)
            # Look for step-like transitions in lightness
            l_diff = np.abs(np.diff(l.astype(float), axis=1))
            banding_score = np.sum((l_diff > 0) & (l_diff < 3)) / l_diff.size
            
            # Score calculation
            score = 0
            
            # Unusual saturation skew
            if abs(s_skew) > 1.5:
                score += 20
            
            # Low hue entropy (limited color palette)
            if h_entropy < 4:
                score += 25
            elif h_entropy < 5:
                score += 10
            
            # Very smooth color gradients
            if gradient_smoothness > 0.8:
                score += 25
            elif gradient_smoothness > 0.6:
                score += 10
            
            # High banding
            if banding_score > 0.3:
                score += 20
            
            score = min(100, score)
            
            details = {
                "saturation_mean": round(float(s_mean), 1),
                "saturation_skew": round(float(s_skew), 2),
                "hue_entropy": round(float(h_entropy), 2),
                "gradient_smoothness": round(float(gradient_smoothness), 3),
                "banding_score": round(float(banding_score), 3)
            }
            
            return AICheckResult.from_score(
                name="color_coherence",
                display_name="Color Coherence Analysis",
                score=score,
                details=details,
                finding_ai="Color distribution shows AI-typical patterns (limited palette, smooth gradients)",
                finding_natural="Color distribution appears natural"
            )
            
        except Exception as e:
            return AICheckResult(
                name="color_coherence",
                display_name="Color Coherence Analysis",
                score=50,
                status="uncertain",
                finding=f"Analysis error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_symmetry(self, gray: np.ndarray) -> AICheckResult:
        """
        Check for unnatural symmetry.
        AI images sometimes exhibit excessive symmetry.
        """
        
        try:
            h, w = gray.shape
            
            # 1. Horizontal symmetry
            left = gray[:, :w//2]
            right = np.fliplr(gray[:, w//2:w//2*2])
            
            min_w = min(left.shape[1], right.shape[1])
            h_symmetry = 1 - np.mean(np.abs(left[:, :min_w].astype(float) - 
                                            right[:, :min_w].astype(float))) / 255
            
            # 2. Vertical symmetry
            top = gray[:h//2, :]
            bottom = np.flipud(gray[h//2:h//2*2, :])
            
            min_h = min(top.shape[0], bottom.shape[0])
            v_symmetry = 1 - np.mean(np.abs(top[:min_h, :].astype(float) - 
                                            bottom[:min_h, :].astype(float))) / 255
            
            # 3. Pattern symmetry using autocorrelation
            # Normalize image
            norm_gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)
            
            # Compute autocorrelation
            f = np.fft.fft2(norm_gray)
            autocorr = np.fft.ifft2(f * np.conj(f)).real
            autocorr = np.fft.fftshift(autocorr)
            
            # Normalize
            autocorr = autocorr / autocorr.max()
            
            # Check for secondary peaks (indicates repetition/symmetry)
            center_y, center_x = h // 2, w // 2
            center_val = autocorr[center_y, center_x]
            
            # Mask out center
            mask = np.ones_like(autocorr)
            mask_size = min(h, w) // 10
            mask[center_y-mask_size:center_y+mask_size, 
                 center_x-mask_size:center_x+mask_size] = 0
            
            secondary_peaks = autocorr * mask
            max_secondary = np.max(secondary_peaks)
            
            # Score calculation
            score = 0
            
            # Very high horizontal symmetry (unusual for natural images)
            if h_symmetry > 0.9:
                score += 35
            elif h_symmetry > 0.8:
                score += 15
            
            # Very high vertical symmetry
            if v_symmetry > 0.9:
                score += 35
            elif v_symmetry > 0.8:
                score += 15
            
            # Strong secondary peaks in autocorrelation
            if max_secondary > 0.5:
                score += 25
            elif max_secondary > 0.3:
                score += 10
            
            score = min(100, score)
            
            details = {
                "horizontal_symmetry": round(float(h_symmetry), 3),
                "vertical_symmetry": round(float(v_symmetry), 3),
                "autocorr_secondary_peak": round(float(max_secondary), 3)
            }
            
            return AICheckResult.from_score(
                name="symmetry_analysis",
                display_name="Symmetry Analysis",
                score=score,
                details=details,
                finding_ai="Image shows excessive symmetry patterns uncommon in natural images",
                finding_natural="Symmetry levels appear natural"
            )
            
        except Exception as e:
            return AICheckResult(
                name="symmetry_analysis",
                display_name="Symmetry Analysis",
                score=50,
                status="uncertain",
                finding=f"Analysis error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_fine_details(self, gray: np.ndarray) -> AICheckResult:
        """
        Check fine detail quality.
        AI often struggles with fine details, especially at edges.
        """
        
        try:
            # 1. High-frequency content analysis
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            h, w = gray.shape
            center_y, center_x = h // 2, w // 2
            
            # Create high-frequency mask
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            high_freq_mask = dist > min(h, w) * 0.3
            
            high_freq_energy = np.sum(magnitude[high_freq_mask])
            total_energy = np.sum(magnitude)
            hf_ratio = high_freq_energy / (total_energy + 1e-6)
            
            # 2. Edge sharpness analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate edge gradient magnitude
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(gx**2 + gy**2)
            
            # Edge sharpness (higher = sharper edges)
            edge_sharpness = np.percentile(edge_mag, 95) / (np.mean(edge_mag) + 1e-6)
            
            # 3. Detail consistency
            # Compare details in different regions
            block_size = min(h, w) // 4
            detail_scores = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    laplacian = cv2.Laplacian(block, cv2.CV_64F)
                    detail_scores.append(np.var(laplacian))
            
            detail_variance = np.std(detail_scores) / (np.mean(detail_scores) + 1e-6) if detail_scores else 0
            
            # Score calculation
            score = 0
            
            # Low high-frequency ratio (AI often lacks fine details)
            if hf_ratio < 0.05:
                score += 30
            elif hf_ratio < 0.1:
                score += 15
            
            # Unusual edge sharpness (AI can have unnaturally sharp or soft edges)
            if edge_sharpness > 10 or edge_sharpness < 2:
                score += 25
            
            # Low detail variance (AI tends to be more uniform)
            if detail_variance < 0.3:
                score += 30
            elif detail_variance < 0.5:
                score += 15
            
            score = min(100, score)
            
            details = {
                "high_freq_ratio": round(float(hf_ratio), 4),
                "edge_density": round(float(edge_density), 4),
                "edge_sharpness": round(float(edge_sharpness), 2),
                "detail_variance": round(float(detail_variance), 3)
            }
            
            return AICheckResult.from_score(
                name="fine_details",
                display_name="Fine Detail Analysis",
                score=score,
                details=details,
                finding_ai="Fine detail patterns suggest AI generation (uniform details, unusual sharpness)",
                finding_natural="Fine details appear natural with expected variation"
            )
            
        except Exception as e:
            return AICheckResult(
                name="fine_details",
                display_name="Fine Detail Analysis",
                score=50,
                status="uncertain",
                finding=f"Analysis error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_background_consistency(self, image: np.ndarray) -> AICheckResult:
        """
        Check background consistency.
        AI often has inconsistent or unrealistic backgrounds.
        """
        
        try:
            h, w = image.shape[:2]
            
            # Convert to LAB for better color comparison
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # 1. Edge-based segmentation (simple foreground/background)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # Dilate edges to connect nearby edges
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours to identify potential foreground
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create foreground mask
            foreground_mask = np.zeros((h, w), dtype=np.uint8)
            if contours:
                # Take largest contours as foreground
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                for cnt in sorted_contours[:5]:
                    if cv2.contourArea(cnt) > (h * w) * 0.01:  # At least 1% of image
                        cv2.drawContours(foreground_mask, [cnt], -1, 255, -1)
            
            background_mask = 255 - foreground_mask
            
            # 2. Analyze background regions
            if np.sum(background_mask > 0) > 100:
                bg_pixels = lab[background_mask > 0]
                
                # Background color variance
                bg_l_var = np.var(bg_pixels[:, 0])
                bg_a_var = np.var(bg_pixels[:, 1])
                bg_b_var = np.var(bg_pixels[:, 2])
                bg_total_var = (bg_l_var + bg_a_var + bg_b_var) / 3
                
                # Check for unnatural uniformity
                bg_uniformity = 1 / (1 + bg_total_var / 100)
            else:
                bg_uniformity = 0.5
                bg_total_var = 0
            
            # 3. Check for repeated patterns in background
            if np.sum(background_mask > 0) > 1000:
                bg_gray = gray.copy()
                bg_gray[foreground_mask > 0] = 0
                
                # Compute autocorrelation on background
                bg_norm = (bg_gray - np.mean(bg_gray)) / (np.std(bg_gray) + 1e-6)
                f = np.fft.fft2(bg_norm)
                autocorr = np.abs(np.fft.ifft2(f * np.conj(f)))
                
                # Check for periodic patterns
                autocorr_centered = np.fft.fftshift(autocorr)
                center = autocorr_centered[h//2, w//2]
                
                # Mask center
                mask_size = min(h, w) // 8
                autocorr_centered[h//2-mask_size:h//2+mask_size, 
                                  w//2-mask_size:w//2+mask_size] = 0
                
                secondary_max = np.max(autocorr_centered)
                pattern_score = secondary_max / (center + 1e-6)
            else:
                pattern_score = 0
            
            # 4. Check edge blur at foreground/background boundary
            if np.sum(foreground_mask > 0) > 100 and np.sum(background_mask > 0) > 100:
                # Find boundary
                boundary = cv2.dilate(foreground_mask, kernel, iterations=1) - foreground_mask
                
                if np.sum(boundary > 0) > 10:
                    boundary_region = gray[boundary > 0]
                    boundary_blur = np.std(boundary_region)
                else:
                    boundary_blur = 50
            else:
                boundary_blur = 50
            
            # Score calculation
            score = 0
            
            # Very uniform background (AI sign)
            if bg_uniformity > 0.8:
                score += 30
            elif bg_uniformity > 0.6:
                score += 15
            
            # Repetitive patterns in background
            if pattern_score > 0.3:
                score += 25
            elif pattern_score > 0.15:
                score += 10
            
            # Unusual boundary blur (too sharp or too soft)
            if boundary_blur < 10 or boundary_blur > 100:
                score += 25
            
            score = min(100, score)
            
            details = {
                "background_uniformity": round(float(bg_uniformity), 3),
                "background_variance": round(float(bg_total_var), 2),
                "pattern_score": round(float(pattern_score), 3),
                "boundary_blur": round(float(boundary_blur), 2)
            }
            
            return AICheckResult.from_score(
                name="background_consistency",
                display_name="Background Consistency",
                score=score,
                details=details,
                finding_ai="Background shows AI-typical patterns (uniform, repetitive, or unnatural boundaries)",
                finding_natural="Background appears natural"
            )
            
        except Exception as e:
            return AICheckResult(
                name="background_consistency",
                display_name="Background Consistency",
                score=50,
                status="uncertain",
                finding=f"Analysis error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_face_artifacts(self, image: np.ndarray, gray: np.ndarray) -> AICheckResult:
        """
        Check for AI face artifacts.
        AI-generated faces often have telltale signs.
        """
        
        try:
            # Try to detect faces
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                return AICheckResult(
                    name="face_artifacts",
                    display_name="Face Artifact Analysis",
                    score=50,
                    status="uncertain",
                    finding="No faces detected - face artifact analysis not applicable",
                    details={"faces_detected": 0, "note": "Analysis requires face detection"}
                )
            
            face_scores = []
            face_details = []
            
            for i, (x, y, w, h) in enumerate(faces[:3]):  # Analyze up to 3 faces
                face_roi = gray[y:y+h, x:x+w]
                face_color = image[y:y+h, x:x+w]
                
                # 1. Symmetry check (AI faces often too symmetric)
                left_half = face_roi[:, :w//2]
                right_half = np.fliplr(face_roi[:, w//2:])
                min_w = min(left_half.shape[1], right_half.shape[1])
                face_symmetry = 1 - np.mean(np.abs(left_half[:, :min_w].astype(float) - 
                                                    right_half[:, :min_w].astype(float))) / 255
                
                # 2. Skin texture analysis
                # AI skin often too smooth
                face_laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
                skin_texture_var = np.var(face_laplacian)
                
                # 3. Eye region analysis (if face is large enough)
                if h > 50:
                    eye_region = face_roi[h//4:h//2, :]
                    eye_symmetry = 1 - np.mean(np.abs(
                        eye_region[:, :w//2].astype(float) - 
                        np.fliplr(eye_region[:, w//2:]).astype(float)
                    )) / 255
                else:
                    eye_symmetry = 0.5
                
                # 4. Color consistency in skin
                hsv_face = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
                skin_hue_std = np.std(hsv_face[:, :, 0])
                
                # Calculate face score
                f_score = 0
                
                # Too symmetric face
                if face_symmetry > 0.85:
                    f_score += 30
                elif face_symmetry > 0.75:
                    f_score += 15
                
                # Too smooth skin
                if skin_texture_var < 100:
                    f_score += 30
                elif skin_texture_var < 200:
                    f_score += 15
                
                # Too symmetric eyes
                if eye_symmetry > 0.9:
                    f_score += 20
                
                # Too uniform skin color
                if skin_hue_std < 10:
                    f_score += 20
                
                face_scores.append(min(100, f_score))
                face_details.append({
                    "face_index": i,
                    "symmetry": round(float(face_symmetry), 3),
                    "skin_texture": round(float(skin_texture_var), 2),
                    "eye_symmetry": round(float(eye_symmetry), 3),
                    "skin_hue_std": round(float(skin_hue_std), 2)
                })
            
            avg_score = np.mean(face_scores)
            
            details = {
                "faces_detected": len(faces),
                "face_analyses": face_details,
                "average_score": round(float(avg_score), 1)
            }
            
            return AICheckResult.from_score(
                name="face_artifacts",
                display_name="Face Artifact Analysis",
                score=avg_score,
                details=details,
                finding_ai="Face(s) show AI-typical artifacts (excessive symmetry, smooth skin, uniform color)",
                finding_natural="Face(s) appear natural with expected asymmetry and texture"
            )
            
        except Exception as e:
            return AICheckResult(
                name="face_artifacts",
                display_name="Face Artifact Analysis",
                score=50,
                status="uncertain",
                finding=f"Analysis error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _generate_assessment(self, checks: List[AICheckResult], 
                             likely_ai: int, possibly_ai: int, 
                             likely_natural: int) -> Dict:
        """Generate overall AI assessment based on all checks."""
        
        total = len(checks)
        
        # Calculate weighted average score
        avg_score = np.mean([c.score for c in checks])
        
        # Collect indicators
        ai_indicators = [c.display_name for c in checks 
                        if c.status in [AIDetectionStatus.LIKELY_AI.value, 
                                        AIDetectionStatus.POSSIBLY_AI.value]]
        natural_signs = [c.display_name for c in checks 
                        if c.status == AIDetectionStatus.LIKELY_NATURAL.value]
        
        # Determine conclusion based on evidence
        if likely_ai >= 3:
            conclusion = "LIKELY_AI_GENERATED"
            confidence = min(95, 60 + likely_ai * 10)
            interpretation = f"Multiple indicators ({likely_ai}/{total}) suggest this content is AI-generated."
        elif likely_ai >= 2 or (likely_ai >= 1 and possibly_ai >= 2):
            conclusion = "POSSIBLY_AI_GENERATED"
            confidence = 50 + (likely_ai * 10) + (possibly_ai * 5)
            interpretation = f"Some indicators suggest possible AI generation, but evidence is not conclusive."
        elif likely_natural >= 4:
            conclusion = "LIKELY_NATURAL"
            confidence = min(90, 50 + likely_natural * 10)
            interpretation = f"Most indicators ({likely_natural}/{total}) suggest natural/camera-captured content."
        else:
            conclusion = "UNCERTAIN"
            confidence = 40 + abs(likely_ai - likely_natural) * 5
            interpretation = "Cannot reliably determine if content is AI-generated or natural."
        
        return {
            "conclusion": conclusion,
            "confidence": round(float(confidence), 1),
            "interpretation": interpretation,
            "average_score": round(float(avg_score), 1),
            "indicators_found": ai_indicators,
            "natural_signs": natural_signs,
            "recommendation": self._get_recommendation(conclusion),
            "disclaimer": "AI generation detection is inherently probabilistic. These results are indicators, not definitive proof."
        }
    
    def _get_recommendation(self, conclusion: str) -> str:
        """Get recommendation based on conclusion."""
        
        recommendations = {
            "LIKELY_AI_GENERATED": "Consider this content as potentially AI-generated. Verify authenticity through other means if critical.",
            "POSSIBLY_AI_GENERATED": "Some AI indicators detected. Exercise caution and seek additional verification if needed.",
            "LIKELY_NATURAL": "Content appears to be naturally captured. Standard verification procedures apply.",
            "UNCERTAIN": "Cannot reliably determine origin. Manual inspection and additional context recommended."
        }
        
        return recommendations.get(conclusion, "Manual inspection recommended.")
