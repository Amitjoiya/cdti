"""
Simple Reliable Forensics Engine v6.0
=====================================
Focus: Accurate Fake/Real Detection (No Heatmaps)

Problem with v5:
- Pretrained model gives random results
- No proper deepfake training data

Solution v6:
- Use MULTIPLE classical forensic signals together
- Weighted voting system
- Conservative thresholds (prefer UNCERTAIN over wrong)
- No heatmaps - just verdict and reasoning
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SimpleForensicsEngine:
    """
    Simple and Reliable Forensics Engine
    
    NO deep learning (unreliable without training)
    NO heatmaps (distracting)
    
    FOCUS on:
    1. Noise consistency analysis
    2. Compression artifact detection
    3. Color histogram analysis
    4. Frequency domain analysis
    5. Texture smoothness detection
    6. Edge coherence analysis
    
    Each gives a score, then weighted voting for final verdict.
    """
    
    def __init__(self):
        self.version = "6.0"
        print(f"  ✅ Simple Forensics v6.0 initialized")
        print(f"     Focus: Reliable Fake/Real detection")
    
    def analyze(self, file_path: str) -> Dict:
        """
        Analyze image/video for authenticity.
        
        Returns simple, reliable results without heatmaps.
        """
        result = {
            "status": "processing",
            "version": self.version,
            "analyzed_at": datetime.now().isoformat(),
            "file_path": file_path,
            
            # Main verdict
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "is_fake": None,  # True/False/None
            
            # Evidence breakdown
            "evidence": {
                "noise": {},
                "compression": {},
                "color": {},
                "frequency": {},
                "texture": {},
                "edge": {}
            },
            
            # Simple explanation
            "explanation": {
                "summary": "",
                "reasons": [],
                "limitations": []
            },
            
            # Scores for each method
            "scores": {},
            
            # For API compatibility
            "prediction": {
                "class": 0,
                "class_name": "UNKNOWN",
                "confidence": 0.0,
                "distribution": {
                    "real": 0.0,
                    "ai_generated": 0.0,
                    "ai_enhanced": 0.0
                },
                "is_uncertain": True
            },
            
            "errors": []
        }
        
        try:
            path = Path(file_path)
            is_video = path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
            # Get image (or frame from video)
            if is_video:
                img = self._extract_video_frame(file_path)
                result["media_type"] = "video"
            else:
                img = cv2.imread(file_path)
                result["media_type"] = "image"
            
            if img is None:
                result["errors"].append("Could not load file")
                result["status"] = "error"
                return result
            
            # Get dimensions
            h, w = img.shape[:2]
            result["dimensions"] = {"width": w, "height": h}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Run all forensic analyses
            noise_result = self._analyze_noise(img, gray)
            compression_result = self._analyze_compression(img, file_path)
            color_result = self._analyze_color(img)
            frequency_result = self._analyze_frequency(gray)
            texture_result = self._analyze_texture(gray)
            edge_result = self._analyze_edges(gray)
            
            # Store evidence
            result["evidence"]["noise"] = noise_result
            result["evidence"]["compression"] = compression_result
            result["evidence"]["color"] = color_result
            result["evidence"]["frequency"] = frequency_result
            result["evidence"]["texture"] = texture_result
            result["evidence"]["edge"] = edge_result
            
            # Store scores
            result["scores"] = {
                "noise_score": noise_result.get("fake_score", 50),
                "compression_score": compression_result.get("fake_score", 50),
                "color_score": color_result.get("fake_score", 50),
                "frequency_score": frequency_result.get("fake_score", 50),
                "texture_score": texture_result.get("fake_score", 50),
                "edge_score": edge_result.get("fake_score", 50)
            }
            
            # Calculate final verdict using weighted voting
            verdict_result = self._calculate_verdict(result["scores"])
            result["verdict"] = verdict_result["verdict"]
            result["confidence"] = verdict_result["confidence"]
            result["is_fake"] = verdict_result["is_fake"]
            
            # Update prediction for API compatibility
            if verdict_result["is_fake"] == True:
                result["prediction"]["class"] = 1
                result["prediction"]["class_name"] = "LIKELY_FAKE"
                result["prediction"]["distribution"]["ai_generated"] = result["confidence"]
                result["prediction"]["distribution"]["real"] = 100 - result["confidence"]
            elif verdict_result["is_fake"] == False:
                result["prediction"]["class"] = 0
                result["prediction"]["class_name"] = "LIKELY_REAL"
                result["prediction"]["distribution"]["real"] = result["confidence"]
                result["prediction"]["distribution"]["ai_generated"] = 100 - result["confidence"]
            else:
                result["prediction"]["class"] = -1
                result["prediction"]["class_name"] = "UNCERTAIN"
                result["prediction"]["distribution"]["real"] = 50
                result["prediction"]["distribution"]["ai_generated"] = 50
            
            result["prediction"]["confidence"] = result["confidence"]
            result["prediction"]["is_uncertain"] = verdict_result["is_fake"] is None
            
            # Generate explanation
            result["explanation"] = self._generate_explanation(result)
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _extract_video_frame(self, file_path: str) -> Optional[np.ndarray]:
        """Extract middle frame from video"""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    # ==========================================
    # FORENSIC ANALYSIS METHODS
    # ==========================================
    
    def _analyze_noise(self, img: np.ndarray, gray: np.ndarray) -> Dict:
        """
        Noise Consistency Analysis
        
        Real images: Natural noise patterns, consistent across image
        Fake images: AI images often have unnaturally consistent noise
                     or inconsistent noise in different regions
        """
        result = {"fake_score": 50, "details": {}}
        
        try:
            # Estimate noise using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_std = np.std(laplacian)
            
            # Check noise consistency across blocks
            h, w = gray.shape
            block_size = 64
            block_stds = []
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    block_lap = cv2.Laplacian(block, cv2.CV_64F)
                    block_stds.append(np.std(block_lap))
            
            if block_stds:
                noise_variance = np.var(block_stds)
                noise_mean = np.mean(block_stds)
                consistency = noise_variance / (noise_mean + 0.001)
            else:
                consistency = 0
            
            result["details"] = {
                "overall_noise": round(noise_std, 2),
                "noise_consistency": round(consistency, 4),
                "block_count": len(block_stds)
            }
            
            # Score: Very low noise OR very inconsistent = suspicious
            if noise_std < 5:  # Unnaturally smooth
                result["fake_score"] = 70
                result["flag"] = "unnaturally_smooth"
            elif consistency > 0.5:  # Inconsistent noise
                result["fake_score"] = 65
                result["flag"] = "inconsistent_noise"
            elif noise_std > 30:  # Natural high noise
                result["fake_score"] = 30
                result["flag"] = "natural_noise"
            else:
                result["fake_score"] = 45
                result["flag"] = "normal"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_compression(self, img: np.ndarray, file_path: str) -> Dict:
        """
        Compression Artifact Analysis
        
        Real images: Usually have consistent compression
        Fake images: May show double compression or unusual artifacts
        """
        result = {"fake_score": 50, "details": {}}
        
        try:
            ext = Path(file_path).suffix.lower()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check for blocking artifacts (8x8 JPEG blocks)
            h, w = gray.shape
            block_diffs = []
            
            for y in range(8, h - 8, 8):
                for x in range(8, w - 8, 8):
                    # Difference at block boundaries
                    diff = abs(int(gray[y-1, x]) - int(gray[y, x]))
                    block_diffs.append(diff)
            
            avg_block_diff = np.mean(block_diffs) if block_diffs else 0
            
            # Check file size vs resolution ratio
            file_size = os.path.getsize(file_path)
            pixels = h * w
            bytes_per_pixel = file_size / (pixels + 1)
            
            result["details"] = {
                "format": ext,
                "block_artifact_score": round(avg_block_diff, 2),
                "bytes_per_pixel": round(bytes_per_pixel, 4),
                "file_size_kb": round(file_size / 1024, 1)
            }
            
            # Score: Strong block artifacts might indicate re-compression (editing)
            if avg_block_diff > 15:
                result["fake_score"] = 60
                result["flag"] = "strong_compression"
            elif avg_block_diff < 2:
                result["fake_score"] = 55  # Too clean for JPEG
                result["flag"] = "minimal_artifacts"
            else:
                result["fake_score"] = 45
                result["flag"] = "normal"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_color(self, img: np.ndarray) -> Dict:
        """
        Color Distribution Analysis
        
        Real images: Natural color distributions
        AI images: Sometimes have unusual color patterns
        """
        result = {"fake_score": 50, "details": {}}
        
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Analyze saturation
            saturation = hsv[:, :, 1]
            sat_mean = np.mean(saturation)
            sat_std = np.std(saturation)
            
            # Analyze value (brightness)
            value = hsv[:, :, 2]
            val_mean = np.mean(value)
            val_std = np.std(value)
            
            # Check for unnatural color uniformity
            color_uniformity = sat_std / (sat_mean + 1)
            
            result["details"] = {
                "saturation_mean": round(sat_mean, 2),
                "saturation_std": round(sat_std, 2),
                "brightness_mean": round(val_mean, 2),
                "brightness_std": round(val_std, 2),
                "color_uniformity": round(color_uniformity, 4)
            }
            
            # Score
            if color_uniformity < 0.1:  # Very uniform (unnatural)
                result["fake_score"] = 60
                result["flag"] = "uniform_colors"
            elif sat_mean > 200:  # Oversaturated
                result["fake_score"] = 55
                result["flag"] = "oversaturated"
            else:
                result["fake_score"] = 45
                result["flag"] = "normal"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_frequency(self, gray: np.ndarray) -> Dict:
        """
        Frequency Domain Analysis
        
        Real images: Natural frequency distribution
        AI images: May have unusual high/low frequency ratios
        """
        result = {"fake_score": 50, "details": {}}
        
        try:
            # FFT
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            
            # Low frequency (center) vs high frequency (edges)
            radius = min(h, w) // 4
            
            # Create masks
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            low_mask = dist <= radius
            high_mask = dist > radius * 2
            
            low_energy = np.mean(magnitude[low_mask])
            high_energy = np.mean(magnitude[high_mask])
            
            freq_ratio = high_energy / (low_energy + 0.001)
            
            result["details"] = {
                "low_freq_energy": round(low_energy, 2),
                "high_freq_energy": round(high_energy, 2),
                "freq_ratio": round(freq_ratio, 4)
            }
            
            # AI images often have lower high-frequency content
            if freq_ratio < 0.3:
                result["fake_score"] = 65
                result["flag"] = "low_high_freq"
            elif freq_ratio > 0.7:
                result["fake_score"] = 35
                result["flag"] = "natural_freq"
            else:
                result["fake_score"] = 50
                result["flag"] = "normal"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_texture(self, gray: np.ndarray) -> Dict:
        """
        Texture Smoothness Analysis
        
        Real images: Natural texture variations
        AI images: Often have unnaturally smooth or repetitive textures
        """
        result = {"fake_score": 50, "details": {}}
        
        try:
            # Local Binary Pattern-like texture analysis
            # Calculate gradient magnitude
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            
            texture_strength = np.mean(gradient_mag)
            texture_std = np.std(gradient_mag)
            
            # Check for texture consistency
            texture_ratio = texture_std / (texture_strength + 1)
            
            result["details"] = {
                "texture_strength": round(texture_strength, 2),
                "texture_variation": round(texture_std, 2),
                "texture_ratio": round(texture_ratio, 4)
            }
            
            # Very smooth textures are suspicious
            if texture_strength < 10:
                result["fake_score"] = 65
                result["flag"] = "very_smooth"
            elif texture_strength > 50:
                result["fake_score"] = 35
                result["flag"] = "strong_texture"
            else:
                result["fake_score"] = 45
                result["flag"] = "normal"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_edges(self, gray: np.ndarray) -> Dict:
        """
        Edge Coherence Analysis
        
        Real images: Natural edge patterns
        AI images: May have unusual edge smoothness or artifacts
        """
        result = {"fake_score": 50, "details": {}}
        
        try:
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            edge_density = np.mean(edges) / 255.0
            
            # Laplacian variance (focus measure)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_score = np.var(laplacian)
            
            result["details"] = {
                "edge_density": round(edge_density, 4),
                "focus_score": round(focus_score, 2)
            }
            
            # Very low edge density might indicate AI smoothing
            if edge_density < 0.02:
                result["fake_score"] = 60
                result["flag"] = "low_edges"
            elif focus_score < 100:
                result["fake_score"] = 55
                result["flag"] = "low_focus"
            elif edge_density > 0.15:
                result["fake_score"] = 35
                result["flag"] = "detailed"
            else:
                result["fake_score"] = 45
                result["flag"] = "normal"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    # ==========================================
    # VERDICT CALCULATION
    # ==========================================
    
    def _calculate_verdict(self, scores: Dict) -> Dict:
        """
        Calculate final verdict using weighted voting.
        
        Conservative approach: Only say FAKE or REAL if confident.
        Otherwise say UNCERTAIN.
        """
        # Weights for each method
        weights = {
            "noise_score": 0.20,
            "compression_score": 0.15,
            "color_score": 0.15,
            "frequency_score": 0.20,
            "texture_score": 0.15,
            "edge_score": 0.15
        }
        
        # Calculate weighted average
        weighted_sum = 0
        total_weight = 0
        
        for name, weight in weights.items():
            score = scores.get(name, 50)
            weighted_sum += score * weight
            total_weight += weight
        
        avg_score = weighted_sum / total_weight if total_weight > 0 else 50
        
        # Count how many methods agree
        fake_votes = sum(1 for s in scores.values() if s >= 60)
        real_votes = sum(1 for s in scores.values() if s <= 40)
        
        # Determine verdict
        # Be conservative - need strong agreement
        if avg_score >= 62 and fake_votes >= 3:
            verdict = "LIKELY FAKE"
            confidence = min(85, avg_score + 10)
            is_fake = True
        elif avg_score <= 40 and real_votes >= 3:
            verdict = "LIKELY REAL"
            confidence = min(85, 100 - avg_score + 10)
            is_fake = False
        elif avg_score >= 55:
            verdict = "POSSIBLY FAKE"
            confidence = avg_score
            is_fake = None  # Uncertain
        elif avg_score <= 45:
            verdict = "PROBABLY REAL"
            confidence = 100 - avg_score
            is_fake = None  # Uncertain
        else:
            verdict = "UNCERTAIN"
            confidence = 50
            is_fake = None
        
        return {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "is_fake": is_fake,
            "avg_score": round(avg_score, 1),
            "fake_votes": fake_votes,
            "real_votes": real_votes
        }
    
    def _generate_explanation(self, result: Dict) -> Dict:
        """Generate simple explanation"""
        explanation = {
            "summary": "",
            "reasons": [],
            "limitations": [
                "Analysis based on statistical patterns only",
                "Modern AI can produce very realistic content",
                "Results should be verified by experts for legal use"
            ]
        }
        
        verdict = result["verdict"]
        confidence = result["confidence"]
        evidence = result["evidence"]
        
        # Summary
        if verdict == "LIKELY FAKE":
            explanation["summary"] = f"This media shows signs of AI generation or manipulation ({confidence:.0f}% confidence)."
        elif verdict == "LIKELY REAL":
            explanation["summary"] = f"This media appears to be authentic ({confidence:.0f}% confidence)."
        elif verdict == "POSSIBLY FAKE":
            explanation["summary"] = f"This media has some suspicious characteristics but is not conclusive ({confidence:.0f}% confidence)."
        elif verdict == "PROBABLY REAL":
            explanation["summary"] = f"This media appears mostly natural ({confidence:.0f}% confidence)."
        else:
            explanation["summary"] = "Analysis is inconclusive. Manual review recommended."
        
        # Collect reasons
        for name, data in evidence.items():
            flag = data.get("flag", "")
            score = data.get("fake_score", 50)
            
            if flag == "unnaturally_smooth":
                explanation["reasons"].append("Unnaturally smooth texture detected (common in AI images)")
            elif flag == "inconsistent_noise":
                explanation["reasons"].append("Noise patterns are inconsistent across the image")
            elif flag == "low_high_freq":
                explanation["reasons"].append("Low high-frequency content (AI images often lack fine details)")
            elif flag == "uniform_colors":
                explanation["reasons"].append("Color distribution is unusually uniform")
            elif flag == "low_edges":
                explanation["reasons"].append("Low edge detail detected")
            elif flag == "natural_noise" and score < 40:
                explanation["reasons"].append("Natural noise patterns detected")
            elif flag == "strong_texture" and score < 40:
                explanation["reasons"].append("Natural texture variations present")
        
        if not explanation["reasons"]:
            explanation["reasons"].append("No strong indicators in either direction")
        
        return explanation


# ==========================================
# WRAPPER CLASS FOR COMPATIBILITY
# ==========================================

class AdvancedVisualForensics:
    """Wrapper for backward compatibility with main.py"""
    
    def __init__(self, model_path: str = None):
        self.engine = SimpleForensicsEngine()
        self.version = "6.0"
    
    def analyze(self, file_path: str) -> Dict:
        return self.engine.analyze(file_path)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Simple Forensics Engine v6.0")
    print("=" * 60)
    print("Focus: Reliable Fake/Real Detection")
    print("NO heatmaps, NO unreliable DL")
    print("=" * 60)
    
    engine = SimpleForensicsEngine()
