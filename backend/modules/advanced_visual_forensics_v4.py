"""
Advanced Visual Forensics Engine v4.0
=====================================
PRIMARY: Deep Learning + Grad-CAM (Decision-based)
SECONDARY: Classical Methods (Supporting evidence)

Architecture Redesign:
- DL model is the PRIMARY decision maker
- Classical methods (ELA, Noise, Edge) are SUPPORTING evidence
- Heatmaps are generated from Grad-CAM (not edge detection)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import cv2
    import numpy as np
    from scipy import ndimage, fftpack
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Deep Learning Engine (PRIMARY)
try:
    from dl_forensics_engine import DLForensicsEngine
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("⚠️ DL Engine not available")


class AdvancedVisualForensics:
    """
    Visual Forensics Engine v4.0
    
    IMPORTANT ARCHITECTURE CHANGE:
    ==============================
    
    PRIMARY EVIDENCE (Final Decision):
    - Deep Learning CNN prediction
    - Grad-CAM heatmaps (decision-based)
    - Regional analysis
    
    SECONDARY EVIDENCE (Supporting only):
    - Error Level Analysis (ELA)
    - Noise Analysis
    - Edge Detection
    - Frequency Analysis
    
    Why Classical Methods are Secondary:
    1. ELA: Designed for JPEG re-saves, not AI generation
    2. Noise: GANs produce consistent noise patterns
    3. Edge: Deepfakes have natural-looking edges
    4. Frequency: Modern GANs match real distributions
    
    Why DL is Primary:
    1. Trained on actual deepfake/real pairs
    2. Learns manipulation-specific patterns
    3. Grad-CAM shows WHY it flagged (not just edges)
    4. Different heatmaps for different images
    """
    
    def __init__(self):
        self.version = "4.0"
        
        # Initialize DL Engine (PRIMARY)
        self.dl_engine = None
        if DL_AVAILABLE:
            try:
                self.dl_engine = DLForensicsEngine()
                print("  ✅ PRIMARY: Deep Learning Engine ready")
            except Exception as e:
                print(f"  ⚠️ DL Engine failed: {e}")
        
        print("  ✅ SECONDARY: Classical methods available")
    
    def analyze(self, file_path: str) -> Dict:
        """
        Run complete forensic analysis.
        
        Returns structured result with:
        - primary_evidence: DL prediction + Grad-CAM (MAIN DECISION)
        - secondary_evidence: Classical methods (SUPPORTING)
        """
        result = {
            "status": "processing",
            "version": self.version,
            "analyzed_at": datetime.now().isoformat(),
            "file_path": file_path,
            
            # ==========================================
            # PRIMARY EVIDENCE (Deep Learning)
            # This is what makes the final decision
            # ==========================================
            "primary_evidence": {
                "available": False,
                "prediction": 0,
                "confidence": 0.0,
                "label": "UNKNOWN",
                "method": "Deep Learning CNN + Grad-CAM",
                "heatmaps": {},
                "regions": {},
                "interpretation": {}
            },
            
            # ==========================================
            # SECONDARY EVIDENCE (Classical Methods)
            # Supporting indicators only
            # ==========================================
            "secondary_evidence": {
                "ela": {},
                "noise": {},
                "edge": {},
                "frequency": {},
                "compression": {},
                "statistical": {}
            },
            
            # Legacy format (for backward compatibility)
            "scores": {},
            "analysis": {},
            "heatmaps": {},
            "detections": {},
            
            # Final verdict (from PRIMARY)
            "verdict": {
                "conclusion": "UNKNOWN",
                "confidence": 0.0,
                "based_on": "primary_evidence"
            },
            
            "errors": []
        }
        
        if not CV2_AVAILABLE:
            result["errors"].append("OpenCV not available")
            result["status"] = "error"
            return result
        
        try:
            # Determine if video or image
            path = Path(file_path)
            is_video = path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
            # ==========================================
            # STEP 1: PRIMARY ANALYSIS (Deep Learning)
            # ==========================================
            if self.dl_engine and self.dl_engine.initialized:
                if is_video:
                    dl_result = self.dl_engine.analyze_video(file_path)
                else:
                    dl_result = self.dl_engine.analyze_image(file_path)
                
                if dl_result.get("status") == "completed":
                    primary = result["primary_evidence"]
                    primary["available"] = True
                    primary["prediction"] = dl_result["primary"]["prediction"]
                    primary["confidence"] = dl_result["primary"]["confidence"]
                    primary["label"] = dl_result["primary"]["label"]
                    primary["heatmaps"] = dl_result.get("heatmaps", {})
                    primary["regions"] = dl_result.get("regions", {})
                    primary["interpretation"] = dl_result.get("interpretation", {})
                    
                    # Video-specific data
                    if is_video:
                        primary["temporal"] = dl_result.get("temporal", {})
                        primary["keyframes"] = dl_result.get("keyframes", {})
                    
                    # Set main heatmaps from DL
                    for name, hm in primary["heatmaps"].items():
                        result["heatmaps"][f"dl_{name}"] = hm
                    
                    # Set verdict from PRIMARY evidence
                    result["verdict"]["conclusion"] = primary["label"]
                    result["verdict"]["confidence"] = primary["confidence"]
                    result["verdict"]["based_on"] = "Deep Learning CNN + Grad-CAM"
            
            # ==========================================
            # STEP 2: SECONDARY ANALYSIS (Classical)
            # ==========================================
            if is_video:
                img = self._extract_video_frame(file_path)
            else:
                img = cv2.imread(file_path)
            
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # ELA (Supporting)
                ela_result = self._analyze_ela(img, file_path)
                result["secondary_evidence"]["ela"] = ela_result
                result["analysis"]["ela"] = ela_result.get("details", {})
                result["scores"]["ela_score"] = ela_result.get("score", 0)
                if ela_result.get("heatmap") is not None:
                    result["heatmaps"]["ela_heatmap"] = ela_result["heatmap"]
                
                # Noise (Supporting)
                noise_result = self._analyze_noise(img, gray)
                result["secondary_evidence"]["noise"] = noise_result
                result["analysis"]["noise"] = noise_result.get("details", {})
                result["scores"]["noise_score"] = noise_result.get("score", 0)
                if noise_result.get("heatmap") is not None:
                    result["heatmaps"]["noise_heatmap"] = noise_result["heatmap"]
                
                # Edge (Supporting)
                edge_result = self._analyze_edges(gray)
                result["secondary_evidence"]["edge"] = edge_result
                result["analysis"]["edge"] = edge_result.get("details", {})
                result["scores"]["edge_score"] = edge_result.get("score", 0)
                if edge_result.get("heatmap") is not None:
                    result["heatmaps"]["edge_heatmap"] = edge_result["heatmap"]
                
                # Frequency (Supporting)
                freq_result = self._analyze_frequency(gray)
                result["secondary_evidence"]["frequency"] = freq_result
                result["analysis"]["frequency"] = freq_result.get("details", {})
                result["scores"]["frequency_score"] = freq_result.get("score", 0)
                if freq_result.get("heatmap") is not None:
                    result["heatmaps"]["frequency_heatmap"] = freq_result["heatmap"]
                
                # Compression (Supporting)
                comp_result = self._analyze_compression(img, file_path)
                result["secondary_evidence"]["compression"] = comp_result
                result["analysis"]["compression"] = comp_result.get("details", {})
                result["scores"]["compression_score"] = comp_result.get("score", 0)
                
                # Statistical (Supporting)
                stat_result = self._analyze_statistical(gray)
                result["secondary_evidence"]["statistical"] = stat_result
                result["analysis"]["statistical"] = stat_result.get("details", {})
                result["scores"]["statistical_score"] = stat_result.get("score", 0)
            
            # ==========================================
            # STEP 3: CALCULATE FINAL SCORES
            # ==========================================
            
            # If DL is available, it dominates the score
            if result["primary_evidence"]["available"]:
                dl_conf = result["primary_evidence"]["confidence"]
                
                # Classical average (supporting)
                classical_scores = [
                    result["scores"].get("ela_score", 0),
                    result["scores"].get("noise_score", 0),
                    result["scores"].get("edge_score", 0),
                    result["scores"].get("frequency_score", 0),
                    result["scores"].get("compression_score", 0),
                ]
                classical_avg = np.mean(classical_scores)
                
                # Overall: 70% DL + 30% Classical
                overall = 0.7 * dl_conf + 0.3 * classical_avg
                
                result["scores"]["deeplearning_score"] = dl_conf
                result["scores"]["classical_avg"] = round(classical_avg, 2)
                result["scores"]["overall_score"] = round(overall, 2)
            else:
                # Fallback to classical only
                classical_scores = [
                    result["scores"].get("ela_score", 0),
                    result["scores"].get("noise_score", 0),
                    result["scores"].get("edge_score", 0),
                    result["scores"].get("frequency_score", 0),
                    result["scores"].get("compression_score", 0),
                ]
                overall = np.mean(classical_scores)
                result["scores"]["overall_score"] = round(overall, 2)
                
                # Set verdict from classical (less reliable)
                if overall > 60:
                    result["verdict"]["conclusion"] = "POSSIBLY MANIPULATED"
                elif overall > 40:
                    result["verdict"]["conclusion"] = "UNCERTAIN"
                else:
                    result["verdict"]["conclusion"] = "LIKELY AUTHENTIC"
                result["verdict"]["confidence"] = round(overall, 2)
                result["verdict"]["based_on"] = "Classical methods only (less reliable)"
            
            # Detections summary
            result["detections"] = {
                "primary_method": "Deep Learning" if result["primary_evidence"]["available"] else "Classical",
                "overall_score": result["scores"].get("overall_score", 0),
                "severity": self._get_severity(result["scores"].get("overall_score", 0)),
                "algorithms_used": {
                    "primary": ["ResNet-18 CNN", "Grad-CAM"] if result["primary_evidence"]["available"] else [],
                    "secondary": ["ELA", "Noise Analysis", "Edge Detection", "Frequency Analysis"]
                }
            }
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _get_severity(self, score: float) -> str:
        """Get severity level from score"""
        if score >= 70:
            return "critical"
        elif score >= 50:
            return "high"
        elif score >= 30:
            return "medium"
        elif score >= 15:
            return "low"
        return "none"
    
    def _extract_video_frame(self, video_path: str, position: float = 0.25) -> Optional[np.ndarray]:
        """Extract a frame from video"""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * position))
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    
    # ==========================================
    # SECONDARY METHODS (Classical - Supporting)
    # ==========================================
    
    def _analyze_ela(self, img: np.ndarray, file_path: str) -> Dict:
        """
        Error Level Analysis (SECONDARY/SUPPORTING)
        
        Limitation: Designed for JPEG re-save detection, not AI generation
        Modern deepfakes often don't trigger ELA
        """
        result = {
            "score": 0,
            "details": {
                "method": "Error Level Analysis",
                "role": "SECONDARY (Supporting evidence only)",
                "limitation": "May not detect AI-generated content"
            },
            "heatmap": None
        }
        
        try:
            import tempfile
            
            # Re-compress and compare
            temp_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            recompressed = cv2.imread(temp_path)
            os.unlink(temp_path)
            
            if recompressed is None:
                return result
            
            # Calculate ELA
            ela = cv2.absdiff(img, recompressed)
            ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
            
            # Enhance
            ela_enhanced = cv2.normalize(ela_gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # Score
            mean_ela = np.mean(ela_enhanced)
            result["score"] = min(100, mean_ela * 2)
            result["details"]["mean_ela"] = round(float(mean_ela), 2)
            
            # Heatmap
            heatmap = cv2.applyColorMap(ela_enhanced.astype(np.uint8), cv2.COLORMAP_JET)
            result["heatmap"] = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    def _analyze_noise(self, img: np.ndarray, gray: np.ndarray) -> Dict:
        """
        Noise Analysis (SECONDARY/SUPPORTING)
        
        Limitation: AI models can produce consistent noise patterns
        """
        result = {
            "score": 0,
            "details": {
                "method": "Noise Pattern Analysis",
                "role": "SECONDARY (Supporting evidence only)",
                "limitation": "GANs can produce consistent noise"
            },
            "heatmap": None
        }
        
        try:
            # Extract noise using median filter
            denoised = cv2.medianBlur(gray, 3)
            noise = cv2.absdiff(gray, denoised)
            
            # Block-wise noise variance
            h, w = gray.shape
            block_size = 32
            noise_map = np.zeros_like(gray, dtype=np.float32)
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = noise[y:y+block_size, x:x+block_size]
                    variance = np.var(block)
                    noise_map[y:y+block_size, x:x+block_size] = variance
            
            # Normalize
            if noise_map.max() > 0:
                noise_map = (noise_map / noise_map.max() * 255).astype(np.uint8)
            
            # Score based on variance inconsistency
            overall_var = np.var(noise_map)
            result["score"] = min(100, overall_var / 10)
            result["details"]["noise_variance"] = round(float(overall_var), 4)
            
            # Heatmap
            heatmap = cv2.applyColorMap(noise_map, cv2.COLORMAP_JET)
            result["heatmap"] = cv2.addWeighted(
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.5, heatmap, 0.5, 0
            )
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    def _analyze_edges(self, gray: np.ndarray) -> Dict:
        """
        Edge Analysis (SECONDARY/SUPPORTING)
        
        Limitation: Deepfakes often have natural-looking edges
        """
        result = {
            "score": 0,
            "details": {
                "method": "Edge Consistency Analysis",
                "role": "SECONDARY (Supporting evidence only)",
                "limitation": "Deepfakes can have natural edges"
            },
            "heatmap": None
        }
        
        try:
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Edge density variance across blocks
            h, w = gray.shape
            block_size = 32
            density_map = np.zeros_like(gray, dtype=np.float32)
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = edges[y:y+block_size, x:x+block_size]
                    density = np.sum(block > 0) / (block_size * block_size)
                    density_map[y:y+block_size, x:x+block_size] = density
            
            # Normalize
            if density_map.max() > 0:
                density_map = (density_map / density_map.max() * 255).astype(np.uint8)
            
            # Score
            variance = np.var(density_map)
            result["score"] = min(100, variance / 5)
            result["details"]["edge_variance"] = round(float(variance), 4)
            
            # Heatmap
            heatmap = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
            result["heatmap"] = cv2.addWeighted(
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.5, heatmap, 0.5, 0
            )
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    def _analyze_frequency(self, gray: np.ndarray) -> Dict:
        """
        Frequency Analysis (SECONDARY/SUPPORTING)
        
        Limitation: Modern GANs learn to match frequency distributions
        """
        result = {
            "score": 0,
            "details": {
                "method": "Frequency Domain Analysis",
                "role": "SECONDARY (Supporting evidence only)",
                "limitation": "Modern GANs match real frequency patterns"
            },
            "heatmap": None
        }
        
        try:
            # FFT
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            
            # Normalize
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
            
            # High frequency ratio
            h, w = gray.shape
            center_h, center_w = h // 2, w // 2
            radius = min(center_h, center_w) // 3
            
            y, x = np.ogrid[:h, :w]
            mask = (x - center_w) ** 2 + (y - center_h) ** 2 > radius ** 2
            
            high_freq_energy = np.sum(np.abs(fshift)[mask])
            total_energy = np.sum(np.abs(fshift))
            
            ratio = high_freq_energy / (total_energy + 1e-8)
            result["score"] = min(100, ratio * 100)
            result["details"]["high_freq_ratio"] = round(float(ratio), 4)
            
            # Heatmap
            heatmap = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
            result["heatmap"] = heatmap
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    def _analyze_compression(self, img: np.ndarray, file_path: str) -> Dict:
        """
        Compression Analysis (SECONDARY/SUPPORTING)
        """
        result = {
            "score": 0,
            "details": {
                "method": "Compression Artifact Analysis",
                "role": "SECONDARY (Supporting evidence only)"
            }
        }
        
        try:
            # Check for 8x8 block artifacts (JPEG)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            h, w = gray.shape
            
            block_diffs = []
            for y in range(0, h - 8, 8):
                for x in range(0, w - 8, 8):
                    block = gray[y:y+8, x:x+8]
                    if x + 8 < w:
                        next_block = gray[y:y+8, x+8:x+16]
                        if next_block.shape == block.shape:
                            diff = np.abs(np.mean(block) - np.mean(next_block))
                            block_diffs.append(diff)
            
            if block_diffs:
                avg_diff = np.mean(block_diffs)
                result["score"] = min(100, avg_diff * 2)
                result["details"]["block_artifact_score"] = round(float(avg_diff), 4)
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    def _analyze_statistical(self, gray: np.ndarray) -> Dict:
        """
        Statistical Analysis (SECONDARY/SUPPORTING)
        """
        result = {
            "score": 0,
            "details": {
                "method": "Statistical Distribution Analysis",
                "role": "SECONDARY (Supporting evidence only)"
            }
        }
        
        try:
            # Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Entropy
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            
            # Deviation from expected
            expected_entropy = 7.0  # Typical for natural images
            deviation = abs(entropy - expected_entropy)
            
            result["score"] = min(100, deviation * 20)
            result["details"]["entropy"] = round(float(entropy), 4)
            result["details"]["deviation"] = round(float(deviation), 4)
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Visual Forensics Engine v4.0")
    print("=" * 60)
    print("PRIMARY: Deep Learning + Grad-CAM")
    print("SECONDARY: Classical Methods (Supporting)")
    print("=" * 60)
    
    engine = AdvancedVisualForensics()
    print("\n✅ Engine ready")
