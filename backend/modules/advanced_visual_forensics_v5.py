"""
Advanced Visual Forensics Engine v5.0
=====================================
Multi-Class AI Detection: Real vs AI Generated vs AI Enhanced

PRIMARY: Deep Learning + Frequency Fusion + Multi-layer Grad-CAM
SECONDARY: Classical Methods (Supporting evidence only)

Upgrades from v4:
- 3-Class classification
- Frequency domain fusion
- Patch + Full image analysis
- Heatmap entropy for confidence
- Hard AI sample detection
"""

import os
import base64
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

# New Advanced AI Detector (PRIMARY)
try:
    from advanced_ai_detector import (
        AdvancedAIDetector, 
        FrequencyExtractor,
        CLASS_NAMES, 
        CLASS_DESCRIPTIONS
    )
    ADVANCED_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        from modules.advanced_ai_detector import (
            AdvancedAIDetector, 
            FrequencyExtractor,
            CLASS_NAMES, 
            CLASS_DESCRIPTIONS
        )
        ADVANCED_DETECTOR_AVAILABLE = True
    except ImportError:
        ADVANCED_DETECTOR_AVAILABLE = False
        CLASS_NAMES = {0: "REAL", 1: "AI_GENERATED", 2: "AI_ENHANCED"}
        CLASS_DESCRIPTIONS = {
            0: "Natural/Camera Image",
            1: "AI Generated (Midjourney, SD, DALL·E)",
            2: "AI Enhanced/Refined"
        }

# Legacy DL Engine (fallback)
try:
    from dl_forensics_engine import DLForensicsEngine
    LEGACY_DL_AVAILABLE = True
except ImportError:
    try:
        from modules.dl_forensics_engine import DLForensicsEngine
        LEGACY_DL_AVAILABLE = True
    except ImportError:
        LEGACY_DL_AVAILABLE = False


class AdvancedVisualForensics:
    """
    Visual Forensics Engine v5.0
    
    MULTI-CLASS DETECTION:
    ======================
    Class 0: REAL (Camera/Natural)
    Class 1: AI GENERATED (Midjourney, SD, DALL·E, etc.)
    Class 2: AI ENHANCED (Gemini enhance, upscale, Photoshop)
    
    ARCHITECTURE:
    =============
    PRIMARY EVIDENCE (60-70% weight):
    - Deep Learning + Frequency Fusion
    - Multi-layer Grad-CAM
    - Patch + Full image analysis
    - Heatmap entropy confidence
    
    SECONDARY EVIDENCE (30-40% weight):
    - ELA (JPEG re-compression)
    - Noise Analysis
    - Frequency Domain
    - Edge Detection
    
    IMPORTANT LIMITATIONS:
    ======================
    - Not pixel-perfect localization
    - New AI models may evade detection
    - Compression affects accuracy
    - Enhanced AI images are harder to detect
    """
    
    def __init__(self, model_path: str = None):
        self.version = "5.0"
        self.model_path = model_path
        
        # Initialize Advanced Detector (PRIMARY)
        self.detector = None
        if ADVANCED_DETECTOR_AVAILABLE:
            try:
                self.detector = AdvancedAIDetector(model_path=model_path)
                if self.detector.initialized:
                    print("  ✅ PRIMARY: Advanced 3-Class Detector ready")
                else:
                    self.detector = None
            except Exception as e:
                print(f"  ⚠️ Advanced Detector failed: {e}")
        
        # Fallback to legacy
        self.legacy_engine = None
        if self.detector is None and LEGACY_DL_AVAILABLE:
            try:
                self.legacy_engine = DLForensicsEngine()
                print("  ⚠️ Using legacy 2-class detector (fallback)")
            except Exception as e:
                print(f"  ⚠️ Legacy engine failed: {e}")
        
        # Frequency extractor for classical analysis
        if ADVANCED_DETECTOR_AVAILABLE:
            self.freq_extractor = FrequencyExtractor()
        else:
            self.freq_extractor = None
        
        print("  ✅ SECONDARY: Classical methods available")
        print(f"  📊 Mode: {'3-Class (Real/Generated/Enhanced)' if self.detector else '2-Class (Real/Fake)'}")
    
    def analyze(self, file_path: str) -> Dict:
        """
        Run complete forensic analysis.
        
        Returns:
        - prediction: 3-class result with confidence
        - primary_evidence: DL + Frequency fusion
        - secondary_evidence: Classical methods
        - heatmaps: Multi-layer Grad-CAM
        - interpretation: Human-readable explanation
        """
        result = {
            "status": "processing",
            "version": self.version,
            "analyzed_at": datetime.now().isoformat(),
            "file_path": file_path,
            
            # ==========================================
            # PREDICTION (3-Class)
            # ==========================================
            "prediction": {
                "class": 0,
                "class_name": "UNKNOWN",
                "description": "",
                "confidence": 0.0,
                "distribution": {
                    "real": 0.0,
                    "ai_generated": 0.0,
                    "ai_enhanced": 0.0
                },
                "is_uncertain": True
            },
            
            # ==========================================
            # PRIMARY EVIDENCE (Deep Learning)
            # ==========================================
            "primary_evidence": {
                "available": False,
                "method": "DL + Frequency Fusion + Multi-layer Grad-CAM",
                "full_image": {},
                "patches": {},
                "frequency": {},
                "confidence_calibration": {}
            },
            
            # ==========================================
            # SECONDARY EVIDENCE (Classical)
            # ==========================================
            "secondary_evidence": {
                "ela": {},
                "noise": {},
                "edge": {},
                "frequency": {},
                "compression": {},
                "statistical": {}
            },
            
            # Heatmaps
            "heatmaps": {},
            "heatmaps_base64": {},  # For frontend
            
            # Interpretation
            "interpretation": {
                "summary": "",
                "evidence": [],
                "limitations": [],
                "recommendation": ""
            },
            
            # Legacy format
            "scores": {},
            "analysis": {},
            "detections": {},
            
            # Verdict
            "verdict": {
                "conclusion": "UNKNOWN",
                "confidence": 0.0,
                "based_on": ""
            },
            
            "errors": []
        }
        
        if not CV2_AVAILABLE:
            result["errors"].append("OpenCV not available")
            result["status"] = "error"
            return result
        
        try:
            path = Path(file_path)
            is_video = path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
            # ==========================================
            # STEP 1: PRIMARY ANALYSIS (Advanced Detector)
            # ==========================================
            if self.detector:
                if is_video:
                    dl_result = self._analyze_video_advanced(file_path)
                else:
                    dl_result = self.detector.analyze(file_path)
                
                if dl_result.get("status") == "completed":
                    self._populate_from_advanced(result, dl_result)
            
            elif self.legacy_engine:
                # Fallback to legacy 2-class
                if is_video:
                    dl_result = self.legacy_engine.analyze_video(file_path)
                else:
                    dl_result = self.legacy_engine.analyze_image(file_path)
                
                if dl_result.get("status") == "completed":
                    self._populate_from_legacy(result, dl_result)
            
            # ==========================================
            # STEP 2: SECONDARY ANALYSIS (Classical)
            # ==========================================
            if is_video:
                img = self._extract_video_frame(file_path)
            else:
                img = cv2.imread(file_path)
            
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # ELA
                ela_result = self._analyze_ela(img, file_path)
                result["secondary_evidence"]["ela"] = ela_result
                result["scores"]["ela_score"] = ela_result.get("score", 0)
                if ela_result.get("heatmap") is not None:
                    result["heatmaps"]["ela"] = ela_result["heatmap"]
                
                # Noise
                noise_result = self._analyze_noise(img, gray)
                result["secondary_evidence"]["noise"] = noise_result
                result["scores"]["noise_score"] = noise_result.get("score", 0)
                if noise_result.get("heatmap") is not None:
                    result["heatmaps"]["noise"] = noise_result["heatmap"]
                
                # Edge
                edge_result = self._analyze_edges(gray)
                result["secondary_evidence"]["edge"] = edge_result
                result["scores"]["edge_score"] = edge_result.get("score", 0)
                if edge_result.get("heatmap") is not None:
                    result["heatmaps"]["edge"] = edge_result["heatmap"]
                
                # Frequency
                freq_result = self._analyze_frequency(gray)
                result["secondary_evidence"]["frequency"] = freq_result
                result["scores"]["frequency_score"] = freq_result.get("score", 0)
                if freq_result.get("heatmap") is not None:
                    result["heatmaps"]["frequency"] = freq_result["heatmap"]
                
                # Compression
                comp_result = self._analyze_compression(img, file_path)
                result["secondary_evidence"]["compression"] = comp_result
                result["scores"]["compression_score"] = comp_result.get("score", 0)
                
                # Statistical
                stat_result = self._analyze_statistical(gray)
                result["secondary_evidence"]["statistical"] = stat_result
                result["scores"]["statistical_score"] = stat_result.get("score", 0)
            
            # ==========================================
            # STEP 3: COMBINE & FINALIZE
            # ==========================================
            self._calculate_final_verdict(result)
            self._convert_heatmaps_to_base64(result)
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _populate_from_advanced(self, result: Dict, dl_result: Dict):
        """Populate result from advanced 3-class detector"""
        pred = dl_result.get("prediction", {})
        
        # Main prediction
        result["prediction"]["class"] = pred.get("class", 0)
        result["prediction"]["class_name"] = pred.get("class_name", "UNKNOWN")
        result["prediction"]["description"] = CLASS_DESCRIPTIONS.get(pred.get("class", 0), "")
        result["prediction"]["confidence"] = pred.get("confidence", 0)
        result["prediction"]["distribution"] = pred.get("distribution", {})
        
        calib = dl_result.get("confidence_calibration", {})
        result["prediction"]["is_uncertain"] = calib.get("is_uncertain", True)
        
        # Primary evidence
        primary = result["primary_evidence"]
        primary["available"] = True
        primary["full_image"] = dl_result.get("full_image_analysis", {})
        primary["patches"] = dl_result.get("patch_analysis", {})
        primary["frequency"] = dl_result.get("frequency_analysis", {})
        primary["confidence_calibration"] = calib
        
        # Heatmaps from DL
        for name, hm in dl_result.get("heatmaps", {}).items():
            if isinstance(hm, np.ndarray):
                result["heatmaps"][f"gradcam_{name}"] = hm
        
        # Interpretation
        interp = dl_result.get("interpretation", {})
        result["interpretation"] = {
            "summary": interp.get("summary", ""),
            "evidence": interp.get("evidence", []),
            "limitations": interp.get("limitations", []),
            "recommendation": interp.get("recommendation", "")
        }
        
        # Verdict from prediction
        result["verdict"]["conclusion"] = pred.get("class_name", "UNKNOWN")
        result["verdict"]["confidence"] = calib.get("calibrated_confidence", pred.get("confidence", 0))
        result["verdict"]["based_on"] = "3-Class DL + Frequency Fusion"
    
    def _populate_from_legacy(self, result: Dict, dl_result: Dict):
        """Populate result from legacy 2-class detector"""
        primary = dl_result.get("primary", {})
        
        # Map 2-class to 3-class
        prediction = primary.get("prediction", 0)
        confidence = primary.get("confidence", 0)
        
        if prediction == 0:
            result["prediction"]["class"] = 0
            result["prediction"]["class_name"] = "REAL"
            result["prediction"]["distribution"] = {"real": confidence, "ai_generated": 100-confidence, "ai_enhanced": 0}
        else:
            # Legacy doesn't distinguish generated vs enhanced
            result["prediction"]["class"] = 1
            result["prediction"]["class_name"] = "AI_GENERATED"
            result["prediction"]["distribution"] = {"real": 100-confidence, "ai_generated": confidence, "ai_enhanced": 0}
        
        result["prediction"]["confidence"] = confidence
        result["prediction"]["description"] = CLASS_DESCRIPTIONS.get(result["prediction"]["class"], "")
        result["prediction"]["is_uncertain"] = confidence < 60
        
        # Primary evidence
        result["primary_evidence"]["available"] = True
        result["primary_evidence"]["full_image"] = {
            "prediction": prediction,
            "confidence": confidence
        }
        
        # Heatmaps
        for name, hm in dl_result.get("heatmaps", {}).items():
            if isinstance(hm, np.ndarray):
                result["heatmaps"][f"legacy_{name}"] = hm
        
        result["verdict"]["conclusion"] = result["prediction"]["class_name"]
        result["verdict"]["confidence"] = confidence
        result["verdict"]["based_on"] = "Legacy 2-Class Detector"
    
    def _analyze_video_advanced(self, file_path: str) -> Dict:
        """Analyze video with advanced detector"""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"status": "error", "error": "Cannot open video"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample frames
        sample_interval = max(1, total_frames // 10)  # 10 samples
        frame_results = []
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % sample_interval == 0:
                # Save temp frame
                temp_path = f"temp_frame_{frame_num}.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Analyze
                frame_result = self.detector.analyze(temp_path)
                frame_results.append(frame_result)
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            frame_num += 1
        
        cap.release()
        
        # Aggregate results
        if not frame_results:
            return {"status": "error", "error": "No frames analyzed"}
        
        # Average predictions
        avg_probs = np.zeros(3)
        for fr in frame_results:
            pred = fr.get("prediction", {})
            dist = pred.get("distribution", {})
            avg_probs[0] += dist.get("real", 0)
            avg_probs[1] += dist.get("ai_generated", 0)
            avg_probs[2] += dist.get("ai_enhanced", 0)
        
        avg_probs /= len(frame_results)
        final_class = int(np.argmax(avg_probs))
        
        return {
            "status": "completed",
            "prediction": {
                "class": final_class,
                "class_name": CLASS_NAMES.get(final_class, "UNKNOWN"),
                "confidence": float(avg_probs[final_class]),
                "distribution": {
                    "real": float(avg_probs[0]),
                    "ai_generated": float(avg_probs[1]),
                    "ai_enhanced": float(avg_probs[2])
                }
            },
            "confidence_calibration": {
                "calibrated_confidence": float(avg_probs[final_class]),
                "is_uncertain": avg_probs[final_class] < 45
            },
            "interpretation": frame_results[-1].get("interpretation", {}),
            "heatmaps": frame_results[-1].get("heatmaps", {}),
            "video_info": {
                "frames_analyzed": len(frame_results),
                "total_frames": total_frames,
                "duration_seconds": duration
            }
        }
    
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
    
    def _calculate_final_verdict(self, result: Dict):
        """Calculate final verdict combining DL and classical"""
        pred = result["prediction"]
        secondary = result["secondary_evidence"]
        
        # If DL prediction available
        if result["primary_evidence"]["available"]:
            dl_conf = pred["confidence"]
            dl_class = pred["class_name"]
            
            # Classical supporting score
            classical_scores = []
            for key in ["ela", "noise", "edge", "frequency"]:
                score = secondary.get(key, {}).get("score", 50)
                classical_scores.append(score)
            
            classical_avg = np.mean(classical_scores) if classical_scores else 50
            
            # Weight: DL 70%, Classical 30%
            final_conf = 0.7 * dl_conf + 0.3 * classical_avg
            
            # Check for hard-AI patterns
            freq_info = result["primary_evidence"].get("frequency", {})
            freq_anomaly = freq_info.get("frequency_anomaly_score", 50)
            
            # If high frequency anomaly detected for AI class, boost confidence
            if dl_class in ["AI_GENERATED", "AI_ENHANCED"] and freq_anomaly > 60:
                final_conf = min(95, final_conf * 1.1)
            
            # If low freq anomaly for REAL class, boost confidence
            if dl_class == "REAL" and freq_anomaly < 40:
                final_conf = min(95, final_conf * 1.1)
            
            # Check if uncertain
            is_uncertain = pred.get("is_uncertain", False)
            if final_conf < 45:
                is_uncertain = True
            
            # Update verdict
            result["verdict"]["conclusion"] = "UNCERTAIN" if is_uncertain else dl_class
            result["verdict"]["confidence"] = round(final_conf, 2)
            
            # Update interpretation if uncertain
            if is_uncertain:
                result["interpretation"]["summary"] = f"UNCERTAIN: Analysis inconclusive ({final_conf:.0f}% confidence)"
                result["interpretation"]["recommendation"] = "Manual expert review required"
        else:
            # Classical only (fallback)
            classical_scores = [
                secondary.get("ela", {}).get("score", 50),
                secondary.get("noise", {}).get("score", 50),
                secondary.get("frequency", {}).get("score", 50)
            ]
            
            avg_score = np.mean(classical_scores)
            
            if avg_score > 60:
                result["verdict"]["conclusion"] = "LIKELY_AI"
            elif avg_score < 40:
                result["verdict"]["conclusion"] = "LIKELY_REAL"
            else:
                result["verdict"]["conclusion"] = "UNCERTAIN"
            
            result["verdict"]["confidence"] = round(abs(avg_score - 50) * 2, 2)
            result["verdict"]["based_on"] = "Classical Methods Only (DL unavailable)"
            
            result["interpretation"]["summary"] = "Analysis based on classical methods only"
            result["interpretation"]["limitations"].append("Deep Learning not available - reduced accuracy")
    
    def _convert_heatmaps_to_base64(self, result: Dict):
        """Convert heatmaps to base64 for frontend"""
        for name, heatmap in result["heatmaps"].items():
            if isinstance(heatmap, np.ndarray):
                # Apply colormap if grayscale
                if len(heatmap.shape) == 2:
                    hm_uint8 = (heatmap * 255).astype(np.uint8) if heatmap.max() <= 1 else heatmap.astype(np.uint8)
                    colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                else:
                    colored = heatmap
                
                # Encode to base64
                _, buffer = cv2.imencode('.png', colored)
                b64 = base64.b64encode(buffer).decode('utf-8')
                result["heatmaps_base64"][name] = f"data:image/png;base64,{b64}"
    
    # =====================================================
    # CLASSICAL METHODS (SECONDARY)
    # =====================================================
    
    def _analyze_ela(self, img: np.ndarray, file_path: str) -> Dict:
        """Error Level Analysis - JPEG re-compression artifacts"""
        result = {"score": 50, "details": {}, "heatmap": None}
        
        try:
            # Save at known quality
            temp_path = "temp_ela.jpg"
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Reload and compare
            recompressed = cv2.imread(temp_path)
            
            if recompressed is not None:
                # Calculate difference
                diff = cv2.absdiff(img, recompressed)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                # Amplify
                diff_amplified = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
                
                # Statistics
                mean_diff = np.mean(diff_amplified)
                max_diff = np.max(diff_amplified)
                std_diff = np.std(diff_amplified)
                
                # Score: Higher diff → more likely modified
                score = min(100, mean_diff * 3 + std_diff)
                
                result["score"] = round(score, 2)
                result["details"] = {
                    "mean_difference": round(float(mean_diff), 2),
                    "max_difference": int(max_diff),
                    "std_difference": round(float(std_diff), 2)
                }
                result["heatmap"] = diff_amplified.astype(np.float32) / 255.0
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_noise(self, img: np.ndarray, gray: np.ndarray) -> Dict:
        """Noise pattern analysis"""
        result = {"score": 50, "details": {}, "heatmap": None}
        
        try:
            # Estimate noise
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, denoised)
            
            # Noise variance
            noise_std = np.std(noise)
            noise_mean = np.mean(noise)
            
            # Check consistency
            blocks = []
            h, w = noise.shape
            block_size = 64
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = noise[y:y+block_size, x:x+block_size]
                    blocks.append(np.std(block))
            
            if blocks:
                block_std = np.std(blocks)
                consistency = block_std / (np.mean(blocks) + 1e-6)
            else:
                consistency = 0
            
            # Score: Inconsistent noise → suspicious
            score = min(100, consistency * 100)
            
            result["score"] = round(score, 2)
            result["details"] = {
                "noise_std": round(float(noise_std), 2),
                "noise_mean": round(float(noise_mean), 2),
                "consistency_ratio": round(float(consistency), 4)
            }
            result["heatmap"] = noise.astype(np.float32) / 255.0
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_edges(self, gray: np.ndarray) -> Dict:
        """Edge analysis"""
        result = {"score": 50, "details": {}, "heatmap": None}
        
        try:
            # Canny edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Edge density
            edge_density = np.mean(edges) / 255.0
            
            # Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = np.var(laplacian)
            
            # Score: Very smooth (low edges) might indicate AI
            score = max(0, 50 - edge_density * 100 + (lap_var / 1000))
            score = min(100, max(0, score))
            
            result["score"] = round(score, 2)
            result["details"] = {
                "edge_density": round(edge_density, 4),
                "laplacian_variance": round(lap_var, 2)
            }
            result["heatmap"] = edges.astype(np.float32) / 255.0
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_frequency(self, gray: np.ndarray) -> Dict:
        """Frequency domain analysis"""
        result = {"score": 50, "details": {}, "heatmap": None}
        
        try:
            # FFT
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            
            # Normalize
            magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
            
            # Analyze frequency distribution
            h, w = magnitude.shape
            center = (h // 2, w // 2)
            
            # Low freq (center) vs high freq (edges)
            radius = min(h, w) // 4
            y, x = np.ogrid[:h, :w]
            mask_low = (x - center[1])**2 + (y - center[0])**2 <= radius**2
            mask_high = (x - center[1])**2 + (y - center[0])**2 > radius**2
            
            low_energy = np.mean(magnitude[mask_low])
            high_energy = np.mean(magnitude[mask_high])
            
            ratio = high_energy / (low_energy + 1e-6)
            
            # AI often has lower high-freq ratio
            score = max(0, 50 - ratio * 100)
            score = min(100, max(0, score))
            
            result["score"] = round(score, 2)
            result["details"] = {
                "low_freq_energy": round(float(low_energy), 2),
                "high_freq_energy": round(float(high_energy), 2),
                "freq_ratio": round(float(ratio), 4)
            }
            result["heatmap"] = magnitude_norm
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_compression(self, img: np.ndarray, file_path: str) -> Dict:
        """Compression artifact analysis"""
        result = {"score": 50, "details": {}}
        
        try:
            # Check file type
            ext = Path(file_path).suffix.lower()
            
            if ext in ['.jpg', '.jpeg']:
                # JPEG block artifact detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Check 8x8 block boundaries
                h, w = gray.shape
                block_diff = 0
                count = 0
                
                for y in range(8, h, 8):
                    for x in range(0, w - 1):
                        block_diff += abs(int(gray[y-1, x]) - int(gray[y, x]))
                        count += 1
                
                avg_block_diff = block_diff / (count + 1)
                
                result["details"]["format"] = "JPEG"
                result["details"]["block_artifact_score"] = round(avg_block_diff, 2)
                result["score"] = min(100, avg_block_diff * 2)
            else:
                result["details"]["format"] = ext
                result["details"]["note"] = "Non-JPEG format"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _analyze_statistical(self, gray: np.ndarray) -> Dict:
        """Statistical analysis"""
        result = {"score": 50, "details": {}}
        
        try:
            # Histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist_norm = hist / hist.sum()
            
            # Entropy
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
            
            # Skewness and kurtosis
            mean = np.mean(gray)
            std = np.std(gray)
            
            if std > 0:
                skewness = np.mean(((gray - mean) / std) ** 3)
                kurtosis = np.mean(((gray - mean) / std) ** 4) - 3
            else:
                skewness = 0
                kurtosis = 0
            
            result["details"] = {
                "entropy": round(entropy, 2),
                "mean": round(float(mean), 2),
                "std": round(float(std), 2),
                "skewness": round(float(skewness), 3),
                "kurtosis": round(float(kurtosis), 3)
            }
            
            # Low entropy might indicate AI (smoother)
            score = max(0, (7 - entropy) * 15)
            result["score"] = round(min(100, max(0, score)), 2)
            
        except Exception as e:
            result["error"] = str(e)
        
        return result


# =====================================================
# INITIALIZATION
# =====================================================

def create_forensics_engine(model_path: str = None):
    """Factory function to create forensics engine"""
    return AdvancedVisualForensics(model_path=model_path)


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Advanced Visual Forensics Engine v5.0")
    print("=" * 60)
    print("Multi-Class Detection: Real | AI Generated | AI Enhanced")
    print("=" * 60)
    
    engine = AdvancedVisualForensics()
    
    print("\nEngine initialized:")
    print(f"  Version: {engine.version}")
    print(f"  Advanced Detector: {'✅' if engine.detector else '❌'}")
    print(f"  Legacy Fallback: {'✅' if engine.legacy_engine else '❌'}")
