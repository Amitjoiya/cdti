"""
Ultra-Advanced Visual Forensics Engine v3.0
25+ Manipulation Detection Algorithms with Detailed Heatmaps
+ Deep Learning Grad-CAM Integration
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import cv2
    import numpy as np
    from scipy import ndimage, fftpack
    from scipy.stats import chi2_contingency
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Deep Learning Grad-CAM
try:
    from deepfake_detector import DeepfakeDetector
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False


class AdvancedVisualForensics:
    """
    Ultra-Advanced Visual Forensics Engine
    
    25+ Algorithms in 8 Categories:
    1. ELA (Error Level Analysis) - 4 methods
    2. Noise Analysis - 4 methods  
    3. Frequency Analysis - 3 methods
    4. Edge Analysis - 4 methods
    5. Clone Detection - 2 methods
    6. Color Analysis - 3 methods
    7. Compression Analysis - 2 methods
    8. Statistical Analysis - 3 methods
    
    + Deep Learning Grad-CAM (Decision-based heatmaps)
    """
    
    def __init__(self):
        self.version = "3.0"
        self.algorithms_count = 25
        
        # Initialize Deep Learning Detector
        self.dl_detector = None
        if DL_AVAILABLE:
            try:
                self.dl_detector = DeepfakeDetector()
                print("  ✓ Deep Learning Grad-CAM enabled")
            except Exception as e:
                print(f"  ⚠️ Grad-CAM disabled: {e}")
    
    def analyze(self, file_path: str) -> Dict:
        """Run complete forensic analysis"""
        result = {
            "status": "processing",
            "version": self.version,
            "analyzed_at": datetime.now().isoformat(),
            "file_path": file_path,
            
            # Scores (0-100, higher = more suspicious)
            "scores": {
                "ela_score": 0,
                "noise_score": 0,
                "edge_score": 0,
                "clone_score": 0,
                "frequency_score": 0,
                "color_score": 0,
                "compression_score": 0,
                "statistical_score": 0,
                "overall_score": 0
            },
            
            # Detailed analysis results
            "analysis": {
                "ela": {},
                "noise": {},
                "edge": {},
                "clone": {},
                "frequency": {},
                "color": {},
                "compression": {},
                "statistical": {}
            },
            
            # Heatmaps (numpy arrays)
            "heatmaps": {},
            
            # Detection results
            "detections": {
                "manipulated_regions": [],
                "manipulation_percentage": 0,
                "severity": "none",
                "algorithms_triggered": 0
            },
            
            # Summary
            "summary": {},
            
            "errors": []
        }
        
        if not CV2_AVAILABLE:
            result["errors"].append("OpenCV not available")
            result["status"] = "error"
            return result
        
        try:
            # Load image
            path = Path(file_path)
            is_video = path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
            if is_video:
                img = self._extract_video_frame(file_path)
            else:
                img = cv2.imread(file_path)
            
            if img is None:
                result["errors"].append("Failed to load image")
                result["status"] = "error"
                return result
            
            # Store original for reference
            original = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # ========================================
            # 1. ERROR LEVEL ANALYSIS (4 methods)
            # ========================================
            ela_result = self._analyze_ela(img, file_path)
            result["analysis"]["ela"] = ela_result["details"]
            result["scores"]["ela_score"] = ela_result["score"]
            if ela_result.get("heatmap") is not None:
                result["heatmaps"]["ela_heatmap"] = ela_result["heatmap"]
            if ela_result.get("ela_multi") is not None:
                result["heatmaps"]["ela_multi_quality"] = ela_result["ela_multi"]
            
            # ========================================
            # 2. NOISE ANALYSIS (4 methods)
            # ========================================
            noise_result = self._analyze_noise(img, gray)
            result["analysis"]["noise"] = noise_result["details"]
            result["scores"]["noise_score"] = noise_result["score"]
            if noise_result.get("heatmap") is not None:
                result["heatmaps"]["noise_heatmap"] = noise_result["heatmap"]
            
            # ========================================
            # 3. EDGE ANALYSIS (4 methods)
            # ========================================
            edge_result = self._analyze_edges(gray)
            result["analysis"]["edge"] = edge_result["details"]
            result["scores"]["edge_score"] = edge_result["score"]
            if edge_result.get("heatmap") is not None:
                result["heatmaps"]["edge_heatmap"] = edge_result["heatmap"]
            
            # ========================================
            # 4. CLONE DETECTION (2 methods)
            # ========================================
            clone_result = self._detect_clones(gray)
            result["analysis"]["clone"] = clone_result["details"]
            result["scores"]["clone_score"] = clone_result["score"]
            if clone_result.get("heatmap") is not None:
                result["heatmaps"]["clone_heatmap"] = clone_result["heatmap"]
            
            # ========================================
            # 5. FREQUENCY ANALYSIS (3 methods)
            # ========================================
            freq_result = self._analyze_frequency(gray)
            result["analysis"]["frequency"] = freq_result["details"]
            result["scores"]["frequency_score"] = freq_result["score"]
            if freq_result.get("heatmap") is not None:
                result["heatmaps"]["frequency_heatmap"] = freq_result["heatmap"]
            
            # ========================================
            # 6. COLOR ANALYSIS (3 methods)
            # ========================================
            color_result = self._analyze_color(img)
            result["analysis"]["color"] = color_result["details"]
            result["scores"]["color_score"] = color_result["score"]
            if color_result.get("heatmap") is not None:
                result["heatmaps"]["color_heatmap"] = color_result["heatmap"]
            
            # ========================================
            # 7. COMPRESSION ANALYSIS (2 methods)
            # ========================================
            comp_result = self._analyze_compression(img, file_path)
            result["analysis"]["compression"] = comp_result["details"]
            result["scores"]["compression_score"] = comp_result["score"]
            if comp_result.get("heatmap") is not None:
                result["heatmaps"]["compression_heatmap"] = comp_result["heatmap"]
            
            # ========================================
            # 8. STATISTICAL ANALYSIS (3 methods)
            # ========================================
            stat_result = self._analyze_statistical(gray)
            result["analysis"]["statistical"] = stat_result["details"]
            result["scores"]["statistical_score"] = stat_result["score"]
            
            # ========================================
            # 9. DEEP LEARNING GRAD-CAM (Decision-based)
            # ========================================
            if self.dl_detector is not None:
                try:
                    dl_result = self.dl_detector.analyze_image(file_path)
                    
                    # Add DL-based heatmaps
                    dl_heatmaps = dl_result.get("heatmaps", {})
                    for name, heatmap in dl_heatmaps.items():
                        if heatmap is not None:
                            result["heatmaps"][f"gradcam_{name}"] = heatmap
                    
                    # Add DL analysis
                    result["analysis"]["deep_learning"] = {
                        "prediction": dl_result.get("prediction", 0),
                        "confidence": dl_result.get("confidence", 0),
                        "label": dl_result.get("label", "UNKNOWN"),
                        "regions": dl_result.get("regions", {}),
                        "forensic_analysis": dl_result.get("forensic_analysis", {}),
                        "algorithms": [
                            {"name": "ResNet-18 CNN", "type": "deep_learning"},
                            {"name": "Grad-CAM", "type": "explainability"},
                            {"name": "Multi-region Analysis", "type": "localization"}
                        ]
                    }
                    
                    # Add DL score to overall calculation
                    result["scores"]["deeplearning_score"] = dl_result.get("confidence", 0)
                    
                except Exception as e:
                    result["analysis"]["deep_learning"] = {"error": str(e)}
            
            # ========================================
            # COMBINED HEATMAP
            # ========================================
            combined = self._create_combined_heatmap(result["heatmaps"], original.shape[:2])
            if combined is not None:
                result["heatmaps"]["combined_heatmap"] = combined
                
                # Create overlay on original
                overlay = self._create_overlay(original, combined)
                result["heatmaps"]["manipulation_overlay"] = overlay
            
            # ========================================
            # DETECT MANIPULATED REGIONS
            # ========================================
            regions, percentage = self._detect_regions(combined if combined is not None else None)
            result["detections"]["manipulated_regions"] = regions
            result["detections"]["manipulation_percentage"] = percentage
            
            # ========================================
            # CALCULATE OVERALL SCORE
            # ========================================
            scores = result["scores"]
            
            # Weighted average (with DL if available)
            if "deeplearning_score" in scores:
                weights = {
                    "ela_score": 0.15,
                    "noise_score": 0.10,
                    "edge_score": 0.08,
                    "clone_score": 0.15,
                    "frequency_score": 0.07,
                    "color_score": 0.05,
                    "compression_score": 0.08,
                    "statistical_score": 0.02,
                    "deeplearning_score": 0.30  # DL has highest weight
                }
            else:
                weights = {
                    "ela_score": 0.25,
                    "noise_score": 0.15,
                    "edge_score": 0.10,
                    "clone_score": 0.20,
                    "frequency_score": 0.10,
                    "color_score": 0.05,
                    "compression_score": 0.10,
                    "statistical_score": 0.05
                }
            
            overall = sum(scores[k] * weights[k] for k in weights.keys())
            result["scores"]["overall_score"] = round(overall, 2)
            
            # Count triggered algorithms
            triggered = sum(1 for k, v in scores.items() if k != "overall_score" and v > 40)
            result["detections"]["algorithms_triggered"] = triggered
            
            # Determine severity
            if overall >= 70 or triggered >= 5:
                result["detections"]["severity"] = "critical"
            elif overall >= 50 or triggered >= 3:
                result["detections"]["severity"] = "high"
            elif overall >= 30 or triggered >= 2:
                result["detections"]["severity"] = "medium"
            elif overall >= 15:
                result["detections"]["severity"] = "low"
            else:
                result["detections"]["severity"] = "none"
            
            # ========================================
            # GENERATE SUMMARY
            # ========================================
            result["summary"] = {
                "overall_score": result["scores"]["overall_score"],
                "severity": result["detections"]["severity"],
                "manipulation_percentage": percentage,
                "regions_detected": len(regions),
                "algorithms_triggered": triggered,
                "highest_score": max(scores.values()),
                "highest_category": max(scores.keys(), key=lambda k: scores[k] if k != "overall_score" else 0),
                "image_dimensions": f"{original.shape[1]}x{original.shape[0]}",
                "analysis_complete": True
            }
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _extract_video_frame(self, video_path: str) -> np.ndarray:
        """Extract middle frame from video"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    
    # ========================================
    # ELA ANALYSIS
    # ========================================
    def _analyze_ela(self, img: np.ndarray, file_path: str) -> Dict:
        """Error Level Analysis - 4 methods"""
        result = {
            "score": 0,
            "details": {
                "method": "Error Level Analysis",
                "algorithms": [],
                "ela_mean": 0,
                "ela_std": 0,
                "ela_max": 0,
                "suspicious_pixels_percentage": 0,
                "quality_levels_tested": []
            },
            "heatmap": None,
            "ela_multi": None
        }
        
        try:
            import tempfile
            
            all_ela = []
            qualities = [95, 90, 85, 75]
            result["details"]["quality_levels_tested"] = qualities
            
            for quality in qualities:
                # Save at quality
                temp_path = tempfile.mktemp(suffix='.jpg')
                cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                # Reload
                resaved = cv2.imread(temp_path)
                os.unlink(temp_path)
                
                if resaved is None:
                    continue
                
                # Calculate ELA
                ela = cv2.absdiff(img, resaved)
                ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
                
                # Enhance
                ela_enhanced = cv2.normalize(ela_gray, None, 0, 255, cv2.NORM_MINMAX)
                all_ela.append(ela_enhanced)
                
                result["details"]["algorithms"].append({
                    "name": f"ELA Q{quality}",
                    "mean": float(np.mean(ela_gray)),
                    "std": float(np.std(ela_gray)),
                    "max": float(np.max(ela_gray))
                })
            
            if all_ela:
                # Use Q90 as main heatmap
                main_ela = all_ela[1] if len(all_ela) > 1 else all_ela[0]
                result["heatmap"] = cv2.applyColorMap(main_ela.astype(np.uint8), cv2.COLORMAP_JET)
                
                # Multi-quality combined
                if len(all_ela) >= 3:
                    multi = np.mean(all_ela, axis=0).astype(np.uint8)
                    result["ela_multi"] = cv2.applyColorMap(multi, cv2.COLORMAP_HOT)
                
                # Calculate statistics
                result["details"]["ela_mean"] = float(np.mean(main_ela))
                result["details"]["ela_std"] = float(np.std(main_ela))
                result["details"]["ela_max"] = float(np.max(main_ela))
                
                # Suspicious pixels (high ELA values)
                threshold = np.mean(main_ela) + 2 * np.std(main_ela)
                suspicious = np.sum(main_ela > threshold)
                total = main_ela.size
                result["details"]["suspicious_pixels_percentage"] = round(suspicious / total * 100, 2)
                
                # Score based on suspicious pixels and variance
                score = min(100, result["details"]["suspicious_pixels_percentage"] * 3 + result["details"]["ela_std"] / 2)
                result["score"] = round(score, 2)
                
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    # ========================================
    # NOISE ANALYSIS
    # ========================================
    def _analyze_noise(self, img: np.ndarray, gray: np.ndarray) -> Dict:
        """Noise Pattern Analysis - 4 methods"""
        result = {
            "score": 0,
            "details": {
                "method": "Noise Pattern Analysis",
                "algorithms": [],
                "noise_level": 0,
                "noise_variance": 0,
                "inconsistency_score": 0
            },
            "heatmap": None
        }
        
        try:
            h, w = gray.shape
            block_size = 64
            noise_levels = []
            
            # 1. Block-wise noise analysis
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size].astype(float)
                    
                    # High-pass filter to extract noise
                    blurred = cv2.GaussianBlur(block, (5, 5), 0)
                    noise = block - blurred
                    noise_std = np.std(noise)
                    noise_levels.append((x, y, noise_std))
            
            result["details"]["algorithms"].append({
                "name": "Block Noise Analysis",
                "blocks_analyzed": len(noise_levels),
                "block_size": block_size
            })
            
            if noise_levels:
                stds = [n[2] for n in noise_levels]
                result["details"]["noise_level"] = float(np.mean(stds))
                result["details"]["noise_variance"] = float(np.var(stds))
                
                # Create noise heatmap
                noise_map = np.zeros((h, w), dtype=np.float32)
                for x, y, std in noise_levels:
                    noise_map[y:y+block_size, x:x+block_size] = std
                
                noise_map = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX)
                result["heatmap"] = cv2.applyColorMap(noise_map.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            
            # 2. Median filter residual
            median = cv2.medianBlur(gray, 5)
            residual = cv2.absdiff(gray, median)
            result["details"]["algorithms"].append({
                "name": "Median Filter Residual",
                "mean_residual": float(np.mean(residual)),
                "max_residual": float(np.max(residual))
            })
            
            # 3. Laplacian variance (focus/blur detection)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = laplacian.var()
            result["details"]["algorithms"].append({
                "name": "Laplacian Variance",
                "variance": float(lap_var),
                "interpretation": "sharp" if lap_var > 500 else "blurry"
            })
            
            # Calculate inconsistency score
            if noise_levels:
                # High variance in noise levels = inconsistent = suspicious
                cv_noise = np.std(stds) / (np.mean(stds) + 1e-6) * 100
                result["details"]["inconsistency_score"] = float(cv_noise)
                result["score"] = min(100, cv_noise * 2)
                
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    # ========================================
    # EDGE ANALYSIS
    # ========================================
    def _analyze_edges(self, gray: np.ndarray) -> Dict:
        """Edge Consistency Analysis - 4 methods"""
        result = {
            "score": 0,
            "details": {
                "method": "Edge Analysis",
                "algorithms": [],
                "edge_density": 0,
                "edge_consistency": 0
            },
            "heatmap": None
        }
        
        try:
            # 1. Canny Edge Detection
            edges_canny = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges_canny > 0) / edges_canny.size * 100
            result["details"]["edge_density"] = round(edge_density, 2)
            result["details"]["algorithms"].append({
                "name": "Canny Edge Detection",
                "edge_pixels": int(np.sum(edges_canny > 0)),
                "density_percent": round(edge_density, 2)
            })
            
            # 2. Sobel Edges
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            result["details"]["algorithms"].append({
                "name": "Sobel Gradient",
                "mean_magnitude": float(np.mean(sobel_mag)),
                "max_magnitude": float(np.max(sobel_mag))
            })
            
            # 3. Laplacian Edges
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            result["details"]["algorithms"].append({
                "name": "Laplacian",
                "mean": float(np.mean(np.abs(laplacian))),
                "variance": float(np.var(laplacian))
            })
            
            # 4. Edge density variation (block-wise)
            h, w = gray.shape
            block_size = 64
            densities = []
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = edges_canny[y:y+block_size, x:x+block_size]
                    block_density = np.sum(block > 0) / block.size
                    densities.append(block_density)
            
            if densities:
                edge_cv = np.std(densities) / (np.mean(densities) + 1e-6) * 100
                result["details"]["edge_consistency"] = round(100 - edge_cv, 2)
                result["details"]["algorithms"].append({
                    "name": "Block Edge Density Variance",
                    "blocks_analyzed": len(densities),
                    "variance": float(np.var(densities))
                })
            
            # Create edge heatmap
            edge_map = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result["heatmap"] = cv2.applyColorMap(edge_map, cv2.COLORMAP_BONE)
            
            # Score based on edge inconsistency
            result["score"] = min(100, max(0, 100 - result["details"]["edge_consistency"]))
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    # ========================================
    # CLONE DETECTION
    # ========================================
    def _detect_clones(self, gray: np.ndarray) -> Dict:
        """Clone (Copy-Move) Detection - 2 methods"""
        result = {
            "score": 0,
            "details": {
                "method": "Clone Detection",
                "algorithms": [],
                "clone_pairs_found": 0,
                "clone_regions": []
            },
            "heatmap": None
        }
        
        try:
            h, w = gray.shape
            clone_map = np.zeros((h, w), dtype=np.uint8)
            
            # 1. ORB Feature Matching
            orb = cv2.ORB_create(nfeatures=1000)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            result["details"]["algorithms"].append({
                "name": "ORB Feature Matching",
                "keypoints_found": len(keypoints) if keypoints else 0
            })
            
            if descriptors is not None and len(descriptors) > 10:
                # Match features to themselves
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(descriptors, descriptors, k=2)
                
                clone_pairs = []
                for m, n in matches:
                    if m.queryIdx != m.trainIdx:  # Not same point
                        # Check if points are far apart (potential clone)
                        pt1 = keypoints[m.queryIdx].pt
                        pt2 = keypoints[m.trainIdx].pt
                        dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                        
                        if dist > 50 and m.distance < 30:  # Far apart but similar
                            clone_pairs.append((pt1, pt2, m.distance))
                            # Mark on heatmap
                            cv2.circle(clone_map, (int(pt1[0]), int(pt1[1])), 20, 255, -1)
                            cv2.circle(clone_map, (int(pt2[0]), int(pt2[1])), 20, 255, -1)
                
                result["details"]["clone_pairs_found"] = len(clone_pairs)
                result["details"]["clone_regions"] = [
                    {"pt1": p[0], "pt2": p[1], "similarity": float(100 - p[2])}
                    for p in clone_pairs[:20]
                ]
                
                result["details"]["algorithms"][0]["matches_found"] = len(clone_pairs)
            
            # 2. Block Matching (DCT-based)
            block_size = 16
            blocks = {}
            
            for y in range(0, h - block_size, block_size // 2):
                for x in range(0, w - block_size, block_size // 2):
                    block = gray[y:y+block_size, x:x+block_size].astype(float)
                    
                    # DCT of block
                    dct = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Quantize DCT coefficients
                    quantized = tuple((dct[:4, :4] / 10).astype(int).flatten())
                    
                    if quantized in blocks:
                        # Found similar block
                        ox, oy = blocks[quantized]
                        dist = np.sqrt((x-ox)**2 + (y-oy)**2)
                        if dist > 50:  # Far apart
                            cv2.rectangle(clone_map, (x, y), (x+block_size, y+block_size), 200, -1)
                            cv2.rectangle(clone_map, (ox, oy), (ox+block_size, oy+block_size), 200, -1)
                    else:
                        blocks[quantized] = (x, y)
            
            result["details"]["algorithms"].append({
                "name": "DCT Block Matching",
                "blocks_analyzed": len(blocks)
            })
            
            if np.max(clone_map) > 0:
                result["heatmap"] = cv2.applyColorMap(clone_map, cv2.COLORMAP_HOT)
            
            # Score based on clone pairs
            clone_score = min(100, result["details"]["clone_pairs_found"] * 5)
            result["score"] = clone_score
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    # ========================================
    # FREQUENCY ANALYSIS
    # ========================================
    def _analyze_frequency(self, gray: np.ndarray) -> Dict:
        """Frequency Domain Analysis - 3 methods"""
        result = {
            "score": 0,
            "details": {
                "method": "Frequency Analysis",
                "algorithms": [],
                "high_freq_ratio": 0,
                "spectrum_anomaly_score": 0
            },
            "heatmap": None
        }
        
        try:
            # 1. FFT Analysis
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            
            # Normalize
            magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result["heatmap"] = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_MAGMA)
            
            result["details"]["algorithms"].append({
                "name": "FFT Spectrum",
                "mean_magnitude": float(np.mean(magnitude)),
                "max_magnitude": float(np.max(magnitude))
            })
            
            # 2. High frequency ratio
            h, w = gray.shape
            center_h, center_w = h // 2, w // 2
            radius = min(h, w) // 4
            
            # Create mask for high frequencies (outside center)
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_w)**2 + (y - center_h)**2) > radius**2
            
            high_freq = np.sum(np.abs(fshift) * mask)
            low_freq = np.sum(np.abs(fshift) * ~mask)
            
            high_freq_ratio = high_freq / (low_freq + 1e-6) * 100
            result["details"]["high_freq_ratio"] = round(high_freq_ratio, 2)
            
            result["details"]["algorithms"].append({
                "name": "High Frequency Ratio",
                "ratio": round(high_freq_ratio, 2),
                "interpretation": "normal" if high_freq_ratio < 50 else "high"
            })
            
            # 3. DCT Analysis
            dct = fftpack.dct(fftpack.dct(gray.astype(float).T, norm='ortho').T, norm='ortho')
            dct_mag = np.log(np.abs(dct) + 1)
            
            result["details"]["algorithms"].append({
                "name": "DCT Analysis",
                "mean_coefficient": float(np.mean(dct_mag)),
                "std_coefficient": float(np.std(dct_mag))
            })
            
            # Score based on frequency anomalies
            result["score"] = min(100, high_freq_ratio)
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    # ========================================
    # COLOR ANALYSIS
    # ========================================
    def _analyze_color(self, img: np.ndarray) -> Dict:
        """Color Consistency Analysis - 3 methods"""
        result = {
            "score": 0,
            "details": {
                "method": "Color Analysis",
                "algorithms": [],
                "color_temperature_variance": 0,
                "saturation_anomaly": 0
            },
            "heatmap": None
        }
        
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            h, w = img.shape[:2]
            block_size = 64
            
            # 1. Block-wise color temperature analysis
            temps = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = lab[y:y+block_size, x:x+block_size]
                    # Color temperature approximation from a-b channels
                    temp = np.mean(block[:, :, 1]) - np.mean(block[:, :, 2])
                    temps.append(temp)
            
            if temps:
                temp_var = np.var(temps)
                result["details"]["color_temperature_variance"] = float(temp_var)
                result["details"]["algorithms"].append({
                    "name": "Color Temperature Analysis",
                    "blocks_analyzed": len(temps),
                    "variance": float(temp_var),
                    "interpretation": "consistent" if temp_var < 50 else "inconsistent"
                })
            
            # 2. Saturation analysis
            saturation = hsv[:, :, 1]
            sat_mean = np.mean(saturation)
            sat_std = np.std(saturation)
            
            result["details"]["algorithms"].append({
                "name": "Saturation Analysis",
                "mean": float(sat_mean),
                "std": float(sat_std)
            })
            
            # Create saturation heatmap
            sat_norm = cv2.normalize(saturation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result["heatmap"] = cv2.applyColorMap(sat_norm, cv2.COLORMAP_RAINBOW)
            
            # 3. Histogram analysis per channel
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
            
            result["details"]["algorithms"].append({
                "name": "Channel Histogram",
                "blue_peak": int(np.argmax(hist_b)),
                "green_peak": int(np.argmax(hist_g)),
                "red_peak": int(np.argmax(hist_r))
            })
            
            # Score based on color inconsistency
            result["score"] = min(100, result["details"]["color_temperature_variance"] * 0.5)
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    # ========================================
    # COMPRESSION ANALYSIS
    # ========================================
    def _analyze_compression(self, img: np.ndarray, file_path: str) -> Dict:
        """JPEG Compression Analysis - 2 methods"""
        result = {
            "score": 0,
            "details": {
                "method": "Compression Analysis",
                "algorithms": [],
                "double_compression_detected": False,
                "blocking_artifact_score": 0
            },
            "heatmap": None
        }
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            h, w = gray.shape
            
            # 1. JPEG Block Artifact Detection (8x8 grid)
            block_artifacts = np.zeros((h, w), dtype=np.float32)
            
            for y in range(7, h-1):
                for x in range(7, w-1):
                    if y % 8 == 7 or x % 8 == 7:
                        # Edge of 8x8 block
                        diff = abs(float(gray[y, x]) - float(gray[y, x+1]))
                        diff += abs(float(gray[y, x]) - float(gray[y+1, x]))
                        block_artifacts[y, x] = diff
            
            artifact_score = np.mean(block_artifacts) / 5
            result["details"]["blocking_artifact_score"] = float(artifact_score)
            
            result["details"]["algorithms"].append({
                "name": "8x8 Block Artifact Detection",
                "artifact_score": float(artifact_score),
                "interpretation": "compressed" if artifact_score > 5 else "clean"
            })
            
            # 2. Double JPEG Detection
            # Analyze DCT coefficient histogram
            gray_float = gray.astype(float)
            block_size = 8
            dct_coeffs = []
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray_float[y:y+block_size, x:x+block_size]
                    dct = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                    dct_coeffs.extend(dct.flatten()[1:10])  # First 10 AC coefficients
            
            if dct_coeffs:
                # Check for periodic patterns in DCT histogram
                hist, _ = np.histogram(dct_coeffs, bins=100)
                peaks = np.sum(hist > np.mean(hist) * 2)
                
                double_jpeg = peaks > 15
                result["details"]["double_compression_detected"] = bool(double_jpeg)
                
                result["details"]["algorithms"].append({
                    "name": "Double JPEG Detection",
                    "detected": bool(double_jpeg),
                    "histogram_peaks": int(peaks)
                })
            
            # Create artifact heatmap
            artifact_norm = cv2.normalize(block_artifacts, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result["heatmap"] = cv2.applyColorMap(artifact_norm, cv2.COLORMAP_TURBO)
            
            # Score
            score = artifact_score * 5
            if result["details"]["double_compression_detected"]:
                score += 30
            result["score"] = min(100, score)
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    # ========================================
    # STATISTICAL ANALYSIS
    # ========================================
    def _analyze_statistical(self, gray: np.ndarray) -> Dict:
        """Statistical Distribution Analysis - 3 methods"""
        result = {
            "score": 0,
            "details": {
                "method": "Statistical Analysis",
                "algorithms": [],
                "benford_deviation": 0,
                "chi_square_score": 0
            }
        }
        
        try:
            # 1. Pixel Value Distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist_normalized = hist / np.sum(hist)
            
            result["details"]["algorithms"].append({
                "name": "Pixel Distribution",
                "mean": float(np.mean(gray)),
                "std": float(np.std(gray)),
                "skewness": float(np.mean(((gray - np.mean(gray)) / np.std(gray))**3)),
                "entropy": float(-np.sum(hist_normalized[hist_normalized > 0] * np.log2(hist_normalized[hist_normalized > 0])))
            })
            
            # 2. First Digit Distribution (Benford's Law)
            pixels = gray[gray > 0].flatten()
            first_digits = (pixels // 10**(np.floor(np.log10(pixels + 0.1)))).astype(int)
            first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]
            
            if len(first_digits) > 0:
                digit_hist = np.histogram(first_digits, bins=range(1, 11))[0]
                digit_hist = digit_hist / np.sum(digit_hist)
                
                # Expected Benford distribution
                benford = np.log10(1 + 1/np.arange(1, 10))
                
                deviation = np.sum(np.abs(digit_hist - benford))
                result["details"]["benford_deviation"] = float(deviation)
                
                result["details"]["algorithms"].append({
                    "name": "Benford's Law",
                    "deviation": float(deviation),
                    "interpretation": "natural" if deviation < 0.3 else "suspicious"
                })
            
            # 3. Chi-Square Test
            expected = np.ones(256) * len(gray.flatten()) / 256
            observed = hist
            
            chi2 = np.sum((observed - expected)**2 / (expected + 1e-6))
            chi2_normalized = chi2 / len(gray.flatten())
            result["details"]["chi_square_score"] = float(chi2_normalized)
            
            result["details"]["algorithms"].append({
                "name": "Chi-Square Test",
                "score": float(chi2_normalized),
                "interpretation": "uniform" if chi2_normalized < 1 else "non-uniform"
            })
            
            # Score
            result["score"] = min(100, result["details"]["benford_deviation"] * 100 + chi2_normalized * 10)
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    # ========================================
    # HELPER METHODS
    # ========================================
    def _create_combined_heatmap(self, heatmaps: Dict, shape: Tuple[int, int]) -> np.ndarray:
        """Create weighted combined heatmap"""
        if not heatmaps:
            return None
        
        h, w = shape
        combined = np.zeros((h, w), dtype=np.float32)
        
        weights = {
            "ela_heatmap": 0.3,
            "noise_heatmap": 0.2,
            "clone_heatmap": 0.25,
            "edge_heatmap": 0.1,
            "frequency_heatmap": 0.1,
            "color_heatmap": 0.05
        }
        
        total_weight = 0
        for name, heatmap in heatmaps.items():
            if heatmap is None or name not in weights:
                continue
            
            # Convert to grayscale if needed
            if len(heatmap.shape) == 3:
                gray_hm = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
            else:
                gray_hm = heatmap
            
            # Resize if needed
            if gray_hm.shape != (h, w):
                gray_hm = cv2.resize(gray_hm, (w, h))
            
            combined += gray_hm.astype(np.float32) * weights[name]
            total_weight += weights[name]
        
        if total_weight > 0:
            combined /= total_weight
        
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    
    def _create_overlay(self, original: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Create semi-transparent overlay"""
        if heatmap.shape[:2] != original.shape[:2]:
            heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        return overlay
    
    def _detect_regions(self, heatmap: np.ndarray) -> Tuple[List[Dict], float]:
        """Detect individual manipulated regions"""
        regions = []
        
        if heatmap is None:
            return regions, 0
        
        # Convert to grayscale
        if len(heatmap.shape) == 3:
            gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        else:
            gray = heatmap
        
        # Threshold to find high-intensity regions
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = gray.shape[0] * gray.shape[1]
        manipulation_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                manipulation_area += area
                
                # Determine severity based on intensity
                roi = gray[y:y+h, x:x+w]
                intensity = np.mean(roi)
                
                if intensity > 200:
                    severity = "critical"
                elif intensity > 150:
                    severity = "high"
                elif intensity > 100:
                    severity = "medium"
                else:
                    severity = "low"
                
                regions.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area),
                    "intensity": float(intensity),
                    "severity": severity
                })
        
        # Sort by area
        regions.sort(key=lambda r: r["area"], reverse=True)
        
        percentage = round(manipulation_area / total_area * 100, 2)
        
        return regions[:20], percentage


# Test
if __name__ == "__main__":
    vf = AdvancedVisualForensics()
    print(f"✅ Advanced Visual Forensics v{vf.version} initialized!")
    print(f"   Algorithms: {vf.algorithms_count}")
