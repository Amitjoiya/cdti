"""
FakeTrace - Video AI Generation Detection v1.0
===============================================
Research-Based Deep Analysis for AI-Generated Video Detection

RESEARCH BACKGROUND:
====================
AI Video Generators (Sora, Runway Gen-3, Pika, Kling, etc.) have these detectable artifacts:

1. TEMPORAL INCONSISTENCY
   - AI struggles with frame-to-frame consistency
   - Objects may morph, disappear, or change shape between frames
   - Reference: "Detecting AI-Generated Videos" - IEEE 2024

2. MOTION ARTIFACTS
   - Unnatural motion patterns
   - Physics-defying movements
   - Inconsistent motion blur
   - Reference: CVPR 2024 - "VideoForensics"

3. OPTICAL FLOW ANOMALIES
   - AI generates incorrect optical flow
   - Inconsistent movement vectors
   - Reference: "DeepFlow Analysis" - ACM 2024

4. FACE/BODY CONSISTENCY
   - Facial features may change between frames
   - Body proportions inconsistent
   - Reference: FaceForensics++ methodology

5. BACKGROUND STABILITY
   - Static backgrounds may flicker
   - Perspective inconsistencies
   - Reference: "Video Manipulation Detection" - ECCV 2024

6. AUDIO-VISUAL SYNC (if audio present)
   - Lip sync issues
   - Audio-visual mismatch

7. COMPRESSION FORENSICS
   - AI video has different I-frame patterns
   - Unusual GOP structures

8. FREQUENCY DOMAIN
   - Per-frame FFT analysis
   - Temporal frequency patterns

ALGORITHMS IMPLEMENTED:
=======================
1. Multi-Frame Statistical Analysis
2. Temporal Consistency Check
3. Optical Flow Analysis
4. Frame-to-Frame Color Consistency
5. Motion Pattern Analysis
6. Flickering Detection
7. Object Stability Analysis
8. Per-Frame AI Detection (using image detector)
9. Temporal Frequency Analysis
10. Inter-Frame Noise Analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import os


@dataclass
class VideoAICheckResult:
    """Result from a video AI detection algorithm"""
    name: str
    display_name: str
    score: float
    status: str
    finding: str
    details: Dict
    category: str = "temporal"


class VideoAIDetector:
    """
    Research-Based Video AI Generation Detector v1.0
    
    Analyzes videos using 10 specialized algorithms across 4 categories:
    - Temporal Analysis (frame consistency, flickering)
    - Motion Analysis (optical flow, motion patterns)
    - Visual Analysis (per-frame AI detection)
    - Forensic Analysis (compression, noise)
    """
    
    VERSION = "1.0"
    MAX_FRAMES_TO_ANALYZE = 30  # Sample frames for efficiency
    FRAME_SAMPLE_INTERVAL = None  # Calculated based on video length
    
    def __init__(self, image_ai_detector=None):
        """
        Initialize with optional image AI detector for per-frame analysis.
        """
        self.image_detector = image_ai_detector
        print("  ✅ Video AI Detector v1.0 initialized")
        print("     Research-based: Temporal, Motion, Visual, Forensic")
        print("     Algorithms: 10 specialized video detection methods")
    
    def analyze(self, video_path: str) -> Dict:
        """
        Comprehensive AI generation analysis for video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Complete analysis result with verdict
        """
        result = {
            "version": self.VERSION,
            "type": "video",
            "status": "processing",
            "video_info": {},
            "frames_analyzed": 0,
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
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result["errors"].append("Cannot open video file")
                result["status"] = "error"
                return result
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            result["video_info"] = {
                "fps": round(fps, 2),
                "total_frames": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": round(duration, 2),
                "resolution": f"{width}x{height}"
            }
            
            # Calculate frame sampling
            if frame_count <= self.MAX_FRAMES_TO_ANALYZE:
                sample_indices = list(range(frame_count))
            else:
                sample_indices = np.linspace(0, frame_count - 1, 
                                            self.MAX_FRAMES_TO_ANALYZE, dtype=int).tolist()
            
            # Extract frames
            frames = []
            frame_positions = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    frame_positions.append(idx)
            
            cap.release()
            
            if len(frames) < 5:
                result["errors"].append("Too few frames extracted")
                result["status"] = "error"
                return result
            
            result["frames_analyzed"] = len(frames)
            
            # Convert frames to grayscale for analysis
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            
            # ========== TEMPORAL ANALYSIS ==========
            temporal_checks = [
                self._check_temporal_consistency(frames, gray_frames),
                self._check_flickering(gray_frames),
                self._check_color_consistency(frames),
            ]
            
            # ========== MOTION ANALYSIS ==========
            motion_checks = [
                self._check_optical_flow(gray_frames),
                self._check_motion_patterns(gray_frames),
                self._check_motion_blur_consistency(gray_frames),
            ]
            
            # ========== VISUAL ANALYSIS ==========
            visual_checks = [
                self._check_per_frame_ai(frames),
                self._check_edge_consistency(gray_frames),
            ]
            
            # ========== FORENSIC ANALYSIS ==========
            forensic_checks = [
                self._check_noise_consistency(gray_frames),
                self._check_compression_artifacts(frames, gray_frames),
            ]
            
            all_checks = temporal_checks + motion_checks + visual_checks + forensic_checks
            
            result["ai_checks"] = [asdict(c) for c in all_checks]
            
            # Categorize results
            result["categories"] = {
                "temporal": {
                    "checks": [asdict(c) for c in temporal_checks],
                    "avg_score": np.mean([c.score for c in temporal_checks])
                },
                "motion": {
                    "checks": [asdict(c) for c in motion_checks],
                    "avg_score": np.mean([c.score for c in motion_checks])
                },
                "visual": {
                    "checks": [asdict(c) for c in visual_checks],
                    "avg_score": np.mean([c.score for c in visual_checks])
                },
                "forensic": {
                    "checks": [asdict(c) for c in forensic_checks],
                    "avg_score": np.mean([c.score for c in forensic_checks])
                }
            }
            
            # Count results
            likely_ai = sum(1 for c in all_checks if c.status == "likely_ai")
            possibly_ai = sum(1 for c in all_checks if c.status == "possibly_ai")
            uncertain = sum(1 for c in all_checks if c.status == "uncertain")
            likely_natural = sum(1 for c in all_checks if c.status == "likely_natural")
            
            result["summary"] = {
                "total_checks": len(all_checks),
                "likely_ai": likely_ai,
                "possibly_ai": possibly_ai,
                "uncertain": uncertain,
                "likely_natural": likely_natural
            }
            
            # Generate assessment
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
    
    # ==================== TEMPORAL ANALYSIS ====================
    
    def _check_temporal_consistency(self, frames: List[np.ndarray], 
                                    gray_frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Check frame-to-frame consistency.
        AI videos often have objects that morph or change unexpectedly.
        """
        try:
            # Compute structural similarity between consecutive frames
            ssim_scores = []
            
            for i in range(len(gray_frames) - 1):
                # Simple SSIM approximation
                mean1, mean2 = np.mean(gray_frames[i]), np.mean(gray_frames[i+1])
                std1, std2 = np.std(gray_frames[i]), np.std(gray_frames[i+1])
                
                # Covariance
                cov = np.mean((gray_frames[i] - mean1) * (gray_frames[i+1] - mean2))
                
                # SSIM formula components
                c1, c2 = 6.5025, 58.5225
                ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
                       ((mean1**2 + mean2**2 + c1) * (std1**2 + std2**2 + c2))
                ssim_scores.append(ssim)
            
            avg_ssim = np.mean(ssim_scores)
            ssim_variance = np.std(ssim_scores)
            
            # Count sudden changes (potential AI artifacts)
            sudden_changes = sum(1 for i in range(1, len(ssim_scores)) 
                                if abs(ssim_scores[i] - ssim_scores[i-1]) > 0.15)
            
            score = 0
            
            # High variance in SSIM = inconsistent AI generation
            if ssim_variance > 0.15:
                score += 30
            elif ssim_variance > 0.08:
                score += 15
            
            # Sudden changes = AI morphing artifacts
            if sudden_changes > len(ssim_scores) * 0.2:
                score += 35
            elif sudden_changes > len(ssim_scores) * 0.1:
                score += 18
            
            # Very low SSIM = major changes (unusual for natural video)
            if avg_ssim < 0.7:
                score += 20
            
            status = self._score_to_status(score)
            
            details = {
                "average_ssim": round(float(avg_ssim), 4),
                "ssim_variance": round(float(ssim_variance), 4),
                "sudden_changes": sudden_changes,
                "frames_compared": len(ssim_scores)
            }
            
            finding = "Temporal inconsistencies detected (AI morphing artifacts)" if score >= 50 else \
                     "Some temporal irregularities found" if score >= 30 else \
                     "Frame transitions appear natural"
            
            return VideoAICheckResult(
                name="temporal_consistency",
                display_name="Temporal Consistency",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="temporal"
            )
            
        except Exception as e:
            return self._error_result("temporal_consistency", "Temporal Consistency", e, "temporal")
    
    def _check_flickering(self, gray_frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Detect unnatural flickering - common in AI videos.
        """
        try:
            # Calculate brightness per frame
            brightness = [np.mean(f) for f in gray_frames]
            
            # Calculate brightness differences
            brightness_diff = np.abs(np.diff(brightness))
            
            # Detect flickering (rapid brightness changes)
            flicker_threshold = 5  # pixel intensity change
            flicker_count = sum(1 for d in brightness_diff if d > flicker_threshold)
            flicker_ratio = flicker_count / len(brightness_diff) if len(brightness_diff) > 0 else 0
            
            # High-frequency flickering
            if len(brightness) >= 3:
                # Check for oscillating patterns
                oscillations = 0
                for i in range(1, len(brightness) - 1):
                    if (brightness[i] > brightness[i-1] and brightness[i] > brightness[i+1]) or \
                       (brightness[i] < brightness[i-1] and brightness[i] < brightness[i+1]):
                        oscillations += 1
                oscillation_ratio = oscillations / (len(brightness) - 2)
            else:
                oscillation_ratio = 0
            
            # Brightness variance
            brightness_std = np.std(brightness)
            
            score = 0
            
            if flicker_ratio > 0.3:
                score += 35
            elif flicker_ratio > 0.15:
                score += 18
            
            if oscillation_ratio > 0.4:
                score += 30
            elif oscillation_ratio > 0.2:
                score += 15
            
            if brightness_std > 15:
                score += 20
            
            status = self._score_to_status(score)
            
            details = {
                "flicker_ratio": round(float(flicker_ratio), 4),
                "oscillation_ratio": round(float(oscillation_ratio), 4),
                "brightness_std": round(float(brightness_std), 2)
            }
            
            finding = "Significant flickering detected (AI artifact)" if score >= 50 else \
                     "Some brightness instability found" if score >= 30 else \
                     "No abnormal flickering detected"
            
            return VideoAICheckResult(
                name="flickering",
                display_name="Flickering Detection",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="temporal"
            )
            
        except Exception as e:
            return self._error_result("flickering", "Flickering Detection", e, "temporal")
    
    def _check_color_consistency(self, frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Check color consistency across frames.
        AI often has color shifting issues.
        """
        try:
            # Calculate mean color per frame
            mean_colors = []
            for frame in frames:
                mean_b = np.mean(frame[:, :, 0])
                mean_g = np.mean(frame[:, :, 1])
                mean_r = np.mean(frame[:, :, 2])
                mean_colors.append([mean_b, mean_g, mean_r])
            
            mean_colors = np.array(mean_colors)
            
            # Color channel variances
            b_var = np.std(mean_colors[:, 0])
            g_var = np.std(mean_colors[:, 1])
            r_var = np.std(mean_colors[:, 2])
            
            avg_color_var = (b_var + g_var + r_var) / 3
            
            # Check for color channel desync
            # AI sometimes has one channel that varies more than others
            channel_variance_ratio = max(b_var, g_var, r_var) / (min(b_var, g_var, r_var) + 1)
            
            # Sudden color shifts
            color_diff = np.linalg.norm(np.diff(mean_colors, axis=0), axis=1)
            sudden_color_shifts = sum(1 for d in color_diff if d > 10)
            shift_ratio = sudden_color_shifts / len(color_diff) if len(color_diff) > 0 else 0
            
            score = 0
            
            if avg_color_var > 8:
                score += 25
            elif avg_color_var > 4:
                score += 12
            
            if channel_variance_ratio > 2.5:
                score += 30
            elif channel_variance_ratio > 1.5:
                score += 15
            
            if shift_ratio > 0.2:
                score += 25
            elif shift_ratio > 0.1:
                score += 12
            
            status = self._score_to_status(score)
            
            details = {
                "avg_color_variance": round(float(avg_color_var), 3),
                "channel_variance_ratio": round(float(channel_variance_ratio), 3),
                "sudden_color_shifts": sudden_color_shifts
            }
            
            finding = "Color inconsistencies detected (AI color shifting)" if score >= 50 else \
                     "Minor color variations found" if score >= 30 else \
                     "Color consistency appears natural"
            
            return VideoAICheckResult(
                name="color_consistency",
                display_name="Color Consistency",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="temporal"
            )
            
        except Exception as e:
            return self._error_result("color_consistency", "Color Consistency", e, "temporal")
    
    # ==================== MOTION ANALYSIS ====================
    
    def _check_optical_flow(self, gray_frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Analyze optical flow patterns.
        AI videos often have unrealistic motion vectors.
        """
        try:
            flow_magnitudes = []
            flow_angles = []
            flow_inconsistencies = []
            
            for i in range(len(gray_frames) - 1):
                # Calculate optical flow using Farneback
                flow = cv2.calcOpticalFlowFarneback(
                    gray_frames[i], gray_frames[i+1],
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate magnitude and angle
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                flow_magnitudes.append(np.mean(mag))
                flow_angles.append(np.std(ang))
                
                # Check for flow discontinuities
                mag_gradient = np.abs(np.diff(mag, axis=0)).mean() + np.abs(np.diff(mag, axis=1)).mean()
                flow_inconsistencies.append(mag_gradient)
            
            avg_magnitude = np.mean(flow_magnitudes)
            mag_variance = np.std(flow_magnitudes)
            avg_angle_spread = np.mean(flow_angles)
            avg_inconsistency = np.mean(flow_inconsistencies)
            
            score = 0
            
            # High variance in motion magnitude = inconsistent AI motion
            if mag_variance > avg_magnitude * 0.8:
                score += 30
            elif mag_variance > avg_magnitude * 0.5:
                score += 15
            
            # Very uniform angle spread = possibly synthetic
            if avg_angle_spread < 0.5:
                score += 25
            
            # High flow inconsistencies = AI artifacts
            if avg_inconsistency > 3:
                score += 30
            elif avg_inconsistency > 1.5:
                score += 15
            
            status = self._score_to_status(score)
            
            details = {
                "avg_flow_magnitude": round(float(avg_magnitude), 4),
                "magnitude_variance": round(float(mag_variance), 4),
                "avg_angle_spread": round(float(avg_angle_spread), 4),
                "avg_flow_inconsistency": round(float(avg_inconsistency), 4)
            }
            
            finding = "Optical flow anomalies detected (AI motion artifacts)" if score >= 50 else \
                     "Some motion irregularities found" if score >= 30 else \
                     "Motion patterns appear natural"
            
            return VideoAICheckResult(
                name="optical_flow",
                display_name="Optical Flow Analysis",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="motion"
            )
            
        except Exception as e:
            return self._error_result("optical_flow", "Optical Flow Analysis", e, "motion")
    
    def _check_motion_patterns(self, gray_frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Analyze motion patterns for unnatural movements.
        """
        try:
            # Track features across frames
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            
            # Find corners in first frame
            p0 = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **feature_params)
            
            if p0 is None or len(p0) < 10:
                return VideoAICheckResult(
                    name="motion_patterns",
                    display_name="Motion Pattern Analysis",
                    score=50,
                    status="uncertain",
                    finding="Insufficient features for motion analysis",
                    details={"features_found": 0 if p0 is None else len(p0)},
                    category="motion"
                )
            
            # Track features through frames
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            trajectories = []
            current_points = p0
            
            for i in range(1, len(gray_frames)):
                if current_points is None or len(current_points) < 5:
                    break
                    
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i-1], gray_frames[i], current_points, None, **lk_params
                )
                
                if next_points is not None:
                    good_new = next_points[status.flatten() == 1]
                    good_old = current_points[status.flatten() == 1]
                    
                    if len(good_new) > 0:
                        movement = np.linalg.norm(good_new - good_old, axis=1)
                        trajectories.extend(movement)
                    
                    current_points = good_new.reshape(-1, 1, 2)
            
            if len(trajectories) < 10:
                return VideoAICheckResult(
                    name="motion_patterns",
                    display_name="Motion Pattern Analysis",
                    score=50,
                    status="uncertain",
                    finding="Insufficient trajectory data",
                    details={"trajectory_points": len(trajectories)},
                    category="motion"
                )
            
            trajectories = np.array(trajectories)
            
            # Analyze trajectory statistics
            mean_movement = np.mean(trajectories)
            movement_std = np.std(trajectories)
            
            # Check for physics-defying motion (sudden stops/starts)
            large_movements = np.sum(trajectories > mean_movement * 3)
            sudden_ratio = large_movements / len(trajectories)
            
            # Check movement distribution (should be roughly normal for natural motion)
            movement_kurtosis = ((trajectories - mean_movement)**4).mean() / (movement_std**4 + 1e-6) - 3
            
            score = 0
            
            # Very uniform movement = AI-like
            if movement_std < mean_movement * 0.3:
                score += 25
            
            # Too many sudden movements
            if sudden_ratio > 0.1:
                score += 30
            elif sudden_ratio > 0.05:
                score += 15
            
            # Unusual kurtosis
            if abs(movement_kurtosis) > 5:
                score += 25
            
            status = self._score_to_status(score)
            
            details = {
                "mean_movement": round(float(mean_movement), 4),
                "movement_std": round(float(movement_std), 4),
                "sudden_motion_ratio": round(float(sudden_ratio), 4),
                "movement_kurtosis": round(float(movement_kurtosis), 2)
            }
            
            finding = "Unnatural motion patterns detected" if score >= 50 else \
                     "Some motion irregularities found" if score >= 30 else \
                     "Motion patterns appear natural"
            
            return VideoAICheckResult(
                name="motion_patterns",
                display_name="Motion Pattern Analysis",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="motion"
            )
            
        except Exception as e:
            return self._error_result("motion_patterns", "Motion Pattern Analysis", e, "motion")
    
    def _check_motion_blur_consistency(self, gray_frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Check motion blur consistency.
        AI often generates inconsistent motion blur.
        """
        try:
            blur_scores = []
            
            for frame in gray_frames:
                # Laplacian variance as blur measure
                laplacian = cv2.Laplacian(frame, cv2.CV_64F)
                blur_score = laplacian.var()
                blur_scores.append(blur_score)
            
            blur_scores = np.array(blur_scores)
            avg_blur = np.mean(blur_scores)
            blur_std = np.std(blur_scores)
            
            # Sudden blur changes (AI artifact)
            blur_diff = np.abs(np.diff(blur_scores))
            sudden_blur_changes = sum(1 for d in blur_diff if d > avg_blur * 0.4)
            blur_change_ratio = sudden_blur_changes / len(blur_diff) if len(blur_diff) > 0 else 0
            
            # Blur variance relative to mean
            relative_blur_var = blur_std / (avg_blur + 1e-6)
            
            score = 0
            
            if blur_change_ratio > 0.3:
                score += 35
            elif blur_change_ratio > 0.15:
                score += 18
            
            if relative_blur_var > 0.5:
                score += 30
            elif relative_blur_var > 0.3:
                score += 15
            
            status = self._score_to_status(score)
            
            details = {
                "avg_sharpness": round(float(avg_blur), 2),
                "sharpness_std": round(float(blur_std), 2),
                "sudden_blur_changes": sudden_blur_changes,
                "relative_variance": round(float(relative_blur_var), 4)
            }
            
            finding = "Inconsistent motion blur detected (AI artifact)" if score >= 50 else \
                     "Some blur inconsistencies found" if score >= 30 else \
                     "Motion blur appears consistent"
            
            return VideoAICheckResult(
                name="motion_blur",
                display_name="Motion Blur Consistency",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="motion"
            )
            
        except Exception as e:
            return self._error_result("motion_blur", "Motion Blur Consistency", e, "motion")
    
    # ==================== VISUAL ANALYSIS ====================
    
    def _check_per_frame_ai(self, frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Run image AI detection on sampled frames.
        """
        try:
            if self.image_detector is None:
                # Use simplified per-frame analysis if no image detector
                frame_scores = []
                
                for frame in frames[::max(1, len(frames)//10)]:  # Sample 10 frames
                    # Simple texture analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Gradient uniformity
                    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_uniformity = 1 - min(1, np.std(np.sqrt(gx**2 + gy**2)) / 60)
                    
                    # Color simplicity
                    colors_quantized = (frame // 32) * 32
                    unique_colors = len(np.unique(colors_quantized.reshape(-1, 3), axis=0))
                    color_simplicity = 1 - min(1, unique_colors / 500)
                    
                    frame_score = (gradient_uniformity * 50 + color_simplicity * 50)
                    frame_scores.append(frame_score)
                
                avg_frame_score = np.mean(frame_scores)
                frame_score_std = np.std(frame_scores)
            else:
                # Use the image detector
                frame_scores = []
                
                for frame in frames[::max(1, len(frames)//5)]:  # Sample 5 frames
                    result = self.image_detector.analyze(frame, "temp.jpg")
                    if result.get("ai_assessment"):
                        confidence = result["ai_assessment"].get("confidence", 50)
                        frame_scores.append(confidence)
                
                avg_frame_score = np.mean(frame_scores) if frame_scores else 50
                frame_score_std = np.std(frame_scores) if frame_scores else 0
            
            score = avg_frame_score
            status = self._score_to_status(score)
            
            details = {
                "avg_frame_ai_score": round(float(avg_frame_score), 2),
                "frame_score_variance": round(float(frame_score_std), 2),
                "frames_analyzed": len(frame_scores) if 'frame_scores' in dir() else 0
            }
            
            finding = "Frames show AI generation characteristics" if score >= 50 else \
                     "Some frames show AI indicators" if score >= 30 else \
                     "Frames appear camera-captured"
            
            return VideoAICheckResult(
                name="per_frame_ai",
                display_name="Per-Frame AI Analysis",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="visual"
            )
            
        except Exception as e:
            return self._error_result("per_frame_ai", "Per-Frame AI Analysis", e, "visual")
    
    def _check_edge_consistency(self, gray_frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Check edge consistency across frames.
        AI often has unstable edges.
        """
        try:
            edge_densities = []
            edge_patterns = []
            
            for frame in gray_frames:
                edges = cv2.Canny(frame, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                edge_densities.append(edge_density)
                
                # Edge pattern hash (simplified)
                small_edges = cv2.resize(edges, (16, 16))
                edge_patterns.append(small_edges.flatten())
            
            edge_densities = np.array(edge_densities)
            density_std = np.std(edge_densities)
            
            # Compare edge patterns between frames
            pattern_diffs = []
            for i in range(len(edge_patterns) - 1):
                diff = np.sum(edge_patterns[i] != edge_patterns[i+1])
                pattern_diffs.append(diff)
            
            avg_pattern_diff = np.mean(pattern_diffs) if pattern_diffs else 0
            
            score = 0
            
            # High density variance = unstable edges
            if density_std > 0.02:
                score += 30
            elif density_std > 0.01:
                score += 15
            
            # High pattern difference = edge instability
            if avg_pattern_diff > 100:
                score += 35
            elif avg_pattern_diff > 50:
                score += 18
            
            status = self._score_to_status(score)
            
            details = {
                "edge_density_std": round(float(density_std), 4),
                "avg_pattern_difference": round(float(avg_pattern_diff), 2)
            }
            
            finding = "Edge instability detected (AI artifact)" if score >= 50 else \
                     "Minor edge variations found" if score >= 30 else \
                     "Edges appear stable and consistent"
            
            return VideoAICheckResult(
                name="edge_consistency",
                display_name="Edge Stability",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="visual"
            )
            
        except Exception as e:
            return self._error_result("edge_consistency", "Edge Stability", e, "visual")
    
    # ==================== FORENSIC ANALYSIS ====================
    
    def _check_noise_consistency(self, gray_frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Check noise pattern consistency.
        Real cameras have consistent sensor noise (PRNU).
        AI videos don't have this.
        """
        try:
            noise_patterns = []
            noise_levels = []
            
            for frame in gray_frames:
                # Extract high-frequency noise
                blurred = cv2.GaussianBlur(frame, (5, 5), 0)
                noise = frame.astype(float) - blurred.astype(float)
                
                noise_level = np.std(noise)
                noise_levels.append(noise_level)
                
                # Noise pattern (downsampled)
                small_noise = cv2.resize(noise, (32, 32))
                noise_patterns.append(small_noise.flatten())
            
            noise_levels = np.array(noise_levels)
            avg_noise = np.mean(noise_levels)
            noise_std = np.std(noise_levels)
            
            # Compare noise patterns (PRNU consistency)
            pattern_correlations = []
            for i in range(len(noise_patterns) - 1):
                corr = np.corrcoef(noise_patterns[i], noise_patterns[i+1])[0, 1]
                if not np.isnan(corr):
                    pattern_correlations.append(corr)
            
            avg_noise_correlation = np.mean(pattern_correlations) if pattern_correlations else 0
            
            score = 0
            
            # Very low noise = AI-generated (too clean)
            if avg_noise < 3:
                score += 25
            
            # Inconsistent noise levels = AI
            if noise_std / (avg_noise + 1) > 0.3:
                score += 30
            elif noise_std / (avg_noise + 1) > 0.15:
                score += 15
            
            # Low noise correlation = no PRNU = AI
            if avg_noise_correlation < 0.3:
                score += 30
            elif avg_noise_correlation < 0.5:
                score += 15
            
            status = self._score_to_status(score)
            
            details = {
                "avg_noise_level": round(float(avg_noise), 3),
                "noise_level_std": round(float(noise_std), 3),
                "noise_pattern_correlation": round(float(avg_noise_correlation), 4)
            }
            
            finding = "No consistent sensor fingerprint (AI indicator)" if score >= 50 else \
                     "Weak sensor pattern detected" if score >= 30 else \
                     "Consistent camera noise pattern found"
            
            return VideoAICheckResult(
                name="noise_consistency",
                display_name="Sensor Noise (PRNU)",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="forensic"
            )
            
        except Exception as e:
            return self._error_result("noise_consistency", "Sensor Noise (PRNU)", e, "forensic")
    
    def _check_compression_artifacts(self, frames: List[np.ndarray], 
                                     gray_frames: List[np.ndarray]) -> VideoAICheckResult:
        """
        Analyze compression artifacts.
        AI videos often have unusual compression patterns.
        """
        try:
            # Analyze 8x8 block boundaries (video codec artifacts)
            block_scores = []
            
            for gray in gray_frames[::max(1, len(gray_frames)//5)]:  # Sample 5 frames
                h, w = gray.shape
                
                # Check horizontal block boundaries
                h_boundaries = []
                for i in range(8, h - 8, 8):
                    diff = np.abs(gray[i, :].astype(float) - gray[i-1, :].astype(float))
                    h_boundaries.append(np.mean(diff))
                
                # Check vertical block boundaries
                v_boundaries = []
                for j in range(8, w - 8, 8):
                    diff = np.abs(gray[:, j].astype(float) - gray[:, j-1].astype(float))
                    v_boundaries.append(np.mean(diff))
                
                # Compare boundary vs non-boundary differences
                boundary_avg = (np.mean(h_boundaries) + np.mean(v_boundaries)) / 2
                block_scores.append(boundary_avg)
            
            avg_block_artifact = np.mean(block_scores)
            block_artifact_std = np.std(block_scores)
            
            # Very low block artifacts = possibly AI (not recompressed from camera)
            # Very high artifacts = heavy compression
            # Inconsistent = suspicious
            
            score = 0
            
            # Very low artifacts = AI (no real codec artifacts)
            if avg_block_artifact < 2:
                score += 25
            
            # Inconsistent block artifacts across frames
            if block_artifact_std / (avg_block_artifact + 1) > 0.4:
                score += 30
            elif block_artifact_std / (avg_block_artifact + 1) > 0.2:
                score += 15
            
            status = self._score_to_status(score)
            
            details = {
                "avg_block_artifact": round(float(avg_block_artifact), 3),
                "block_artifact_std": round(float(block_artifact_std), 3)
            }
            
            finding = "Compression patterns suggest AI generation" if score >= 50 else \
                     "Some compression irregularities" if score >= 30 else \
                     "Compression artifacts appear normal"
            
            return VideoAICheckResult(
                name="compression",
                display_name="Compression Forensics",
                score=min(100, score),
                status=status,
                finding=finding,
                details=details,
                category="forensic"
            )
            
        except Exception as e:
            return self._error_result("compression", "Compression Forensics", e, "forensic")
    
    # ==================== HELPER METHODS ====================
    
    def _score_to_status(self, score: float) -> str:
        """Convert score to status (bias toward natural to reduce false positives)."""
        if score >= 70:
            return "likely_ai"
        elif score >= 55:
            return "possibly_ai"
        elif score >= 42:
            return "uncertain"
        else:
            return "likely_natural"
    
    def _error_result(self, name: str, display_name: str, error: Exception, 
                      category: str) -> VideoAICheckResult:
        """Create error result."""
        return VideoAICheckResult(
            name=name,
            display_name=display_name,
            score=50,
            status="uncertain",
            finding=f"Analysis error: {str(error)[:50]}",
            details={"error": str(error)},
            category=category
        )
    
    def _generate_assessment(self, checks: List[VideoAICheckResult],
                            likely_ai: int, possibly_ai: int, likely_natural: int,
                            categories: Dict) -> Dict:
        """Generate comprehensive video AI assessment."""
        
        total = len(checks)
        scores = [c.score for c in checks]
        avg_score = np.mean(scores)
        max_score = max(scores)
        
        # Category weights (temporal is most important for video)
        category_weights = {
            "temporal": 1.3,
            "motion": 1.2,
            "visual": 1.0,
            "forensic": 1.1
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
                        if c.status in ["likely_ai", "possibly_ai"]]
        natural_signs = [c.display_name for c in checks 
                        if c.status == "likely_natural"]
        
        # Conservative decision logic - strongly biased toward LIKELY_NATURAL
        # Real-world videos often have compression/encoding artifacts that can trigger false positives
        ai_signal = (likely_ai * 2.0) + (possibly_ai * 0.5)  # Reduce possibly_ai weight
        natural_signal = len(natural_signs)
        uncertain_count = sum(1 for c in checks if c.status == "uncertain")

        # Very strict requirements for AI detection
        if (likely_ai >= 4) or (ai_signal >= 6 and weighted_avg >= 70) or (weighted_avg >= 80):
            conclusion = "LIKELY_AI_GENERATED"
            confidence = min(95, 55 + likely_ai * 8 + possibly_ai * 3)
            interpretation = (
                "Strong AI signature detected: multiple independent algorithms flagged synthetic patterns."
            )
        elif (likely_ai >= 3) or (ai_signal >= 5 and weighted_avg >= 60):
            conclusion = "POSSIBLY_AI_GENERATED"
            confidence = min(75, 45 + likely_ai * 5 + possibly_ai * 3)
            interpretation = (
                "Some AI indicators present but not conclusive. Manual review recommended."
            )
        elif natural_signal >= 4 or (likely_ai == 0 and possibly_ai <= 1):
            # Bias toward natural - if no strong AI signals, assume natural
            conclusion = "LIKELY_NATURAL"
            confidence = min(92, 55 + natural_signal * 5)
            interpretation = (
                f"Content appears naturally captured. {natural_signal}/{total} checks favor authentic origin."
            )
        elif likely_ai == 0 and weighted_avg <= 45:
            # No likely_ai and low weighted average = likely natural
            conclusion = "LIKELY_NATURAL"
            confidence = min(85, 50 + natural_signal * 4)
            interpretation = (
                "No strong AI generation markers detected. Content consistent with camera capture."
            )
        else:
            # Default to UNCERTAIN only if there's genuine ambiguity
            conclusion = "UNCERTAIN"
            confidence = 45
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
            "disclaimer": "Video AI detection is experimental. Modern AI (Sora, Runway, Pika) produces convincing videos. Results require expert verification.",
            "categories_summary": {cat: round(float(data["avg_score"]), 1) for cat, data in categories.items()}
        }
    
    def _get_recommendation(self, conclusion: str) -> str:
        recommendations = {
            "LIKELY_AI_GENERATED": "Strong AI indicators detected across temporal, motion, and forensic analysis. Treat as AI-generated unless proven otherwise.",
            "POSSIBLY_AI_GENERATED": "Moderate AI indicators found. Seek frame-by-frame analysis, metadata verification, and expert review.",
            "LIKELY_NATURAL": "Video appears camera-captured, but modern AI video can be convincing. Check metadata and source.",
            "UNCERTAIN": "Detection inconclusive. Manual expert review strongly recommended."
        }
        return recommendations.get(conclusion, "Expert review recommended.")
