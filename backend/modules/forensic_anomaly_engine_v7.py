"""
FakeTrace v7.0 - Forensic Anomaly Detection Engine
===================================================
PHILOSOPHY: Anomaly-First, Evidence-Driven, Legally Safe

This is a CLASSICAL FORENSIC ASSISTANCE TOOL,
NOT a definitive AI detector.
Designed to SUPPORT investigators, NOT replace them.

Core Principles:
1. Never claim certainty when evidence is weak
2. If no anomaly detected → explicitly say so
3. Prioritize "what is unusual" over "fake vs real"
4. Better UNCERTAIN than wrong prediction
5. Each algorithm is independent forensic evidence
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


# ==========================================
# ENUMS & DATA CLASSES
# ==========================================

class AnomalyStatus(Enum):
    """Anomaly detection status for each algorithm"""
    DETECTED = "detected"           # Score >= 60: Clear anomaly found
    NOT_DETECTED = "not_detected"   # Score <= 45: No anomaly
    INCONCLUSIVE = "inconclusive"   # Score 46-59: Cannot determine


class VerdictType(Enum):
    """Final verdict categories"""
    NO_ANOMALY = "NO_FORENSIC_ANOMALY_DETECTED"
    ANOMALY_DETECTED = "FORENSIC_MANIPULATION_INDICATORS_DETECTED"
    INCONCLUSIVE = "INCONCLUSIVE_MINOR_IRREGULARITIES"


@dataclass
class AlgorithmResult:
    """Result from a single forensic algorithm"""
    name: str
    display_name: str
    score: float                    # 0-100
    anomaly_status: str             # detected/not_detected/inconclusive
    reason: str                     # Human-readable explanation
    details: Dict                   # Technical details
    
    @staticmethod
    def from_score(name: str, display_name: str, score: float, 
                   details: Dict, reason_detected: str, 
                   reason_not_detected: str) -> 'AlgorithmResult':
        """Create result with automatic anomaly classification"""
        
        if score >= 60:
            status = AnomalyStatus.DETECTED.value
            reason = reason_detected
        elif score <= 45:
            status = AnomalyStatus.NOT_DETECTED.value
            reason = reason_not_detected
        else:
            status = AnomalyStatus.INCONCLUSIVE.value
            reason = "Patterns observed but insufficient for reliable conclusion"
        
        return AlgorithmResult(
            name=name,
            display_name=display_name,
            score=round(score, 1),
            anomaly_status=status,
            reason=reason,
            details=details
        )


@dataclass
class ForensicVerdict:
    """Final verdict based on anomaly count"""
    verdict: str
    verdict_code: str
    interpretation: str
    anomaly_count: int
    inconclusive_count: int
    clean_count: int
    total_checks: int
    recommendation: str
    legal_disclaimer: str


# ==========================================
# MAIN FORENSIC ENGINE
# ==========================================

class ForensicAnomalyEngine:
    """
    Forensic Anomaly Detection Engine v7.0
    
    Key Differences from v6.0:
    - NO weighted averages as primary decision
    - Anomaly count based verdict
    - Each algorithm is independent evidence
    - Conservative, legally-safe language
    """
    
    VERSION = "7.0"
    
    # Thresholds
    ANOMALY_THRESHOLD = 60      # Score >= this = anomaly detected
    CLEAN_THRESHOLD = 45        # Score <= this = no anomaly
    MIN_ANOMALIES_FOR_DETECTION = 3  # Need at least 3 for "detected"
    
    # Video analysis
    VIDEO_FRAME_SAMPLE_COUNT = 12
    VIDEO_CONSISTENCY_THRESHOLD = 0.6  # 60% of frames must agree
    
    def __init__(self):
        print(f"  ✅ Forensic Anomaly Engine v{self.VERSION} initialized")
        print(f"     Mode: Anomaly-First Evidence-Driven Analysis")
        print(f"     Philosophy: Support investigators, not replace them")
    
    def analyze(self, file_path: str) -> Dict:
        """
        Main analysis entry point.
        
        Returns comprehensive forensic report with:
        - Individual algorithm evidence
        - Anomaly-based verdict
        - Legal-safe explanations
        """
        
        result = {
            "status": "processing",
            "version": self.VERSION,
            "analyzed_at": datetime.now().isoformat(),
            "file_path": file_path,
            
            # System identity
            "system_identity": {
                "name": "FakeTrace Forensic Assistance Tool",
                "type": "Classical Forensic Analysis",
                "purpose": "Support investigators, not replace them",
                "disclaimer": "Results are probabilistic, not definitive"
            },
            
            # Evidence section (per-algorithm)
            "evidence": {
                "algorithms": [],
                "summary": {
                    "total_checks": 0,
                    "anomalies_detected": 0,
                    "inconclusive": 0,
                    "clean": 0
                }
            },
            
            # Verdict section (based on anomaly count ONLY)
            "verdict": None,
            
            # Supporting statistics (SECONDARY, never overrides verdict)
            "supporting_statistics": {
                "note": "These are supporting metrics only. They do NOT determine the verdict.",
                "average_score": 0.0,
                "interpretation": ""
            },
            
            # Explanation section
            "explanation": {
                "summary": "",
                "findings": [],
                "limitations": [],
                "recommendation": ""
            },
            
            "errors": []
        }
        
        try:
            path = Path(file_path)
            is_video = path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            result["media_type"] = "video" if is_video else "image"
            
            if is_video:
                result = self._analyze_video(file_path, result)
            else:
                result = self._analyze_image(file_path, result)
            
            result["status"] = "completed"
            
        except Exception as e:
            result["errors"].append(str(e))
            result["status"] = "error"
            import traceback
            traceback.print_exc()
        
        return result
    
    def _analyze_image(self, file_path: str, result: Dict) -> Dict:
        """Analyze a single image"""
        
        img = cv2.imread(file_path)
        if img is None:
            result["errors"].append("Could not load image file")
            return result
        
        h, w = img.shape[:2]
        result["dimensions"] = {"width": w, "height": h}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Run all forensic algorithms
        algorithms = [
            self._analyze_noise(img, gray),
            self._analyze_compression(img, file_path),
            self._analyze_color(img),
            self._analyze_frequency(gray),
            self._analyze_texture(gray),
            self._analyze_edge(gray)
        ]
        
        # Store algorithm results
        result["evidence"]["algorithms"] = [asdict(a) for a in algorithms]
        
        # Count anomalies
        anomaly_count = sum(1 for a in algorithms if a.anomaly_status == AnomalyStatus.DETECTED.value)
        inconclusive_count = sum(1 for a in algorithms if a.anomaly_status == AnomalyStatus.INCONCLUSIVE.value)
        clean_count = sum(1 for a in algorithms if a.anomaly_status == AnomalyStatus.NOT_DETECTED.value)
        
        result["evidence"]["summary"] = {
            "total_checks": len(algorithms),
            "anomalies_detected": anomaly_count,
            "inconclusive": inconclusive_count,
            "clean": clean_count
        }
        
        # Calculate verdict based on anomaly count ONLY
        result["verdict"] = asdict(self._calculate_verdict(
            anomaly_count, inconclusive_count, clean_count, len(algorithms)
        ))
        
        # Supporting statistics (secondary only)
        avg_score = np.mean([a.score for a in algorithms])
        result["supporting_statistics"]["average_score"] = round(avg_score, 1)
        result["supporting_statistics"]["interpretation"] = (
            f"{avg_score:.0f}% average anomaly indication across {len(algorithms)} checks "
            "(supporting metric only, does not determine verdict)"
        )
        
        # Generate explanation
        result["explanation"] = self._generate_explanation(algorithms, result["verdict"])
        
        return result
    
    def _analyze_video(self, file_path: str, result: Dict) -> Dict:
        """
        Analyze video with multi-frame sampling.
        
        An anomaly is valid ONLY if it appears in >= 60% of sampled frames.
        """
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            result["errors"].append("Could not open video file")
            return result
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        result["video_info"] = {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 2),
            "frames_analyzed": 0
        }
        
        # Sample frames evenly distributed
        sample_indices = np.linspace(0, total_frames - 1, self.VIDEO_FRAME_SAMPLE_COUNT, dtype=int)
        
        # Store per-frame results for each algorithm
        frame_results = {
            "noise": [],
            "compression": [],
            "color": [],
            "frequency": [],
            "texture": [],
            "edge": []
        }
        
        frames_analyzed = 0
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Run algorithms on this frame
            frame_results["noise"].append(self._analyze_noise(frame, gray))
            frame_results["compression"].append(self._analyze_compression_frame(frame))
            frame_results["color"].append(self._analyze_color(frame))
            frame_results["frequency"].append(self._analyze_frequency(gray))
            frame_results["texture"].append(self._analyze_texture(gray))
            frame_results["edge"].append(self._analyze_edge(gray))
            
            frames_analyzed += 1
        
        cap.release()
        result["video_info"]["frames_analyzed"] = frames_analyzed
        
        # Aggregate results with consistency requirement
        algorithms = []
        for algo_name, results_list in frame_results.items():
            if not results_list:
                continue
            
            # Calculate consistency
            anomaly_frames = sum(1 for r in results_list if r.anomaly_status == AnomalyStatus.DETECTED.value)
            consistency_ratio = anomaly_frames / len(results_list)
            
            avg_score = np.mean([r.score for r in results_list])
            
            # Only mark as anomaly if consistent across frames
            if consistency_ratio >= self.VIDEO_CONSISTENCY_THRESHOLD and avg_score >= self.ANOMALY_THRESHOLD:
                status = AnomalyStatus.DETECTED.value
                reason = f"Anomaly detected consistently in {anomaly_frames}/{len(results_list)} frames ({consistency_ratio*100:.0f}%)"
            elif avg_score <= self.CLEAN_THRESHOLD:
                status = AnomalyStatus.NOT_DETECTED.value
                reason = "No consistent anomaly patterns across frames"
            else:
                status = AnomalyStatus.INCONCLUSIVE.value
                reason = f"Inconsistent patterns: anomaly in {anomaly_frames}/{len(results_list)} frames"
            
            algorithms.append(AlgorithmResult(
                name=algo_name,
                display_name=results_list[0].display_name if results_list else algo_name,
                score=round(avg_score, 1),
                anomaly_status=status,
                reason=reason,
                details={
                    "frames_with_anomaly": anomaly_frames,
                    "total_frames": len(results_list),
                    "consistency_ratio": round(consistency_ratio, 2),
                    "per_frame_scores": [round(r.score, 1) for r in results_list]
                }
            ))
        
        # Store algorithm results
        result["evidence"]["algorithms"] = [asdict(a) for a in algorithms]
        
        # Count anomalies
        anomaly_count = sum(1 for a in algorithms if a.anomaly_status == AnomalyStatus.DETECTED.value)
        inconclusive_count = sum(1 for a in algorithms if a.anomaly_status == AnomalyStatus.INCONCLUSIVE.value)
        clean_count = sum(1 for a in algorithms if a.anomaly_status == AnomalyStatus.NOT_DETECTED.value)
        
        result["evidence"]["summary"] = {
            "total_checks": len(algorithms),
            "anomalies_detected": anomaly_count,
            "inconclusive": inconclusive_count,
            "clean": clean_count
        }
        
        # Calculate verdict
        result["verdict"] = asdict(self._calculate_verdict(
            anomaly_count, inconclusive_count, clean_count, len(algorithms)
        ))
        
        # Supporting statistics
        if algorithms:
            avg_score = np.mean([a.score for a in algorithms])
            result["supporting_statistics"]["average_score"] = round(avg_score, 1)
            result["supporting_statistics"]["interpretation"] = (
                f"{avg_score:.0f}% average anomaly indication across {frames_analyzed} frames "
                "(supporting metric only)"
            )
        
        # Generate explanation
        result["explanation"] = self._generate_explanation(algorithms, result["verdict"])
        
        return result
    
    # ==========================================
    # FORENSIC ALGORITHMS
    # ==========================================
    
    def _analyze_noise(self, img: np.ndarray, gray: np.ndarray) -> AlgorithmResult:
        """
        Noise Consistency Analysis
        
        Real images: Natural noise patterns, consistent across image
        AI images: Unnaturally smooth OR inconsistent noise
        """
        
        details = {}
        
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
            
            details = {
                "overall_noise_level": round(noise_std, 2),
                "noise_consistency_ratio": round(consistency, 4),
                "blocks_analyzed": len(block_stds)
            }
            
            # Scoring
            if noise_std < 5:  # Unnaturally smooth
                score = 70
                reason_detected = "Unnaturally low noise level detected (common in AI-generated content)"
            elif consistency > 0.5:  # Inconsistent noise
                score = 65
                reason_detected = "Noise patterns are inconsistent across image regions"
            elif noise_std > 25:  # Natural high noise
                score = 30
                reason_detected = ""
            else:
                score = 45
                reason_detected = ""
            
            reason_not_detected = "Noise patterns appear natural and consistent"
            
        except Exception as e:
            score = 50
            details["error"] = str(e)
            reason_detected = "Analysis error"
            reason_not_detected = "Analysis error"
        
        return AlgorithmResult.from_score(
            name="noise",
            display_name="Noise Consistency Analysis",
            score=score,
            details=details,
            reason_detected=reason_detected,
            reason_not_detected=reason_not_detected
        )
    
    def _analyze_compression(self, img: np.ndarray, file_path: str) -> AlgorithmResult:
        """Compression Artifact Analysis"""
        
        details = {}
        
        try:
            ext = Path(file_path).suffix.lower()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            h, w = gray.shape
            block_diffs = []
            
            for y in range(8, h - 8, 8):
                for x in range(8, w - 8, 8):
                    diff = abs(int(gray[y-1, x]) - int(gray[y, x]))
                    block_diffs.append(diff)
            
            avg_block_diff = np.mean(block_diffs) if block_diffs else 0
            
            file_size = os.path.getsize(file_path)
            pixels = h * w
            bytes_per_pixel = file_size / (pixels + 1)
            
            details = {
                "format": ext,
                "block_artifact_level": round(avg_block_diff, 2),
                "bytes_per_pixel": round(bytes_per_pixel, 4),
                "file_size_kb": round(file_size / 1024, 1)
            }
            
            if avg_block_diff > 15:
                score = 60
                reason_detected = "Strong compression artifacts suggest possible re-encoding or editing"
            elif avg_block_diff < 2 and ext in ['.jpg', '.jpeg']:
                score = 55
                reason_detected = "Unusually clean for JPEG format"
            else:
                score = 42
                reason_detected = ""
            
            reason_not_detected = "Compression artifacts appear normal for format"
            
        except Exception as e:
            score = 50
            details["error"] = str(e)
            reason_detected = "Analysis error"
            reason_not_detected = "Analysis error"
        
        return AlgorithmResult.from_score(
            name="compression",
            display_name="Compression Artifact Analysis",
            score=score,
            details=details,
            reason_detected=reason_detected,
            reason_not_detected=reason_not_detected
        )
    
    def _analyze_compression_frame(self, frame: np.ndarray) -> AlgorithmResult:
        """Compression analysis for video frame (no file path)"""
        
        details = {}
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            block_diffs = []
            
            for y in range(8, h - 8, 8):
                for x in range(8, w - 8, 8):
                    diff = abs(int(gray[y-1, x]) - int(gray[y, x]))
                    block_diffs.append(diff)
            
            avg_block_diff = np.mean(block_diffs) if block_diffs else 0
            
            details = {"block_artifact_level": round(avg_block_diff, 2)}
            
            if avg_block_diff > 15:
                score = 60
                reason_detected = "Strong compression artifacts detected"
            else:
                score = 42
                reason_detected = ""
            
            reason_not_detected = "Compression artifacts appear normal"
            
        except Exception as e:
            score = 50
            details["error"] = str(e)
            reason_detected = "Analysis error"
            reason_not_detected = "Analysis error"
        
        return AlgorithmResult.from_score(
            name="compression",
            display_name="Compression Artifact Analysis",
            score=score,
            details=details,
            reason_detected=reason_detected,
            reason_not_detected=reason_not_detected
        )
    
    def _analyze_color(self, img: np.ndarray) -> AlgorithmResult:
        """Color Distribution Analysis"""
        
        details = {}
        
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            saturation = hsv[:, :, 1]
            sat_mean = np.mean(saturation)
            sat_std = np.std(saturation)
            
            value = hsv[:, :, 2]
            val_mean = np.mean(value)
            val_std = np.std(value)
            
            color_uniformity = sat_std / (sat_mean + 1)
            
            details = {
                "saturation_mean": round(sat_mean, 2),
                "saturation_std": round(sat_std, 2),
                "brightness_mean": round(val_mean, 2),
                "brightness_std": round(val_std, 2),
                "color_uniformity_ratio": round(color_uniformity, 4)
            }
            
            if color_uniformity < 0.1:
                score = 62
                reason_detected = "Unusually uniform color distribution (uncommon in natural photos)"
            elif sat_mean > 200:
                score = 58
                reason_detected = "Oversaturation detected"
            else:
                score = 40
                reason_detected = ""
            
            reason_not_detected = "Color distribution appears natural"
            
        except Exception as e:
            score = 50
            details["error"] = str(e)
            reason_detected = "Analysis error"
            reason_not_detected = "Analysis error"
        
        return AlgorithmResult.from_score(
            name="color",
            display_name="Color Distribution Analysis",
            score=score,
            details=details,
            reason_detected=reason_detected,
            reason_not_detected=reason_not_detected
        )
    
    def _analyze_frequency(self, gray: np.ndarray) -> AlgorithmResult:
        """Frequency Domain Analysis (FFT)"""
        
        details = {}
        
        try:
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 4
            
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            low_mask = dist <= radius
            high_mask = dist > radius * 2
            
            low_energy = np.mean(magnitude[low_mask])
            high_energy = np.mean(magnitude[high_mask])
            
            freq_ratio = high_energy / (low_energy + 0.001)
            
            details = {
                "low_frequency_energy": round(low_energy, 2),
                "high_frequency_energy": round(high_energy, 2),
                "high_to_low_ratio": round(freq_ratio, 4)
            }
            
            if freq_ratio < 0.3:
                score = 68
                reason_detected = "Low high-frequency content detected (AI-generated images often lack fine details)"
            elif freq_ratio > 0.7:
                score = 32
                reason_detected = ""
            else:
                score = 48
                reason_detected = ""
            
            reason_not_detected = "Frequency distribution appears natural with expected detail levels"
            
        except Exception as e:
            score = 50
            details["error"] = str(e)
            reason_detected = "Analysis error"
            reason_not_detected = "Analysis error"
        
        return AlgorithmResult.from_score(
            name="frequency",
            display_name="Frequency Domain Analysis",
            score=score,
            details=details,
            reason_detected=reason_detected,
            reason_not_detected=reason_not_detected
        )
    
    def _analyze_texture(self, gray: np.ndarray) -> AlgorithmResult:
        """Texture Smoothness Analysis"""
        
        details = {}
        
        try:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            
            texture_strength = np.mean(gradient_mag)
            texture_std = np.std(gradient_mag)
            texture_ratio = texture_std / (texture_strength + 1)
            
            details = {
                "texture_strength": round(texture_strength, 2),
                "texture_variation": round(texture_std, 2),
                "texture_ratio": round(texture_ratio, 4)
            }
            
            if texture_strength < 10:
                score = 67
                reason_detected = "Unnaturally smooth texture detected (common in AI-generated content)"
            elif texture_strength > 50:
                score = 30
                reason_detected = ""
            else:
                score = 44
                reason_detected = ""
            
            reason_not_detected = "Texture patterns appear natural with expected variations"
            
        except Exception as e:
            score = 50
            details["error"] = str(e)
            reason_detected = "Analysis error"
            reason_not_detected = "Analysis error"
        
        return AlgorithmResult.from_score(
            name="texture",
            display_name="Texture Smoothness Analysis",
            score=score,
            details=details,
            reason_detected=reason_detected,
            reason_not_detected=reason_not_detected
        )
    
    def _analyze_edge(self, gray: np.ndarray) -> AlgorithmResult:
        """Edge Coherence Analysis"""
        
        details = {}
        
        try:
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_score = np.var(laplacian)
            
            details = {
                "edge_density": round(edge_density, 4),
                "focus_variance": round(focus_score, 2)
            }
            
            if edge_density < 0.02:
                score = 63
                reason_detected = "Very low edge density (may indicate AI smoothing)"
            elif focus_score < 100:
                score = 58
                reason_detected = "Low focus variance detected"
            elif edge_density > 0.15:
                score = 32
                reason_detected = ""
            else:
                score = 43
                reason_detected = ""
            
            reason_not_detected = "Edge patterns appear natural with expected detail"
            
        except Exception as e:
            score = 50
            details["error"] = str(e)
            reason_detected = "Analysis error"
            reason_not_detected = "Analysis error"
        
        return AlgorithmResult.from_score(
            name="edge",
            display_name="Edge Coherence Analysis",
            score=score,
            details=details,
            reason_detected=reason_detected,
            reason_not_detected=reason_not_detected
        )
    
    # ==========================================
    # VERDICT CALCULATION (ANOMALY-BASED ONLY)
    # ==========================================
    
    def _calculate_verdict(self, anomaly_count: int, inconclusive_count: int, 
                          clean_count: int, total: int) -> ForensicVerdict:
        """
        Calculate verdict based on ANOMALY COUNT ONLY.
        
        This is the core decision logic - NO weighted averages.
        """
        
        if anomaly_count == 0:
            verdict = VerdictType.NO_ANOMALY.value
            interpretation = (
                "This media does not show statistical signs of AI generation or manipulation. "
                "All forensic checks passed without detecting anomalies."
            )
            recommendation = (
                "No forensic irregularities found. If authenticity concerns remain, "
                "consider additional context verification or expert review."
            )
        
        elif anomaly_count >= self.MIN_ANOMALIES_FOR_DETECTION:
            verdict = VerdictType.ANOMALY_DETECTED.value
            interpretation = (
                f"Multiple independent forensic checks ({anomaly_count} of {total}) "
                "detected suspicious statistical patterns. This does not confirm "
                "manipulation but indicates irregularities warrant further investigation."
            )
            recommendation = (
                "Forensic anomalies detected. Recommend expert review, "
                "source verification, and additional contextual analysis."
            )
        
        else:
            verdict = VerdictType.INCONCLUSIVE.value
            interpretation = (
                f"Some irregularities were observed ({anomaly_count} anomaly, "
                f"{inconclusive_count} inconclusive), but not enough for a reliable conclusion. "
                "The evidence is insufficient for a definitive assessment."
            )
            recommendation = (
                "Results are inconclusive. Additional evidence or expert "
                "analysis may be needed for a reliable determination."
            )
        
        legal_disclaimer = (
            "DISCLAIMER: This analysis is probabilistic, not definitive. "
            "Results should be interpreted by qualified professionals. "
            "This tool supports investigators but does not provide legal proof."
        )
        
        return ForensicVerdict(
            verdict=verdict,
            verdict_code=verdict.replace("_", " "),
            interpretation=interpretation,
            anomaly_count=anomaly_count,
            inconclusive_count=inconclusive_count,
            clean_count=clean_count,
            total_checks=total,
            recommendation=recommendation,
            legal_disclaimer=legal_disclaimer
        )
    
    def _generate_explanation(self, algorithms: List[AlgorithmResult], 
                             verdict: Dict) -> Dict:
        """Generate human-readable explanation"""
        
        findings = []
        
        # Group by status
        anomalies = [a for a in algorithms if a.anomaly_status == AnomalyStatus.DETECTED.value]
        inconclusive = [a for a in algorithms if a.anomaly_status == AnomalyStatus.INCONCLUSIVE.value]
        clean = [a for a in algorithms if a.anomaly_status == AnomalyStatus.NOT_DETECTED.value]
        
        if anomalies:
            findings.append(f"❌ {len(anomalies)} algorithm(s) detected anomalies:")
            for a in anomalies:
                findings.append(f"   • {a.display_name}: {a.reason}")
        
        if inconclusive:
            findings.append(f"⚠️ {len(inconclusive)} algorithm(s) returned inconclusive results:")
            for a in inconclusive:
                findings.append(f"   • {a.display_name}: {a.reason}")
        
        if clean:
            findings.append(f"✅ {len(clean)} algorithm(s) found no anomalies:")
            for a in clean:
                findings.append(f"   • {a.display_name}: {a.reason}")
        
        limitations = [
            "This analysis uses classical forensic methods, not deep learning",
            "Modern AI-generated content may evade detection",
            "Results are statistical indications, not proof",
            "High compression can mask or create artifacts",
            "Some natural photos may trigger false positives"
        ]
        
        summary = verdict.get("interpretation", "")
        
        return {
            "summary": summary,
            "findings": findings,
            "limitations": limitations,
            "recommendation": verdict.get("recommendation", "")
        }


# ==========================================
# WRAPPER FOR BACKWARD COMPATIBILITY
# ==========================================

class AdvancedVisualForensics:
    """Wrapper for backward compatibility with main.py"""
    
    def __init__(self, model_path: str = None):
        self.engine = ForensicAnomalyEngine()
        self.version = ForensicAnomalyEngine.VERSION
    
    def analyze(self, file_path: str) -> Dict:
        result = self.engine.analyze(file_path)
        
        # Add compatibility fields for existing frontend
        result["scores"] = {}
        result["prediction"] = {
            "class": 0,
            "class_name": result.get("verdict", {}).get("verdict", "UNKNOWN"),
            "confidence": result.get("supporting_statistics", {}).get("average_score", 0),
            "distribution": {"real": 50, "ai_generated": 50, "ai_enhanced": 0},
            "is_uncertain": result.get("verdict", {}).get("verdict") == VerdictType.INCONCLUSIVE.value
        }
        
        # Map algorithms to scores format
        for algo in result.get("evidence", {}).get("algorithms", []):
            result["scores"][f"{algo['name']}_score"] = algo["score"]
        
        return result


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Forensic Anomaly Detection Engine v7.0")
    print("=" * 60)
    print("Mode: Anomaly-First Evidence-Driven Analysis")
    print("Philosophy: Support investigators, not replace them")
    print("=" * 60)
    
    engine = ForensicAnomalyEngine()
