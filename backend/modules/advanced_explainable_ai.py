"""
Ultra-Advanced Explainable AI System v3.0
Detailed Scientific Explanations with Court-Ready Reports
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class AdvancedExplainableAI:
    """
    Ultra-Advanced Explainable AI
    
    Features:
    - Simple explanations (for everyone)
    - Technical explanations (for experts)
    - Legal statements (for court)
    - AI visual analysis (Google Gemini)
    - Evidence-based reasoning
    - Red flag detection
    - Counter-arguments
    - Recommendations
    """
    
    def __init__(self):
        self.genai_key = os.getenv("GOOGLE_GENAI_API_KEY")
        self.genai_enabled = False
        
        if GENAI_AVAILABLE and self.genai_key:
            try:
                genai.configure(api_key=self.genai_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.genai_enabled = True
            except:
                pass
    
    def explain(self, visual_forensics: dict, file_path: str = None, db_info: dict = None) -> dict:
        """Generate comprehensive explanation"""
        result = {
            "explanation_time": datetime.now().isoformat(),
            "status": "completed",
            "version": "3.0",
            
            "verdict": {
                "conclusion": "",
                "confidence": 0,
                "confidence_level": "",
                "summary": "",
            },
            
            "evidence": {
                "supporting_manipulation": [],
                "supporting_authenticity": [],
                "inconclusive": [],
            },
            
            "findings": [],
            "red_flags": [],
            "counter_arguments": [],
            "confidence_breakdown": {},
            
            "explanations": {
                "simple": "",
                "technical": "",
                "legal": "",
            },
            
            "ai_analysis": None,
            "history_context": {},
            "recommendations": [],
            "errors": []
        }
        
        try:
            if not visual_forensics:
                result["errors"].append("No visual forensics data")
                return result
            
            # Collect evidence
            result["evidence"] = self._collect_evidence(visual_forensics)
            
            # Generate findings
            result["findings"] = self._generate_findings(visual_forensics)
            
            # Identify red flags
            result["red_flags"] = self._identify_red_flags(visual_forensics)
            
            # Counter arguments
            result["counter_arguments"] = self._generate_counter_arguments(result["findings"])
            
            # Confidence breakdown
            result["confidence_breakdown"] = self._get_confidence_breakdown(visual_forensics)
            
            # Determine verdict
            result["verdict"] = self._determine_verdict(result, visual_forensics)
            
            # Generate explanations
            result["explanations"]["simple"] = self._generate_simple_explanation(result)
            result["explanations"]["technical"] = self._generate_technical_explanation(visual_forensics, result)
            result["explanations"]["legal"] = self._generate_legal_statement(result, visual_forensics, db_info)
            
            # AI Visual Analysis
            if self.genai_enabled and file_path and Path(file_path).exists():
                result["ai_analysis"] = self._get_ai_analysis(file_path, result["verdict"])
            
            # History context
            if db_info:
                result["history_context"] = {
                    "is_repeat": db_info.get("analysis_count", 1) > 1,
                    "times_analyzed": db_info.get("analysis_count", 1),
                    "first_seen": db_info.get("first_seen")
                }
            
            # Recommendations
            result["recommendations"] = self._generate_recommendations(result)
            
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    def _collect_evidence(self, vf: dict) -> dict:
        """Collect and categorize evidence"""
        evidence = {
            "supporting_manipulation": [],
            "supporting_authenticity": [],
            "inconclusive": [],
        }
        
        scores = vf.get("scores", {})
        analysis = vf.get("analysis", {})
        
        # ELA Evidence
        ela_score = scores.get("ela_score", 0)
        ela_details = analysis.get("ela", {})
        
        if ela_score > 50:
            evidence["supporting_manipulation"].append({
                "category": "Error Level Analysis",
                "score": ela_score,
                "finding": f"High compression inconsistency detected (Score: {ela_score:.1f}%)",
                "details": f"ELA mean: {ela_details.get('ela_mean', 0):.2f}, "
                          f"Suspicious pixels: {ela_details.get('suspicious_pixels_percentage', 0):.2f}%",
                "severity": "high" if ela_score > 70 else "medium",
                "explanation": "Different parts of the image show different compression levels, "
                              "indicating they may have been edited or added later."
            })
        elif ela_score < 25:
            evidence["supporting_authenticity"].append({
                "category": "Error Level Analysis",
                "finding": "Consistent compression levels across image",
                "score": ela_score,
                "severity": "low"
            })
        else:
            evidence["inconclusive"].append({
                "category": "Error Level Analysis",
                "finding": f"Moderate ELA score ({ela_score:.1f}%)",
                "score": ela_score
            })
        
        # Clone Detection Evidence
        clone_score = scores.get("clone_score", 0)
        clone_details = analysis.get("clone", {})
        clone_pairs = clone_details.get("clone_pairs_found", 0)
        
        if clone_score > 30 or clone_pairs > 5:
            evidence["supporting_manipulation"].append({
                "category": "Clone Detection",
                "score": clone_score,
                "finding": f"Copy-paste manipulation detected ({clone_pairs} similar regions found)",
                "details": f"ORB feature matching found {clone_pairs} duplicate areas",
                "severity": "high" if clone_pairs > 15 else "medium",
                "explanation": "Parts of the image appear to be duplicated, a common technique "
                              "to hide or replicate content."
            })
        else:
            evidence["supporting_authenticity"].append({
                "category": "Clone Detection",
                "finding": "No significant copy-paste regions detected",
                "score": clone_score,
                "severity": "low"
            })
        
        # Noise Evidence
        noise_score = scores.get("noise_score", 0)
        noise_details = analysis.get("noise", {})
        
        if noise_score > 40:
            evidence["supporting_manipulation"].append({
                "category": "Noise Analysis",
                "score": noise_score,
                "finding": f"Inconsistent noise patterns (Score: {noise_score:.1f}%)",
                "details": f"Noise variance: {noise_details.get('noise_variance', 0):.4f}",
                "severity": "medium",
                "explanation": "Different parts of the image have different noise characteristics, "
                              "suggesting they may come from different sources."
            })
        
        # Edge Evidence
        edge_score = scores.get("edge_score", 0)
        if edge_score > 45:
            evidence["supporting_manipulation"].append({
                "category": "Edge Analysis",
                "score": edge_score,
                "finding": f"Unnatural edge boundaries detected (Score: {edge_score:.1f}%)",
                "severity": "medium",
                "explanation": "Some edges in the image show unnatural patterns that could indicate "
                              "splicing or object insertion."
            })
        
        # Compression Evidence
        comp_score = scores.get("compression_score", 0)
        comp_details = analysis.get("compression", {})
        
        if comp_details.get("double_compression_detected"):
            evidence["supporting_manipulation"].append({
                "category": "Compression Analysis",
                "score": comp_score,
                "finding": "Double JPEG compression detected",
                "severity": "medium",
                "explanation": "The image has been saved as JPEG multiple times, which often "
                              "happens during editing."
            })
        
        return evidence
    
    def _generate_findings(self, vf: dict) -> list:
        """Generate detailed findings"""
        findings = []
        scores = vf.get("scores", {})
        analysis = vf.get("analysis", {})
        
        categories = [
            ("ela", "Error Level Analysis", "Compression consistency"),
            ("noise", "Noise Analysis", "Noise pattern uniformity"),
            ("edge", "Edge Analysis", "Edge consistency"),
            ("clone", "Clone Detection", "Copy-move detection"),
            ("frequency", "Frequency Analysis", "Frequency domain"),
            ("color", "Color Analysis", "Color consistency"),
            ("compression", "Compression Analysis", "JPEG artifacts"),
            ("statistical", "Statistical Analysis", "Distribution analysis"),
        ]
        
        for key, name, desc in categories:
            score = scores.get(f"{key}_score", 0)
            details = analysis.get(key, {})
            algorithms = details.get("algorithms", [])
            
            is_suspicious = score > 40
            
            findings.append({
                "category": key,
                "name": name,
                "description": desc,
                "score": round(score, 2),
                "is_suspicious": is_suspicious,
                "algorithms_used": len(algorithms),
                "algorithm_details": algorithms[:3],  # Top 3
                "interpretation": self._interpret_score(key, score, details)
            })
        
        findings.sort(key=lambda x: x["score"], reverse=True)
        return findings
    
    def _interpret_score(self, category: str, score: float, details: dict) -> str:
        """Generate human-readable interpretation"""
        interpretations = {
            "ela": {
                "high": f"Strong evidence of editing. ELA reveals compression inconsistencies "
                       f"({details.get('suspicious_pixels_percentage', 0):.1f}% suspicious pixels).",
                "medium": "Moderate ELA variations detected. Some areas may have been modified.",
                "low": "ELA shows consistent compression. No obvious editing detected."
            },
            "noise": {
                "high": f"Significant noise inconsistencies. Different image regions show "
                       f"different noise patterns (variance: {details.get('noise_variance', 0):.4f}).",
                "medium": "Some noise variation detected across image regions.",
                "low": "Noise patterns are consistent throughout the image."
            },
            "clone": {
                "high": f"Copy-paste manipulation detected! {details.get('clone_pairs_found', 0)} "
                       f"duplicate regions found.",
                "medium": "Some similar regions detected that may indicate cloning.",
                "low": "No significant duplicate regions found."
            },
            "edge": {
                "high": "Unnatural edge patterns suggest possible splicing or insertion.",
                "medium": "Some edge inconsistencies detected.",
                "low": "Edge patterns appear natural and consistent."
            },
            "frequency": {
                "high": "Frequency spectrum shows anomalies indicating manipulation.",
                "medium": "Some frequency irregularities detected.",
                "low": "Frequency patterns appear normal."
            },
            "color": {
                "high": "Color temperature varies significantly across the image.",
                "medium": "Some color inconsistencies detected.",
                "low": "Color distribution is consistent."
            },
            "compression": {
                "high": "Strong JPEG artifacts suggest multiple compression cycles.",
                "medium": "Some compression artifacts detected.",
                "low": "Compression artifacts are minimal."
            },
            "statistical": {
                "high": "Statistical distribution deviates from natural image patterns.",
                "medium": "Some statistical anomalies detected.",
                "low": "Statistical distribution matches natural patterns."
            }
        }
        
        level = "high" if score > 60 else "medium" if score > 30 else "low"
        return interpretations.get(category, {}).get(level, "No interpretation available.")
    
    def _identify_red_flags(self, vf: dict) -> list:
        """Identify critical red flags"""
        red_flags = []
        scores = vf.get("scores", {})
        detections = vf.get("detections", {})
        analysis = vf.get("analysis", {})
        
        overall = scores.get("overall_score", 0)
        
        # High overall score
        if overall >= 70:
            red_flags.append({
                "flag": f"Very high manipulation score: {overall:.1f}%",
                "severity": "critical",
                "explanation": "Multiple forensic indicators strongly suggest this content has been manipulated."
            })
        elif overall >= 50:
            red_flags.append({
                "flag": f"Elevated manipulation score: {overall:.1f}%",
                "severity": "high",
                "explanation": "Several forensic indicators suggest possible manipulation."
            })
        
        # Clone detection
        clone_pairs = analysis.get("clone", {}).get("clone_pairs_found", 0)
        if clone_pairs > 10:
            red_flags.append({
                "flag": f"Significant copy-paste detected: {clone_pairs} duplicate regions",
                "severity": "critical",
                "explanation": "Multiple areas of the image appear to be copied and pasted."
            })
        
        # Critical regions
        critical = [r for r in detections.get("manipulated_regions", []) if r.get("severity") == "critical"]
        if critical:
            red_flags.append({
                "flag": f"{len(critical)} critical manipulation region(s) identified",
                "severity": "critical",
                "explanation": "Specific areas show very strong signs of tampering."
            })
        
        # High ELA
        if scores.get("ela_score", 0) > 60:
            red_flags.append({
                "flag": "Major compression inconsistencies in ELA",
                "severity": "high",
                "explanation": "Error Level Analysis reveals significant editing traces."
            })
        
        # Double compression
        if analysis.get("compression", {}).get("double_compression_detected"):
            red_flags.append({
                "flag": "Double JPEG compression detected",
                "severity": "medium",
                "explanation": "Image has been saved multiple times, common in edited images."
            })
        
        # Large manipulation area
        manip_pct = detections.get("manipulation_percentage", 0)
        if manip_pct > 30:
            red_flags.append({
                "flag": f"{manip_pct:.1f}% of image shows manipulation signs",
                "severity": "high" if manip_pct > 50 else "medium",
                "explanation": "A significant portion of the image appears altered."
            })
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        red_flags.sort(key=lambda x: severity_order.get(x["severity"], 4))
        
        return red_flags
    
    def _generate_counter_arguments(self, findings: list) -> list:
        """Generate counter-arguments"""
        counter_args = []
        
        counter_templates = {
            "ela": [
                "ELA differences can occur from legitimate image processing like brightness adjustments",
                "Social media compression can cause ELA artifacts without malicious editing",
                "Different camera processing modes can create ELA variations"
            ],
            "noise": [
                "Camera sensor noise naturally varies with lighting conditions",
                "High ISO settings cause non-uniform noise patterns",
                "Image compression affects noise characteristics"
            ],
            "clone": [
                "Natural repetitive patterns (tiles, fabric) trigger false positives",
                "Panorama stitching creates duplicate regions legitimately",
                "Some textures naturally contain similar patterns"
            ],
            "edge": [
                "Motion blur creates natural edge variations",
                "JPEG compression affects edges differently",
                "Depth of field causes edge inconsistencies"
            ],
            "color": [
                "Mixed lighting causes natural color temperature variations",
                "Camera white balance issues create color differences",
                "Legitimate color grading affects regions differently"
            ]
        }
        
        for finding in findings:
            if finding["is_suspicious"] and finding["category"] in counter_templates:
                for counter in counter_templates[finding["category"]][:2]:
                    counter_args.append({
                        "against": finding["name"],
                        "argument": counter,
                        "strength": "moderate"
                    })
        
        return counter_args
    
    def _get_confidence_breakdown(self, vf: dict) -> dict:
        """Get confidence for each category"""
        scores = vf.get("scores", {})
        return {
            "ela": round(scores.get("ela_score", 0), 2),
            "noise": round(scores.get("noise_score", 0), 2),
            "edge": round(scores.get("edge_score", 0), 2),
            "clone": round(scores.get("clone_score", 0), 2),
            "frequency": round(scores.get("frequency_score", 0), 2),
            "color": round(scores.get("color_score", 0), 2),
            "compression": round(scores.get("compression_score", 0), 2),
            "statistical": round(scores.get("statistical_score", 0), 2),
            "overall": round(scores.get("overall_score", 0), 2)
        }
    
    def _determine_verdict(self, result: dict, vf: dict) -> dict:
        """Determine final verdict"""
        verdict = {
            "conclusion": "INCONCLUSIVE",
            "confidence": 50,
            "confidence_level": "Low",
            "summary": ""
        }
        
        overall = vf.get("scores", {}).get("overall_score", 0)
        severity = vf.get("detections", {}).get("severity", "none")
        critical_flags = len([f for f in result.get("red_flags", []) if f["severity"] == "critical"])
        high_flags = len([f for f in result.get("red_flags", []) if f["severity"] == "high"])
        manip_evidence = len(result.get("evidence", {}).get("supporting_manipulation", []))
        
        if critical_flags > 0 or overall >= 70 or severity == "critical":
            verdict["conclusion"] = "LIKELY MANIPULATED"
            verdict["confidence"] = min(95, overall + 15)
        elif high_flags >= 2 or overall >= 50 or severity == "high":
            verdict["conclusion"] = "POSSIBLY MANIPULATED"
            verdict["confidence"] = overall + 5
        elif overall < 25 and manip_evidence == 0:
            verdict["conclusion"] = "LIKELY AUTHENTIC"
            verdict["confidence"] = min(90, 100 - overall)
        else:
            verdict["conclusion"] = "INCONCLUSIVE"
            verdict["confidence"] = 50
        
        # Confidence level
        if verdict["confidence"] >= 85:
            verdict["confidence_level"] = "Very High"
        elif verdict["confidence"] >= 70:
            verdict["confidence_level"] = "High"
        elif verdict["confidence"] >= 55:
            verdict["confidence_level"] = "Moderate"
        else:
            verdict["confidence_level"] = "Low"
        
        # Summary
        if verdict["conclusion"] == "LIKELY MANIPULATED":
            verdict["summary"] = (
                f"Based on {manip_evidence} manipulation indicators and {len(result.get('red_flags', []))} red flags, "
                f"this content shows strong signs of digital alteration. "
                f"Analysis confidence: {verdict['confidence_level']} ({verdict['confidence']:.0f}%)."
            )
        elif verdict["conclusion"] == "POSSIBLY MANIPULATED":
            verdict["summary"] = (
                f"Analysis found {manip_evidence} potential indicators of manipulation. "
                f"While not definitive, several methods suggest possible tampering."
            )
        elif verdict["conclusion"] == "LIKELY AUTHENTIC":
            verdict["summary"] = (
                f"No significant manipulation patterns detected. "
                f"The content appears to be genuine with {verdict['confidence']:.0f}% confidence."
            )
        else:
            verdict["summary"] = (
                f"Analysis results are mixed. Additional verification is recommended."
            )
        
        return verdict
    
    def _generate_simple_explanation(self, result: dict) -> str:
        """Generate simple explanation"""
        verdict = result.get("verdict", {})
        red_flags = result.get("red_flags", [])
        
        conclusion = verdict.get("conclusion", "UNKNOWN")
        confidence = verdict.get("confidence", 0)
        
        if conclusion == "LIKELY MANIPULATED":
            emoji = "🔴"
            status = "FAKE or EDITED"
            action = "Do NOT share this content without verification."
        elif conclusion == "POSSIBLY MANIPULATED":
            emoji = "🟠"
            status = "POSSIBLY EDITED"
            action = "Verify from original sources before sharing."
        elif conclusion == "LIKELY AUTHENTIC":
            emoji = "🟢"
            status = "LIKELY REAL"
            action = "No manipulation detected."
        else:
            emoji = "🟡"
            status = "UNCLEAR"
            action = "Get a second opinion."
        
        lines = [
            f"{emoji} THIS CONTENT IS {status}",
            "",
            f"We are {confidence:.0f}% confident about this.",
            ""
        ]
        
        if red_flags:
            lines.append("⚠️ WARNING SIGNS:")
            for flag in red_flags[:5]:
                lines.append(f"  • [{flag['severity'].upper()}] {flag['flag']}")
            lines.append("")
        
        lines.append(f"💡 RECOMMENDATION: {action}")
        
        return "\n".join(lines)
    
    def _generate_technical_explanation(self, vf: dict, result: dict) -> str:
        """Generate technical explanation"""
        lines = ["=" * 60]
        lines.append("TECHNICAL FORENSIC ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Scores
        scores = vf.get("scores", {})
        lines.append("ANALYSIS SCORES:")
        lines.append("-" * 40)
        for key, value in scores.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}%")
        lines.append("")
        
        # Findings
        lines.append("DETAILED FINDINGS:")
        lines.append("-" * 40)
        for finding in result.get("findings", [])[:8]:
            status = "⚠️ SUSPICIOUS" if finding["is_suspicious"] else "✓ OK"
            lines.append(f"  [{status}] {finding['name']}: {finding['score']:.1f}%")
            lines.append(f"      → {finding['interpretation'][:80]}...")
        lines.append("")
        
        # Algorithms
        lines.append("ALGORITHMS USED:")
        lines.append("-" * 40)
        analysis = vf.get("analysis", {})
        for cat, data in analysis.items():
            algos = data.get("algorithms", [])
            if algos:
                lines.append(f"  {cat.upper()}: {len(algos)} algorithms")
                for algo in algos[:2]:
                    lines.append(f"    - {algo.get('name', 'Unknown')}")
        lines.append("")
        
        # Regions
        regions = vf.get("detections", {}).get("manipulated_regions", [])
        if regions:
            lines.append(f"MANIPULATED REGIONS: {len(regions)}")
            lines.append("-" * 40)
            for i, region in enumerate(regions[:5], 1):
                lines.append(f"  Region {i}: ({region['x']}, {region['y']}) "
                           f"Size: {region['width']}x{region['height']} "
                           f"Severity: {region.get('severity', 'unknown')}")
        
        return "\n".join(lines)
    
    def _generate_legal_statement(self, result: dict, vf: dict, db_info: dict = None) -> str:
        """Generate court-ready legal statement"""
        lines = []
        lines.append("=" * 70)
        lines.append("DIGITAL FORENSIC ANALYSIS CERTIFICATE")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Report ID: {db_info.get('content_id', 'N/A') if db_info else 'N/A'}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 70)
        
        verdict = result.get("verdict", {})
        lines.append(f"Conclusion: {verdict.get('conclusion', 'N/A')}")
        lines.append(f"Confidence: {verdict.get('confidence', 0):.0f}% ({verdict.get('confidence_level', 'N/A')})")
        lines.append("")
        lines.append(verdict.get("summary", ""))
        lines.append("")
        
        # Evidence
        evidence = result.get("evidence", {})
        lines.append("-" * 70)
        lines.append("EVIDENCE SUMMARY")
        lines.append("-" * 70)
        lines.append(f"Manipulation Indicators: {len(evidence.get('supporting_manipulation', []))}")
        lines.append(f"Authenticity Indicators: {len(evidence.get('supporting_authenticity', []))}")
        lines.append("")
        
        if evidence.get("supporting_manipulation"):
            lines.append("Key Evidence of Manipulation:")
            for e in evidence["supporting_manipulation"][:5]:
                lines.append(f"  • [{e.get('severity', 'medium').upper()}] {e.get('finding', '')}")
        lines.append("")
        
        # Methodology
        lines.append("-" * 70)
        lines.append("FORENSIC METHODOLOGY")
        lines.append("-" * 70)
        lines.append("This analysis employed 25+ digital forensic algorithms:")
        lines.append("  1. Error Level Analysis (ELA) - JPEG compression forensics")
        lines.append("  2. Noise Pattern Analysis - Sensor noise verification")
        lines.append("  3. Clone Detection - Copy-move forgery detection")
        lines.append("  4. Edge Analysis - Boundary consistency")
        lines.append("  5. Frequency Domain Analysis - FFT/DCT examination")
        lines.append("  6. Color Consistency Analysis - Illumination check")
        lines.append("  7. Compression Artifact Analysis - Double-JPEG detection")
        lines.append("  8. Statistical Analysis - Distribution verification")
        lines.append("")
        
        # Disclaimer
        lines.append("-" * 70)
        lines.append("LEGAL DISCLAIMER")
        lines.append("-" * 70)
        lines.append("This forensic analysis represents automated examination results.")
        lines.append("While based on established methodologies, this report supports")
        lines.append("expert testimony, not standalone legal determination.")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _get_ai_analysis(self, file_path: str, verdict: dict) -> dict:
        """Get AI visual analysis"""
        result = {
            "available": False,
            "reasoning": None,
            "suspicious_areas": []
        }
        
        if not self.genai_enabled or not PIL_AVAILABLE:
            return result
        
        try:
            is_video = Path(file_path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
            if is_video and CV2_AVAILABLE:
                cap = cv2.VideoCapture(file_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 4))
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                else:
                    return result
            else:
                img = Image.open(file_path)
            
            prompt = f"""You are a digital forensics expert. Our automated analysis concluded: {verdict.get('conclusion', 'Unknown')} with {verdict.get('confidence', 0):.0f}% confidence.

Analyze this image for:
1. Visual anomalies (smoothing, blurring, unnatural edges)
2. Lighting/shadow inconsistencies
3. Signs of AI generation (artifacts, merged features)
4. Suspicious boundaries or compositing

Provide your assessment in 2-3 sentences. Be specific about what you observe."""

            response = self.model.generate_content([prompt, img])
            
            if response and response.text:
                result["available"] = True
                result["reasoning"] = response.text
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _generate_recommendations(self, result: dict) -> list:
        """Generate recommendations"""
        recs = []
        verdict = result.get("verdict", {})
        conclusion = verdict.get("conclusion", "")
        
        if conclusion == "LIKELY MANIPULATED":
            recs.append({
                "priority": "high",
                "action": "Do not share this content without verification",
                "reason": "Strong manipulation indicators detected"
            })
            recs.append({
                "priority": "high",
                "action": "Try to find the original source",
                "reason": "Compare with unedited version"
            })
            recs.append({
                "priority": "medium",
                "action": "Report if spreading misinformation",
                "reason": "High confidence of manipulation"
            })
        elif conclusion == "POSSIBLY MANIPULATED":
            recs.append({
                "priority": "medium",
                "action": "Verify with reverse image search",
                "reason": "Find original versions"
            })
            recs.append({
                "priority": "medium",
                "action": "Check source credibility",
                "reason": "Context matters for verification"
            })
        else:
            recs.append({
                "priority": "low",
                "action": "Content appears genuine",
                "reason": "No significant manipulation detected"
            })
        
        return recs


if __name__ == "__main__":
    xai = AdvancedExplainableAI()
    print("✅ Advanced Explainable AI v3.0 loaded!")
    print(f"   Google AI: {'✓ Enabled' if xai.genai_enabled else '✗ Not configured'}")
