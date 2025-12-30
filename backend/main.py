"""
FakeTrace v7.0 - Forensic Anomaly Detection System
===================================================
PHILOSOPHY: Anomaly-First, Evidence-Driven, Legally Safe

This is a CLASSICAL FORENSIC ASSISTANCE TOOL,
NOT a definitive AI detector.
Designed to SUPPORT investigators, NOT replace them.

Systems:
1. Content Database (Unique ID + Storage + Dashboard)
2. Forensic Anomaly Engine v7 (Anomaly-first analysis)
3. Advanced Explainable AI (Multi-level explanations)

Core Principles:
- Never claim certainty when evidence is weak
- If no anomaly detected → explicitly say so
- Prioritize "what is unusual" over "fake vs real"
- Better UNCERTAIN than wrong prediction
"""

import os
import sys
import json
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import numpy as np

# Load environment
load_dotenv()

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.void,)):
            return None
        return super().default(obj)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.void,)):
        return None
    else:
        return obj

# =====================
# UPLOAD FOLDER
# =====================

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# =====================
# INITIALIZE MODULES
# =====================

print("\n" + "="*60)
print("🔍 FakeTrace v7.0 - Forensic Anomaly Detection")
print("   Anomaly-First | Evidence-Driven | Legally Safe")
print("="*60)
print(f"📁 Upload Folder: {UPLOAD_FOLDER}")

# 1. Content Database
content_db = None
try:
    from content_database import ContentDatabase
    content_db = ContentDatabase()
    print("✅ Content Database initialized (SQLite)")
except Exception as e:
    print(f"❌ Content Database failed: {e}")
    import traceback
    traceback.print_exc()

# 2. Forensic Anomaly Engine v7 (Anomaly-first analysis)
visual_forensics = None
try:
    from forensic_anomaly_engine_v7 import AdvancedVisualForensics
    visual_forensics = AdvancedVisualForensics()
    print("✅ Forensic Anomaly Engine v7.0 (Anomaly-first analysis)")
except Exception as e:
    print(f"⚠️ v7 failed, trying v6 fallback: {e}")
    try:
        from simple_forensics_v6 import AdvancedVisualForensics
        visual_forensics = AdvancedVisualForensics()
        print("✅ Simple Forensics v6.0 fallback")
    except Exception as e2:
        print(f"⚠️ v6 failed, trying v5: {e2}")
        try:
            from advanced_visual_forensics_v5 import AdvancedVisualForensics
            visual_forensics = AdvancedVisualForensics()
            print("✅ Visual Forensics v5.0 fallback")
        except Exception as e3:
            print(f"❌ Visual Forensics failed: {e3}")

# 3. Advanced Explainable AI
explainable_ai = None
try:
    from advanced_explainable_ai import AdvancedExplainableAI
    explainable_ai = AdvancedExplainableAI()
    print(f"✅ Advanced Explainable AI initialized (Gemini: {'✓' if explainable_ai.genai_enabled else '✗'})")
except Exception as e:
    print(f"❌ Explainable AI failed: {e}")

# 4. Claim Verification System
claim_verifier = None
decision_fusion = None
try:
    from claim_verification import ClaimVerificationEngine, DecisionFusion, create_user_context
    claim_verifier = ClaimVerificationEngine()
    decision_fusion = DecisionFusion()
    print("✅ Claim Verification System initialized")
except Exception as e:
    print(f"⚠️ Claim Verification failed: {e}")

# 5. AI Generation Detector v3 (Advanced - 17 algorithms)
ai_detector = None
try:
    from ai_generation_detector_v3 import AdvancedAIDetector
    ai_detector = AdvancedAIDetector()
    print("✅ AI Generation Detector v3.0 ADVANCED (17 algorithms across 5 categories)")
except Exception as e:
    print(f"⚠️ v3 failed, trying v2: {e}")
    try:
        from ai_generation_detector_v2 import AIGenerationDetectorV2
        ai_detector = AIGenerationDetectorV2()
        print("✅ AI Generation Detector v2.0 fallback")
    except Exception as e2:
        print(f"⚠️ v2 failed, trying v1: {e2}")
        try:
            from ai_generation_detector import AIGenerationDetector
            ai_detector = AIGenerationDetector()
            print("✅ AI Generation Detector v1.0 fallback")
        except Exception as e3:
            print(f"⚠️ AI Generation Detector failed: {e3}")

# 6. Video AI Generation Detector v3 (BALANCED - catches AI, ignores real)
video_ai_detector = None
try:
    from video_ai_detector_v3 import VideoAIDetectorV3
    video_ai_detector = VideoAIDetectorV3(image_ai_detector=ai_detector)
    print("✅ Video AI Detector v3.0 (BALANCED - catches Gemini, Sora)")
except Exception as e:
    print(f"⚠️ v3 failed, trying v2: {e}")
    try:
        from video_ai_detector_v2 import VideoAIDetectorV2
        video_ai_detector = VideoAIDetectorV2(image_ai_detector=ai_detector)
        print("✅ Video AI Detector v2.0 fallback")
    except Exception as e2:
        print(f"⚠️ v2 failed, trying v1: {e2}")
        try:
            from video_ai_detector import VideoAIDetector
            video_ai_detector = VideoAIDetector(image_ai_detector=ai_detector)
            print("✅ Video AI Detector v1.0 fallback")
        except Exception as e3:
            print(f"⚠️ Video AI Detector failed: {e3}")

# 7. Fraud Message Analyzer
fraud_analyzer = None
try:
    from fraud_analyzer import FraudAnalyzer
    fraud_analyzer = FraudAnalyzer()
    print("✅ Fraud Analyzer v1.0 (SMS/WhatsApp/Instagram fraud detection)")
except Exception as e:
    print(f"⚠️ Fraud Analyzer failed: {e}")

# 8. EasyOCR Reader (Pre-load for fast image analysis)
ocr_reader = None
try:
    import easyocr
    print("⏳ Loading OCR engine (one-time setup)...")
    ocr_reader = easyocr.Reader(['en', 'hi'], gpu=False, verbose=False)
    print("✅ EasyOCR initialized (English + Hindi)")
except ImportError:
    print("⚠️ EasyOCR not installed - image OCR will use pytesseract")
except Exception as e:
    print(f"⚠️ EasyOCR failed to load: {e}")

# 9. Location Database (Separate database for location tracking)
location_db = None
try:
    from modules.location_database import LocationDatabase
    location_db = LocationDatabase()
    print("✅ Location Database initialized (Separate DB)")
except Exception as e:
    print(f"⚠️ Location Database failed: {e}")

print("="*60 + "\n")

# =====================
# FASTAPI APP
# =====================

app = FastAPI(
    title="FakeTrace v7.0",
    description="Forensic Anomaly Detection - Anomaly-First, Evidence-Driven, Legally Safe",
    version="7.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files
from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_FOLDER)), name="uploads")

# =====================
# HELPER FUNCTIONS
# =====================

def save_heatmaps_to_base64(heatmaps: dict) -> dict:
    """Convert heatmap images to base64 for frontend"""
    import cv2
    import base64
    
    result = {}
    
    for name, data in heatmaps.items():
        if data is None:
            continue
        
        try:
            if isinstance(data, np.ndarray):
                # Encode as PNG
                _, buffer = cv2.imencode('.png', data)
                b64 = base64.b64encode(buffer).decode('utf-8')
                result[name] = f"data:image/png;base64,{b64}"
        except Exception as e:
            print(f"Error encoding heatmap {name}: {e}")
    
    return result


def build_legal_profile(verdict, ai_generation, ai_detection, evidence) -> dict:
    """Classify content as legal/illegal signals for law-enforcement triage."""

    cues = []
    risk_points = 0.0
    protective_points = 0.0

    def register(label: str, detail: str, *, level: str = "major", source: str = None,
                 metric: str = None, weight: float = 10.0, positive: bool = False):
        nonlocal risk_points, protective_points
        cue = {
            "label": label,
            "detail": detail,
            "level": level,
            "kind": "protective" if positive else "risk"
        }
        if source:
            cue["source"] = source
        if metric:
            cue["metric"] = metric
        cues.append(cue)
        if positive:
            protective_points += weight
        else:
            risk_points += weight

    def metric_text(details: dict, fields: Sequence[str]) -> str:
        if not isinstance(details, dict) or not fields:
            return None
        formatted = []
        for key in fields:
            if key not in details:
                continue
            value = details.get(key)
            if isinstance(value, float):
                formatted.append(f"{key.replace('_', ' ')} {value:.3f}")
            else:
                formatted.append(f"{key.replace('_', ' ')} {value}")
        return ", ".join(formatted) if formatted else None

    summary = evidence.get("summary") if isinstance(evidence, dict) else {}
    anomalies = summary.get("anomalies_detected", 0) if isinstance(summary, dict) else 0
    total_checks = summary.get("total_checks", 0) if isinstance(summary, dict) else 0
    clean_checks = summary.get("clean", 0) if isinstance(summary, dict) else 0

    verdict_code = None
    if isinstance(verdict, dict):
        verdict_code = verdict.get("verdict") or verdict.get("verdict_code")
    elif isinstance(verdict, str):
        verdict_code = verdict

    # Explicit deepfake verdicts - these are strong signals for illegal content
    if verdict_code in ("LIKELY_DEEPFAKE", "LIKELY_FAKE", "DEEPFAKE_DETECTED", "SYNTHETIC_FACE_DETECTED"):
        register(
            label="Deepfake detected",
            detail="Content identified as deepfake or synthetic face manipulation.",
            level="critical",
            source="Forensic Analysis",
            weight=40  # Very high weight for clear deepfakes
        )

    if verdict_code == "FORENSIC_MANIPULATION_INDICATORS_DETECTED" and total_checks:
        register(
            label="Multiple forensic anomalies",
            detail=f"{anomalies} of {total_checks} anomaly checks triggered manipulation indicators.",
            level="critical",
            source="Forensic Anomaly Engine v7",
            metric=f"{anomalies}/{total_checks} checks",
            weight=28
        )
    elif verdict_code == "NO_FORENSIC_ANOMALY_DETECTED" and clean_checks:
        register(
            label="Clean forensic sweep",
            detail=f"{clean_checks} checks stayed clean with no statistical irregularities detected.",
            level="supportive",
            source="Forensic Anomaly Engine v7",
            metric=f"clean {clean_checks}",
            weight=18,
            positive=True
        )
    elif verdict_code == "INCONCLUSIVE_MINOR_IRREGULARITIES":
        # Minor irregularities are normal in real-world media - treat as protective
        register(
            label="Normal compression artifacts",
            detail="Minor statistical variations consistent with normal camera/compression processing.",
            level="supportive",
            source="Forensic Anomaly Engine v7",
            weight=6,
            positive=True
        )

    ai_conclusion = None
    ai_confidence = 0.0
    ai_is_video = False
    if isinstance(ai_generation, dict):
        ai_conclusion = ai_generation.get("conclusion")
        ai_confidence = ai_generation.get("confidence", 0)
        ai_is_video = ai_generation.get("is_video", False)

    if ai_conclusion == "LIKELY_AI_GENERATED":
        # AI generated content - this is a significant risk signal
        if ai_confidence >= 70:
            register(
                label="AI detector escalation",
                detail="Multi-algorithm detector marked the file as synthetic with high confidence.",
                level="critical",
                source="AI Generation Detector",
                metric=f"confidence {ai_confidence}%",
                weight=28
            )
        else:
            register(
                label="AI generation detected",
                detail="AI detector identified synthetic generation patterns.",
                level="major",
                source="AI Generation Detector",
                metric=f"confidence {ai_confidence}%",
                weight=18
            )
    elif ai_conclusion == "POSSIBLY_AI_GENERATED":
        # Possible AI - moderate risk signal
        register(
            label="Possible AI generation",
            detail="Detector found AI-style patterns that indicate potential synthetic content.",
            level="major",
            source="AI Generation Detector",
            metric=f"confidence {ai_confidence}%",
            weight=12
        )
    elif ai_conclusion == "UNCERTAIN":
        # Uncertain - slight risk (not protective)
        register(
            label="Inconclusive AI check",
            detail="AI detector found mixed signals - content origin unclear.",
            level="minor",
            source="AI Generation Detector",
            metric=f"confidence {ai_confidence}%",
            weight=4
        )
    elif ai_conclusion == "LIKELY_NATURAL":
        register(
            label="Detector leans natural",
            detail="AI detector favored natural/camera capture signatures.",
            level="supportive",
            source="AI Generation Detector",
            metric=f"confidence {ai_confidence}%",
            weight=22,
            positive=True
        )

    ai_checks = []
    total_ai_checks = 0
    natural_ai_checks = 0
    if isinstance(ai_detection, dict):
        ai_checks = ai_detection.get("ai_checks") or []
    check_lookup = {}
    for check in ai_checks:
        if isinstance(check, dict) and check.get("name"):
            check_lookup[check["name"]] = check
            total_ai_checks += 1
            if check.get("status") == "likely_natural":
                natural_ai_checks += 1

    risk_registry = {
        "frequency_grid": {
            "label": "GAN spectral grid",
            "detail": "FFT shows checkerboard lattice typical of diffusion upscaling.",
            "source": "Spectral Grid Analysis",
            "weight": 18,
            "level_map": {"likely_ai": "critical", "possibly_ai": "major"},
            "metric_fields": ["checker_energy", "cross_energy"],
            "trigger_statuses": {"likely_ai", "possibly_ai"}
        },
        "heartbeat_coherence": {
            "label": "Missing physiological rhythm",
            "detail": "PPG trace lacks human heartbeat coherence used in live impersonations.",
            "source": "Heartbeat Coherence",
            "weight": 17,
            "level_map": {"likely_ai": "critical", "possibly_ai": "major"},
            "metric_fields": ["coherence", "pulse_strength"]
        },
        "object_morphing": {
            "label": "Temporal morphing",
            "detail": "Geometry shifts frame-to-frame beyond natural motion.",
            "source": "Object Morphing Detection",
            "weight": 14,
            "level_map": {"likely_ai": "major", "possibly_ai": "minor"}
        },
        "edge_bleeding": {
            "label": "Edge halo seams",
            "detail": "Neural render left haloing along subject boundaries.",
            "source": "Edge Bleeding Detection",
            "weight": 12,
            "level_map": {"likely_ai": "major", "possibly_ai": "minor"}
        },
        "physics": {
            "label": "Impossible motion physics",
            "detail": "Acceleration profile violated natural motion constraints.",
            "source": "Physics Analysis",
            "weight": 16,
            "level_map": {"likely_ai": "critical", "possibly_ai": "major"},
            "metric_fields": ["max_acceleration", "impossible_motions"]
        },
        "noise_authenticity": {
            "label": "Synthetic noise profile",
            "detail": "Noise statistics look procedural instead of sensor-derived.",
            "source": "Noise Authenticity",
            "weight": 13,
            "level_map": {"likely_ai": "major", "possibly_ai": "minor"},
            "metric_fields": ["noise_uniformity"]
        },
        "prnu_quality": {
            "label": "Sensor fingerprint missing",
            "detail": "PRNU correlation outside camera norms.",
            "source": "Sensor Fingerprint Quality",
            "weight": 20,
            "level_map": {"likely_ai": "critical", "possibly_ai": "major"},
            "metric_fields": ["prnu_correlation"]
        },
        "prnu_absence": {
            "label": "Still lacks PRNU",
            "detail": "Image failed camera sensor fingerprint test.",
            "source": "Sensor Fingerprint (PRNU) Analysis",
            "weight": 20,
            "level_map": {"likely_ai": "critical", "possibly_ai": "major"},
            "metric_fields": ["spatial_correlation"]
        },
        "cfa_consistency": {
            "label": "Bayer CFA mismatch",
            "detail": "Mosaic periodicity deviates from real demosaicing.",
            "source": "CFA Pattern Consistency",
            "weight": 17,
            "level_map": {"likely_ai": "critical", "possibly_ai": "major"},
            "metric_fields": ["green_delta", "rb_correlation"]
        },
        "radial_power": {
            "label": "Radial spectrum anomaly",
            "detail": "Power spectrum slope breaks natural 1/f decay.",
            "source": "Radial Power Signature",
            "weight": 15,
            "level_map": {"likely_ai": "major", "possibly_ai": "minor"},
            "metric_fields": ["spectrum_slope", "high_freq_ratio"]
        }
    }

    protective_registry = {
        "prnu_quality": {
            "label": "Stable camera PRNU",
            "detail": "Sensor fingerprint consistent across frames.",
            "source": "Sensor Fingerprint Quality",
            "weight": 14,
            "trigger_statuses": {"likely_natural"},
            "metric_fields": ["prnu_correlation"]
        },
        "noise_authenticity": {
            "label": "Natural noise field",
            "detail": "Noise variance matches in-camera acquisition.",
            "source": "Noise Authenticity",
            "weight": 12,
            "trigger_statuses": {"likely_natural"},
            "metric_fields": ["noise_uniformity"]
        },
        "prnu_absence": {
            "label": "Camera fingerprint present",
            "detail": "PRNU analysis supports authentic capture.",
            "source": "Sensor Fingerprint (PRNU) Analysis",
            "weight": 12,
            "trigger_statuses": {"likely_natural"},
            "metric_fields": ["spatial_correlation"]
        },
        "cfa_consistency": {
            "label": "Bayer CFA intact",
            "detail": "Color filter mosaic matches real optics.",
            "source": "CFA Pattern Consistency",
            "weight": 10,
            "trigger_statuses": {"likely_natural"},
            "metric_fields": ["green_delta", "rb_correlation"]
        }
    }

    for name, meta in risk_registry.items():
        check = check_lookup.get(name)
        if not check:
            continue
        status = check.get("status")
        trigger_statuses = meta.get("trigger_statuses", {"likely_ai"})
        if status not in trigger_statuses:
            continue
        details = check.get("details") or {}
        metric = metric_text(details, meta.get("metric_fields"))
        level = meta.get("level_map", {}).get(status, meta.get("level", "major"))
        finding = check.get("finding")
        detail = meta.get("detail", "")
        if finding:
            detail = f"{detail} {finding}"
        register(
            label=meta.get("label", name),
            detail=detail.strip(),
            level=level,
            source=meta.get("source"),
            metric=metric,
            weight=meta.get("weight", 10)
        )

    for name, meta in protective_registry.items():
        check = check_lookup.get(name)
        if not check:
            continue
        status = check.get("status")
        trigger_statuses = meta.get("trigger_statuses", {"likely_natural"})
        if status not in trigger_statuses:
            continue
        details = check.get("details") or {}
        metric = metric_text(details, meta.get("metric_fields"))
        register(
            label=meta.get("label", name),
            detail=meta.get("detail", ""),
            level="supportive",
            source=meta.get("source"),
            metric=metric,
            weight=meta.get("weight", 10),
            positive=True
        )

    natural_ratio = (natural_ai_checks / total_ai_checks) if total_ai_checks else 0
    if natural_ratio >= 0.5:
        register(
            label="Consistent natural cues",
            detail=f"{int(natural_ratio * 100)}% of AI detectors favored natural capture.",
            level="supportive",
            source="AI Detection Ensemble",
            metric=f"natural ratio {natural_ratio:.2f}",
            weight=20,
            positive=True
        )

    level_priority = {"critical": 0, "major": 1, "minor": 2, "supportive": 3}
    cues.sort(key=lambda cue: (level_priority.get(cue.get("level"), 4), cue.get("kind")))
    display_cues = cues[:8]

    net_score = risk_points - protective_points
    
    # Determine legal status - balanced approach
    has_critical = any(c["level"] == "critical" for c in cues if c.get("kind") == "risk")
    has_major = any(c["level"] == "major" for c in cues if c.get("kind") == "risk")
    
    if risk_points == 0 and protective_points == 0 and not verdict_code and not ai_conclusion:
        status = "NOT_ENOUGH_DATA"
    elif has_critical and risk_points >= 35:
        # Critical evidence with very high risk = ILLEGAL (deepfake with manipulation intent)
        status = "ILLEGAL_SIGNAL"
    elif net_score >= 50 or (risk_points >= 55 and has_critical):
        # Very high risk = ILLEGAL
        status = "ILLEGAL_SIGNAL"
    elif has_critical or has_major or risk_points >= 12:
        # Any critical/major risk signal = REVIEW (AI-generated content needs human check)
        status = "REQUIRES_REVIEW"
    elif risk_points > 0 and protective_points < risk_points * 1.5:
        # Some risk and not strongly protected = REVIEW
        status = "REQUIRES_REVIEW"
    elif protective_points > risk_points * 2 and not has_major:
        # Strong protective signals and no major risk = LEGAL
        status = "LEGAL_CLEAR"
    elif risk_points == 0 and protective_points > 0:
        # No risk, only protective = LEGAL
        status = "LEGAL_CLEAR"
    else:
        # Default to REVIEW for ambiguous cases
        status = "REQUIRES_REVIEW"

    status_catalog = {
        "ILLEGAL_SIGNAL": {
            "label": "Escalate: Potential Illegal Synthetic Media",
            "action": "Preserve evidence, notify cybercrime or prosecutors, and document cue metrics in the case log."
        },
        "REQUIRES_REVIEW": {
            "label": "Hold: Needs Analyst Review",
            "action": "Hold content for senior analyst sign-off before dissemination."
        },
        "LEGAL_CLEAR": {
            "label": "Clears Legal Triage",
            "action": "Document findings, keep hashes on file, and release if no other evidence contradicts."
        },
        "NOT_ENOUGH_DATA": {
            "label": "Insufficient Forensic Data",
            "action": "Re-run acquisition (better quality upload or metadata pull) before making legal calls."
        }
    }

    magnitude = abs(net_score)
    if status == "NOT_ENOUGH_DATA":
        confidence = 30.0
    else:
        base_conf = 52 + min(30, magnitude * 0.8)
        if len(display_cues) >= 3:
            base_conf += 4
        if status == "LEGAL_CLEAR" and protective_points > risk_points:
            base_conf = min(base_conf, 88)
        confidence = round(min(96, max(40, base_conf)), 1)

    risk_highlights = [c["label"] for c in display_cues if c.get("kind") == "risk" and c.get("level") in {"critical", "major"}]
    protective_highlights = [c["label"] for c in display_cues if c.get("kind") == "protective"]

    if status == "ILLEGAL_SIGNAL" and risk_highlights:
        summary_text = f"High-priority cues: {', '.join(risk_highlights[:3])}."
        if anomalies:
            summary_text += f" {anomalies} forensic checks backed the anomaly call."
    elif status == "LEGAL_CLEAR" and protective_highlights:
        summary_text = f"Protective cues dominate ({', '.join(protective_highlights[:3])})."
    elif status == "NOT_ENOUGH_DATA":
        summary_text = "Legal triage skipped because detectors returned insufficient evidence."
    else:
        summary_text = "Mixed cues detected; escalate for human confirmation."

    notes = [
        "Hash and archive the asset before acting on these cues.",
        "Corroborate statistical cues with interview, device, or metadata evidence."
    ]
    if ai_is_video and check_lookup.get("heartbeat_coherence"):
        notes.append("If pursuing impersonation charges, capture the source stream; re-encodes can erase physiological cues.")
    notes = notes[:3]

    status_payload = status_catalog.get(status, status_catalog["REQUIRES_REVIEW"])

    return {
        "status": status,
        "status_label": status_payload["label"],
        "summary": summary_text,
        "confidence": confidence,
        "recommended_action": status_payload["action"],
        "cues": display_cues,
        "supporting_metrics": {
            "risk_points": round(risk_points, 1),
            "protective_points": round(protective_points, 1),
            "net_score": round(net_score, 1)
        },
        "notes": notes,
        "disclaimer": "Legal labels are investigative triage signals only; FakeTrace does not render statutory determinations."
    }


# =====================
# MAIN ANALYSIS ENDPOINT
# =====================

@app.post("/api/analyze")
async def analyze_content(
    file: UploadFile = File(...),
    # Optional user context (form fields)
    source: Optional[str] = None,
    is_ai_generated: Optional[str] = None,
    is_enhanced: Optional[str] = None,
    workflow: Optional[str] = None,
    user_notes: Optional[str] = None
):
    """
    Main analysis endpoint with optional user context.
    
    Flow:
    1. Generate unique fingerprint ID
    2. Run visual forensics (3-class detection)
    3. Verify user claims against forensic evidence
    4. Generate AI explanations
    5. Store in database
    6. Return comprehensive results with claim verification
    
    User Context (optional):
    - source: "camera", "social_media", "web_download", "unknown"
    - is_ai_generated: "yes", "no", "not_sure"
    - is_enhanced: "yes", "no", "not_sure"
    - workflow: "camera_direct", "ai_generate", "ai_generate_enhance"
    - user_notes: Additional context
    """
    result = {
        "status": "processing",
        "timestamp": datetime.now().isoformat(),
        "version": "7.0",
        
        # System Identity (v7)
        "system_identity": {
            "name": "FakeTrace Forensic Assistance Tool",
            "type": "Classical Forensic Analysis",
            "purpose": "Support investigators, not replace them",
            "disclaimer": "Results are probabilistic, not definitive"
        },
        
        # Content identification
        "content_id": None,
        "fingerprint": None,
        "is_known": False,
        "analysis_count": 1,
        
        # Evidence (v7 format)
        "evidence": {
            "algorithms": [],
            "summary": {
                "total_checks": 0,
                "anomalies_detected": 0,
                "inconclusive": 0,
                "clean": 0
            }
        },
        
        # Verdict (v7 format - object, not string)
        "verdict": {
            "verdict": "UNKNOWN",
            "verdict_code": "UNKNOWN",
            "interpretation": "Analysis in progress...",
            "anomaly_count": 0,
            "inconclusive_count": 0,
            "clean_count": 0,
            "total_checks": 0,
            "recommendation": "",
            "legal_disclaimer": "Results are probabilistic, not definitive."
        },
        
        # Supporting statistics (v7 - secondary only)
        "supporting_statistics": {
            "note": "These are supporting metrics only. They do NOT determine the verdict.",
            "average_score": 0.0,
            "interpretation": ""
        },
        
        # Explanation (v7)
        "explanation": {
            "summary": "",
            "findings": [],
            "limitations": [],
            "recommendation": ""
        },
        
        # AI Generation Detection
        "ai_generation": {
            "conclusion": "UNKNOWN",
            "confidence": 0,
            "interpretation": "Analysis pending...",
            "indicators": [],
            "natural_signs": [],
            "recommendation": "",
            "disclaimer": "AI generation detection is probabilistic, not definitive."
        },
        "ai_detection": None,
        
        # Legacy fields for backward compatibility
        "visual_forensics": None,
        "prediction": None,
        "confidence": 0,
        
        # Claim Verification
        "claim_verification": None,
        "user_context": None,
        
        "legal_profile": None,

        "errors": []
    }
    
    # Parse user context
    user_context = None
    if decision_fusion and any([source, is_ai_generated, is_enhanced, workflow]):
        try:
            # Convert string inputs to proper types
            ai_gen = None
            if is_ai_generated:
                ai_gen = is_ai_generated.lower() == "yes" if is_ai_generated.lower() != "not_sure" else None
            
            enhanced = None
            if is_enhanced:
                enhanced = is_enhanced.lower() == "yes" if is_enhanced.lower() != "not_sure" else None
            
            user_context = create_user_context(
                source=source,
                is_ai_generated=ai_gen,
                is_enhanced=enhanced,
                workflow=workflow,
                notes=user_notes
            )
            
            result["user_context"] = {
                "source": source,
                "is_ai_generated": is_ai_generated,
                "is_enhanced": is_enhanced,
                "workflow": workflow,
                "notes": user_notes
            }
        except Exception as e:
            result["errors"].append(f"User context parsing: {str(e)}")
    
    temp_path = None
    saved_path = None
    
    try:
        # Save uploaded file to temp first
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        filename = file.filename
        
        # =====================
        # STEP 1: DATABASE REGISTRATION
        # =====================
        if content_db:
            try:
                db_result = content_db.register_content(temp_path, filename)
                result["content_id"] = db_result.get("content_id")
                result["fingerprint"] = db_result.get("fingerprint")
                result["is_known"] = db_result.get("is_known", False)
                result["analysis_count"] = db_result.get("analysis_count", 1)
                
                # Save file permanently to uploads folder
                if result["content_id"]:
                    save_filename = f"{result['content_id']}{Path(filename).suffix}"
                    saved_path = UPLOAD_FOLDER / save_filename
                    shutil.copy2(temp_path, saved_path)
                    result["saved_path"] = str(saved_path)
                    print(f"📁 Saved: {saved_path}")
                    
            except Exception as e:
                result["errors"].append(f"Database registration: {str(e)}")
                traceback.print_exc()
        
        # =====================
        # STEP 2: VISUAL FORENSICS (v7)
        # =====================
        if visual_forensics:
            try:
                vf_result = visual_forensics.analyze(temp_path)
                
                # Store raw result for legacy compatibility
                vf_cleaned = {k: v for k, v in vf_result.items() 
                              if k not in ["heatmaps", "heatmaps_base64"]}
                result["visual_forensics"] = convert_numpy_types(vf_cleaned)
                
                # ===== V7 FORMAT HANDLING =====
                
                # 1. System Identity
                if vf_result.get("system_identity"):
                    result["system_identity"] = vf_result["system_identity"]
                
                # 2. Evidence (algorithms + summary)
                if vf_result.get("evidence"):
                    result["evidence"] = convert_numpy_types(vf_result["evidence"])
                
                # 3. Verdict (object format)
                if vf_result.get("verdict"):
                    result["verdict"] = convert_numpy_types(vf_result["verdict"])
                    # Also set confidence for legacy
                    result["confidence"] = float(vf_result.get("supporting_statistics", {}).get("average_score", 0))
                
                # 4. Supporting Statistics
                if vf_result.get("supporting_statistics"):
                    result["supporting_statistics"] = convert_numpy_types(vf_result["supporting_statistics"])
                
                # 5. Explanation
                if vf_result.get("explanation"):
                    result["explanation"] = convert_numpy_types(vf_result["explanation"])
                
                # ===== LEGACY FALLBACK =====
                # If no v7 verdict, try old format
                if not vf_result.get("verdict") and vf_result.get("prediction"):
                    pred = vf_result["prediction"]
                    result["prediction"] = convert_numpy_types(pred)
                    result["confidence"] = float(pred.get("confidence", 0))
                    
                    # Create v7 verdict from old format
                    class_name = pred.get("class_name", "UNKNOWN")
                    result["verdict"] = {
                        "verdict": class_name,
                        "verdict_code": class_name,
                        "interpretation": f"Legacy analysis: {class_name}",
                        "anomaly_count": 0,
                        "inconclusive_count": 0,
                        "clean_count": 0,
                        "total_checks": 6,
                        "recommendation": "Consider re-analyzing with v7 engine",
                        "legal_disclaimer": "Results are probabilistic, not definitive."
                    }
                        
            except Exception as e:
                result["errors"].append(f"Visual forensics: {str(e)}")
                traceback.print_exc()
        
        # =====================
        # STEP 2.5: AI GENERATION DETECTION (Image + Video)
        # =====================
        # Check if it's a video file
        is_video = Path(temp_path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif']
        
        if is_video and video_ai_detector:
            # VIDEO AI DETECTION
            try:
                print(f"🎬 Running Video AI Detection on {Path(temp_path).name}...")
                video_result = video_ai_detector.analyze(temp_path)
                result["ai_detection"] = convert_numpy_types(video_result)
                
                # Add AI assessment to result
                if video_result.get("ai_assessment"):
                    result["ai_generation"] = {
                        "conclusion": video_result["ai_assessment"].get("conclusion", "UNKNOWN"),
                        "confidence": video_result["ai_assessment"].get("confidence", 0),
                        "interpretation": video_result["ai_assessment"].get("interpretation", ""),
                        "indicators_found": video_result["ai_assessment"].get("indicators_found", []),
                        "natural_signs": video_result["ai_assessment"].get("natural_signs", []),
                        "recommendation": video_result["ai_assessment"].get("recommendation", ""),
                        "category_analysis": video_result["ai_assessment"].get("category_analysis", []),
                        "categories_summary": video_result["ai_assessment"].get("categories_summary", {}),
                        "weighted_score": video_result["ai_assessment"].get("weighted_score", 0),
                        "disclaimer": "Video AI detection is experimental. Results require expert verification.",
                        "is_video": True,
                        "frames_analyzed": video_result.get("frames_analyzed", 0),
                        "video_info": video_result.get("video_info", {}),
                        "key_findings": video_result["ai_assessment"].get("key_findings", {})
                    }
                
                print(f"🎬 Video AI Detection: {result.get('ai_generation', {}).get('conclusion', 'N/A')} ({video_result.get('frames_analyzed', 0)} frames)")
                
            except Exception as e:
                result["errors"].append(f"Video AI detection: {str(e)}")
                traceback.print_exc()
        
        elif ai_detector and temp_path:
            # IMAGE AI DETECTION
            try:
                import cv2
                img = cv2.imread(temp_path)
                if img is not None:
                    ai_result = ai_detector.analyze(img, temp_path)
                    result["ai_detection"] = convert_numpy_types(ai_result)
                    
                    # Add AI assessment to result
                    if ai_result.get("ai_assessment"):
                        result["ai_generation"] = {
                            "conclusion": ai_result["ai_assessment"].get("conclusion", "UNKNOWN"),
                            "confidence": ai_result["ai_assessment"].get("confidence", 0),
                            "interpretation": ai_result["ai_assessment"].get("interpretation", ""),
                            "indicators_found": ai_result["ai_assessment"].get("indicators_found", []),
                            "natural_signs": ai_result["ai_assessment"].get("natural_signs", []),
                            "recommendation": ai_result["ai_assessment"].get("recommendation", ""),
                            "category_analysis": ai_result["ai_assessment"].get("category_analysis", []),
                            "categories_summary": ai_result["ai_assessment"].get("categories_summary", {}),
                            "weighted_score": ai_result["ai_assessment"].get("weighted_score", 0),
                            "disclaimer": "AI generation detection is probabilistic, not definitive.",
                            "is_video": False,
                            "key_findings": ai_result["ai_assessment"].get("key_findings", {})
                        }
                    
                    print(f"🤖 Image AI Detection: {result.get('ai_generation', {}).get('conclusion', 'N/A')}")
                    
            except Exception as e:
                result["errors"].append(f"AI generation detection: {str(e)}")
                traceback.print_exc()
        
        # =====================
        # STEP 3: CLAIM VERIFICATION
        # =====================
        if decision_fusion and user_context and result.get("visual_forensics"):
            try:
                # Fuse forensic evidence with user claims
                fusion_result = decision_fusion.fuse_decision(
                    forensic_result=result["visual_forensics"],
                    user_context=user_context
                )
                
                result["claim_verification"] = convert_numpy_types(
                    fusion_result.get("claim_verification")
                )
                
                # Add context warning to v7 verdict if claims contradict
                final_verdict = fusion_result.get("final_verdict", {})
                if "CONTEXT DISCREPANCY" in final_verdict.get("verdict", ""):
                    # Add warning to v7 verdict object instead of replacing it
                    if isinstance(result["verdict"], dict):
                        result["verdict"]["context_warning"] = True
                        result["verdict"]["claim_discrepancy"] = final_verdict.get("verdict", "")
                    result["context_warning"] = True
                
                # Add decision explanation
                result["decision_explanation"] = fusion_result.get("decision_explanation")
                
            except Exception as e:
                result["errors"].append(f"Claim verification: {str(e)}")
                traceback.print_exc()
        
        # =====================
        # STEP 4: EXPLAINABLE AI
        # =====================
        if explainable_ai and result.get("visual_forensics"):
            try:
                db_info = {
                    "content_id": result.get("content_id"),
                    "analysis_count": result.get("analysis_count", 1),
                    "first_seen": result.get("fingerprint", {}).get("first_seen") if isinstance(result.get("fingerprint"), dict) else None
                }
                
                explanation = explainable_ai.explain(
                    visual_forensics=result["visual_forensics"],
                    file_path=temp_path,
                    db_info=db_info
                )
                
                # Merge explanation into v7 explanation (don't override)
                if explanation:
                    xai_explanation = convert_numpy_types(explanation)
                    
                    # Keep v7 explanation structure, add XAI insights
                    if xai_explanation.get("simple"):
                        result["explanation"]["ai_summary"] = xai_explanation.get("simple")
                    if xai_explanation.get("technical"):
                        result["explanation"]["ai_technical"] = xai_explanation.get("technical")
                    if xai_explanation.get("red_flags"):
                        result["explanation"]["ai_red_flags"] = xai_explanation.get("red_flags")
                    if xai_explanation.get("recommendations"):
                        result["explanation"]["ai_recommendations"] = xai_explanation.get("recommendations")
                    
                    # Store full XAI result separately
                    result["xai_analysis"] = xai_explanation
                
                # NOTE: Do NOT override v7 verdict with XAI verdict
                # v7 verdict is based on anomaly count, not XAI interpretation
                    
            except Exception as e:
                result["errors"].append(f"Explainable AI: {str(e)}")
                traceback.print_exc()
        
        # =====================
        # STEP 4.5: LEGAL CLASSIFICATION
        # =====================
        try:
            legal_profile = build_legal_profile(
                verdict=result.get("verdict"),
                ai_generation=result.get("ai_generation"),
                ai_detection=result.get("ai_detection"),
                evidence=result.get("evidence")
            )
            result["legal_profile"] = legal_profile
            print(f"[DEBUG] Legal profile built: status={legal_profile.get('status')}, confidence={legal_profile.get('confidence')}")
        except Exception as e:
            result["errors"].append(f"Legal profile: {str(e)}")
            traceback.print_exc()

        # =====================
        # STEP 5: UPDATE DATABASE
        # =====================
        if content_db and result.get("content_id"):
            try:
                # Include claim verification in stored results
                analysis_data = {
                    "visual_forensics_scores": result.get("visual_forensics", {}).get("scores", {}),
                    "red_flags": result.get("explanation", {}).get("red_flags", []),
                    "claim_verification": result.get("claim_verification"),
                    "user_context": result.get("user_context")
                }
                
                legal_prof = result.get("legal_profile")
                print(f"[DEBUG] Saving to DB: content_id={result['content_id']}, legal_profile exists={legal_prof is not None}")
                if legal_prof:
                    print(f"[DEBUG] legal_profile.status = {legal_prof.get('status')}")
                
                # Extract verdict string if it's a dict
                verdict_value = result["verdict"]
                if isinstance(verdict_value, dict):
                    verdict_value = verdict_value.get("verdict") or verdict_value.get("verdict_code") or str(verdict_value)
                
                content_db.update_analysis(
                    content_id=result["content_id"],
                    verdict=verdict_value,
                    confidence=result["confidence"],
                    analysis_results=analysis_data,
                    legal_profile=legal_prof
                )
                print(f"[DEBUG] Database updated successfully")
            except Exception as e:
                result["errors"].append(f"Database update: {str(e)}")
                print(f"[ERROR] Database update failed: {e}")
                traceback.print_exc()
        
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        traceback.print_exc()
    
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
    
    return convert_numpy_types(result)


# =====================
# DASHBOARD ENDPOINTS
# =====================

@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard statistics"""
    if not content_db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        stats = content_db.get_dashboard_stats()
        return {
            "status": "success",
            "stats": convert_numpy_types(stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/content")
async def get_all_content(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    """Get paginated content list"""
    if not content_db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        result = content_db.get_all_content(page=page, limit=limit)
        return {
            "status": "success",
            **convert_numpy_types(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/content/{content_id}")
async def get_content_details(content_id: str):
    """Get detailed content info by ID"""
    if not content_db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        info = content_db.get_content_info(content_id)
        if not info:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "status": "success",
            "content": convert_numpy_types(info)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/search")
async def search_content(
    q: str = Query(..., min_length=1)
):
    """Search content by ID or filename"""
    if not content_db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        results = content_db.search_content(q)
        return {
            "status": "success",
            "results": convert_numpy_types(results),
            "query": q
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/similar/{content_id}")
async def find_similar(content_id: str):
    """Find similar content"""
    if not content_db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        similar = content_db.find_similar(content_id)
        return {
            "status": "success",
            "similar": convert_numpy_types(similar),
            "content_id": content_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# UTILITY ENDPOINTS
# =====================

@app.get("/api/context-questions")
async def get_context_questions():
    """
    Get available context questions for user input.
    
    These questions help users provide context that will be
    cross-verified against forensic evidence.
    """
    return {
        "version": "5.0",
        "description": "Optional context questions - answers will be verified against forensic evidence",
        "disclaimer": "Your answers are treated as 'soft claims' and will be independently verified.",
        "questions": [
            {
                "id": "source",
                "question": "What is the source of this media?",
                "type": "select",
                "required": False,
                "options": [
                    {"value": "camera", "label": "Camera (took it myself)"},
                    {"value": "social_media", "label": "Social Media (Facebook, Instagram, etc.)"},
                    {"value": "web_download", "label": "Downloaded from web"},
                    {"value": "messaging", "label": "Received via messaging app"},
                    {"value": "screenshot", "label": "Screenshot"},
                    {"value": "unknown", "label": "Don't know / Unsure"}
                ]
            },
            {
                "id": "is_ai_generated",
                "question": "Was this media generated by AI?",
                "type": "select",
                "required": False,
                "options": [
                    {"value": "yes", "label": "Yes, I believe it was AI-generated"},
                    {"value": "no", "label": "No, I believe it's a real photo/video"},
                    {"value": "not_sure", "label": "Not sure"}
                ]
            },
            {
                "id": "is_enhanced",
                "question": "Was this media enhanced or edited?",
                "type": "select",
                "required": False,
                "options": [
                    {"value": "yes", "label": "Yes (filters, AI enhance, Photoshop, etc.)"},
                    {"value": "no", "label": "No, it's unedited"},
                    {"value": "not_sure", "label": "Not sure"}
                ]
            },
            {
                "id": "workflow",
                "question": "What workflow was used to create this?",
                "type": "select",
                "required": False,
                "options": [
                    {"value": "camera_direct", "label": "Camera → Direct upload"},
                    {"value": "camera_edited", "label": "Camera → Edited → Upload"},
                    {"value": "ai_generate", "label": "AI generated (Midjourney, SD, etc.)"},
                    {"value": "ai_generate_enhance", "label": "AI generated → Enhanced"},
                    {"value": "download_reupload", "label": "Downloaded → Re-uploaded"},
                    {"value": "download_enhance", "label": "Downloaded → Enhanced → Upload"},
                    {"value": "unknown", "label": "Unknown"}
                ]
            },
            {
                "id": "user_notes",
                "question": "Any additional context?",
                "type": "text",
                "required": False,
                "placeholder": "E.g., 'I received this from a friend' or 'Found on Twitter'"
            }
        ],
        "trust_notice": (
            "Important: Your answers will be cross-verified against forensic analysis. "
            "If discrepancies are found, they will be noted in the report. "
            "Forensic evidence always takes precedence over user claims."
        )
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "5.0",
        "modules": {
            "content_database": content_db is not None,
            "visual_forensics": visual_forensics is not None,
            "explainable_ai": explainable_ai is not None,
            "claim_verification": claim_verifier is not None,
            "ai_enabled": explainable_ai.genai_enabled if explainable_ai else False
        }
    }


@app.get("/api/systems")
async def get_systems():
    """Get available systems info"""
    return {
        "version": "5.0",
        "name": "FakeTrace - Multi-Class AI Detection",
        "systems": [
            {
                "id": "content_database",
                "name": "Content Database",
                "description": "Unique ID generation, SQLite storage, analysis history",
                "status": "active" if content_db else "error",
                "features": [
                    "7 hash algorithms (MD5, SHA256, pHash, aHash, dHash, wHash, colorHash)",
                    "SQLite persistent storage",
                    "Analysis history tracking",
                    "Similar content detection",
                    "Dashboard statistics"
                ]
            },
            {
                "id": "visual_forensics",
                "name": "Advanced Visual Forensics",
                "description": "20+ manipulation detection algorithms",
                "status": "active" if visual_forensics else "error",
                "features": [
                    "Error Level Analysis (Multi-quality, Adaptive, Variance)",
                    "Noise Analysis (Level, Variance, Residual, Median filter)",
                    "Frequency Analysis (DCT, FFT, High-frequency anomaly)",
                    "Edge Analysis (Canny, Sobel, Laplacian, Density)",
                    "Clone Detection (ORB keypoints, Block matching)",
                    "Color Analysis (Histogram, Chrominance, Illumination)",
                    "Compression Analysis (Double JPEG, Block artifacts)",
                    "Statistical Analysis (Chi-square, LBP, Benford's law)"
                ]
            },
            {
                "id": "explainable_ai",
                "name": "Advanced Explainable AI",
                "description": "Multi-level scientific explanations",
                "status": "active" if explainable_ai else "error",
                "features": [
                    "Evidence-based reasoning with weighted scoring",
                    "Simple explanations for non-experts",
                    "Technical explanations for experts",
                    "Court-ready legal statements",
                    "AI visual analysis (Google Gemini)",
                    "Red flag detection with severity",
                    "Counter-argument generation",
                    "Actionable recommendations"
                ]
            },
            {
                "id": "claim_verification",
                "name": "Claim Verification System",
                "description": "Cross-verifies user claims against forensic evidence",
                "status": "active" if claim_verifier else "error",
                "features": [
                    "User context collection (source, workflow, etc.)",
                    "Independent verification of each claim",
                    "Claim consistency scoring",
                    "Discrepancy detection and reporting",
                    "Trust level advisory",
                    "Forensic evidence always takes precedence"
                ]
            }
        ]
    }


# =====================
# FRAUD ANALYZER API
# =====================

from pydantic import BaseModel
from typing import Optional as OptionalType

class FraudAnalyzeRequest(BaseModel):
    """Request model for fraud analysis"""
    message: str
    platform: OptionalType[str] = None  # whatsapp, sms, instagram, email
    message_type: OptionalType[str] = None  # banking, payment, offer, government, delivery, job

@app.post("/api/fraud/analyze")
async def analyze_fraud_message(request: FraudAnalyzeRequest):
    """
    Analyze a message for fraud/phishing indicators.
    
    Supports:
    - SMS
    - WhatsApp
    - Instagram
    - Email
    
    Categories:
    - Banking
    - Payment (UPI, Paytm, etc.)
    - Offers/Lottery
    - Government
    - Delivery
    - Job/Work-from-home
    """
    if not fraud_analyzer:
        raise HTTPException(status_code=500, detail="Fraud Analyzer not initialized")
    
    if not request.message or len(request.message.strip()) < 5:
        raise HTTPException(status_code=400, detail="Message too short for analysis")
    
    try:
        result = fraud_analyzer.analyze(
            message=request.message,
            platform=request.platform,
            message_type=request.message_type
        )
        
        # Save to database for dashboard
        if content_db:
            try:
                save_result = content_db.save_fraud_analysis(
                    fraud_id=result.get("fraud_id"),
                    analysis_result=result
                )
                result["database"] = save_result
            except Exception as db_err:
                print(f"⚠️ Failed to save fraud analysis to DB: {db_err}")
                result["database"] = {"saved": False, "error": str(db_err)}
        
        return JSONResponse(content=convert_numpy_types(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# FRAUD IMAGE ANALYSIS (OCR)
# =====================

@app.post("/api/fraud/analyze-image")
async def analyze_fraud_image(
    file: UploadFile = File(...),
    platform: str = Form(None),
    message_type: str = Form(None)
):
    """
    Analyze an image (screenshot) for fraud/phishing indicators.
    Extracts text using OCR and then performs fraud analysis.
    
    Supports image formats: PNG, JPEG, JPG, BMP, GIF, WEBP
    """
    if not fraud_analyzer:
        raise HTTPException(status_code=500, detail="Fraud Analyzer not initialized")
    
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: PNG, JPEG, JPG, BMP, GIF, WEBP"
        )
    
    try:
        import io
        from PIL import Image
        
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed (for RGBA images like PNG)
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Try OCR - use pre-loaded reader for speed
        extracted_text = ""
        ocr_method = "none"
        ocr_success = False
        
        # Use pre-loaded EasyOCR reader (fast!)
        if ocr_reader is not None:
            try:
                print("📷 Extracting text with pre-loaded EasyOCR...")
                results = ocr_reader.readtext(np.array(image))
                extracted_text = "\n".join([text for _, text, _ in results])
                ocr_method = "easyocr"
                ocr_success = True
                print(f"✅ OCR extracted {len(extracted_text)} characters")
            except Exception as ocr_err:
                print(f"⚠️ EasyOCR error: {ocr_err}")
                ocr_success = False
        
        # Fallback to pytesseract if EasyOCR not available
        if not ocr_success:
            print("⚠️ Trying pytesseract fallback...")
            try:
                import pytesseract
                
                # Configure pytesseract for Hindi + English
                custom_config = r'--oem 3 --psm 6 -l eng+hin'
                
                try:
                    extracted_text = pytesseract.image_to_string(image, config=custom_config)
                except:
                    # Fallback to English only
                    extracted_text = pytesseract.image_to_string(image)
                
                ocr_method = "pytesseract"
                ocr_success = True
                print(f"✅ Pytesseract extracted {len(extracted_text)} characters")
                
            except ImportError:
                # No OCR library available
                ocr_success = False
                ocr_method = "unavailable"
            except Exception as ocr_error:
                print(f"⚠️ Pytesseract error: {ocr_error}")
                ocr_success = False
                ocr_method = f"error: {str(ocr_error)}"
        
        # Clean extracted text
        extracted_text = extracted_text.strip() if extracted_text else ""
        
        if not ocr_success or len(extracted_text) < 5:
            return JSONResponse(content={
                "success": False,
                "error": "Could not extract text from image",
                "ocr_method": ocr_method,
                "extracted_text": extracted_text,
                "suggestion": "Please ensure the image is clear and contains readable text, or paste the message directly"
            })
        
        # Perform fraud analysis on extracted text
        result = fraud_analyzer.analyze(
            message=extracted_text,
            platform=platform,
            message_type=message_type
        )
        
        # Add OCR metadata
        result["ocr_info"] = {
            "method": ocr_method,
            "success": ocr_success,
            "extracted_text": extracted_text,
            "image_filename": file.filename,
            "image_size": f"{image.size[0]}x{image.size[1]}"
        }
        result["input_type"] = "image"
        
        # Save to database
        if content_db:
            try:
                save_result = content_db.save_fraud_analysis(
                    fraud_id=result.get("fraud_id"),
                    analysis_result=result
                )
                result["database"] = save_result
            except Exception as db_err:
                print(f"⚠️ Failed to save fraud analysis to DB: {db_err}")
                result["database"] = {"saved": False, "error": str(db_err)}
        
        return JSONResponse(content=convert_numpy_types(result))
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@app.get("/api/fraud/platforms")
async def get_fraud_platforms():
    """Get supported platforms and message types"""
    return {
        "platforms": [
            {"id": "whatsapp", "name": "WhatsApp", "icon": "💬"},
            {"id": "sms", "name": "SMS/Text Message", "icon": "📱"},
            {"id": "instagram", "name": "Instagram", "icon": "📸"},
            {"id": "email", "name": "Email", "icon": "📧"},
            {"id": "telegram", "name": "Telegram", "icon": "✈️"},
            {"id": "other", "name": "Other", "icon": "📝"}
        ],
        "message_types": [
            {"id": "banking", "name": "Banking/Account", "icon": "🏦", "description": "Bank account, credit card, loan related"},
            {"id": "payment", "name": "Payment/UPI", "icon": "💳", "description": "Paytm, PhonePe, GPay, UPI related"},
            {"id": "offer", "name": "Offer/Prize/Lottery", "icon": "🎁", "description": "Lottery, prize, cashback, discount"},
            {"id": "government", "name": "Government/Tax", "icon": "🏛️", "description": "Income tax, Aadhar, PAN, passport"},
            {"id": "delivery", "name": "Delivery/Shipping", "icon": "📦", "description": "Package, courier, order tracking"},
            {"id": "job", "name": "Job/Work-from-Home", "icon": "💼", "description": "Job offers, work from home, income opportunities"},
            {"id": "tech_support", "name": "Tech Support", "icon": "🔧", "description": "Microsoft, Apple, antivirus related"},
            {"id": "other", "name": "Other", "icon": "❓", "description": "Other type of message"}
        ]
    }


# =====================
# FRAUD DASHBOARD API
# =====================

@app.get("/api/fraud/dashboard")
async def get_fraud_dashboard():
    """Get fraud analysis dashboard statistics"""
    if not content_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        stats = content_db.get_fraud_dashboard_stats()
        return JSONResponse(content=convert_numpy_types(stats))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fraud/messages")
async def get_fraud_messages(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    legal_status: str = Query(None, description="Filter by: LEGAL, ILLEGAL, NEEDS_REVIEW")
):
    """Get all analyzed fraud messages with optional filtering"""
    if not content_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        result = content_db.get_all_fraud_messages(page=page, limit=limit, legal_status=legal_status)
        return JSONResponse(content=convert_numpy_types(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fraud/messages/{fraud_id}")
async def get_fraud_message_detail(fraud_id: str):
    """Get detailed fraud analysis by ID"""
    if not content_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        result = content_db.get_fraud_analysis(fraud_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Fraud analysis {fraud_id} not found")
        return JSONResponse(content=convert_numpy_types(result))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fraud/legal")
async def get_legal_messages(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    """Get all LEGAL (safe) messages"""
    if not content_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        result = content_db.get_all_fraud_messages(page=page, limit=limit, legal_status="LEGAL")
        return JSONResponse(content=convert_numpy_types(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fraud/illegal")
async def get_illegal_messages(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    """Get all ILLEGAL (fraud) messages"""
    if not content_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        result = content_db.get_all_fraud_messages(page=page, limit=limit, legal_status="ILLEGAL")
        return JSONResponse(content=convert_numpy_types(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fraud/review")
async def get_review_messages(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    """Get all NEEDS_REVIEW messages"""
    if not content_db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        result = content_db.get_all_fraud_messages(page=page, limit=limit, legal_status="NEEDS_REVIEW")
        return JSONResponse(content=convert_numpy_types(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# LOCATION TRACKING API
# =====================

from pydantic import BaseModel
from typing import Optional as OptionalType, Any, Dict, List, Union

class LocationData(BaseModel):
    """Location data from user (with consent)"""
    latitude: OptionalType[Union[float, int]] = None
    longitude: OptionalType[Union[float, int]] = None
    accuracy: OptionalType[Union[float, int]] = None
    altitude: OptionalType[Union[float, int]] = None
    altitude_accuracy: OptionalType[Union[float, int]] = None
    heading: OptionalType[Union[float, int]] = None
    speed: OptionalType[Union[float, int]] = None
    city: OptionalType[str] = None
    region: OptionalType[str] = None
    country: OptionalType[str] = None
    country_code: OptionalType[str] = None
    postal_code: OptionalType[str] = None
    timezone: OptionalType[str] = None
    isp: OptionalType[str] = None
    ip_address: OptionalType[str] = None
    user_agent: OptionalType[str] = None
    device_type: OptionalType[str] = None
    browser: OptionalType[str] = None
    os: OptionalType[str] = None
    screen_resolution: OptionalType[str] = None
    language: OptionalType[str] = None
    referrer: OptionalType[str] = None
    page_url: OptionalType[str] = None
    session_id: OptionalType[str] = None
    consent_given: OptionalType[bool] = True
    location_method: OptionalType[str] = "gps"
    # New fields for fingerprinting
    device_id: OptionalType[str] = None
    fingerprint: OptionalType[Dict[str, Any]] = None
    gps_denied: OptionalType[bool] = False
    gps_error: OptionalType[str] = None
    ip_failed: OptionalType[bool] = False
    asn: OptionalType[str] = None
    no_geolocation_api: OptionalType[bool] = False
    
    class Config:
        extra = "allow"  # Allow extra fields not defined in model


@app.post("/api/location/track")
async def track_user_location(request: Request):
    """
    Track user location (with consent).
    Called when user allows location permission.
    Accepts raw JSON to handle any data structure.
    """
    if not location_db:
        raise HTTPException(status_code=500, detail="Location Database not initialized")
    
    try:
        # Get raw JSON body
        location_dict = await request.json()
        
        # Get IP from request if not provided
        if not location_dict.get("ip_address"):
            # Get client IP
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                location_dict["ip_address"] = forwarded.split(",")[0].strip()
            else:
                location_dict["ip_address"] = request.client.host if request.client else None
        
        # Get user agent from request if not provided
        if not location_dict.get("user_agent"):
            location_dict["user_agent"] = request.headers.get("user-agent")
        
        result = location_db.save_location(location_dict)
        return JSONResponse(content=convert_numpy_types(result))
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/location/dashboard")
async def get_location_dashboard():
    """Get location tracking statistics"""
    if not location_db:
        raise HTTPException(status_code=500, detail="Location Database not initialized")
    
    try:
        stats = location_db.get_stats()
        return JSONResponse(content=convert_numpy_types(stats))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/location/all")
async def get_all_locations(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100)
):
    """Get all tracked locations with pagination"""
    if not location_db:
        raise HTTPException(status_code=500, detail="Location Database not initialized")
    
    try:
        result = location_db.get_all_locations(page=page, limit=limit)
        return JSONResponse(content=convert_numpy_types(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/location/{location_id}")
async def get_location_detail(location_id: str):
    """Get detailed location info by ID (LOC-XXXXXXXX format)"""
    if not location_db:
        raise HTTPException(status_code=500, detail="Location Database not initialized")
    
    try:
        result = location_db.get_location_by_id(location_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Location {location_id} not found")
        return JSONResponse(content=convert_numpy_types(result))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# DEVICE FINGERPRINT ENDPOINTS
# =====================

@app.get("/api/devices/all")
async def get_all_devices(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100)
):
    """Get all tracked devices with fingerprints"""
    if not location_db:
        raise HTTPException(status_code=500, detail="Location Database not initialized")
    
    try:
        result = location_db.get_all_devices(page=page, limit=limit)
        return JSONResponse(content=convert_numpy_types(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/devices/{device_id}")
async def get_device_detail(device_id: str):
    """Get detailed device fingerprint by ID (DEV-XXXXXXXXXX format)"""
    if not location_db:
        raise HTTPException(status_code=500, detail="Location Database not initialized")
    
    try:
        result = location_db.get_device_by_id(device_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
        return JSONResponse(content=convert_numpy_types(result))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🔍 FakeTrace v7.0 - Multi-Class AI Detection + Fraud Analysis",
        "status": "running",
        "systems": 5,
        "docs": "/docs"
    }


# =====================
# RUN SERVER
# =====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting FakeTrace v7.0 Backend...")
    print("📡 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
