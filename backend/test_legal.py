"""Quick test for legal profile logic"""
import sys
sys.path.insert(0, '.')
from main import build_legal_profile

def test_profile(name, **kwargs):
    result = build_legal_profile(**kwargs)
    print(f"=== {name} ===")
    print(f"Status: {result['status']} | Confidence: {result['confidence']}%")
    print(f"Cues: {len(result.get('cues', []))}")
    print()

# Test 1: Skull image (AI-generated)
test_profile(
    "SKULL IMAGE (AI-Generated)",
    verdict="UNCERTAIN",
    ai_generation={
        "conclusion": "LIKELY_AI_GENERATED",
        "confidence": 84,
        "ai_detection": [
            {"verdict": "likely_ai", "confidence": 75, "algorithm": "frequency_artifacts"},
            {"verdict": "likely_ai", "confidence": 68, "algorithm": "dct_analysis"}
        ]
    },
    ai_detection=[
        {"verdict": "likely_ai", "confidence": 75},
        {"verdict": "likely_ai", "confidence": 68}
    ],
    evidence={"summary": {}, "content_type": "image"}
)

# Test 2: Natural video
test_profile(
    "NATURAL VIDEO",
    verdict="LIKELY_AUTHENTIC",
    ai_generation={
        "conclusion": "LIKELY_NATURAL",
        "confidence": 90,
        "ai_detection": [
            {"verdict": "likely_natural", "confidence": 90},
            {"verdict": "likely_natural", "confidence": 85},
            {"verdict": "likely_natural", "confidence": 88}
        ]
    },
    ai_detection=[
        {"verdict": "likely_natural", "confidence": 90},
        {"verdict": "likely_natural", "confidence": 85}
    ],
    evidence={"summary": {}, "content_type": "video"}
)

# Test 3: Deepfake video
test_profile(
    "DEEPFAKE VIDEO",
    verdict="LIKELY_DEEPFAKE",
    ai_generation={
        "conclusion": "LIKELY_AI_GENERATED",
        "confidence": 95,
        "ai_detection": [
            {"verdict": "likely_ai", "confidence": 95},
            {"verdict": "likely_ai", "confidence": 92},
            {"verdict": "likely_ai", "confidence": 88},
            {"verdict": "likely_ai", "confidence": 90}
        ]
    },
    ai_detection=[
        {"verdict": "likely_ai", "confidence": 95},
        {"verdict": "likely_ai", "confidence": 92},
        {"verdict": "likely_ai", "confidence": 88}
    ],
    evidence={"summary": {}, "content_type": "video"}
)
