# FakeTrace v7.0 - Forensic Anomaly Detection System

## 🎯 System Identity

**FakeTrace v7.0** is a **Classical Forensic Assistance Tool**:
- NOT a definitive AI detector
- Designed to **SUPPORT investigators**, NOT replace them
- Results are **probabilistic**, NOT definitive

---

## 🧠 Core Philosophy (MANDATORY)

| Principle | Description |
|-----------|-------------|
| **Conservative** | Better to say UNCERTAIN than give wrong prediction |
| **Evidence-Driven** | Each algorithm is independent forensic evidence |
| **Anomaly-First** | Prioritize "what is unusual" over "fake vs real" |
| **Legally Safe** | Never claim certainty when evidence is weak |
| **Transparent** | Every decision is explainable |

---

## 🔬 Technical Architecture

### Anomaly-First Decision Logic

```
┌─────────────────────────────────────────────────────────┐
│            FORENSIC ANOMALY ENGINE v7.0                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Image/Video                                       │
│        │                                                 │
│        ├──► Noise Analysis ──────┐                       │
│        ├──► Compression Analysis ┤                       │
│        ├──► Color Analysis ──────┤                       │
│        ├──► Frequency Analysis ──┼──► Per-Algorithm:     │
│        ├──► Texture Analysis ────┤    {                  │
│        └──► Edge Analysis ───────┘      score: 0-100,    │
│                                         anomaly: ✅/❌/⚠️, │
│                                         reason: "..."    │
│                                       }                  │
│                                                          │
│                      │                                   │
│                      ▼                                   │
│           COUNT ANOMALIES (NOT weighted avg)             │
│                      │                                   │
│                      ▼                                   │
│        ┌─────────────────────────────┐                   │
│        │  anomaly_count == 0         │──► NO ANOMALY     │
│        │  anomaly_count >= 3         │──► DETECTED       │
│        │  else                       │──► INCONCLUSIVE   │
│        └─────────────────────────────┘                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Algorithm Score Thresholds

| Score Range | Anomaly Status | Meaning |
|-------------|----------------|---------|
| **60-100** | `detected` ❌ | Clear anomaly found |
| **46-59** | `inconclusive` ⚠️ | Cannot determine |
| **0-45** | `not_detected` ✅ | No anomaly |

---

## 📊 6 Forensic Algorithms

| Algorithm | What It Analyzes | Anomaly Signature |
|-----------|------------------|-------------------|
| **Noise Analysis** | Noise consistency across blocks | Unnaturally smooth OR inconsistent |
| **Compression Analysis** | JPEG blocking artifacts | Double compression from editing |
| **Color Analysis** | Color distribution | Uniform colors, oversaturation |
| **Frequency Analysis** | FFT high/low frequency ratio | Low high-frequency content |
| **Texture Analysis** | Gradient/smoothness | Unnaturally smooth textures |
| **Edge Analysis** | Edge density & coherence | Low edge density |

---

## ⚖️ Final Verdict Logic (STRICT)

### Decision Rules

```python
if anomaly_count == 0:
    verdict = "NO_FORENSIC_ANOMALY_DETECTED"
    # All checks passed - no suspicious patterns found

elif anomaly_count >= 3:
    verdict = "FORENSIC_MANIPULATION_INDICATORS_DETECTED"
    # Multiple independent checks found anomalies

else:
    verdict = "INCONCLUSIVE_MINOR_IRREGULARITIES"
    # Some irregularities, but not enough for conclusion
```

### Verdict Meanings

| Verdict | Icon | Interpretation |
|---------|------|----------------|
| **NO_FORENSIC_ANOMALY_DETECTED** | ✅ | Media does not show statistical signs of AI generation or manipulation |
| **FORENSIC_MANIPULATION_INDICATORS_DETECTED** | ❌ | Multiple independent forensic checks detected suspicious patterns |
| **INCONCLUSIVE_MINOR_IRREGULARITIES** | ⚠️ | Some irregularities observed, but insufficient for reliable conclusion |

---

## 🎥 Video-Specific Analysis

For videos, the system:

1. **Samples 12 frames** evenly distributed across video
2. **Runs all 6 algorithms** on each frame
3. **Requires 60% consistency** - anomaly only valid if detected in ≥60% of frames
4. **Tracks temporal consistency** - frame-to-frame noise analysis

---

## 🚫 Forbidden Language

**NEVER use these phrases:**
- ❌ "This image is fake"
- ❌ "This video is AI-generated"
- ❌ "Confirmed manipulation"
- ❌ "Definitely authentic"

**ALWAYS use these phrases:**
- ✅ "Forensic anomaly detected"
- ✅ "Statistical irregularities observed"
- ✅ "No conclusive forensic evidence found"
- ✅ "Results are probabilistic, not definitive"

---

## 📤 API Response Structure

```json
{
  "status": "completed",
  "version": "7.0",
  
  "system_identity": {
    "name": "FakeTrace Forensic Assistance Tool",
    "type": "Classical Forensic Analysis",
    "purpose": "Support investigators, not replace them",
    "disclaimer": "Results are probabilistic, not definitive"
  },
  
  "evidence": {
    "algorithms": [
      {
        "name": "noise",
        "display_name": "Noise Consistency Analysis",
        "score": 70,
        "anomaly_status": "detected",
        "reason": "Unnaturally low noise level detected (common in AI-generated content)",
        "details": { ... }
      },
      {
        "name": "frequency",
        "display_name": "Frequency Domain Analysis",
        "score": 68,
        "anomaly_status": "detected",
        "reason": "Low high-frequency content detected (AI images often lack fine details)",
        "details": { ... }
      },
      ...
    ],
    "summary": {
      "total_checks": 6,
      "anomalies_detected": 3,
      "inconclusive": 1,
      "clean": 2
    }
  },
  
  "verdict": {
    "verdict": "FORENSIC_MANIPULATION_INDICATORS_DETECTED",
    "verdict_code": "FORENSIC MANIPULATION INDICATORS DETECTED",
    "interpretation": "Multiple independent forensic checks (3 of 6) detected suspicious statistical patterns...",
    "anomaly_count": 3,
    "inconclusive_count": 1,
    "clean_count": 2,
    "total_checks": 6,
    "recommendation": "Forensic anomalies detected. Recommend expert review...",
    "legal_disclaimer": "DISCLAIMER: This analysis is probabilistic, not definitive..."
  },
  
  "supporting_statistics": {
    "note": "These are supporting metrics only. They do NOT determine the verdict.",
    "average_score": 55.2,
    "interpretation": "55% average anomaly indication across 6 checks (supporting metric only)"
  },
  
  "explanation": {
    "summary": "Multiple independent forensic checks detected suspicious patterns...",
    "findings": [
      "❌ 3 algorithm(s) detected anomalies:",
      "   • Noise Consistency Analysis: Unnaturally low noise level...",
      "   • Frequency Domain Analysis: Low high-frequency content...",
      "   • Texture Smoothness Analysis: Unnaturally smooth texture...",
      "⚠️ 1 algorithm(s) returned inconclusive results:",
      "   • Color Distribution Analysis: Patterns observed but insufficient...",
      "✅ 2 algorithm(s) found no anomalies:",
      "   • Compression Artifact Analysis: Compression artifacts appear normal",
      "   • Edge Coherence Analysis: Edge patterns appear natural"
    ],
    "limitations": [
      "This analysis uses classical forensic methods, not deep learning",
      "Modern AI-generated content may evade detection",
      "Results are statistical indications, not proof",
      "High compression can mask or create artifacts",
      "Some natural photos may trigger false positives"
    ],
    "recommendation": "Forensic anomalies detected. Recommend expert review..."
  }
}
```

---

## ⚖️ Legal Triage Output (NEW)

FakeTrace now emits a `legal_profile` object for law-enforcement friendly triage. It never makes a legal determination; it summarizes forensic cues in the language investigators expect.

### `legal_profile` fields

| Field | Description |
|-------|-------------|
| `status` | `ILLEGAL_SIGNAL`, `REQUIRES_REVIEW`, `LEGAL_CLEAR`, or `NOT_ENOUGH_DATA`. |
| `status_label` | Human readable badge shown in the UI. |
| `summary` | One-line explanation of why the score landed in that band. |
| `confidence` | 0-100 heuristic confidence based on cue agreement. |
| `recommended_action` | Next best step (seize, hold, or release). |
| `cues[]` | Up to 8 law-enforcement cues (label, detail, level, metric, source, kind). |
| `supporting_metrics` | Raw `risk_points`, `protective_points`, `net_score`. |
| `notes[]` | Short operational reminders (hash files, corroborate evidence, etc.). |
| `disclaimer` | Legal-safe reminder that this is triage only. |

### Status meanings

| Status | Meaning | Typical Action |
|--------|---------|----------------|
| `ILLEGAL_SIGNAL` | High-risk cues (heartbeat failure, spectral grid, missing PRNU) exceed the escalation threshold. | Preserve evidence, notify prosecutors/cybercrime, document metrics. |
| `REQUIRES_REVIEW` | Mixed cues; needs human sign-off before use. | Hold asset; senior analyst reviews cues/metadata. |
| `LEGAL_CLEAR` | Protective cues (camera PRNU, natural CFA, authentic noise) dominate. | Document chain-of-custody and release unless other evidence contradicts. |
| `NOT_ENOUGH_DATA` | Upload quality or detectors did not provide enough signal. | Re-acquire source or request higher quality copy. |

Every cue references a measurable forensic source (FFT spectral grid, heartbeat coherence, CFA, PRNU, etc.) so officers can cite them inside affidavits without saying “AI said so.”

---

## 🖥️ UI/UX Guidelines

### Visual Indicators

| Status | Icon | Background | Use When |
|--------|------|------------|----------|
| **Clean** | ✅ | Green | No anomaly detected |
| **Anomaly** | ❌ | Red | Anomaly detected |
| **Inconclusive** | ⚠️ | Yellow | Cannot determine |

### Display Priority

1. **System Identity** - Always show disclaimer first
2. **Anomaly Counts** - Primary metrics (3 boxes: Anomalies/Inconclusive/Clean)
3. **Per-Algorithm Evidence** - Show each algorithm's result with icon
4. **Supporting Statistics** - Secondary only, clearly labeled
5. **Explanation** - Full findings and limitations

---

## 🚀 Quick Start

### Start Backend
```bash
cd FakeTrace/backend
python main.py
# Server: http://localhost:8000
```

### Start Frontend
```bash
cd FakeTrace/frontend
npm run dev
# UI: http://localhost:5173
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Main forensic analysis |
| `/api/health` | GET | System health check |
| `/api/history` | GET | Analysis history |
| `/docs` | GET | API documentation |

---

## ⚖️ Legal/Forensic Presentation

### For Court/Evidence

1. **Present as probabilistic** - "3 of 6 forensic checks detected anomalies"
2. **Include limitations** - Classical methods have known blind spots
3. **Show individual evidence** - Each algorithm's result separately
4. **Never claim certainty** - "Indicators detected" not "Confirmed fake"
5. **Recommend expert review** - Tool supports, doesn't replace experts

### Example Court Language

✅ **Good**: "The forensic analysis detected statistical irregularities in 3 of 6 independent checks. These findings indicate patterns commonly associated with AI-generated content, though this is not definitive proof."

❌ **Bad**: "The system confirmed this image is AI-generated."

---

## 🔧 System Components

### Backend Files

| File | Purpose |
|------|---------|
| `forensic_anomaly_engine_v7.py` | Main forensic engine (anomaly-first) |
| `content_database.py` | SQLite database for tracking |
| `advanced_explainable_ai.py` | Gemini AI explanations |
| `main.py` | FastAPI backend server |

### Frontend Components

| Component | Purpose |
|-----------|---------|
| `SystemIdentityCard` | Shows system disclaimer |
| `AnomalyVerdictCard` | Main verdict with anomaly counts |
| `ForensicEvidenceCard` | Per-algorithm results |
| `AlgorithmResultRow` | Single algorithm display |
| `SupportingStatsCard` | Secondary statistics |
| `ExplanationCard` | Findings and limitations |

---

## 📋 Demo Script

### For Police/Forensics Demo

1. **Show System Identity**
   - "This is a forensic assistance tool, not a definitive detector"
   - "Designed to support your investigation"

2. **Upload Real Photo**
   - Expected: "NO FORENSIC ANOMALY DETECTED"
   - Show: All 6 algorithms with ✅ icons
   - Emphasize: "No suspicious patterns found"

3. **Upload AI Image**
   - Expected: "FORENSIC MANIPULATION INDICATORS DETECTED"
   - Show: Which specific algorithms found anomalies
   - Emphasize: "3 independent checks detected irregularities"

4. **Upload Edge Case**
   - Expected: "INCONCLUSIVE"
   - Show: Mixed results (some ✅, some ⚠️)
   - Emphasize: "Not enough evidence for conclusion"

5. **Explain Limitations**
   - Classical methods, not deep learning
   - Modern AI can evade detection
   - Should supplement expert analysis

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| **7.0** | Dec 2025 | Anomaly-first logic, legally-safe language, no weighted averages |
| 6.0 | Dec 2025 | Classical forensics, removed unreliable DL |
| 5.0 | Dec 2025 | 3-class detection, claim verification |
| 4.0 | Dec 2025 | Deep learning with Grad-CAM |
| 3.0 | Dec 2025 | Advanced heatmaps |
| 2.0 | Dec 2025 | ELA + noise analysis |
| 1.0 | Dec 2025 | Basic content database |

---

## 🛡️ Reducing False Positives

### Strategies Implemented

1. **Conservative Thresholds** - Score must be ≥60 for anomaly detection
2. **Multi-Algorithm Agreement** - Need ≥3 anomalies for "DETECTED" verdict
3. **Video Temporal Consistency** - 60% of frames must agree
4. **Clear INCONCLUSIVE Category** - When evidence is mixed
5. **Per-Algorithm Reasoning** - Explains WHY each flagged

### Future Improvements (Without Deep Learning)

1. **EXIF/Metadata Analysis** - Check for AI generator signatures
2. **Statistical Distribution Analysis** - More sophisticated histogram checks
3. **Wavelet Analysis** - Multi-resolution frequency decomposition
4. **Chroma Subsampling Analysis** - JPEG chroma pattern detection
5. **Quantization Table Analysis** - JPEG compression fingerprinting

---

**Version**: 7.0  
**Date**: December 2025  
**Status**: Production Ready  
**Philosophy**: Anomaly-First, Evidence-Driven, Legally Safe

---

*"A classical forensic assistance tool, designed to support investigators, not replace them."*
