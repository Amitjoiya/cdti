# 🔍 FakeTrace v7.0 - CDTI Digital Forensics Platform

<p align="center">
  <img src="frontend/public/cdti-logo.png" alt="CDTI Logo" width="200" />
</p>

<p align="center">
  <strong>Central Detective Training Institute, Jaipur</strong><br/>
  <em>Bureau of Police Research & Development (BPRD)</em><br/>
  <em>तेजस्वि नावधीतमस्तु</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-7.0-e91e63?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-1e3a5f?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/TailwindCSS-3.4-38B2AC?style=for-the-badge&logo=tailwindcss" />
</p>

> **Advanced Digital Forensics Platform** - An anomaly-first, evidence-driven forensic investigation tool designed for law enforcement to detect deepfakes, AI-generated content, analyze fraud messages, and track digital evidence.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Location Tracking](#-location-tracking-privacy-compliant)
- [Fraud Analyzer](#-fraud-analyzer)
- [Technical Details](#-technical-details)
- [Legal Compliance](#-legal-compliance)
- [Credits](#-credits)

---

## 🎯 Features

### 🔬 Forensic Anomaly Engine v7.0
- **Anomaly-First Analysis** - Detects irregularities before making conclusions
- **Evidence-Driven Approach** - Every finding backed by measurable data
- **20+ Detection Algorithms** - Comprehensive visual forensics
- **Explainable AI** - Scientific reasoning for all verdicts

### 🤖 AI Generation Detection v3.0
| Category | Algorithms | Purpose |
|----------|------------|---------|
| **Visual** | 5 algorithms | Pattern, texture, artifact analysis |
| **Frequency** | 4 algorithms | DCT, FFT, wavelet analysis |
| **Statistical** | 3 algorithms | Noise, histogram, entropy analysis |
| **Forensic** | 3 algorithms | Compression, EXIF, consistency checks |
| **Semantic** | 2 algorithms | Face, object coherence analysis |

### 📹 Video AI Detector v1.0
- **Temporal Analysis** - Frame-to-frame consistency
- **Motion Forensics** - Optical flow anomalies
- **Face Tracking** - Identity persistence across frames
- **Lip-Sync Detection** - Audio-visual synchronization

### 💬 Fraud Analyzer v2.0
- **SMS/WhatsApp Fraud Detection** - Lottery, OTP, job scams
- **URL Deep Scan** - Phishing, malware, suspicious domains
- **Phone Number Analysis** - Spam, fraud caller identification
- **Instagram Scam Detection** - Fake giveaways, impersonation
- **Pattern Recognition** - 50+ fraud patterns database

### 📍 Location Tracking (Privacy-Compliant)
- **Pseudonymous User IDs** - SHA-256 hashed, non-reversible
- **Consent-Based Tracking** - Separate handling for Allow/Deny
- **Distance-Based History** - 100m threshold for location reuse
- **No Raw IP Storage** - Legal compliance ensured
- **Device Fingerprinting** - Canvas, WebGL, Audio fingerprints

### 📊 Dashboard & Analytics
- **Media Dashboard** - All analyzed content with verdicts
- **Fraud Dashboard** - Fraud patterns, risk distribution
- **Location Analytics** - User visit patterns (privacy-safe)
- **Export Reports** - PDF generation for legal proceedings

---

## 🏗 System Architecture

```
FakeTrace/
├── backend/                    # Python FastAPI Backend
│   ├── main.py                 # Main API server
│   ├── modules/
│   │   ├── forensic_anomaly_engine_v7.py    # Core forensic engine
│   │   ├── ai_generation_detector_v3.py     # AI detection algorithms
│   │   ├── advanced_explainable_ai.py       # Gemini AI integration
│   │   ├── fraud_analyzer.py                # Fraud detection
│   │   ├── location_database.py             # Privacy-compliant location DB
│   │   ├── content_database.py              # SQLite content storage
│   │   └── video_ai_detector.py             # Video analysis
│   ├── uploads/                # Analyzed media storage
│   ├── location/               # Location JSON files
│   ├── database.db             # Main content database
│   └── location_database.db    # Location tracking database
│
├── frontend/                   # React + Vite Frontend
│   ├── src/
│   │   ├── App.jsx             # Main application
│   │   ├── index.css           # CDTI theme styles
│   │   └── main.jsx            # React entry point
│   ├── public/
│   │   └── cdti-logo.png       # CDTI logo
│   └── package.json
│
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites
- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **Windows 10/11** or Linux

### Step 1: Clone/Download
```bash
cd FakeTrace
```

### Step 2: Backend Setup
```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure API keys
copy .env.example .env
# Edit .env with your API keys (see below)
```

### Step 3: Frontend Setup
```bash
cd frontend

# Install dependencies
npm install
```

### Step 4: Configure API Keys

Edit `backend/.env`:

```env
# Required APIs
GOOGLE_API_KEY=your_gemini_api_key
SERPAPI_KEY=your_serpapi_key

# Optional APIs (enhance accuracy)
HIVE_API_KEY=your_hive_ai_key
SIGHTENGINE_USER=your_sightengine_user
SIGHTENGINE_SECRET=your_sightengine_secret
```

| API | Purpose | Get Key |
|-----|---------|---------|
| **Google AI (Gemini)** | AI analysis & explanations | [aistudio.google.com](https://aistudio.google.com/apikey) |
| **SerpAPI** | Reverse image search | [serpapi.com](https://serpapi.com/) |
| Hive AI | Deepfake detection | [thehive.ai](https://thehive.ai/) |
| SightEngine | Content moderation | [sightengine.com](https://sightengine.com/) |

> **Minimum Required:** Google AI + SerpAPI

---

## 📖 Usage Guide

### Starting the Application

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```
Backend starts at: `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
Frontend starts at: `http://localhost:5173`

### Using FakeTrace

#### 1. Analyze Media
1. Go to **Analyze** tab
2. Drag & drop or click to upload image/video
3. Wait for forensic analysis (10-30 seconds)
4. Review detailed findings:
   - **Verdict**: Authentic/Suspicious/Manipulated
   - **AI Detection**: Human-made vs AI-generated
   - **Visual Forensics**: 20+ anomaly checks
   - **Origin Tracking**: Reverse image search results

#### 2. Fraud Analyzer
1. Go to **Fraud Analyzer** tab
2. Paste suspicious message (SMS/WhatsApp/Email)
3. Click "Analyze for Fraud"
4. Get detailed risk assessment:
   - Risk Level (Critical/High/Medium/Low)
   - Fraud Type (Lottery, OTP, Phishing, etc.)
   - Suspicious Elements highlighted
   - Safe/Unsafe indicators

#### 3. Fraud Dashboard
- View all analyzed fraud messages
- Filter by risk level
- Export for reporting

#### 4. Media Dashboard
- Browse all analyzed media
- Filter by verdict
- View detailed analysis history

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Media Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Analyze image/video |
| GET | `/api/dashboard` | Get dashboard stats |
| GET | `/api/dashboard/content` | List analyzed content |

#### Fraud Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/fraud/analyze` | Analyze suspicious message |
| GET | `/api/fraud/dashboard` | Fraud statistics |
| GET | `/api/fraud/messages` | List analyzed messages |

#### Location Tracking
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/location/track` | Track user location |
| GET | `/api/location/stats` | Location statistics |
| GET | `/api/location/all` | List all locations |

### Example: Analyze Image
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@suspicious_image.jpg"
```

### Example: Analyze Fraud Message
```bash
curl -X POST http://localhost:8000/api/fraud/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "Congratulations! You won Rs 50 Lakhs lottery. Send Rs 5000 to claim."}'
```

---

## 📍 Location Tracking (Privacy-Compliant)

### How It Works

1. **User Visits Website** → Location permission popup appears

2. **If User ALLOWS:**
   - GPS coordinates captured (or IP fallback on Desktop)
   - Pseudonymous User ID generated: `USR-XXXXXXXX`
   - Location saved with city, country, coordinates
   - File created: `LOC-XXXXXXXX.json`

3. **If User DENIES:**
   - NO coordinates stored
   - Anonymous ID generated: `ANON-XXXXXXXX`
   - Only timezone-based region saved
   - File created: `ANON-session.json`

### Privacy Features
| Feature | Implementation |
|---------|----------------|
| User ID | SHA-256 hash of device fingerprint (non-reversible) |
| IP Address | Never stored |
| Coordinates | Only with explicit consent |
| Fingerprint | Hashed, not raw |
| Distance Threshold | 100m (same location reuse) |

### Database Schema
```sql
-- Users (pseudonymous)
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    consent_status TEXT,
    fingerprint_hash TEXT
);

-- Locations (allowed users only)
CREATE TABLE location_records (
    location_id TEXT PRIMARY KEY,
    user_id TEXT,
    latitude REAL,
    longitude REAL,
    city TEXT,
    country TEXT
);

-- Anonymous visits (denied users)
CREATE TABLE anonymous_visits (
    anonymous_id TEXT,
    session_id TEXT,
    approximate_region TEXT,
    denial_reason TEXT
);
```

---

## 💬 Fraud Analyzer

### Supported Fraud Types

| Type | Detection Patterns |
|------|-------------------|
| **Lottery Scam** | Prize money, claim now, winner selected |
| **OTP Fraud** | Share OTP, verify account, bank alert |
| **Job Scam** | Work from home, easy money, Rs/day |
| **Loan Fraud** | Instant loan, no documents, low interest |
| **KYC Scam** | Update KYC, account blocked, verify |
| **Investment Fraud** | Double money, guaranteed returns |
| **Phishing** | Suspicious URLs, fake domains |
| **Delivery Scam** | Package pending, pay charges |

### Risk Levels
- 🔴 **CRITICAL** (80-100): Definite fraud
- 🟠 **HIGH** (60-79): Very likely fraud
- 🟡 **MEDIUM** (40-59): Suspicious
- 🟢 **LOW** (0-39): Probably safe

---

## 🔧 Technical Details

### Detection Algorithms

#### Visual Forensics
1. **ELA (Error Level Analysis)** - Compression artifact detection
2. **Noise Analysis** - Inconsistent noise patterns
3. **Edge Detection** - Unnatural edge artifacts
4. **Clone Detection** - Copy-paste regions
5. **Metadata Analysis** - EXIF inconsistencies

#### AI Generation Detection
1. **GAN Fingerprint** - StyleGAN, ProGAN patterns
2. **Diffusion Artifacts** - DALL-E, Midjourney, SD signatures
3. **Face Analysis** - Asymmetry, eye reflection
4. **Texture Analysis** - Unnatural smoothness/patterns
5. **Frequency Analysis** - FFT, DCT anomalies

#### Video Forensics
1. **Temporal Consistency** - Frame-to-frame changes
2. **Motion Vectors** - Optical flow anomalies
3. **Face Tracking** - Identity persistence
4. **Lip Sync** - Audio-visual mismatch
5. **Compression Analysis** - Re-encoding detection

### Technology Stack
| Component | Technology |
|-----------|------------|
| Backend | Python 3.10, FastAPI, SQLite |
| Frontend | React 18, Vite, TailwindCSS |
| AI/ML | Google Gemini, OpenCV, NumPy |
| OCR | EasyOCR (English + Hindi) |
| Database | SQLite (content + location) |

---

## ⚖️ Legal Compliance

### Data Protection
- ✅ No personally identifiable information stored
- ✅ Pseudonymous user identification
- ✅ Consent-based data collection
- ✅ No raw IP addresses stored
- ✅ Hashed fingerprints (non-reversible)

### Evidence Handling
- ✅ SHA-256 file hashes for integrity
- ✅ Timestamped analysis records
- ✅ Chain of custody maintenance
- ✅ PDF report generation for legal use

### Applicable Laws
- Information Technology Act, 2000 (India)
- IT (Reasonable Security Practices) Rules, 2011
- GDPR compliance (for international use)

---

## 👥 Credits

### Developed For
**Central Detective Training Institute (CDTI)**
- Bureau of Police Research & Development
- Ministry of Home Affairs, Government of India
- Jaipur, Rajasthan

### Third-Party Services
- Google Gemini AI
- SerpAPI
- OpenStreetMap Nominatim
- IP Geolocation APIs

---

## 📞 Support

For technical support or queries:
- **Website**: https://bprd.nic.in/cdti

---

## 📄 License

This software is developed for law enforcement training and investigation purposes under CDTI, BPRD. Unauthorized commercial use is prohibited.

---

<p align="center">
  <strong>🔍 FakeTrace v7.0</strong><br/>
  <em>Trace Deepfakes, Expose Truth</em><br/>
  <em>तेजस्वि नावधीतमस्तु</em>
</p>
