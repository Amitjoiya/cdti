"""
Advanced Content Database System v3.0
Ultra-Detailed Fingerprinting & Storage
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    import imagehash
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ContentDatabase:
    """
    Ultra-Detailed Content Database
    
    Features:
    - 7 Hash Algorithms for fingerprinting
    - SQLite persistent storage
    - Detailed analysis history
    - Dashboard statistics
    - Similar content detection
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content_database.db")
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_analyzed TIMESTAMP,
                analysis_count INTEGER DEFAULT 1,
                original_filename TEXT,
                file_size INTEGER,
                file_extension TEXT,
                md5_hash TEXT,
                sha256_hash TEXT,
                phash TEXT,
                ahash TEXT,
                dhash TEXT,
                whash TEXT,
                colorhash TEXT,
                width INTEGER,
                height INTEGER,
                is_video BOOLEAN,
                duration REAL,
                fps REAL,
                frame_count INTEGER,
                verdict TEXT,
                confidence REAL,
                fingerprint_json TEXT,
                legal_status TEXT,
                legal_confidence REAL,
                legal_profile_json TEXT
            )
        """)

        # Add new columns if database was created before legal fields existed
        for statement in [
            "ALTER TABLE content ADD COLUMN legal_status TEXT",
            "ALTER TABLE content ADD COLUMN legal_confidence REAL",
            "ALTER TABLE content ADD COLUMN legal_profile_json TEXT"
        ]:
            try:
                cursor.execute(statement)
            except sqlite3.OperationalError:
                pass
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verdict TEXT,
                confidence REAL,
                scores_json TEXT,
                FOREIGN KEY (content_id) REFERENCES content(id)
            )
        """)

        # Fraud Message Analysis Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fraud_messages (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_preview TEXT,
                message_hash TEXT,
                platform TEXT,
                message_type TEXT,
                urls_found TEXT,
                contact_numbers TEXT,
                matched_templates TEXT,
                entities_json TEXT,
                verdict TEXT,
                risk_level TEXT,
                risk_score REAL,
                confidence REAL,
                legal_status TEXT,
                checks_json TEXT,
                recommendations_json TEXT,
                full_result_json TEXT
            )
        """)

        # User Location Tracking Table (with consent)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                latitude REAL,
                longitude REAL,
                accuracy REAL,
                altitude REAL,
                altitude_accuracy REAL,
                heading REAL,
                speed REAL,
                city TEXT,
                region TEXT,
                country TEXT,
                country_code TEXT,
                postal_code TEXT,
                timezone TEXT,
                isp TEXT,
                ip_address TEXT,
                user_agent TEXT,
                device_type TEXT,
                browser TEXT,
                os TEXT,
                screen_resolution TEXT,
                language TEXT,
                referrer TEXT,
                page_url TEXT,
                consent_given BOOLEAN DEFAULT TRUE,
                location_method TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_content(self, file_path: str, filename: str = None) -> Dict:
        """Register content and generate fingerprint"""
        result = {
            "content_id": None,
            "fingerprint": {},
            "is_known": False,
            "analysis_count": 1,
            "first_seen": None,
            "registration_time": datetime.now().isoformat()
        }
        
        path = Path(file_path)
        if not path.exists():
            result["error"] = "File not found"
            return result
        
        fingerprint = self._generate_fingerprint(file_path, filename)
        result["fingerprint"] = fingerprint
        result["content_id"] = fingerprint.get("content_id")
        
        if not result["content_id"]:
            result["error"] = "Failed to generate content ID"
            return result
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, analysis_count, created_at, verdict, confidence FROM content WHERE id = ?",
            (result["content_id"],)
        )
        existing = cursor.fetchone()
        
        if existing:
            result["is_known"] = True
            result["analysis_count"] = existing[1] + 1
            result["first_seen"] = existing[2]
            result["previous_verdict"] = existing[3]
            result["previous_confidence"] = existing[4]
            
            cursor.execute(
                "UPDATE content SET analysis_count = ?, last_analyzed = ? WHERE id = ?",
                (result["analysis_count"], datetime.now().isoformat(), result["content_id"])
            )
        else:
            result["first_seen"] = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO content (
                    id, original_filename, file_size, file_extension,
                    md5_hash, sha256_hash, phash, ahash, dhash, whash, colorhash,
                    width, height, is_video, duration, fps, frame_count,
                    fingerprint_json, last_analyzed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result["content_id"],
                fingerprint.get("filename"),
                fingerprint.get("file_size"),
                fingerprint.get("extension"),
                fingerprint.get("md5"),
                fingerprint.get("sha256"),
                fingerprint.get("phash"),
                fingerprint.get("ahash"),
                fingerprint.get("dhash"),
                fingerprint.get("whash"),
                fingerprint.get("colorhash"),
                fingerprint.get("width"),
                fingerprint.get("height"),
                fingerprint.get("is_video"),
                fingerprint.get("duration"),
                fingerprint.get("fps"),
                fingerprint.get("frame_count"),
                json.dumps(fingerprint),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
        
        return result
    
    def _generate_fingerprint(self, file_path: str, filename: str = None) -> Dict:
        """Generate ultra-detailed fingerprint"""
        path = Path(file_path)
        
        fingerprint = {
            "content_id": None,
            "generated_at": datetime.now().isoformat(),
            "filename": filename or path.name,
            "file_size": path.stat().st_size,
            "file_size_formatted": self._format_size(path.stat().st_size),
            "extension": path.suffix.lower(),
            "is_video": path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif'],
            "md5": None,
            "sha256": None,
            "phash": None,
            "ahash": None,
            "dhash": None,
            "whash": None,
            "colorhash": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "megapixels": None,
            "color_mode": None,
            "duration": None,
            "fps": None,
            "frame_count": None,
        }
        
        # Cryptographic Hashes
        with open(file_path, 'rb') as f:
            content = f.read()
            fingerprint["md5"] = hashlib.md5(content).hexdigest()
            fingerprint["sha256"] = hashlib.sha256(content).hexdigest()
        
        img = None
        
        if fingerprint["is_video"] and CV2_AVAILABLE:
            cap = cv2.VideoCapture(file_path)
            fingerprint["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
            fingerprint["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fingerprint["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fingerprint["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fingerprint["fps"] > 0 and fingerprint["frame_count"] > 0:
                fingerprint["duration"] = round(fingerprint["frame_count"] / fingerprint["fps"], 2)
            
            mid_frame = fingerprint["frame_count"] // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
        
        elif PIL_AVAILABLE:
            try:
                img = Image.open(file_path)
                fingerprint["width"] = img.width
                fingerprint["height"] = img.height
                fingerprint["color_mode"] = img.mode
            except Exception as e:
                fingerprint["image_error"] = str(e)
        
        if img and PIL_AVAILABLE:
            try:
                fingerprint["phash"] = str(imagehash.phash(img))
                fingerprint["ahash"] = str(imagehash.average_hash(img))
                fingerprint["dhash"] = str(imagehash.dhash(img))
                fingerprint["whash"] = str(imagehash.whash(img))
                try:
                    fingerprint["colorhash"] = str(imagehash.colorhash(img))
                except:
                    pass
            except Exception as e:
                fingerprint["hash_error"] = str(e)
        
        if fingerprint["width"] and fingerprint["height"]:
            fingerprint["aspect_ratio"] = f"{fingerprint['width']}:{fingerprint['height']}"
            fingerprint["megapixels"] = round((fingerprint["width"] * fingerprint["height"]) / 1000000, 2)
        
        if fingerprint["phash"]:
            fingerprint["content_id"] = f"FT-{fingerprint['phash'][:8].upper()}"
        elif fingerprint["md5"]:
            fingerprint["content_id"] = f"FT-{fingerprint['md5'][:8].upper()}"
        
        return fingerprint
    
    def _format_size(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"
    
    def update_analysis(self, content_id: str, verdict: str, confidence: float,
                       analysis_results: dict = None, legal_profile: dict = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        legal_status = None
        legal_conf = None
        legal_profile_json = None
        if legal_profile:
            # Map status from build_legal_profile to dashboard-friendly labels
            raw_status = legal_profile.get("status")
            status_map = {
                "LEGAL_CLEAR": "LEGAL",
                "ILLEGAL_SIGNAL": "ILLEGAL",
                "REQUIRES_REVIEW": "NEEDS_REVIEW",
                "NOT_ENOUGH_DATA": "NOT_ENOUGH_DATA"
            }
            legal_status = status_map.get(raw_status, raw_status)
            legal_conf = legal_profile.get("confidence")
            try:
                legal_profile_json = json.dumps(legal_profile, default=str)
            except TypeError:
                legal_profile_json = json.dumps(str(legal_profile))
        
        cursor.execute(
            "UPDATE content SET verdict = ?, confidence = ?, legal_status = ?, legal_confidence = ?, legal_profile_json = ?, last_analyzed = ? WHERE id = ?",
            (verdict, confidence, legal_status, legal_conf, legal_profile_json, datetime.now().isoformat(), content_id)
        )
        
        cursor.execute(
            "INSERT INTO analysis_history (content_id, verdict, confidence, scores_json) VALUES (?, ?, ?, ?)",
            (content_id, verdict, confidence, json.dumps(analysis_results) if analysis_results else None)
        )
        
        conn.commit()
        conn.close()
    
    def get_content_info(self, content_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, original_filename, file_size, created_at, last_analyzed,
                   analysis_count, verdict, confidence, width, height,
                   is_video, fingerprint_json, legal_status, legal_confidence,
                   legal_profile_json
            FROM content WHERE id = ?
        """, (content_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "content_id": row[0],
            "filename": row[1],
            "file_size": row[2],
            "first_seen": row[3],
            "last_analyzed": row[4],
            "analysis_count": row[5],
            "verdict": row[6],
            "confidence": row[7],
            "width": row[8],
            "height": row[9],
            "is_video": row[10],
            "fingerprint": json.loads(row[11]) if row[11] else {},
            "legal_status": row[12],
            "legal_confidence": row[13],
            "legal_profile": json.loads(row[14]) if row[14] else None
        }
    
    def get_dashboard_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {
            "total_content": 0,
            "total_analyses": 0,
            "today_count": 0,
            "verdict_distribution": {},
            "legal_status_distribution": {},
            "top_analyzed": [],
            "recent_analyses": []
        }
        
        cursor.execute("SELECT COUNT(*) FROM content")
        stats["total_content"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(analysis_count) FROM content")
        result = cursor.fetchone()[0]
        stats["total_analyses"] = result if result else 0
        
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM content WHERE date(last_analyzed) = ?", (today,))
        stats["today_count"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT verdict, COUNT(*) FROM content WHERE verdict IS NOT NULL GROUP BY verdict")
        for row in cursor.fetchall():
            stats["verdict_distribution"][row[0]] = row[1]
        
        # Legal status distribution for law enforcement dashboard
        cursor.execute("SELECT legal_status, COUNT(*) FROM content WHERE legal_status IS NOT NULL GROUP BY legal_status")
        for row in cursor.fetchall():
            stats["legal_status_distribution"][row[0]] = row[1]
        
        cursor.execute("""
            SELECT id, original_filename, analysis_count, verdict, confidence, legal_status, legal_confidence
            FROM content ORDER BY analysis_count DESC LIMIT 10
        """)
        for row in cursor.fetchall():
            stats["top_analyzed"].append({
                "content_id": row[0], "filename": row[1], "analysis_count": row[2],
                "verdict": row[3], "confidence": row[4], "legal_status": row[5], "legal_confidence": row[6]
            })
        
        conn.close()
        return stats
    
    def get_all_content(self, page: int = 1, limit: int = 20) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        offset = (page - 1) * limit
        
        cursor.execute("""
            SELECT id, original_filename, file_size, created_at, last_analyzed,
                   analysis_count, verdict, confidence, width, height, is_video,
                   legal_status, legal_confidence
            FROM content ORDER BY last_analyzed DESC LIMIT ? OFFSET ?
        """, (limit, offset))
        
        content = []
        for row in cursor.fetchall():
            content.append({
                "content_id": row[0], "filename": row[1], "file_size": row[2],
                "first_seen": row[3], "last_analyzed": row[4], "analysis_count": row[5],
                "verdict": row[6], "confidence": row[7], "width": row[8],
                "height": row[9], "is_video": row[10], "legal_status": row[11], "legal_confidence": row[12]
            })
        
        cursor.execute("SELECT COUNT(*) FROM content")
        total = cursor.fetchone()[0]
        conn.close()
        
        return {"content": content, "page": page, "limit": limit, "total": total}
    
    def search_content(self, query: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        search_term = f"%{query}%"
        
        cursor.execute("""
            SELECT id, original_filename, analysis_count, verdict, confidence, last_analyzed,
                   legal_status, legal_confidence
            FROM content WHERE id LIKE ? OR original_filename LIKE ?
            ORDER BY analysis_count DESC LIMIT 50
        """, (search_term, search_term))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "content_id": row[0], "filename": row[1], "analysis_count": row[2],
                "verdict": row[3], "confidence": row[4], "last_analyzed": row[5],
                "legal_status": row[6], "legal_confidence": row[7]
            })
        
        conn.close()
        return results
    
    def find_similar(self, content_id: str, threshold: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT phash FROM content WHERE id = ?", (content_id,))
        row = cursor.fetchone()
        
        if not row or not row[0]:
            conn.close()
            return []
        
        target_hash = row[0]
        cursor.execute("SELECT id, original_filename, phash, verdict, confidence FROM content WHERE phash IS NOT NULL")
        
        similar = []
        for row in cursor.fetchall():
            if row[0] == content_id:
                continue
            try:
                hash1 = imagehash.hex_to_hash(target_hash)
                hash2 = imagehash.hex_to_hash(row[2])
                distance = hash1 - hash2
                
                if distance <= threshold:
                    similar.append({
                        "content_id": row[0], "filename": row[1],
                        "similarity": round(100 - (distance / 64 * 100), 2),
                        "distance": distance, "verdict": row[3], "confidence": row[4]
                    })
            except:
                pass
        
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        conn.close()
        return similar[:10]

    # ==================== FRAUD MESSAGE METHODS ====================

    def save_fraud_analysis(self, fraud_id: str, analysis_result: Dict) -> Dict:
        """Save fraud message analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        assessment = analysis_result.get("fraud_assessment", {})
        verdict = assessment.get("verdict", "UNKNOWN")
        risk_level = assessment.get("risk_level", "unknown")
        risk_score = assessment.get("risk_score", 0)
        confidence = assessment.get("confidence", 0)

        # Map verdict to legal status for dashboard
        legal_status_map = {
            "HIGH_RISK_FRAUD": "ILLEGAL",
            "LIKELY_FRAUD": "ILLEGAL",
            "SUSPICIOUS": "NEEDS_REVIEW",
            "LIKELY_SAFE": "LEGAL",
            "UNCERTAIN": "NEEDS_REVIEW",
            "INSUFFICIENT_DATA": "NOT_ENOUGH_DATA"
        }
        legal_status = legal_status_map.get(verdict, "NEEDS_REVIEW")

        cursor.execute("""
            INSERT OR REPLACE INTO fraud_messages (
                id, message_preview, message_hash, platform, message_type,
                urls_found, contact_numbers, matched_templates, entities_json,
                verdict, risk_level, risk_score, confidence, legal_status,
                checks_json, recommendations_json, full_result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fraud_id,
            analysis_result.get("message_preview", ""),
            hashlib.md5(analysis_result.get("message_preview", "").encode()).hexdigest(),
            analysis_result.get("platform"),
            analysis_result.get("message_type") or analysis_result.get("detected_type"),
            json.dumps(analysis_result.get("urls_found", [])),
            json.dumps(analysis_result.get("contact_numbers", [])),
            json.dumps(analysis_result.get("matched_templates", [])),
            json.dumps(analysis_result.get("entities", {})),
            verdict,
            risk_level,
            risk_score,
            confidence,
            legal_status,
            json.dumps(analysis_result.get("checks", [])),
            json.dumps(analysis_result.get("recommendations", [])),
            json.dumps(analysis_result, default=str),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return {
            "fraud_id": fraud_id,
            "legal_status": legal_status,
            "saved": True
        }

    def get_fraud_analysis(self, fraud_id: str) -> Optional[Dict]:
        """Get a specific fraud analysis by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, created_at, message_preview, platform, message_type,
                   urls_found, contact_numbers, matched_templates, entities_json,
                   verdict, risk_level, risk_score, confidence, legal_status,
                   checks_json, recommendations_json, full_result_json
            FROM fraud_messages WHERE id = ?
        """, (fraud_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "fraud_id": row[0],
            "created_at": row[1],
            "message_preview": row[2],
            "platform": row[3],
            "message_type": row[4],
            "urls_found": json.loads(row[5]) if row[5] else [],
            "contact_numbers": json.loads(row[6]) if row[6] else [],
            "matched_templates": json.loads(row[7]) if row[7] else [],
            "entities": json.loads(row[8]) if row[8] else {},
            "verdict": row[9],
            "risk_level": row[10],
            "risk_score": row[11],
            "confidence": row[12],
            "legal_status": row[13],
            "checks": json.loads(row[14]) if row[14] else [],
            "recommendations": json.loads(row[15]) if row[15] else [],
            "full_result": json.loads(row[16]) if row[16] else {}
        }

    def get_all_fraud_messages(self, page: int = 1, limit: int = 20, 
                                legal_status: str = None) -> Dict:
        """Get all fraud messages with optional filter"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        offset = (page - 1) * limit

        if legal_status:
            cursor.execute("""
                SELECT id, created_at, message_preview, platform, message_type,
                       verdict, risk_level, risk_score, confidence, legal_status,
                       matched_templates
                FROM fraud_messages WHERE legal_status = ?
                ORDER BY created_at DESC LIMIT ? OFFSET ?
            """, (legal_status, limit, offset))
        else:
            cursor.execute("""
                SELECT id, created_at, message_preview, platform, message_type,
                       verdict, risk_level, risk_score, confidence, legal_status,
                       matched_templates
                FROM fraud_messages ORDER BY created_at DESC LIMIT ? OFFSET ?
            """, (limit, offset))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                "fraud_id": row[0],
                "created_at": row[1],
                "message_preview": row[2],
                "platform": row[3],
                "message_type": row[4],
                "verdict": row[5],
                "risk_level": row[6],
                "risk_score": row[7],
                "confidence": row[8],
                "legal_status": row[9],
                "matched_templates": json.loads(row[10]) if row[10] else []
            })

        if legal_status:
            cursor.execute("SELECT COUNT(*) FROM fraud_messages WHERE legal_status = ?", (legal_status,))
        else:
            cursor.execute("SELECT COUNT(*) FROM fraud_messages")
        total = cursor.fetchone()[0]
        conn.close()

        return {"messages": messages, "page": page, "limit": limit, "total": total}

    def get_fraud_dashboard_stats(self) -> Dict:
        """Get fraud analysis statistics for dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {
            "total_analyzed": 0,
            "today_count": 0,
            "legal_count": 0,
            "illegal_count": 0,
            "needs_review_count": 0,
            "verdict_distribution": {},
            "platform_distribution": {},
            "risk_level_distribution": {},
            "recent_analyses": [],
            "top_scam_templates": []
        }

        cursor.execute("SELECT COUNT(*) FROM fraud_messages")
        stats["total_analyzed"] = cursor.fetchone()[0]

        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM fraud_messages WHERE date(created_at) = ?", (today,))
        stats["today_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fraud_messages WHERE legal_status = 'LEGAL'")
        stats["legal_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fraud_messages WHERE legal_status = 'ILLEGAL'")
        stats["illegal_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fraud_messages WHERE legal_status = 'NEEDS_REVIEW'")
        stats["needs_review_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT verdict, COUNT(*) FROM fraud_messages WHERE verdict IS NOT NULL GROUP BY verdict")
        for row in cursor.fetchall():
            stats["verdict_distribution"][row[0]] = row[1]

        cursor.execute("SELECT platform, COUNT(*) FROM fraud_messages WHERE platform IS NOT NULL GROUP BY platform")
        for row in cursor.fetchall():
            stats["platform_distribution"][row[0]] = row[1]

        cursor.execute("SELECT risk_level, COUNT(*) FROM fraud_messages WHERE risk_level IS NOT NULL GROUP BY risk_level")
        for row in cursor.fetchall():
            stats["risk_level_distribution"][row[0]] = row[1]

        cursor.execute("""
            SELECT id, message_preview, platform, verdict, risk_score, legal_status, created_at
            FROM fraud_messages ORDER BY created_at DESC LIMIT 10
        """)
        for row in cursor.fetchall():
            stats["recent_analyses"].append({
                "fraud_id": row[0],
                "message_preview": row[1],
                "platform": row[2],
                "verdict": row[3],
                "risk_score": row[4],
                "legal_status": row[5],
                "created_at": row[6]
            })

        # Count template matches
        cursor.execute("SELECT matched_templates FROM fraud_messages WHERE matched_templates IS NOT NULL")
        template_counts = {}
        for row in cursor.fetchall():
            try:
                templates = json.loads(row[0])
                for t in templates:
                    template_counts[t] = template_counts.get(t, 0) + 1
            except:
                pass
        stats["top_scam_templates"] = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        conn.close()
        return stats

    # =====================
    # LOCATION TRACKING METHODS
    # =====================

    def save_user_location(self, location_data: Dict) -> Dict:
        """
        Save user location data (with consent).
        
        Args:
            location_data: Dictionary containing location and device info
            
        Returns:
            Dictionary with save status and location_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            import uuid
            session_id = location_data.get("session_id") or str(uuid.uuid4())[:8]
            
            cursor.execute("""
                INSERT INTO user_locations (
                    session_id, latitude, longitude, accuracy, altitude,
                    altitude_accuracy, heading, speed, city, region,
                    country, country_code, postal_code, timezone, isp,
                    ip_address, user_agent, device_type, browser, os,
                    screen_resolution, language, referrer, page_url,
                    consent_given, location_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                location_data.get("latitude"),
                location_data.get("longitude"),
                location_data.get("accuracy"),
                location_data.get("altitude"),
                location_data.get("altitude_accuracy"),
                location_data.get("heading"),
                location_data.get("speed"),
                location_data.get("city"),
                location_data.get("region"),
                location_data.get("country"),
                location_data.get("country_code"),
                location_data.get("postal_code"),
                location_data.get("timezone"),
                location_data.get("isp"),
                location_data.get("ip_address"),
                location_data.get("user_agent"),
                location_data.get("device_type"),
                location_data.get("browser"),
                location_data.get("os"),
                location_data.get("screen_resolution"),
                location_data.get("language"),
                location_data.get("referrer"),
                location_data.get("page_url"),
                location_data.get("consent_given", True),
                location_data.get("location_method", "gps")
            ))
            
            location_id = cursor.lastrowid
            conn.commit()
            
            return {
                "saved": True,
                "location_id": location_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "saved": False,
                "error": str(e)
            }
        finally:
            conn.close()

    def get_all_locations(self, page: int = 1, limit: int = 50) -> Dict:
        """Get all tracked locations with pagination"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        offset = (page - 1) * limit
        
        cursor.execute("SELECT COUNT(*) FROM user_locations")
        total = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT id, session_id, created_at, latitude, longitude, accuracy,
                   city, region, country, ip_address, user_agent, device_type,
                   browser, os, location_method
            FROM user_locations
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        locations = []
        for row in cursor.fetchall():
            locations.append({
                "id": row[0],
                "session_id": row[1],
                "created_at": row[2],
                "latitude": row[3],
                "longitude": row[4],
                "accuracy": row[5],
                "city": row[6],
                "region": row[7],
                "country": row[8],
                "ip_address": row[9],
                "user_agent": row[10],
                "device_type": row[11],
                "browser": row[12],
                "os": row[13],
                "location_method": row[14]
            })
        
        conn.close()
        
        return {
            "locations": locations,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit
        }

    def get_location_by_id(self, location_id: int) -> Optional[Dict]:
        """Get detailed location info by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM user_locations WHERE id = ?
        """, (location_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        columns = [
            "id", "session_id", "created_at", "latitude", "longitude",
            "accuracy", "altitude", "altitude_accuracy", "heading", "speed",
            "city", "region", "country", "country_code", "postal_code",
            "timezone", "isp", "ip_address", "user_agent", "device_type",
            "browser", "os", "screen_resolution", "language", "referrer",
            "page_url", "consent_given", "location_method"
        ]
        
        return dict(zip(columns, row))

    def get_location_stats(self) -> Dict:
        """Get location tracking statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {
            "total_tracked": 0,
            "today_count": 0,
            "unique_sessions": 0,
            "country_distribution": {},
            "device_distribution": {},
            "browser_distribution": {},
            "recent_locations": []
        }
        
        cursor.execute("SELECT COUNT(*) FROM user_locations")
        stats["total_tracked"] = cursor.fetchone()[0]
        
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM user_locations WHERE date(created_at) = ?", (today,))
        stats["today_count"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM user_locations")
        stats["unique_sessions"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT country, COUNT(*) FROM user_locations WHERE country IS NOT NULL GROUP BY country")
        for row in cursor.fetchall():
            stats["country_distribution"][row[0]] = row[1]
        
        cursor.execute("SELECT device_type, COUNT(*) FROM user_locations WHERE device_type IS NOT NULL GROUP BY device_type")
        for row in cursor.fetchall():
            stats["device_distribution"][row[0]] = row[1]
        
        cursor.execute("SELECT browser, COUNT(*) FROM user_locations WHERE browser IS NOT NULL GROUP BY browser")
        for row in cursor.fetchall():
            stats["browser_distribution"][row[0]] = row[1]
        
        cursor.execute("""
            SELECT id, session_id, latitude, longitude, city, country, device_type, created_at
            FROM user_locations ORDER BY created_at DESC LIMIT 10
        """)
        for row in cursor.fetchall():
            stats["recent_locations"].append({
                "id": row[0],
                "session_id": row[1],
                "latitude": row[2],
                "longitude": row[3],
                "city": row[4],
                "country": row[5],
                "device_type": row[6],
                "created_at": row[7]
            })
        
        conn.close()
        return stats


if __name__ == "__main__":
    db = ContentDatabase()
    print("✅ Content Database v3.0 initialized!")
