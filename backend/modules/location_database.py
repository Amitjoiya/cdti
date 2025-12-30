"""
Location Database - Privacy-Compliant User Location Tracking
===========================================================
- Pseudonymous User IDs (SHA-256 hashed, cannot identify real person)
- NO raw IP addresses stored
- Consent-based separation (allowed vs denied)
- Location history with distance comparison
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import math


class LocationDatabase:
    """
    Privacy-Compliant Location Database
    - SQLite for quick queries
    - JSON files in 'location' folder for detailed data
    - Pseudonymous user IDs (hashed fingerprints)
    - Distance-based location history (100m threshold)
    """
    
    DISTANCE_THRESHOLD_METERS = 100  # Same location if within 100m
    
    def __init__(self, db_path: str = None, location_folder: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "location_database.db")
        if location_folder is None:
            location_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "location")
        
        self.db_path = db_path
        self.location_folder = Path(location_folder)
        self.location_folder.mkdir(exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with privacy-compliant schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table (pseudonymous)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                visit_count INTEGER DEFAULT 1,
                consent_status TEXT DEFAULT 'unknown',
                is_anonymous BOOLEAN DEFAULT FALSE,
                fingerprint_hash TEXT
            )
        """)
        
        # Location records (for users who ALLOWED)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location_records (
                location_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                visit_count INTEGER DEFAULT 1,
                latitude REAL,
                longitude REAL,
                accuracy_meters REAL,
                location_name TEXT,
                city TEXT,
                region TEXT,
                country TEXT,
                country_code TEXT,
                json_file TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Visit history (each visit logged with privacy)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                location_id TEXT,
                session_id TEXT,
                visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                consent_status TEXT,
                device_type TEXT,
                browser TEXT,
                os TEXT,
                timezone TEXT,
                language TEXT,
                location_method TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (location_id) REFERENCES location_records(location_id)
            )
        """)
        
        # Anonymous visits (for users who DENIED - no location)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anonymous_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anonymous_id TEXT,
                session_id TEXT,
                visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approximate_region TEXT,
                timezone TEXT,
                device_type TEXT,
                browser TEXT,
                os TEXT,
                language TEXT,
                fingerprint_hash TEXT,
                denial_reason TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        print("  ✅ Privacy-Compliant Location Database initialized")

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        Returns distance in meters
        """
        if None in [lat1, lon1, lat2, lon2]:
            return float('inf')
        
        R = 6371000  # Earth's radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi/2)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    def _find_existing_location(self, user_id: str, latitude: float, longitude: float) -> Optional[str]:
        """
        Find if user has been to this location before (within threshold)
        Returns location_id if found, None otherwise
        """
        if latitude is None or longitude is None:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT location_id, latitude, longitude 
            FROM location_records 
            WHERE user_id = ?
        """, (user_id,))
        
        for row in cursor.fetchall():
            loc_id, loc_lat, loc_lon = row
            distance = self._calculate_distance(latitude, longitude, loc_lat, loc_lon)
            if distance <= self.DISTANCE_THRESHOLD_METERS:
                conn.close()
                return loc_id
        
        conn.close()
        return None

    def _generate_location_id(self, user_id: str, latitude: float, longitude: float) -> str:
        """Generate unique location ID for this user+location combination"""
        # Round to 4 decimal places (~11m precision)
        lat_rounded = round(latitude, 4) if latitude else 0
        lon_rounded = round(longitude, 4) if longitude else 0
        
        coord_string = f"{user_id}:{lat_rounded},{lon_rounded}"
        hash_hex = hashlib.sha256(coord_string.encode()).hexdigest()[:12].upper()
        
        return f"LOC-{hash_hex}"

    def save_location(self, location_data: Dict) -> Dict:
        """
        Save user location data with privacy compliance.
        
        For ALLOWED users: Save GPS coordinates with user history
        For DENIED users: Save anonymous visit with no location
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            user_id = location_data.get("user_id")
            session_id = location_data.get("session_id")
            consent_status = location_data.get("consent_status", "unknown")
            is_anonymous = location_data.get("anonymous", False)
            now = datetime.now().isoformat()
            
            # Extract device metadata (privacy-safe, no identifying info)
            device_meta = location_data.get("device_metadata", {})
            
            # ========================================
            # DENIED USERS: Save anonymous visit only
            # ========================================
            if consent_status == "denied" or is_anonymous:
                approx_region = location_data.get("approximate_region", {})
                
                cursor.execute("""
                    INSERT INTO anonymous_visits (
                        anonymous_id, session_id, approximate_region, timezone,
                        device_type, browser, os, language, fingerprint_hash, denial_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    session_id,
                    json.dumps(approx_region) if approx_region else None,
                    device_meta.get("timezone"),
                    device_meta.get("device_type"),
                    device_meta.get("browser"),
                    device_meta.get("os"),
                    device_meta.get("language"),
                    location_data.get("fingerprint_hash"),
                    location_data.get("denial_reason")
                ))
                
                conn.commit()
                conn.close()
                
                # Create JSON file for anonymous visit
                json_filename = f"ANON-{session_id}.json"
                json_path = self.location_folder / json_filename
                json_data = {
                    "anonymous_id": user_id,
                    "session_id": session_id,
                    "created_at": now,
                    "consent_status": "denied",
                    "approximate_region": approx_region,
                    "device_metadata": device_meta,
                    "fingerprint_hash": location_data.get("fingerprint_hash"),
                    "denial_reason": location_data.get("denial_reason")
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                print(f"  📍 Anonymous visit saved: {json_filename}")
                return {
                    "status": "success",
                    "anonymous_id": user_id,
                    "consent_status": "denied",
                    "file": json_filename
                }
            
            # ========================================
            # ALLOWED USERS: Save full location data
            # ========================================
            latitude = location_data.get("latitude")
            longitude = location_data.get("longitude")
            accuracy = location_data.get("accuracy_meters")
            
            # Update or create user
            cursor.execute("SELECT user_id, visit_count FROM users WHERE user_id = ?", (user_id,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                cursor.execute("""
                    UPDATE users SET last_seen = ?, visit_count = ?, consent_status = ?
                    WHERE user_id = ?
                """, (now, existing_user[1] + 1, consent_status, user_id))
            else:
                cursor.execute("""
                    INSERT INTO users (user_id, consent_status, fingerprint_hash)
                    VALUES (?, ?, ?)
                """, (user_id, consent_status, location_data.get("fingerprint_hash")))
            
            # Check if user has been to this location before (within 100m)
            existing_location = self._find_existing_location(user_id, latitude, longitude)
            
            if existing_location:
                # Update existing location
                location_id = existing_location
                cursor.execute("""
                    UPDATE location_records 
                    SET last_seen = ?, visit_count = visit_count + 1
                    WHERE location_id = ?
                """, (now, location_id))
                is_new_location = False
                
                # Update JSON file
                json_filename = f"{location_id}.json"
                json_path = self.location_folder / json_filename
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    json_data["last_seen"] = now
                    json_data["visit_count"] = json_data.get("visit_count", 1) + 1
                    json_data["visits"].append({
                        "session_id": session_id,
                        "visited_at": now,
                        "accuracy_meters": accuracy,
                        "device_metadata": device_meta
                    })
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
            else:
                # Create new location
                location_id = self._generate_location_id(user_id, latitude, longitude)
                json_filename = f"{location_id}.json"
                
                cursor.execute("""
                    INSERT INTO location_records (
                        location_id, user_id, latitude, longitude, accuracy_meters,
                        location_name, city, region, country, country_code, json_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    location_id,
                    user_id,
                    latitude,
                    longitude,
                    accuracy,
                    location_data.get("location_name"),
                    location_data.get("city"),
                    location_data.get("region"),
                    location_data.get("country"),
                    location_data.get("country_code"),
                    json_filename
                ))
                is_new_location = True
                
                # Create JSON file
                json_path = self.location_folder / json_filename
                json_data = {
                    "location_id": location_id,
                    "user_id": user_id,
                    "created_at": now,
                    "last_seen": now,
                    "visit_count": 1,
                    "consent_status": "allowed",
                    "location": {
                        "latitude": latitude,
                        "longitude": longitude,
                        "accuracy_meters": accuracy,
                        "location_name": location_data.get("location_name"),
                        "city": location_data.get("city"),
                        "area": location_data.get("area"),
                        "region": location_data.get("region"),
                        "country": location_data.get("country"),
                        "country_code": location_data.get("country_code")
                    },
                    "location_method": location_data.get("location_method", "gps"),
                    "distance_threshold_meters": self.DISTANCE_THRESHOLD_METERS,
                    "visits": [{
                        "session_id": session_id,
                        "visited_at": now,
                        "accuracy_meters": accuracy,
                        "device_metadata": device_meta
                    }]
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Log visit
            cursor.execute("""
                INSERT INTO visits (
                    user_id, location_id, session_id, consent_status,
                    device_type, browser, os, timezone, language, location_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                location_id,
                session_id,
                consent_status,
                device_meta.get("device_type"),
                device_meta.get("browser"),
                device_meta.get("os"),
                device_meta.get("timezone"),
                device_meta.get("language"),
                location_data.get("location_method")
            ))
            
            conn.commit()
            conn.close()
            
            print(f"  📍 Location saved: {location_id} (New: {is_new_location})")
            return {
                "status": "success",
                "location_id": location_id,
                "is_new_location": is_new_location,
                "user_id": user_id,
                "file": json_filename
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            conn.close()
            return {"status": "error", "message": str(e)}

    def get_all_locations(self, page: int = 1, limit: int = 50) -> Dict:
        """Get all unique locations with pagination"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        offset = (page - 1) * limit
        
        cursor.execute("SELECT COUNT(*) FROM location_records")
        total = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT location_id, user_id, created_at, last_seen, visit_count,
                   latitude, longitude, city, region, country, country_code
            FROM location_records
            ORDER BY last_seen DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        locations = []
        for row in cursor.fetchall():
            locations.append({
                "location_id": row[0],
                "user_id": row[1],
                "created_at": row[2],
                "last_seen": row[3],
                "visit_count": row[4],
                "latitude": row[5],
                "longitude": row[6],
                "city": row[7],
                "region": row[8],
                "country": row[9],
                "country_code": row[10]
            })
        
        conn.close()
        
        return {
            "locations": locations,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit
        }

    def get_location_by_id(self, location_id: str) -> Optional[Dict]:
        """Get detailed location info from JSON file"""
        json_path = self.location_folder / f"{location_id}.json"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None

    def get_stats(self) -> Dict:
        """Get location tracking statistics - Privacy compliant"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {
            "total_users": 0,
            "total_locations": 0,
            "total_visits": 0,
            "anonymous_visits": 0,
            "allowed_visits": 0,
            "denied_visits": 0,
            "today_visits": 0,
            "country_distribution": {},
            "device_distribution": {},
            "browser_distribution": {},
            "recent_locations": []
        }
        
        # Count users
        cursor.execute("SELECT COUNT(*) FROM users")
        stats["total_users"] = cursor.fetchone()[0]
        
        # Count locations
        cursor.execute("SELECT COUNT(*) FROM location_records")
        stats["total_locations"] = cursor.fetchone()[0]
        
        # Count visits
        cursor.execute("SELECT COUNT(*) FROM visits")
        stats["allowed_visits"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM anonymous_visits")
        stats["anonymous_visits"] = cursor.fetchone()[0]
        stats["denied_visits"] = stats["anonymous_visits"]
        
        stats["total_visits"] = stats["allowed_visits"] + stats["anonymous_visits"]
        
        # Today visits
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM visits WHERE date(visited_at) = ?", (today,))
        stats["today_visits"] = cursor.fetchone()[0]
        
        # Country distribution
        cursor.execute("""
            SELECT country, COUNT(*) FROM location_records 
            WHERE country IS NOT NULL GROUP BY country
        """)
        for row in cursor.fetchall():
            stats["country_distribution"][row[0]] = row[1]
        
        # Device distribution
        cursor.execute("""
            SELECT device_type, COUNT(*) FROM visits 
            WHERE device_type IS NOT NULL GROUP BY device_type
        """)
        for row in cursor.fetchall():
            stats["device_distribution"][row[0]] = row[1]
        
        # Browser distribution
        cursor.execute("""
            SELECT browser, COUNT(*) FROM visits 
            WHERE browser IS NOT NULL GROUP BY browser
        """)
        for row in cursor.fetchall():
            stats["browser_distribution"][row[0]] = row[1]
        
        # Recent locations
        cursor.execute("""
            SELECT location_id, user_id, latitude, longitude, city, country, 
                   visit_count, last_seen
            FROM location_records
            ORDER BY last_seen DESC LIMIT 20
        """)
        for row in cursor.fetchall():
            stats["recent_locations"].append({
                "location_id": row[0],
                "user_id": row[1],
                "latitude": row[2],
                "longitude": row[3],
                "city": row[4],
                "country": row[5],
                "visit_count": row[6],
                "last_seen": row[7]
            })
        
        conn.close()
        return stats


if __name__ == "__main__":
    db = LocationDatabase()
    print("✅ Privacy-Compliant Location Database initialized!")
