"""
FakeTrace - Fraud Message Analyzer v1.0
========================================
Advanced SMS/WhatsApp/Instagram Fraud Detection System

RESEARCH-BASED DETECTION:
==========================
Based on FTC, Cloudflare, KnowBe4 and other security research:

1. URL/LINK ANALYSIS:
   - Domain spoofing (amaz0n.com, paypa1.com)
   - Suspicious TLDs (.xyz, .top, .tk, .ml)
   - IP-based URLs (http://192.168.1.1/login)
   - URL shorteners hiding real destinations
   - Typosquatting (paypaI.com - capital I)
   - Excessive subdomains
   - Homograph attacks (unicode lookalikes)
   - Missing HTTPS for sensitive actions

2. MESSAGE PATTERN ANALYSIS:
   - Urgency language ("Act NOW!", "Immediately")
   - Threat language ("Account suspended", "Legal action")
   - Request for sensitive data (OTP, PIN, Password)
   - Generic greetings ("Dear Customer")
   - Grammar/spelling errors
   - Too-good-to-be-true offers
   - Impersonation patterns

3. PLATFORM-SPECIFIC DETECTION:
   - WhatsApp: Forward chains, unknown numbers
   - SMS: Sender ID spoofing, bank impersonation
   - Instagram: Fake verification, prize scams

4. CATEGORY-BASED CROSS-VERIFICATION:
   - Banking: ICICI, HDFC, SBI patterns
   - Payment: UPI, Paytm, PhonePe patterns
   - E-commerce: Amazon, Flipkart patterns
   - Government: Income Tax, TRAI patterns
   - Job/Offer scams: Work-from-home, lottery
"""

import re
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import json


@dataclass
class FraudCheckResult:
    """Result from a fraud detection check"""
    name: str
    display_name: str
    score: float  # 0-100, higher = more suspicious
    severity: str  # critical, high, medium, low, safe
    finding: str
    details: Dict
    category: str


class FraudAnalyzer:
    """
    Advanced Fraud Message Analyzer v1.0
    
    Analyzes SMS, WhatsApp, Instagram and other messages for:
    - Phishing links
    - Scam patterns
    - Impersonation attempts
    - Urgency manipulation
    - Financial fraud indicators
    """
    
    VERSION = "2.0"
    
    # ==================== INDICATOR DATABASES ====================
    
    # Suspicious TLDs commonly used in phishing
    SUSPICIOUS_TLDS = {
        '.xyz', '.top', '.club', '.info', '.tk', '.ml', '.ga', '.cf',
        '.gq', '.work', '.click', '.link', '.pro', '.live', '.life',
        '.online', '.site', '.website', '.space', '.fun', '.icu',
        '.buzz', '.best', '.rest', '.monster', '.cam', '.bar'
    }
    
    # Trusted TLDs (give positive score)
    TRUSTED_TLDS = {
        '.gov', '.gov.in', '.nic.in', '.edu', '.edu.in', '.ac.in',
        '.org', '.mil'
    }
    
    # Known legitimate domains (partial list for common services)
    LEGITIMATE_DOMAINS = {
        # Banks - India
        'icicibank.com', 'hdfcbank.com', 'sbi.co.in', 'axisbank.com',
        'kotak.com', 'pnbindia.in', 'bankofbaroda.in', 'canarabank.com',
        # Payment Apps
        'paytm.com', 'phonepe.com', 'gpay.com', 'bhimupi.org.in',
        # E-commerce
        'amazon.in', 'amazon.com', 'flipkart.com', 'myntra.com',
        # Government
        'incometax.gov.in', 'uidai.gov.in', 'trai.gov.in',
        # Social Media
        'facebook.com', 'instagram.com', 'whatsapp.com', 'twitter.com',
        # Others
        'google.com', 'microsoft.com', 'apple.com', 'linkedin.com'
    }
    
    # URL shorteners that hide real destinations
    URL_SHORTENERS = {
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'j.mp', 'tr.im', 'tiny.cc', 'lnkd.in',
        'rebrand.ly', 'cutt.ly', 'shorturl.at', 'rb.gy', 'clck.ru',
        'short.io', 't.ly', 'v.gd'
    }
    
    # Urgency words/phrases
    URGENCY_PATTERNS = [
        r'\bact\s*now\b', r'\bimmediately\b', r'\burgent\b', r'\bASAP\b',
        r'\blimited\s*time\b', r'\bexpir(e|es|ing|ed)\b', r'\bdeadline\b',
        r'\blast\s*chance\b', r'\bfinal\s*(notice|warning)\b',
        r'\bwithin\s*\d+\s*(hours?|minutes?|hrs?|mins?)\b',
        r'\btoday\s*only\b', r'\bdon\'?t\s*(miss|wait|delay)\b',
        r'\bhurry\b', r'\bquick(ly)?\b', r'\bfast\b',
        r'\binstant(ly)?\b', r'\bright\s*now\b'
    ]
    
    # Threat/Fear patterns
    THREAT_PATTERNS = [
        r'\b(account|card|service)\s*(will\s*be\s*)?(suspend|block|deactivat|terminat|cancel|clos)',
        r'\b(legal|police|court)\s*action\b', r'\bwarrant\b',
        r'\barrest\b', r'\bpenalty\b', r'\bfine\s*(of)?\s*₹?\s*\d+',
        r'\bfraudulent\s*activity\b', r'\bunauthorized\s*(access|transaction)',
        r'\bsecurity\s*alert\b', r'\bcompromised\b', r'\bhacked\b',
        r'\bverify\s*(immediately|urgently|now)\b',
        r'\bfailure\s*to\s*comply\b', r'\blast\s*warning\b'
    ]
    
    # Request for sensitive data patterns
    SENSITIVE_DATA_PATTERNS = [
        r'\b(enter|share|provide|send|confirm|verify)\s*(your)?\s*(otp|pin|password|cvv)\b',
        r'\botp\s*(is|:)?\s*\d{4,8}\b',  # OTP mentioned with number
        r'\bcard\s*(number|no|#)\b', r'\bexpiry\s*date\b', r'\bcvv\b',
        r'\bbank\s*account\s*(no|number|details)\b',
        r'\b(aadhar|aadhaar|pan)\s*(card)?\s*(no|number)?\b',
        r'\bsocial\s*security\b', r'\blogin\s*credentials\b',
        r'\bpassword\b', r'\bpin\s*(number|code)?\b',
        r'\bupi\s*(pin|id)\b', r'\batm\s*pin\b'
    ]
    
    # Too-good-to-be-true offers
    OFFER_PATTERNS = [
        r'\b(you|u)\s*(have\s*)?(won|win)\b', r'\blottery\b', r'\bjackpot\b',
        r'\bcongrat(s|ulations)\b', r'\bselected\s*(as\s*)?(winner|lucky)\b',
        r'\bprize\s*(of|worth)\b', r'\bfree\s*(gift|iphone|phone|laptop)\b',
        r'\b₹\s*\d+\s*(lakh|crore|lakhs|crores)\b',  # Large INR amounts
        r'\bcashback\s*of\s*\d+%?\b', r'\bearn\s*₹?\s*\d+\s*(daily|weekly|monthly)\b',
        r'\bwork\s*from\s*home\b.*₹', r'\bguaranteed\s*(income|returns)\b',
        r'\bdouble\s*your\s*(money|investment)\b', r'\brisk\s*free\b'
    ]
    
    # Impersonation patterns (common entities being impersonated)
    IMPERSONATION_ENTITIES = {
        'bank': ['icici', 'hdfc', 'sbi', 'axis', 'kotak', 'pnb', 'canara', 'bob', 'idbi', 'rbi'],
        'payment': ['paytm', 'phonepe', 'gpay', 'google pay', 'bhim', 'upi'],
        'ecommerce': ['amazon', 'flipkart', 'myntra', 'snapdeal', 'meesho'],
        'government': ['income tax', 'trai', 'uidai', 'aadhar', 'pan card', 'passport', 'police'],
        'tech': ['microsoft', 'apple', 'google', 'facebook', 'whatsapp', 'instagram'],
        'telecom': ['jio', 'airtel', 'vodafone', 'vi', 'bsnl']
    }
    
    # Generic greeting patterns (sign of mass phishing)
    GENERIC_GREETINGS = [
        r'^dear\s+(customer|user|sir|madam|member|valued\s*customer)',
        r'^respected\s+(customer|user|sir|madam)',
        r'^hello\s+(customer|user)',
        r'^dear\s+account\s*holder'
    ]
    
    # Platform-specific patterns
    PLATFORM_PATTERNS = {
        'whatsapp': {
            'suspicious': [
                r'forward(ed)?\s*(from|message)', r'broadcast', 
                r'share\s*(with|to)\s*\d+\s*(contacts|groups|friends)',
                r'(send|forward)\s*to\s*\d+\s*people'
            ]
        },
        'sms': {
            'suspicious': [
                r'^[A-Z]{2}-[A-Z]+',  # Sender ID pattern
                r'click\s*(here|link|below)',
                r'call\s*(this|the)\s*number'
            ]
        },
        'instagram': {
            'suspicious': [
                r'verification\s*(badge|tick)',
                r'your\s*account\s*(will\s*be)?\s*(verified|removed)',
                r'influencer\s*program', r'brand\s*ambassador'
            ]
        }
    }
    
    # Typosquatting patterns for major brands
    TYPOSQUAT_PATTERNS = {
        'amazon': ['amaz0n', 'amazn', 'amazonn', 'amazzon', 'amaazon', 'amazon-', 'arnazon'],
        'google': ['googl', 'gooogle', 'g00gle', 'googgle', 'qoogle', 'gogle'],
        'facebook': ['faceb00k', 'facebok', 'faecbook', 'faceboook', 'facebock'],
        'paypal': ['paypa1', 'paypall', 'paypa|', 'payypal', 'paypaI'],
        'paytm': ['paytim', 'payt1m', 'payttm', 'payytm', 'patym'],
        'icici': ['1cici', 'icic1', 'icicci', 'lcici', 'icicii'],
        'hdfc': ['hdtc', 'hdf c', 'hdfcc', 'hdfcbank-'],
        'sbi': ['sb1', 'sbl', 'sbii', 'sbi-online', 'onlinesbi-']
    }

    # Phone number patterns (focus on Indian fraud calls)
    PHONE_REGEX = re.compile(r'(?:\+?91[-\s]?|0)?[6-9]\d{9}')
    OFFICIAL_SHORTCODES = {'155260', '155266', '155359', '1930', '112', '1092', '1091', '1947', '139'}

    # Hindi / Hinglish scam phrases
    HINDI_HINGLISH_PATTERNS = [
        r'aapka\s+account\s+(?:band|block|suspend)',
        r'tur?ant\s+karna\s+hoga',
        r'otp\s+bhej',
        r'kyc\s+verify\s+karo',
        r'bank\s+se\s+call\s+kar',
        r'adhar\s+update',
        r'paisa\s+transfer',
        r'tur?ant\s+pay',
        r'f?raud\s+department',
        r'aapko\s+\d+\s*(?:ghante|hours)',
        r'gift\s+jeeta',
    ]

    # Known scam templates
    KNOWN_SCAM_TEMPLATES = [
        {
            "name": "KYC Suspension Scam",
            "keywords": ['kyc', 'suspend', 'account', 'verify', 'update'],
            "description": "Classic bank KYC suspension alert"
        },
        {
            "name": "Lottery / Prize Scam",
            "keywords": ['congratulations', 'won', 'prize', 'lakh', 'crore', 'lucky'],
            "description": "Fake lottery or lucky draw claim"
        },
        {
            "name": "Delivery Reattempt Scam",
            "keywords": ['courier', 'delivery', 'reschedule', 'charges', 'link'],
            "description": "Fake courier/delivery payment request"
        },
        {
            "name": "Job / Work From Home Scam",
            "keywords": ['part time', 'work from home', 'income', 'daily payment', 'task'],
            "description": "Fake job promise with upfront payment"
        }
    ]

    SUSPICIOUS_LINK_KEYWORDS = ['kyc', 'verify', 'secure', 'update', 'account', 'bank', 'upi', 'login', 'otp', 'unlock', 'prize', 'bonus', 'gift']
    
    def __init__(self):
        """Initialize the Fraud Analyzer"""
        print("  ✅ Fraud Analyzer v2.0 initialized")
        print("     Detection: URL deep scan, Pattern, Phone/Entity analysis")
        print("     Supported: WhatsApp, SMS, Instagram, Email")

    def _generate_fraud_id(self, message: str) -> str:
        """Generate a unique fraud ID based on message content"""
        # Use message hash + timestamp for uniqueness
        msg_hash = hashlib.md5(message.encode()).hexdigest()[:8].upper()
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        return f"FM-{msg_hash}-{timestamp}"
    
    def analyze(self, message: str, platform: Optional[str] = None, 
                message_type: Optional[str] = None) -> Dict:
        """
        Analyze a message for fraud indicators.
        
        Args:
            message: The full message text including any links
            platform: Optional - 'whatsapp', 'sms', 'instagram', 'email'
            message_type: Optional - 'banking', 'payment', 'offer', 'government', 'delivery', 'job'
            
        Returns:
            Comprehensive fraud analysis result
        """
        # Generate unique fraud ID based on message content + timestamp
        fraud_id = self._generate_fraud_id(message)
        
        result = {
            "fraud_id": fraud_id,
            "version": self.VERSION,
            "status": "processing",
            "analyzed_at": datetime.now().isoformat(),
            "message_preview": message[:100] + "..." if len(message) > 100 else message,
            "platform": platform,
            "message_type": message_type,
            "urls_found": [],
            "contact_numbers": [],
            "matched_templates": [],
            "entities": {"amounts": [], "organizations": [], "deadlines": []},
            "checks": [],
            "categories": {},
            "summary": {
                "total_checks": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "safe": 0
            },
            "fraud_assessment": {},
            "recommendations": []
        }
        
        try:
            # Extract URLs from message
            urls = self._extract_urls(message)
            result["urls_found"] = urls

            # Extract phone numbers and entities for downstream checks
            contact_numbers = self._extract_phone_numbers(message)
            result["contact_numbers"] = contact_numbers
            result["entities"] = self._extract_entities(message)

            # Template recognition helps explain verdicts
            template_matches, template_check = self._match_known_templates(message)
            result["matched_templates"] = template_matches
            
            # ========== URL ANALYSIS ==========
            url_checks = []
            for url in urls:
                url_checks.extend(self._analyze_url(url))
            
            # ========== MESSAGE PATTERN ANALYSIS ==========
            pattern_checks = [
                self._check_urgency_language(message),
                self._check_threat_language(message),
                self._check_sensitive_data_request(message),
                self._check_too_good_offers(message),
                self._check_generic_greeting(message),
                self._check_grammar_issues(message),
            ]
            
            # ========== PLATFORM-SPECIFIC ANALYSIS ==========
            platform_checks = []
            if platform:
                platform_checks = self._check_platform_specific(message, platform)
            
            # ========== PHONE & LANGUAGE ANALYSIS ==========
            phone_checks = self._analyze_phone_numbers(contact_numbers) if contact_numbers else []
            language_checks = [self._check_hindi_hinglish(message)]

            # ========== CATEGORY CROSS-VERIFICATION ==========
            category_checks = []
            if message_type:
                category_checks = self._verify_against_category(message, urls, message_type)
            else:
                # Auto-detect category and verify
                detected_type = self._detect_message_type(message)
                if detected_type:
                    result["detected_type"] = detected_type
                    category_checks = self._verify_against_category(message, urls, detected_type)
            
            # ========== IMPERSONATION CHECK ==========
            impersonation_checks = self._check_impersonation(message, urls)

            # ========== TEMPLATE MATCHING ==========
            template_checks = [template_check] if template_check else []
            
            # Combine all checks
            all_checks = (
                url_checks
                + pattern_checks
                + platform_checks
                + phone_checks
                + language_checks
                + category_checks
                + impersonation_checks
                + template_checks
            )
            all_checks = [c for c in all_checks if c is not None]
            
            result["checks"] = [asdict(c) for c in all_checks]
            
            # Categorize results
            result["categories"] = self._categorize_checks(all_checks)
            
            # Count by severity
            for check in all_checks:
                result["summary"]["total_checks"] += 1
                result["summary"][check.severity] += 1
            
            # Generate final assessment
            result["fraud_assessment"] = self._generate_assessment(all_checks, urls, message)
            result["recommendations"] = self._generate_recommendations(all_checks, result["fraud_assessment"])
            
            result["status"] = "completed"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    # ==================== URL ANALYSIS ====================
    
    def _extract_urls(self, message: str) -> List[str]:
        """Extract all URLs from message"""
        # URL regex pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, message, re.IGNORECASE)
        
        # Also catch urls without protocol
        domain_pattern = r'(?<!\S)(www\.[^\s<>"{}|\\^`\[\]]+)'
        www_urls = re.findall(domain_pattern, message, re.IGNORECASE)
        urls.extend(['http://' + u for u in www_urls])
        
        # Clean up URLs (remove trailing punctuation)
        cleaned = []
        for url in urls:
            url = url.rstrip('.,;:!?)')
            if url and url not in cleaned:
                cleaned.append(url)
        
        return cleaned

    def _extract_phone_numbers(self, message: str) -> List[str]:
        """Extract likely contact numbers to flag unofficial callers"""
        numbers = []
        seen = set()
        for match in re.finditer(self.PHONE_REGEX, message):
            raw = match.group()
            digits = re.sub(r'\D', '', raw)
            if len(digits) > 10 and digits.startswith('91'):
                digits = digits[-10:]
            normalized = digits
            if len(digits) == 10:
                normalized = f"+91-{digits}"
            if digits in self.OFFICIAL_SHORTCODES:
                normalized = digits
            if normalized and normalized not in seen:
                seen.add(normalized)
                numbers.append(normalized)
        return numbers
    
    def _analyze_url(self, url: str) -> List[FraudCheckResult]:
        """Comprehensive URL analysis"""
        checks = []
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # 1. Check for IP-based URL
            ip_check = self._check_ip_url(url, domain)
            if ip_check:
                checks.append(ip_check)
            
            # 2. Check for suspicious TLD
            tld_check = self._check_suspicious_tld(url, domain)
            if tld_check:
                checks.append(tld_check)
            
            # 3. Check for URL shortener
            shortener_check = self._check_url_shortener(url, domain)
            if shortener_check:
                checks.append(shortener_check)
            
            # 4. Check for typosquatting
            typo_check = self._check_typosquatting(url, domain)
            if typo_check:
                checks.append(typo_check)
            
            # 5. Check for excessive subdomains
            subdomain_check = self._check_excessive_subdomains(url, domain)
            if subdomain_check:
                checks.append(subdomain_check)
            
            # 6. Check for suspicious path patterns
            path_check = self._check_suspicious_path(url, path)
            if path_check:
                checks.append(path_check)
            
            # 7. Check for missing HTTPS
            https_check = self._check_https(url, parsed.scheme)
            if https_check:
                checks.append(https_check)
            
            # 8. Check against legitimate domains
            legit_check = self._check_legitimate_domain(url, domain)
            if legit_check:
                checks.append(legit_check)

            # 9. Inspect keyword intent within the URL
            keyword_check = self._check_suspicious_link_keywords(url, domain, path, parsed.query)
            if keyword_check:
                checks.append(keyword_check)
                
        except Exception as e:
            checks.append(FraudCheckResult(
                name="url_parse_error",
                display_name="URL Analysis Error",
                score=50,
                severity="medium",
                finding=f"Could not fully analyze URL: {str(e)[:50]}",
                details={"url": url, "error": str(e)},
                category="url"
            ))
        
        return checks
    
    def _check_ip_url(self, url: str, domain: str) -> Optional[FraudCheckResult]:
        """Check if URL uses IP address instead of domain"""
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}(:\d+)?$'
        if re.match(ip_pattern, domain):
            return FraudCheckResult(
                name="ip_based_url",
                display_name="IP-Based URL",
                score=85,
                severity="critical",
                finding="URL uses IP address instead of domain name - highly suspicious!",
                details={"url": url, "ip": domain},
                category="url"
            )
        return None
    
    def _check_suspicious_tld(self, url: str, domain: str) -> Optional[FraudCheckResult]:
        """Check for suspicious top-level domains"""
        for tld in self.SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                return FraudCheckResult(
                    name="suspicious_tld",
                    display_name="Suspicious Domain Extension",
                    score=70,
                    severity="high",
                    finding=f"Domain uses suspicious TLD '{tld}' commonly used in phishing",
                    details={"url": url, "tld": tld},
                    category="url"
                )
        
        for tld in self.TRUSTED_TLDS:
            if domain.endswith(tld):
                return FraudCheckResult(
                    name="trusted_tld",
                    display_name="Trusted Domain Extension",
                    score=10,
                    severity="safe",
                    finding=f"Domain uses trusted TLD '{tld}'",
                    details={"url": url, "tld": tld},
                    category="url"
                )
        return None
    
    def _check_url_shortener(self, url: str, domain: str) -> Optional[FraudCheckResult]:
        """Check for URL shorteners"""
        for shortener in self.URL_SHORTENERS:
            if shortener in domain:
                return FraudCheckResult(
                    name="url_shortener",
                    display_name="Shortened URL",
                    score=65,
                    severity="high",
                    finding=f"URL uses shortener '{shortener}' which hides the real destination",
                    details={"url": url, "shortener": shortener},
                    category="url"
                )
        return None
    
    def _check_typosquatting(self, url: str, domain: str) -> Optional[FraudCheckResult]:
        """Check for typosquatting attempts"""
        for brand, typos in self.TYPOSQUAT_PATTERNS.items():
            for typo in typos:
                if typo.lower() in domain:
                    return FraudCheckResult(
                        name="typosquatting",
                        display_name="Typosquatting Detected",
                        score=90,
                        severity="critical",
                        finding=f"Domain appears to impersonate '{brand}' using typo '{typo}'",
                        details={"url": url, "brand": brand, "typo": typo},
                        category="url"
                    )
        return None
    
    def _check_excessive_subdomains(self, url: str, domain: str) -> Optional[FraudCheckResult]:
        """Check for excessive subdomains (hiding real domain)"""
        parts = domain.split('.')
        if len(parts) > 4:
            return FraudCheckResult(
                name="excessive_subdomains",
                display_name="Excessive Subdomains",
                score=60,
                severity="medium",
                finding=f"Domain has {len(parts)} parts - may be hiding real domain",
                details={"url": url, "parts": parts},
                category="url"
            )
        return None
    
    def _check_suspicious_path(self, url: str, path: str) -> Optional[FraudCheckResult]:
        """Check for suspicious URL paths"""
        suspicious_paths = [
            r'/login', r'/signin', r'/verify', r'/confirm', r'/secure',
            r'/update.*account', r'/banking', r'/account.*update',
            r'/password.*reset', r'/otp', r'/kyc', r'/aadhar'
        ]
        
        for pattern in suspicious_paths:
            if re.search(pattern, path):
                return FraudCheckResult(
                    name="suspicious_path",
                    display_name="Suspicious URL Path",
                    score=55,
                    severity="medium",
                    finding=f"URL path contains sensitive-looking segment",
                    details={"url": url, "path": path},
                    category="url"
                )
        return None
    
    def _check_https(self, url: str, scheme: str) -> Optional[FraudCheckResult]:
        """Check for HTTPS"""
        if scheme == 'http':
            return FraudCheckResult(
                name="no_https",
                display_name="No HTTPS",
                score=50,
                severity="medium",
                finding="URL uses insecure HTTP - legitimate sites use HTTPS",
                details={"url": url, "scheme": scheme},
                category="url"
            )
        return None
    
    def _is_legitimate_domain(self, domain: str) -> bool:
        """Helper to see if domain belongs to trusted list"""
        clean_domain = domain.replace('www.', '')
        for legit in self.LEGITIMATE_DOMAINS:
            if clean_domain == legit or clean_domain.endswith('.' + legit):
                return True
        return False

    def _check_legitimate_domain(self, url: str, domain: str) -> Optional[FraudCheckResult]:
        """Check if domain is in known legitimate list"""
        # Remove www. prefix
        clean_domain = domain.replace('www.', '')
        if self._is_legitimate_domain(domain):
            return FraudCheckResult(
                name="legitimate_domain",
                display_name="Known Legitimate Domain",
                score=5,
                severity="safe",
                finding=f"Domain '{clean_domain}' is a known legitimate service",
                details={"url": url, "domain": clean_domain},
                category="url"
            )
        return None

    def _check_suspicious_link_keywords(self, url: str, domain: str, path: str, query: str) -> Optional[FraudCheckResult]:
        """Score URLs that combine sensitive keywords with unknown domains"""
        combined = f"{domain}{path}{query}".lower()
        matches = [kw for kw in self.SUSPICIOUS_LINK_KEYWORDS if kw in combined]
        if not matches:
            return None

        is_legit = self._is_legitimate_domain(domain)
        severity = "medium"
        score = 55
        if len(matches) >= 2 and not is_legit:
            severity = "high"
            score = 75
        elif is_legit:
            severity = "low"
            score = 25
        return FraudCheckResult(
            name="keyword_loaded_url",
            display_name="Sensitive Keywords In URL",
            score=score,
            severity=severity,
            finding="URL embeds high-risk keywords often used in credential phishing",
            details={"url": url, "keywords": matches[:5]},
            category="url"
        )
    
    # ==================== MESSAGE PATTERN ANALYSIS ====================
    
    def _check_urgency_language(self, message: str) -> FraudCheckResult:
        """Check for urgency manipulation"""
        message_lower = message.lower()
        found_patterns = []
        
        for pattern in self.URGENCY_PATTERNS:
            matches = re.findall(pattern, message_lower, re.IGNORECASE)
            if matches:
                found_patterns.extend(matches)
        
        if len(found_patterns) >= 3:
            return FraudCheckResult(
                name="urgency_high",
                display_name="High Urgency Language",
                score=75,
                severity="high",
                finding=f"Message contains multiple urgency phrases ({len(found_patterns)}) - manipulation tactic",
                details={"patterns": found_patterns[:5]},
                category="pattern"
            )
        elif len(found_patterns) >= 1:
            return FraudCheckResult(
                name="urgency_moderate",
                display_name="Urgency Language",
                score=40,
                severity="medium",
                finding=f"Message contains urgency language - common in scams",
                details={"patterns": found_patterns},
                category="pattern"
            )
        else:
            return FraudCheckResult(
                name="no_urgency",
                display_name="No Urgency Language",
                score=10,
                severity="safe",
                finding="No urgency manipulation detected",
                details={},
                category="pattern"
            )
    
    def _check_threat_language(self, message: str) -> FraudCheckResult:
        """Check for threat/fear language"""
        message_lower = message.lower()
        found_patterns = []
        
        for pattern in self.THREAT_PATTERNS:
            matches = re.findall(pattern, message_lower, re.IGNORECASE)
            if matches:
                found_patterns.extend([str(m) for m in matches])
        
        if len(found_patterns) >= 2:
            return FraudCheckResult(
                name="threat_high",
                display_name="Threat/Fear Language",
                score=80,
                severity="critical",
                finding="Message uses multiple threat/fear tactics - classic scam pattern",
                details={"patterns": found_patterns[:5]},
                category="pattern"
            )
        elif len(found_patterns) >= 1:
            return FraudCheckResult(
                name="threat_moderate",
                display_name="Threat Language Detected",
                score=55,
                severity="high",
                finding="Message contains threatening language",
                details={"patterns": found_patterns},
                category="pattern"
            )
        else:
            return FraudCheckResult(
                name="no_threat",
                display_name="No Threat Language",
                score=10,
                severity="safe",
                finding="No threat/fear manipulation detected",
                details={},
                category="pattern"
            )
    
    def _check_sensitive_data_request(self, message: str) -> FraudCheckResult:
        """Check if message requests sensitive data"""
        message_lower = message.lower()
        found_patterns = []
        
        for pattern in self.SENSITIVE_DATA_PATTERNS:
            matches = re.findall(pattern, message_lower, re.IGNORECASE)
            if matches:
                found_patterns.extend([str(m) for m in matches])
        
        if len(found_patterns) >= 1:
            return FraudCheckResult(
                name="sensitive_data_request",
                display_name="Requests Sensitive Data",
                score=90,
                severity="critical",
                finding="Message requests sensitive information (OTP/PIN/Password) - NEVER share these!",
                details={"patterns": found_patterns[:5]},
                category="pattern"
            )
        else:
            return FraudCheckResult(
                name="no_sensitive_request",
                display_name="No Data Request",
                score=10,
                severity="safe",
                finding="No request for sensitive data detected",
                details={},
                category="pattern"
            )
    
    def _check_too_good_offers(self, message: str) -> FraudCheckResult:
        """Check for too-good-to-be-true offers"""
        message_lower = message.lower()
        found_patterns = []
        
        for pattern in self.OFFER_PATTERNS:
            matches = re.findall(pattern, message_lower, re.IGNORECASE)
            if matches:
                found_patterns.extend([str(m) for m in matches])
        
        if len(found_patterns) >= 2:
            return FraudCheckResult(
                name="scam_offer_high",
                display_name="Too-Good-To-Be-True Offer",
                score=85,
                severity="critical",
                finding="Message contains unrealistic offers - classic lottery/prize scam",
                details={"patterns": found_patterns[:5]},
                category="pattern"
            )
        elif len(found_patterns) >= 1:
            return FraudCheckResult(
                name="scam_offer_moderate",
                display_name="Suspicious Offer",
                score=50,
                severity="medium",
                finding="Message contains suspicious offer language",
                details={"patterns": found_patterns},
                category="pattern"
            )
        else:
            return FraudCheckResult(
                name="no_scam_offer",
                display_name="No Suspicious Offers",
                score=10,
                severity="safe",
                finding="No unrealistic offers detected",
                details={},
                category="pattern"
            )
    
    def _check_generic_greeting(self, message: str) -> FraudCheckResult:
        """Check for generic greetings"""
        message_lower = message.lower().strip()
        
        for pattern in self.GENERIC_GREETINGS:
            if re.match(pattern, message_lower, re.IGNORECASE):
                return FraudCheckResult(
                    name="generic_greeting",
                    display_name="Generic Greeting",
                    score=45,
                    severity="medium",
                    finding="Message uses generic greeting - legitimate companies usually use your name",
                    details={"pattern": pattern},
                    category="pattern"
                )
        
        return FraudCheckResult(
            name="personalized_greeting",
            display_name="No Generic Greeting",
            score=10,
            severity="safe",
            finding="No generic greeting pattern detected",
            details={},
            category="pattern"
        )
    
    def _check_grammar_issues(self, message: str) -> FraudCheckResult:
        """Check for grammar/spelling issues common in scams"""
        issues = []
        
        # Common grammar issues in Indian scam messages
        grammar_patterns = [
            (r'\b(your|ur)\s+account\s+(is|has)\s+been?\s+(block|suspend|terminate)', 'blocked/suspended grammar'),
            (r'\byour\s+request\s+(is|has)\s+been\s+recieved', 'received misspelling'),
            (r'\bkindly\s+do\s+the\s+needful\b', 'awkward phrasing'),
            (r'\brevert\s+back\b', 'redundant phrasing'),
            (r'\bplease\s+to\s+click\b', 'grammar error'),
            (r'\bfor\s+to\s+verify\b', 'grammar error'),
            (r'\bclick\s+on\s+below\s+link\b', 'awkward phrasing'),
            (r'\bdo\s+not\s+ignore\s+this\s+message\b', 'pressure tactic'),
        ]
        
        message_lower = message.lower()
        for pattern, issue_type in grammar_patterns:
            if re.search(pattern, message_lower):
                issues.append(issue_type)
        
        # Check for excessive capitals
        caps_ratio = sum(1 for c in message if c.isupper()) / (len(message) + 1)
        if caps_ratio > 0.3:
            issues.append("excessive capitals")
        
        if len(issues) >= 2:
            return FraudCheckResult(
                name="grammar_issues",
                display_name="Grammar/Style Issues",
                score=55,
                severity="medium",
                finding="Message has grammar issues common in scam messages",
                details={"issues": issues},
                category="pattern"
            )
        
        return FraudCheckResult(
            name="grammar_ok",
            display_name="Grammar OK",
            score=10,
            severity="safe",
            finding="No significant grammar issues detected",
            details={},
            category="pattern"
        )

    def _check_hindi_hinglish(self, message: str) -> FraudCheckResult:
        """Detect scam wording in Hindi or Hinglish to cover regional attacks"""
        message_lower = message.lower()
        matches = []
        for pattern in self.HINDI_HINGLISH_PATTERNS:
            if re.search(pattern, message_lower):
                matches.append(pattern)
        if len(matches) >= 2:
            return FraudCheckResult(
                name="hindi_scam_language",
                display_name="Hindi/Hinglish Scam Tone",
                score=70,
                severity="high",
                finding="Message mirrors known Hindi scam wording",
                details={"patterns": matches[:5]},
                category="language"
            )
        elif matches:
            return FraudCheckResult(
                name="hindi_caution",
                display_name="Hindi Scam Signal",
                score=45,
                severity="medium",
                finding="Contains Hindi phrases used in common scam templates",
                details={"patterns": matches},
                category="language"
            )
        return FraudCheckResult(
            name="no_hindi_signal",
            display_name="No Hindi Scam Signals",
            score=10,
            severity="safe",
            finding="No risky Hindi/Hinglish scam phrases detected",
            details={},
            category="language"
        )

    def _match_known_templates(self, message: str) -> Tuple[List[str], Optional[FraudCheckResult]]:
        """Compare message to curated scam templates"""
        matches = []
        message_lower = message.lower()
        for template in self.KNOWN_SCAM_TEMPLATES:
            if all(keyword in message_lower for keyword in template["keywords"]):
                matches.append(template["name"])
        if not matches:
            return [], None
        severity = "high" if len(matches) >= 2 else "medium"
        score = 75 if severity == "high" else 60
        return matches, FraudCheckResult(
            name="template_match",
            display_name="Matches Known Scam Template",
            score=score,
            severity=severity,
            finding="Message structure matches popular scam scripts",
            details={"templates": matches},
            category="template"
        )

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """Pull out money amounts, org names, and deadlines for UI context"""
        message_lower = message.lower()
        amount_pattern = re.compile(r'(?:₹|rs\.?|inr)?\s*\d{3,}(?:,\d{2,3})*(?:\.\d+)?\s*(?:lakh|lac|crore|cr|rs|₹)?', re.IGNORECASE)
        deadline_pattern = re.compile(r'(?:within|inside)\s*\d+\s*(?:hours?|hrs?|minutes?|days?)|by\s+\d{1,2}(?:st|nd|rd|th)?\s+[a-z]{3,}', re.IGNORECASE)
        amounts = [m.group().strip() for m in amount_pattern.finditer(message)]
        deadlines = [m.group().strip() for m in deadline_pattern.finditer(message)]
        organizations = set()
        for entities in self.IMPERSONATION_ENTITIES.values():
            for entity in entities:
                if re.search(r'\b' + re.escape(entity.lower()) + r'\b', message_lower):
                    organizations.add(entity.title())
        return {
            "amounts": [amt.strip() for amt in amounts[:5]],
            "organizations": sorted(list(organizations))[:5],
            "deadlines": [ddl.strip() for ddl in deadlines[:5]]
        }
    
    # ==================== PLATFORM-SPECIFIC ANALYSIS ====================
    
    def _check_platform_specific(self, message: str, platform: str) -> List[FraudCheckResult]:
        """Check for platform-specific scam patterns"""
        checks = []
        platform = platform.lower()
        
        if platform not in self.PLATFORM_PATTERNS:
            return checks
        
        patterns = self.PLATFORM_PATTERNS[platform].get('suspicious', [])
        message_lower = message.lower()
        found = []
        
        for pattern in patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                found.append(pattern)
        
        if found:
            checks.append(FraudCheckResult(
                name=f"{platform}_suspicious",
                display_name=f"{platform.title()} Suspicious Patterns",
                score=50,
                severity="medium",
                finding=f"Message contains {platform.title()}-specific scam patterns",
                details={"patterns_matched": len(found)},
                category="platform"
            ))
        
        return checks

    def _analyze_phone_numbers(self, numbers: List[str]) -> List[FraudCheckResult]:
        """Evaluate extracted phone numbers for legitimacy"""
        checks = []
        if not numbers:
            return checks

        suspicious = []
        official = []
        for number in numbers:
            digits = re.sub(r'\D', '', number)
            if digits in self.OFFICIAL_SHORTCODES:
                official.append(number)
            elif len(digits) == 10:
                suspicious.append(number)
            elif len(digits) <= 6:
                # Weird short codes not in official list are suspicious as well
                suspicious.append(number)

        if suspicious:
            severity = "high" if len(suspicious) >= 2 else "medium"
            score = 80 if severity == "high" else 55
            checks.append(FraudCheckResult(
                name="unofficial_numbers",
                display_name="Unofficial Contact Numbers",
                score=score,
                severity=severity,
                finding="Message asks to contact unofficial personal numbers",
                details={"numbers": suspicious},
                category="phone"
            ))

        if official and not suspicious:
            checks.append(FraudCheckResult(
                name="official_contact",
                display_name="Official Hotline Listed",
                score=15,
                severity="safe",
                finding="Only trusted government helpline numbers detected",
                details={"numbers": official},
                category="phone"
            ))

        return checks
    
    # ==================== CATEGORY CROSS-VERIFICATION ====================
    
    def _detect_message_type(self, message: str) -> Optional[str]:
        """Auto-detect the category of message"""
        message_lower = message.lower()
        
        category_keywords = {
            'banking': ['bank', 'account', 'icici', 'hdfc', 'sbi', 'axis', 'kotak', 'loan', 'credit', 'debit'],
            'payment': ['upi', 'paytm', 'phonepe', 'gpay', 'payment', 'transfer', 'rupees', '₹', 'transaction'],
            'offer': ['offer', 'discount', 'sale', 'cashback', 'prize', 'won', 'winner', 'lottery', 'reward'],
            'government': ['income tax', 'trai', 'aadhar', 'pan card', 'passport', 'police', 'court', 'government'],
            'delivery': ['delivery', 'shipment', 'package', 'courier', 'amazon', 'flipkart', 'order', 'track'],
            'job': ['job', 'work from home', 'vacancy', 'hiring', 'salary', 'income', 'earn', 'recruitment']
        }
        
        scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores, key=scores.get)
        return None
    
    def _verify_against_category(self, message: str, urls: List[str], 
                                  message_type: str) -> List[FraudCheckResult]:
        """Verify message authenticity against its claimed category"""
        checks = []
        message_type = message_type.lower()
        
        # Category-specific legitimate domains
        category_domains = {
            'banking': ['icicibank.com', 'hdfcbank.com', 'sbi.co.in', 'axisbank.com', 'kotak.com'],
            'payment': ['paytm.com', 'phonepe.com', 'gpay.com', 'bhimupi.org.in'],
            'offer': ['amazon.in', 'flipkart.com', 'myntra.com'],
            'government': ['incometax.gov.in', 'uidai.gov.in', 'trai.gov.in'],
            'delivery': ['amazon.in', 'flipkart.com', 'bluedart.com', 'delhivery.com']
        }
        
        expected_domains = category_domains.get(message_type, [])
        
        if expected_domains and urls:
            legit_url_found = False
            for url in urls:
                parsed = urlparse(url)
                domain = parsed.netloc.lower().replace('www.', '')
                if any(exp in domain for exp in expected_domains):
                    legit_url_found = True
                    break
            
            if not legit_url_found:
                checks.append(FraudCheckResult(
                    name="category_mismatch",
                    display_name="Category-URL Mismatch",
                    score=70,
                    severity="high",
                    finding=f"Message claims to be about {message_type} but links don't match known {message_type} domains",
                    details={"expected": expected_domains, "urls": urls},
                    category="verification"
                ))
            else:
                checks.append(FraudCheckResult(
                    name="category_match",
                    display_name="Category-URL Match",
                    score=15,
                    severity="safe",
                    finding=f"Links match expected {message_type} domains",
                    details={"category": message_type},
                    category="verification"
                ))
        
        return checks
    
    # ==================== IMPERSONATION CHECK ====================
    
    def _check_impersonation(self, message: str, urls: List[str]) -> List[FraudCheckResult]:
        """Check for entity impersonation"""
        checks = []
        message_lower = message.lower()
        
        for entity_type, entities in self.IMPERSONATION_ENTITIES.items():
            for entity in entities:
                if entity.lower() in message_lower:
                    # Check if the URL matches the entity
                    legit_url = False
                    for url in urls:
                        if entity.lower() in url.lower():
                            # Could be legit or could be typosquat
                            parsed = urlparse(url)
                            domain = parsed.netloc.lower()
                            
                            # Simple check - is it the real domain?
                            if f"{entity}.com" in domain or f"{entity}.in" in domain or f"{entity}bank.com" in domain:
                                legit_url = True
                    
                    if urls and not legit_url:
                        checks.append(FraudCheckResult(
                            name=f"impersonation_{entity}",
                            display_name=f"Possible {entity.title()} Impersonation",
                            score=75,
                            severity="high",
                            finding=f"Message mentions '{entity}' but links don't go to official {entity} domain",
                            details={"entity": entity, "type": entity_type, "urls": urls},
                            category="impersonation"
                        ))
                    break
        
        return checks
    
    # ==================== RESULT GENERATION ====================
    
    def _categorize_checks(self, checks: List[FraudCheckResult]) -> Dict:
        """Categorize checks by type"""
        categories = {}
        for check in checks:
            cat = check.category
            if cat not in categories:
                categories[cat] = {"checks": [], "avg_score": 0}
            categories[cat]["checks"].append(asdict(check))
        
        for cat in categories:
            scores = [c["score"] for c in categories[cat]["checks"]]
            categories[cat]["avg_score"] = sum(scores) / len(scores) if scores else 0
        
        return categories
    
    def _generate_assessment(self, checks: List[FraudCheckResult], 
                            urls: List[str], message: str) -> Dict:
        """Generate final fraud assessment"""
        
        if not checks:
            return {
                "verdict": "INSUFFICIENT_DATA",
                "confidence": 30,
                "risk_level": "unknown",
                "interpretation": "Not enough data to analyze"
            }
        
        # Calculate risk score
        total_score = sum(c.score for c in checks)
        avg_score = total_score / len(checks)
        max_score = max(c.score for c in checks)
        
        critical_count = sum(1 for c in checks if c.severity == "critical")
        high_count = sum(1 for c in checks if c.severity == "high")
        safe_count = sum(1 for c in checks if c.severity == "safe")
        
        # Weighted scoring
        risk_score = (
            critical_count * 25 +
            high_count * 15 +
            avg_score * 0.5 +
            (max_score if max_score >= 80 else 0) * 0.3
        )
        
        # Normalize to 0-100
        risk_score = min(100, risk_score)
        
        # Determine verdict
        if critical_count >= 2 or risk_score >= 70:
            verdict = "HIGH_RISK_FRAUD"
            risk_level = "critical"
            confidence = min(95, 65 + critical_count * 10)
            interpretation = "🚨 DANGER: This message shows multiple strong fraud indicators. DO NOT click any links or share any information!"
        elif critical_count >= 1 or high_count >= 2 or risk_score >= 50:
            verdict = "LIKELY_FRAUD"
            risk_level = "high"
            confidence = min(85, 55 + high_count * 8)
            interpretation = "⚠️ WARNING: This message has significant fraud indicators. Be very cautious and verify through official channels."
        elif high_count >= 1 or risk_score >= 35:
            verdict = "SUSPICIOUS"
            risk_level = "medium"
            confidence = min(70, 45 + high_count * 5)
            interpretation = "🔶 CAUTION: This message has some suspicious elements. Verify the sender through official channels before taking action."
        elif safe_count >= len(checks) * 0.6:
            verdict = "LIKELY_SAFE"
            risk_level = "low"
            confidence = min(80, 50 + safe_count * 5)
            interpretation = "✅ This message appears relatively safe, but always verify through official channels for financial matters."
        else:
            verdict = "UNCERTAIN"
            risk_level = "unknown"
            confidence = 40
            interpretation = "🔍 Unable to determine with confidence. Exercise caution and verify independently."
        
        # Identify key risks
        key_risks = [c.finding for c in checks if c.severity in ["critical", "high"]]
        safe_signs = [c.finding for c in checks if c.severity == "safe" and c.score <= 15]
        
        return {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "risk_level": risk_level,
            "risk_score": round(risk_score, 1),
            "interpretation": interpretation,
            "key_risks": key_risks[:5],
            "safe_signs": safe_signs[:3],
            "statistics": {
                "total_checks": len(checks),
                "critical_flags": critical_count,
                "high_flags": high_count,
                "safe_flags": safe_count,
                "urls_analyzed": len(urls)
            }
        }
    
    def _generate_recommendations(self, checks: List[FraudCheckResult], 
                                  assessment: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        verdict = assessment.get("verdict", "")
        
        if verdict in ["HIGH_RISK_FRAUD", "LIKELY_FRAUD"]:
            recommendations.extend([
                "🚫 DO NOT click any links in this message",
                "🚫 DO NOT share any personal information (OTP, PIN, Password, Card details)",
                "🚫 DO NOT call any phone numbers mentioned in the message",
                "📞 If concerned, contact the organization directly using official contact details from their website",
                "📱 Block the sender on your messaging platform",
                "🚨 Report this message to the platform (WhatsApp: long press > Report)"
            ])
        elif verdict == "SUSPICIOUS":
            recommendations.extend([
                "⚠️ Do not click links directly - visit the official website instead",
                "⚠️ Verify any claims by calling official customer service",
                "⚠️ Never share OTP, PIN or passwords - legitimate companies never ask for these",
                "📱 Be cautious of unsolicited messages even from known contacts"
            ])
        else:
            recommendations.extend([
                "✅ Message appears safe, but always verify financial matters independently",
                "📌 When in doubt, contact the organization through official channels",
                "🔒 Never share sensitive information via messaging apps"
            ])
        
        # Check-specific recommendations
        for check in checks:
            if check.name == "url_shortener" and check.severity in ["high", "critical"]:
                recommendations.append("🔗 Shortened URLs hide the real destination - use a URL expander tool to check before clicking")
            elif check.name == "sensitive_data_request":
                recommendations.append("🔐 Banks and companies NEVER ask for OTP/PIN via SMS/WhatsApp")
            elif check.name == "typosquatting":
                recommendations.append("👁️ Look carefully at the URL - scammers use similar-looking domains")
        
        return list(dict.fromkeys(recommendations))  # Remove duplicates while preserving order
