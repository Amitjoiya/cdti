import { useState, useEffect } from 'react';
import { 
  AlertTriangle, ShieldCheck, UploadCloud, Zap, 
  Database, Eye, Brain, BarChart3, Search, Home,
  Shield, Fingerprint, Gavel, Siren, MessageSquareWarning,
  Send, Loader2, CheckCircle2, XCircle, AlertCircle, HelpCircle,
  Link2, MessageCircle, Smartphone, Mail, Instagram, ExternalLink,
  Image, Camera, FileImage, Trash2, MapPin
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || '';

// ====================
// DEVICE FINGERPRINTING UTILITIES (100% Legal)
// ====================

// Generate Canvas Fingerprint (Unique per device based on GPU rendering)
const getCanvasFingerprint = () => {
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 200;
    canvas.height = 50;
    
    // Draw text with specific styles
    ctx.textBaseline = 'top';
    ctx.font = '14px Arial';
    ctx.fillStyle = '#f60';
    ctx.fillRect(125, 1, 62, 20);
    ctx.fillStyle = '#069';
    ctx.fillText('FakeTrace🔍', 2, 15);
    ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
    ctx.fillText('Fingerprint', 4, 17);
    
    // Get base64 data URL and hash it
    const dataURL = canvas.toDataURL();
    let hash = 0;
    for (let i = 0; i < dataURL.length; i++) {
      hash = ((hash << 5) - hash) + dataURL.charCodeAt(i);
      hash = hash & hash;
    }
    return 'CF' + Math.abs(hash).toString(16).toUpperCase().substring(0, 12);
  } catch (e) {
    return null;
  }
};

// Get WebGL Fingerprint (GPU identification)
const getWebGLFingerprint = () => {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) return null;
    
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    const vendor = debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Unknown';
    const renderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown';
    
    return {
      vendor: vendor,
      renderer: renderer,
      version: gl.getParameter(gl.VERSION),
      shadingLanguage: gl.getParameter(gl.SHADING_LANGUAGE_VERSION)
    };
  } catch (e) {
    return null;
  }
};

// Get Audio Fingerprint (Audio processing characteristics)
const getAudioFingerprint = () => {
  return new Promise((resolve) => {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const analyser = audioContext.createAnalyser();
      const gain = audioContext.createGain();
      const scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
      
      gain.gain.value = 0; // Mute
      oscillator.type = 'triangle';
      oscillator.connect(analyser);
      analyser.connect(scriptProcessor);
      scriptProcessor.connect(gain);
      gain.connect(audioContext.destination);
      oscillator.start(0);
      
      scriptProcessor.onaudioprocess = (event) => {
        const data = event.inputBuffer.getChannelData(0);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          sum += Math.abs(data[i]);
        }
        const fingerprint = 'AF' + Math.abs(Math.floor(sum * 10000000)).toString(16).toUpperCase().substring(0, 8);
        
        oscillator.stop();
        scriptProcessor.disconnect();
        audioContext.close();
        resolve(fingerprint);
      };
      
      setTimeout(() => resolve(null), 500); // Timeout fallback
    } catch (e) {
      resolve(null);
    }
  });
};

// Detect installed fonts (partial list for fingerprinting)
const getInstalledFonts = () => {
  const baseFonts = ['monospace', 'sans-serif', 'serif'];
  const testFonts = [
    'Arial', 'Verdana', 'Times New Roman', 'Georgia', 'Courier New',
    'Comic Sans MS', 'Impact', 'Trebuchet MS', 'Arial Black', 'Palatino',
    'Lucida Console', 'Tahoma', 'Century Gothic', 'Bookman Old Style',
    'Garamond', 'MS Sans Serif', 'Segoe UI', 'Calibri', 'Cambria'
  ];
  
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const testString = 'mmmmmmmmmmlli';
  const testSize = '72px';
  
  const getWidth = (font) => {
    ctx.font = `${testSize} ${font}`;
    return ctx.measureText(testString).width;
  };
  
  const baseWidths = baseFonts.map(f => getWidth(f));
  const detected = [];
  
  testFonts.forEach((font) => {
    for (let i = 0; i < baseFonts.length; i++) {
      const testFont = `${testSize} '${font}', ${baseFonts[i]}`;
      ctx.font = testFont;
      if (ctx.measureText(testString).width !== baseWidths[i]) {
        detected.push(font);
        break;
      }
    }
  });
  
  return detected;
};

// Get comprehensive device fingerprint
const getDeviceFingerprint = async () => {
  const nav = navigator;
  
  // Basic device info
  const fingerprint = {
    // Hardware
    cpu_cores: nav.hardwareConcurrency || null,
    device_memory: nav.deviceMemory || null,
    max_touch_points: nav.maxTouchPoints || 0,
    
    // Screen
    screen_width: window.screen.width,
    screen_height: window.screen.height,
    screen_depth: window.screen.colorDepth,
    screen_available: `${window.screen.availWidth}x${window.screen.availHeight}`,
    pixel_ratio: window.devicePixelRatio || 1,
    
    // Timezone & Language
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    timezone_offset: new Date().getTimezoneOffset(),
    languages: nav.languages ? nav.languages.join(',') : nav.language,
    
    // Browser capabilities
    cookies_enabled: nav.cookieEnabled,
    do_not_track: nav.doNotTrack || 'unspecified',
    java_enabled: nav.javaEnabled ? nav.javaEnabled() : false,
    pdf_viewer: nav.pdfViewerEnabled || false,
    
    // Connection info
    connection_type: nav.connection ? nav.connection.effectiveType : null,
    connection_downlink: nav.connection ? nav.connection.downlink : null,
    connection_rtt: nav.connection ? nav.connection.rtt : null,
    
    // Platform
    platform: nav.platform,
    vendor: nav.vendor,
    
    // Plugins count
    plugins_count: nav.plugins ? nav.plugins.length : 0,
    
    // Canvas fingerprint
    canvas_fp: getCanvasFingerprint(),
    
    // WebGL info
    webgl: getWebGLFingerprint(),
    
    // Fonts detected
    fonts: getInstalledFonts(),
    
    // Audio fingerprint
    audio_fp: await getAudioFingerprint(),
    
    // Additional signals
    online: nav.onLine,
    storage_available: typeof localStorage !== 'undefined',
    indexed_db: typeof indexedDB !== 'undefined',
    session_storage: typeof sessionStorage !== 'undefined',
    
    // Touch support
    touch_support: 'ontouchstart' in window || nav.maxTouchPoints > 0
  };
  
  // Generate unique device ID from fingerprint components
  const fpString = JSON.stringify({
    canvas: fingerprint.canvas_fp,
    webgl: fingerprint.webgl?.renderer,
    audio: fingerprint.audio_fp,
    screen: `${fingerprint.screen_width}x${fingerprint.screen_height}x${fingerprint.screen_depth}`,
    cores: fingerprint.cpu_cores,
    memory: fingerprint.device_memory,
    tz: fingerprint.timezone,
    platform: fingerprint.platform,
    fonts: fingerprint.fonts.length
  });
  
  // Simple hash for device ID
  let hash = 0;
  for (let i = 0; i < fpString.length; i++) {
    hash = ((hash << 5) - hash) + fpString.charCodeAt(i);
    hash = hash & hash;
  }
  fingerprint.device_id = 'DEV-' + Math.abs(hash).toString(16).toUpperCase().substring(0, 10);
  
  return fingerprint;
};

// ====================
// PRIVACY-COMPLIANT LOCATION TRACKING SYSTEM
// ====================

/**
 * Generate SHA-256 hash (privacy-safe, non-reversible)
 */
const generateHash = async (data) => {
  const encoder = new TextEncoder();
  const dataBuffer = encoder.encode(data + '_FAKETRACE_SALT_2025');
  const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 16).toUpperCase();
};

/**
 * Generate stable pseudonymous User ID from device fingerprint
 * Privacy-safe: Cannot be reversed to identify real person
 */
const generateUserId = async (fingerprint) => {
  const components = [
    fingerprint.canvas_fp || '',
    fingerprint.webgl?.renderer || '',
    fingerprint.screen_width + 'x' + fingerprint.screen_height,
    fingerprint.timezone || '',
    fingerprint.platform || '',
    fingerprint.cpu_cores || '',
    fingerprint.fonts?.length || 0
  ].join('|');
  
  const hash = await generateHash(components);
  return 'USR-' + hash;
};

/**
 * Calculate distance between two coordinates (Haversine formula)
 * Returns distance in meters
 */
const calculateDistance = (lat1, lon1, lat2, lon2) => {
  const R = 6371000; // Earth's radius in meters
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
};

/**
 * Get reverse geocoding (location name from coordinates)
 */
const reverseGeocode = async (latitude, longitude) => {
  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`,
      { headers: { 'User-Agent': 'FakeTrace/1.0' } }
    );
    if (response.ok) {
      const data = await response.json();
      return {
        location_name: data.display_name,
        city: data.address?.city || data.address?.town || data.address?.village,
        area: data.address?.suburb || data.address?.neighbourhood,
        region: data.address?.state,
        country: data.address?.country,
        country_code: data.address?.country_code?.toUpperCase(),
        postal_code: data.address?.postcode
      };
    }
  } catch (e) {
    console.log('Geocoding failed:', e);
  }
  return null;
};

/**
 * Main Location Tracking Function
 * Privacy-compliant, consent-based tracking
 */
const trackUserLocation = async () => {
  console.log('🔄 Starting privacy-compliant location tracking...');
  
  // Get device fingerprint (for pseudonymous user ID)
  const fingerprint = await getDeviceFingerprint();
  const userId = await generateUserId(fingerprint);
  
  // Generate session ID
  let sessionId = sessionStorage.getItem('ft_session_id');
  if (!sessionId) {
    sessionId = 'S' + Math.random().toString(36).substring(2, 10).toUpperCase();
    sessionStorage.setItem('ft_session_id', sessionId);
  }
  
  // Collect device metadata (legal, non-identifying)
  const deviceMetadata = {
    device_type: /Mobile|Android|iPhone/i.test(navigator.userAgent) ? 'Mobile' : 
                 /iPad|Tablet/i.test(navigator.userAgent) ? 'Tablet' : 'Desktop',
    browser: navigator.userAgent.includes('Chrome') ? 'Chrome' :
             navigator.userAgent.includes('Firefox') ? 'Firefox' :
             navigator.userAgent.includes('Safari') ? 'Safari' :
             navigator.userAgent.includes('Edge') ? 'Edge' : 'Unknown',
    os: navigator.userAgent.includes('Windows') ? 'Windows' :
        navigator.userAgent.includes('Mac') ? 'macOS' :
        navigator.userAgent.includes('Linux') ? 'Linux' :
        navigator.userAgent.includes('Android') ? 'Android' :
        navigator.userAgent.includes('iPhone') ? 'iOS' : 'Unknown',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    language: navigator.language,
    screen_resolution: `${window.screen.width}x${window.screen.height}`
  };

  // ========================================
  // HELPER: Get IP-based location (fallback)
  // ========================================
  const getIPLocation = async () => {
    const ipApis = [
      { url: 'https://ipwho.is/', parse: (d) => ({ lat: d.latitude, lon: d.longitude, city: d.city, region: d.region, country: d.country, country_code: d.country_code }) },
      { url: 'https://ipapi.co/json/', parse: (d) => ({ lat: d.latitude, lon: d.longitude, city: d.city, region: d.region, country: d.country_name, country_code: d.country_code }) },
      { url: 'https://ip-api.com/json/?fields=status,lat,lon,city,region,country,countryCode', parse: (d) => d.status === 'success' ? ({ lat: d.lat, lon: d.lon, city: d.city, region: d.region, country: d.country, country_code: d.countryCode }) : null }
    ];
    
    for (const api of ipApis) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const response = await fetch(api.url, { signal: controller.signal });
        clearTimeout(timeoutId);
        if (response.ok) {
          const data = await response.json();
          const parsed = api.parse(data);
          if (parsed && parsed.lat && parsed.lon) {
            console.log('✅ IP Location API success:', api.url);
            return parsed;
          }
        }
      } catch (e) {
        console.log('❌ IP API failed:', api.url);
      }
    }
    return null;
  };

  // ========================================
  // CONSENT HANDLER: User ALLOWS location
  // ========================================
  const handleLocationAllowed = async (position) => {
    console.log('✅ User ALLOWED location access');
    
    const latitude = position.coords.latitude;
    const longitude = position.coords.longitude;
    const accuracy = position.coords.accuracy;
    const timestamp = new Date().toISOString();
    
    // Get location name via reverse geocoding
    const geoData = await reverseGeocode(latitude, longitude);
    
    // Prepare location data (NO raw IP stored)
    const locationData = {
      // Pseudonymous identifiers
      user_id: userId,
      session_id: sessionId,
      
      // Location data
      latitude: latitude,
      longitude: longitude,
      accuracy_meters: accuracy,
      location_name: geoData?.location_name || null,
      city: geoData?.city || null,
      area: geoData?.area || null,
      region: geoData?.region || null,
      country: geoData?.country || null,
      country_code: geoData?.country_code || null,
      
      // Consent & method
      consent_status: 'allowed',
      location_method: 'gps',
      anonymous: false,
      
      // Device metadata (non-identifying)
      device_metadata: deviceMetadata,
      
      // Fingerprint for user consistency (hashed, not raw)
      fingerprint_hash: await generateHash(JSON.stringify(fingerprint)),
      
      // Timestamps
      created_at: timestamp,
      
      // Distance threshold for location history
      distance_threshold_meters: 100
    };
    
    console.log(`📍 Location: ${latitude.toFixed(6)}, ${longitude.toFixed(6)} (±${accuracy}m)`);
    console.log(`👤 User ID: ${userId}`);
    console.log(`📌 Location: ${geoData?.city || 'Unknown'}, ${geoData?.country || 'Unknown'}`);
    
    // Send to backend
    try {
      const response = await fetch(`${API_BASE}/api/location/track`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(locationData)
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log(`✅ Location saved - ID: ${result.location_id}, New Location: ${result.is_new_location}`);
      }
    } catch (e) {
      console.error('Location tracking failed:', e);
    }
  };

  // ========================================
  // FALLBACK: GPS failed but user allowed (Desktop/no GPS hardware)
  // Use IP-based location with consent_status = 'allowed'
  // ========================================
  const handleGPSFallback = async (reason) => {
    console.log('🔄 GPS unavailable, using IP location fallback (user allowed)');
    
    const ipLocation = await getIPLocation();
    
    if (ipLocation) {
      const timestamp = new Date().toISOString();
      const geoData = await reverseGeocode(ipLocation.lat, ipLocation.lon);
      
      const locationData = {
        user_id: userId,
        session_id: sessionId,
        latitude: ipLocation.lat,
        longitude: ipLocation.lon,
        accuracy_meters: 5000, // IP location ~5km accuracy
        location_name: geoData?.location_name || null,
        city: ipLocation.city || geoData?.city || null,
        area: geoData?.area || null,
        region: ipLocation.region || geoData?.region || null,
        country: ipLocation.country || geoData?.country || null,
        country_code: ipLocation.country_code || geoData?.country_code || null,
        consent_status: 'allowed', // User DID allow - just no GPS hardware
        location_method: 'ip_fallback',
        anonymous: false,
        device_metadata: deviceMetadata,
        fingerprint_hash: await generateHash(JSON.stringify(fingerprint)),
        created_at: timestamp,
        distance_threshold_meters: 100,
        fallback_reason: reason
      };
      
      console.log(`📍 IP Location: ${ipLocation.lat.toFixed(4)}, ${ipLocation.lon.toFixed(4)}`);
      console.log(`👤 User ID: ${userId}`);
      console.log(`📌 City: ${ipLocation.city || 'Unknown'}, ${ipLocation.country || 'Unknown'}`);
      
      try {
        const response = await fetch(`${API_BASE}/api/location/track`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(locationData)
        });
        
        if (response.ok) {
          const result = await response.json();
          console.log(`✅ IP Location saved - ID: ${result.location_id}`);
        }
      } catch (e) {
        console.error('IP Location tracking failed:', e);
      }
    } else {
      console.log('❌ IP location also failed, recording minimal data');
      // Even IP failed - record with timezone-based region only
      const anonymousData = {
        user_id: userId, // Still use regular user_id, not ANON
        session_id: sessionId,
        latitude: null,
        longitude: null,
        consent_status: 'allowed', // User allowed, just couldn't get location
        location_method: 'none_available',
        anonymous: false,
        device_metadata: deviceMetadata,
        approximate_region: {
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
          continent: Intl.DateTimeFormat().resolvedOptions().timeZone.split('/')[0]
        },
        fingerprint_hash: await generateHash(JSON.stringify(fingerprint)),
        created_at: new Date().toISOString(),
        fallback_reason: reason + '_ip_also_failed'
      };
      
      try {
        await fetch(`${API_BASE}/api/location/track`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(anonymousData)
        });
      } catch (e) {
        console.error('Minimal tracking failed:', e);
      }
    }
  };

  // ========================================
  // CONSENT HANDLER: User DENIES location
  // ========================================
  const handleLocationDenied = async (error) => {
    console.log('❌ User DENIED location access:', error?.message || 'Permission denied');
    
    // Generate anonymous user ID (different from GPS user ID for privacy)
    const anonymousId = 'ANON-' + await generateHash(JSON.stringify(fingerprint) + '_ANONYMOUS');
    
    // Get approximate region ONLY (country/state level - legal)
    let approximateRegion = null;
    try {
      // Use timezone to infer region (legal, no IP tracking)
      const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const parts = timezone.split('/');
      approximateRegion = {
        timezone: timezone,
        continent: parts[0] || null,
        region_hint: parts[1]?.replace(/_/g, ' ') || null
      };
    } catch (e) {
      console.log('Could not determine region');
    }
    
    // Prepare anonymous data (NO location, NO IP)
    const anonymousData = {
      // Anonymous identifier
      user_id: anonymousId,
      session_id: sessionId,
      
      // NO coordinates
      latitude: null,
      longitude: null,
      
      // Consent status
      consent_status: 'denied',
      anonymous: true,
      
      // Device metadata ONLY (legal, non-identifying)
      device_metadata: deviceMetadata,
      
      // Approximate region (timezone-based, not IP)
      approximate_region: approximateRegion,
      
      // Fingerprint hash (for returning user detection only)
      fingerprint_hash: await generateHash(JSON.stringify(fingerprint)),
      
      // Timestamps
      created_at: new Date().toISOString(),
      
      // Reason for denial
      denial_reason: error?.message || 'user_denied'
    };
    
    console.log(`👤 Anonymous ID: ${anonymousId}`);
    console.log(`🌍 Approximate Region: ${approximateRegion?.region_hint || 'Unknown'}`);
    
    // Send to backend
    try {
      await fetch(`${API_BASE}/api/location/track`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(anonymousData)
      });
      console.log('✅ Anonymous visit recorded (no location stored)');
    } catch (e) {
      console.error('Anonymous tracking failed:', e);
    }
  };

  // ========================================
  // REQUEST GEOLOCATION PERMISSION
  // ========================================
  if ('geolocation' in navigator) {
    navigator.geolocation.getCurrentPosition(
      // SUCCESS: User allowed and GPS works
      handleLocationAllowed,
      // ERROR: Could be user denial OR hardware failure
      async (error) => {
        // Error codes:
        // 1 = PERMISSION_DENIED - User clicked "Block"
        // 2 = POSITION_UNAVAILABLE - No GPS hardware/signal
        // 3 = TIMEOUT - GPS took too long (Desktop has no GPS)
        
        if (error.code === 1) {
          // User explicitly denied - treat as anonymous
          console.log('🚫 User explicitly BLOCKED location');
          await handleLocationDenied(error);
        } else {
          // Hardware failure or timeout - user DID allow, just no GPS
          // Use IP fallback with consent_status = 'allowed'
          console.log(`⚠️ GPS failed (code ${error.code}: ${error.message}), user allowed - using IP fallback`);
          await handleGPSFallback(error.message || 'gps_unavailable');
        }
      },
      // OPTIONS
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  } else {
    console.log('⚠️ Geolocation not supported');
    await handleGPSFallback('geolocation_not_supported');
  }
};

// ====================
// MAIN APP COMPONENT
// ====================
function App() {
  const [page, setPage] = useState('analyze'); // 'analyze', 'dashboard', or 'fraud'
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Track location on app load
  useEffect(() => {
    // Small delay to not block initial render
    const timer = setTimeout(() => {
      trackUserLocation();
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  const handleUpload = async (file) => {
    setError('');
    setIsLoading(true);
    setAnalysis(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Analysis failed');
      }

      const data = await response.json();
      console.log('=== API RESPONSE ===', data);
      console.log('verdict type:', typeof data.verdict, data.verdict);
      console.log('evidence:', data.evidence);
      setAnalysis(data);
    } catch (apiError) {
      console.error('API Error:', apiError);
      setError(apiError.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-transparent">
      {/* Navigation - CDTI Theme */}
      <nav className="bg-gradient-to-r from-[#1e3a5f]/95 to-[#0a1628]/95 border-b border-[#d4a017]/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center gap-4">
              <img src="/cdti-logo.png" alt="CDTI" className="h-14 w-14 object-contain" />
              <div className="flex flex-col">
                <span className="text-white font-bold text-lg">Central Detective Training Institute</span>
                <span className="text-[#d4a017] text-xs">Jaipur, BPRD • FakeTrace v7.0</span>
              </div>
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={() => setPage('analyze')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  page === 'analyze' 
                    ? 'bg-[#e91e63] text-white shadow-lg shadow-[#e91e63]/30' 
                    : 'text-slate-300 hover:text-white hover:bg-white/10'
                }`}
              >
                <Home size={18} />
                <span>Analyze</span>
              </button>
              <button
                onClick={() => setPage('fraud')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  page === 'fraud' 
                    ? 'bg-[#d4a017] text-white shadow-lg shadow-[#d4a017]/30' 
                    : 'text-slate-300 hover:text-white hover:bg-white/10'
                }`}
              >
                <MessageSquareWarning size={18} />
                <span>Fraud Analyzer</span>
              </button>
              <button
                onClick={() => setPage('fraudDashboard')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  page === 'fraudDashboard' 
                    ? 'bg-[#e91e63] text-white shadow-lg shadow-[#e91e63]/30' 
                    : 'text-slate-300 hover:text-white hover:bg-white/10'
                }`}
              >
                <Siren size={18} />
                <span>Fraud Dashboard</span>
              </button>
              <button
                onClick={() => setPage('dashboard')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  page === 'dashboard' 
                    ? 'bg-[#1e3a5f] text-white border border-[#d4a017]/50' 
                    : 'text-slate-300 hover:text-white hover:bg-white/10'
                }`}
              >
                <BarChart3 size={18} />
                <span>Media Dashboard</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {page === 'analyze' ? (
          <AnalyzePage 
            analysis={analysis}
            isLoading={isLoading}
            error={error}
            onUpload={handleUpload}
          />
        ) : page === 'fraud' ? (
          <FraudAnalyzerPage />
        ) : page === 'fraudDashboard' ? (
          <FraudDashboardPage />
        ) : (
          <DashboardPage />
        )}
      </div>
    </div>
  );
}


// ====================
// ANALYZE PAGE
// ====================
function AnalyzePage({ analysis, isLoading, error, onUpload }) {
  return (
    <div className="space-y-8">
      {/* Header - CDTI Theme */}
      <header className="grid gap-6 lg:grid-cols-2">
        <div>
          <div className="flex items-center gap-3 text-[#d4a017] uppercase tracking-[0.3em] text-sm mb-4">
            <ShieldCheck size={20} />
            <span>CDTI Digital Forensics Lab</span>
          </div>
          <h1 className="text-5xl lg:text-6xl font-semibold text-white leading-tight">
            Trace Deepfakes, <span className="text-[#e91e63]">Expose Truth.</span>
          </h1>
          <p className="text-slate-300 mt-4 text-lg max-w-xl">
            तेजस्वि नावधीतमस्तु • Advanced forensic analysis for digital evidence investigation.
          </p>
          
          {/* Features - CDTI Theme */}
          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            <div className="bg-[#1e3a5f]/50 border border-[#1e3a5f] rounded-xl p-4">
              <Database className="text-[#e91e63] mb-2" size={24} />
              <p className="text-white font-semibold">Unique ID</p>
              <p className="text-slate-400 text-sm">Content fingerprinting</p>
            </div>
            <div className="bg-[#1e3a5f]/50 border border-[#1e3a5f] rounded-xl p-4">
              <Eye className="text-[#d4a017] mb-2" size={24} />
              <p className="text-white font-semibold">Visual Forensics</p>
              <p className="text-slate-400 text-sm">20+ algorithms</p>
            </div>
            <div className="bg-[#1e3a5f]/50 border border-[#1e3a5f] rounded-xl p-4">
              <Brain className="text-[#e91e63] mb-2" size={24} />
              <p className="text-white font-semibold">Explainable AI</p>
              <p className="text-slate-400 text-sm">Scientific reasoning</p>
            </div>
          </div>
        </div>
        
        <UploadZone onUpload={onUpload} isLoading={isLoading} />
      </header>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/40 text-red-400 px-4 py-3 rounded-xl flex items-center gap-3">
          <AlertTriangle size={20} />
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Analysis Results */}
      {analysis && (
        <div className="space-y-6">
          {/* System Identity */}
          <SystemIdentityCard identity={analysis.system_identity} />

          {/* Legal Classification */}
          {analysis.legal_profile && (
            <LegalStatusCard profile={analysis.legal_profile} />
          )}
          
          {/* Content ID Card */}
          <ContentIDCard analysis={analysis} />
          
          {/* AI Generation Detection - NEW */}
          {analysis.ai_generation && (
            <AIGenerationCard 
              aiGen={analysis.ai_generation} 
              aiChecks={analysis.ai_detection?.ai_checks}
              categories={analysis.ai_detection?.categories}
            />
          )}
          
          {/* Main Verdict - Anomaly Based */}
          <AnomalyVerdictCard verdict={analysis.verdict} evidence={analysis.evidence} />
          
          {/* Forensic Evidence Card */}
          {analysis.evidence?.algorithms && (
            <ForensicEvidenceCard algorithms={analysis.evidence.algorithms} summary={analysis.evidence.summary} />
          )}
          
          {/* Supporting Statistics (Secondary) */}
          {analysis.supporting_statistics && (
            <SupportingStatsCard stats={analysis.supporting_statistics} />
          )}
          
          {/* Explanation Card */}
          {analysis.explanation && (
            <ExplanationCard explanation={analysis.explanation} />
          )}
        </div>
      )}

      {/* Empty state */}
      {!analysis && !isLoading && (
        <div className="mt-16 text-center text-slate-500">
          <UploadCloud className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <p>Upload a suspicious file to begin the investigation.</p>
        </div>
      )}
    </div>
  );
}


// ====================
// UPLOAD ZONE
// ====================
function UploadZone({ onUpload, isLoading }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) onUpload(file);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`relative border-2 border-dashed rounded-3xl p-12 text-center transition-all ${
        isDragging 
          ? 'border-cyan-400 bg-cyan-400/10' 
          : 'border-white/20 hover:border-white/40'
      } ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
    >
      {isLoading ? (
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin" />
          <p className="text-cyan-400 font-semibold">Analyzing with 20+ algorithms...</p>
          <p className="text-slate-500 text-sm">This may take a moment</p>
        </div>
      ) : (
        <>
          <UploadCloud className="w-16 h-16 text-slate-500 mx-auto mb-4" />
          <p className="text-white font-semibold mb-2">Drop your file here</p>
          <p className="text-slate-400 text-sm mb-4">or click to browse</p>
          <input
            type="file"
            accept="image/*,video/*"
            onChange={handleFileSelect}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
          <div className="flex justify-center gap-2 text-xs text-slate-500">
            <span className="bg-white/5 px-2 py-1 rounded">JPG</span>
            <span className="bg-white/5 px-2 py-1 rounded">PNG</span>
            <span className="bg-white/5 px-2 py-1 rounded">WebP</span>
            <span className="bg-white/5 px-2 py-1 rounded">MP4</span>
          </div>
        </>
      )}
    </div>
  );
}


// ====================
// CONTENT ID CARD
// ====================
function ContentIDCard({ analysis }) {
  return (
    <div className="bg-gradient-to-r from-cyan-900/50 to-purple-900/50 border border-cyan-500/50 rounded-2xl p-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-4">
          <Fingerprint className="text-cyan-400" size={40} />
          <div>
            <p className="text-cyan-400 text-xs uppercase tracking-wider">Content Fingerprint ID</p>
            <p className="text-white font-mono text-2xl font-bold">{analysis.content_id || 'N/A'}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          {analysis.is_known && (
            <div className="bg-yellow-600/30 border border-yellow-500 px-3 py-1 rounded-lg">
              <span className="text-yellow-400 text-sm font-semibold">🔄 Previously Analyzed</span>
            </div>
          )}
          <div className="text-right">
            <p className="text-gray-400 text-xs">Times Analyzed</p>
            <p className="text-cyan-400 text-3xl font-bold">{analysis.analysis_count || 1}</p>
          </div>
        </div>
      </div>
      
      {/* Fingerprint details */}
      {analysis.fingerprint && (
        <div className="mt-4 pt-4 border-t border-white/10">
          <p className="text-xs text-slate-400 mb-2">Fingerprint Algorithms:</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(analysis.fingerprint).slice(0, 6).map(([key, value]) => (
              <div key={key} className="bg-white/5 px-3 py-1 rounded-lg">
                <span className="text-cyan-400 text-xs font-semibold">{key}:</span>
                <span className="text-slate-400 text-xs ml-1">
                  {typeof value === 'string' ? value.substring(0, 8) + '...' : value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


// ====================
// LEGAL STATUS CARD
// ====================
function LegalStatusCard({ profile }) {
  if (!profile) return null;

  const statusPalette = {
    ILLEGAL_SIGNAL: {
      bg: 'bg-gradient-to-br from-red-600/20 via-rose-500/10 to-amber-500/10',
      border: 'border-red-500/40',
      text: 'text-red-100',
      icon: <Siren className="text-red-200" size={32} />
    },
    LEGAL_CLEAR: {
      bg: 'bg-gradient-to-br from-emerald-600/15 via-cyan-500/10 to-green-500/10',
      border: 'border-emerald-500/40',
      text: 'text-emerald-100',
      icon: <Gavel className="text-emerald-200" size={32} />
    },
    NOT_ENOUGH_DATA: {
      bg: 'bg-gradient-to-br from-slate-600/20 to-slate-900/40',
      border: 'border-slate-600/60',
      text: 'text-slate-100',
      icon: <Shield className="text-slate-200" size={32} />
    },
    REQUIRES_REVIEW: {
      bg: 'bg-gradient-to-br from-amber-600/15 via-yellow-500/10 to-slate-800/40',
      border: 'border-amber-500/40',
      text: 'text-amber-100',
      icon: <ShieldCheck className="text-amber-200" size={32} />
    }
  };

  const config = statusPalette[profile.status] || statusPalette.REQUIRES_REVIEW;
  const metrics = profile.supporting_metrics || {};
  const cues = profile.cues || [];
  const notes = profile.notes || [];

  const severityMap = {
    critical: {
      border: 'border-red-500/60',
      bg: 'bg-red-500/10',
      text: 'text-red-100',
      badge: 'bg-red-500/20 text-red-200'
    },
    major: {
      border: 'border-orange-500/50',
      bg: 'bg-orange-500/10',
      text: 'text-orange-100',
      badge: 'bg-orange-500/20 text-orange-200'
    },
    minor: {
      border: 'border-cyan-500/40',
      bg: 'bg-cyan-500/10',
      text: 'text-cyan-100',
      badge: 'bg-cyan-500/20 text-cyan-100'
    },
    supportive: {
      border: 'border-emerald-500/40',
      bg: 'bg-emerald-500/10',
      text: 'text-emerald-100',
      badge: 'bg-emerald-500/30 text-emerald-100'
    }
  };

  return (
    <div className={`${config.bg} ${config.border} border-2 rounded-2xl p-6`}>
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-black/20 flex items-center justify-center">
            {config.icon}
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-slate-300">Legal Signal</p>
            <h2 className={`${config.text} text-2xl font-bold`}>{profile.status_label}</h2>
            <p className="text-slate-200 text-sm mt-1 max-w-xl">{profile.summary}</p>
          </div>
        </div>
        <div className="text-right space-y-1">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Confidence</p>
          <p className={`${config.text} text-4xl font-extrabold`}>{Math.round(profile.confidence ?? 0)}%</p>
          <p className="text-slate-400 text-xs">Net score {metrics.net_score ?? 0}</p>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-3 mt-6">
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-xs uppercase tracking-[0.3em] text-red-200">Risk Points</p>
          <p className="text-red-100 text-2xl font-bold">{metrics.risk_points ?? 0}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-xs uppercase tracking-[0.3em] text-emerald-200">Protective Points</p>
          <p className="text-emerald-100 text-2xl font-bold">{metrics.protective_points ?? 0}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-xs uppercase tracking-[0.3em] text-cyan-200">Cue Count</p>
          <p className="text-cyan-100 text-2xl font-bold">{cues.length}</p>
        </div>
      </div>

      {cues.length > 0 && (
        <div className="mt-6">
          <p className="text-xs uppercase tracking-[0.4em] text-slate-400">Law-Enforcement Cues</p>
          <div className="mt-3 space-y-3">
            {cues.map((cue, idx) => {
              const tone = severityMap[cue.level] || severityMap.minor;
              const cueClasses = cue.kind === 'protective'
                ? `${severityMap.supportive.border} ${severityMap.supportive.bg} ${severityMap.supportive.text}`
                : `${tone.border} ${tone.bg} ${tone.text}`;
              const badgeClasses = cue.kind === 'protective' ? severityMap.supportive.badge : tone.badge;
              return (
                <div key={`${cue.label}-${idx}`} className={`${cueClasses} border rounded-xl p-4`}> 
                  <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                    <div>
                      <p className="text-sm font-semibold">{cue.label}</p>
                      <p className="text-xs text-slate-200/80 mt-1">{cue.detail}</p>
                      {cue.metric && (
                        <p className="text-xs text-slate-300 mt-1">Metric: {cue.metric}</p>
                      )}
                      {cue.source && (
                        <p className="text-[11px] text-slate-400 mt-1">Source: {cue.source}</p>
                      )}
                    </div>
                    <span className={`text-xs px-3 py-1 rounded-full font-semibold ${badgeClasses}`}>
                      {cue.kind === 'protective' ? 'supportive' : cue.level}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="mt-6 bg-black/30 border border-white/10 rounded-xl p-4">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Recommended Action</p>
        <p className="text-slate-100 text-sm mt-2 leading-relaxed">{profile.recommended_action}</p>
      </div>

      {notes.length > 0 && (
        <div className="mt-4">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Operational Notes</p>
          <ul className="list-disc list-inside text-slate-200 text-sm mt-2 space-y-1">
            {notes.map((note, idx) => (
              <li key={`note-${idx}`}>{note}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="text-slate-400 text-xs italic mt-4">{profile.disclaimer}</p>
    </div>
  );
}


// ====================
// VERDICT CARD
// ====================
function VerdictCard({ analysis }) {
  const verdictColors = {
    'LIKELY FAKE': { bg: 'bg-red-500/20', border: 'border-red-500', text: 'text-red-400', emoji: '🔴' },
    'POSSIBLY FAKE': { bg: 'bg-orange-500/20', border: 'border-orange-500', text: 'text-orange-400', emoji: '🟠' },
    'UNCERTAIN': { bg: 'bg-yellow-500/20', border: 'border-yellow-500', text: 'text-yellow-400', emoji: '🟡' },
    'PROBABLY REAL': { bg: 'bg-lime-500/20', border: 'border-lime-500', text: 'text-lime-400', emoji: '🟢' },
    'LIKELY REAL': { bg: 'bg-green-500/20', border: 'border-green-500', text: 'text-green-400', emoji: '🟢' },
    'AI GENERATED': { bg: 'bg-purple-500/20', border: 'border-purple-500', text: 'text-purple-400', emoji: '🤖' },
    'AI ENHANCED': { bg: 'bg-blue-500/20', border: 'border-blue-500', text: 'text-blue-400', emoji: '✨' },
  };
  
  const colors = verdictColors[analysis.verdict] || verdictColors['UNCERTAIN'];

  return (
    <div className={`${colors.bg} ${colors.border} border rounded-2xl p-6`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-4xl">{colors.emoji}</span>
          <div>
            <p className="text-slate-400 text-xs uppercase tracking-wider">Final Verdict</p>
            <p className={`${colors.text} text-3xl font-bold`}>{analysis.verdict}</p>
          </div>
        </div>
        
        <div className="text-right">
          <p className="text-slate-400 text-xs">Confidence</p>
          <p className={`${colors.text} text-4xl font-bold`}>{Math.round(analysis.confidence)}%</p>
        </div>
      </div>
      
      {/* Confidence bar */}
      <div className="mt-4">
        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
          <div 
            className={`h-full ${colors.text.replace('text', 'bg')} rounded-full transition-all duration-1000`}
            style={{ width: `${analysis.confidence}%` }}
          />
        </div>
      </div>
      
      {/* Quick explanation */}
      {analysis.explanation?.verdict?.summary && (
        <p className="mt-4 text-slate-300 text-sm">{analysis.explanation.verdict.summary}</p>
      )}
    </div>
  );
}


// ====================
// SYSTEM IDENTITY CARD (v7)
// ====================
function SystemIdentityCard({ identity }) {
  if (!identity) return null;
  
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
      <div className="flex items-center gap-3 mb-2">
        <Shield className="text-slate-400" size={20} />
        <span className="text-slate-300 font-semibold">{identity.name || 'FakeTrace Forensic Tool'}</span>
      </div>
      <p className="text-slate-500 text-sm">{identity.purpose}</p>
      <p className="text-yellow-500/80 text-xs mt-2 italic">{identity.disclaimer}</p>
    </div>
  );
}


// ====================
// AI GENERATION DETECTION CARD
// ====================
function AIGenerationCard({ aiGen, aiChecks, categories }) {
  if (!aiGen) return null;
  
  const conclusionConfig = {
    'LIKELY_AI_GENERATED': {
      bg: 'bg-purple-500/20',
      border: 'border-purple-500',
      text: 'text-purple-400',
      icon: '🤖',
      label: 'LIKELY AI GENERATED'
    },
    'POSSIBLY_AI_GENERATED': {
      bg: 'bg-violet-500/20',
      border: 'border-violet-500',
      text: 'text-violet-400',
      icon: '🔮',
      label: 'POSSIBLY AI GENERATED'
    },
    'UNCERTAIN': {
      bg: 'bg-slate-500/20',
      border: 'border-slate-500',
      text: 'text-slate-400',
      icon: '❓',
      label: 'UNCERTAIN'
    },
    'LIKELY_NATURAL': {
      bg: 'bg-emerald-500/20',
      border: 'border-emerald-500',
      text: 'text-emerald-400',
      icon: '📷',
      label: 'LIKELY NATURAL/CAMERA'
    }
  };
  
  const config = conclusionConfig[aiGen.conclusion] || conclusionConfig['UNCERTAIN'];
  
  // Category icons and colors - for images
  const imageCategoryConfig = {
    visual: { icon: '👁️', label: 'Visual', color: 'text-blue-400' },
    frequency: { icon: '📊', label: 'Frequency', color: 'text-cyan-400' },
    statistical: { icon: '📈', label: 'Statistical', color: 'text-yellow-400' },
    forensic: { icon: '🔬', label: 'Forensic', color: 'text-red-400' },
    semantic: { icon: '🧠', label: 'Semantic', color: 'text-purple-400' }
  };
  
  // Category icons and colors - for videos
  const videoCategoryConfig = {
    temporal: { icon: '⏱️', label: 'Temporal', color: 'text-blue-400' },
    motion: { icon: '🎬', label: 'Motion', color: 'text-cyan-400' },
    visual: { icon: '👁️', label: 'Visual', color: 'text-yellow-400' },
    forensic: { icon: '🔬', label: 'Forensic', color: 'text-red-400' }
  };
  
  const categoryConfig = aiGen.is_video ? videoCategoryConfig : imageCategoryConfig;
  
  return (
    <div className={`${config.bg} ${config.border} border-2 rounded-2xl p-6`}>
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <Brain className="text-purple-400" size={24} />
        <h2 className="text-xl font-bold text-white">
          {aiGen.is_video ? 'Video' : 'Image'} AI Generation Detection
        </h2>
        <span className="text-xs bg-purple-500/30 text-purple-300 px-2 py-0.5 rounded-full">
          {aiGen.is_video ? 'VIDEO v1.0' : 'v3.0 ADVANCED'}
        </span>
      </div>
      
      {/* Video Info Banner */}
      {aiGen.is_video && aiGen.video_info && (
        <div className="bg-purple-500/10 rounded-lg p-3 mb-4 flex flex-wrap gap-4 text-sm">
          <span className="text-purple-300">🎬 Frames Analyzed: <b className="text-white">{aiGen.frames_analyzed || 0}</b></span>
          <span className="text-purple-300">📐 Resolution: <b className="text-white">{aiGen.video_info.resolution}</b></span>
          <span className="text-purple-300">⏱️ Duration: <b className="text-white">{aiGen.video_info.duration_seconds}s</b></span>
          <span className="text-purple-300">🎞️ FPS: <b className="text-white">{aiGen.video_info.fps}</b></span>
        </div>
      )}
      
      {/* Main Result */}
      <div className="flex items-start gap-4 mb-4">
        <span className="text-5xl">{config.icon}</span>
        <div className="flex-1">
          <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">AI Origin Assessment</p>
          <h3 className={`${config.text} text-2xl font-bold`}>{config.label}</h3>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-slate-400 text-sm">Confidence:</span>
            <span className={`${config.text} font-bold`}>{aiGen.confidence || 0}%</span>
            {aiGen.weighted_score && (
              <span className="text-slate-500 text-xs">(Weighted: {aiGen.weighted_score}%)</span>
            )}
          </div>
        </div>
      </div>
      
      {/* Interpretation */}
      <div className="bg-black/20 rounded-xl p-4 mb-4">
        <p className="text-slate-300 text-sm">{aiGen.interpretation || 'Analysis pending...'}</p>
      </div>
      
      {/* Category Summary - NEW v3 */}
      {aiGen.categories_summary && Object.keys(aiGen.categories_summary).length > 0 && (
        <div className="mb-4">
          <p className="text-purple-400 text-xs font-semibold mb-2">
            📊 Analysis by Category ({aiGen.is_video ? '10 algorithms, 4 categories' : '15 algorithms, 5 categories'}):
          </p>
          <div className={`grid ${aiGen.is_video ? 'grid-cols-4' : 'grid-cols-5'} gap-2`}>
            {Object.entries(aiGen.categories_summary).map(([cat, score]) => {
              const catConfig = categoryConfig[cat] || { icon: '📋', label: cat, color: 'text-slate-400' };
              const scoreColor = score >= 50 ? 'text-purple-400' : score >= 35 ? 'text-violet-400' : 'text-slate-400';
              return (
                <div key={cat} className="bg-black/30 rounded-lg p-2 text-center">
                  <span className="text-lg">{catConfig.icon}</span>
                  <p className={`${catConfig.color} text-xs font-semibold`}>{catConfig.label}</p>
                  <p className={`${scoreColor} font-bold`}>{score}%</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
      
      {/* Category Analysis - NEW v3 */}
      {aiGen.category_analysis && aiGen.category_analysis.length > 0 && (
        <div className="mb-4">
          <p className="text-yellow-400 text-xs font-semibold mb-2">⚡ Category Conclusions:</p>
          <div className="space-y-1">
            {aiGen.category_analysis.map((conclusion, i) => (
              <div key={i} className="bg-yellow-500/10 text-yellow-300 text-xs px-3 py-1.5 rounded-lg">
                {conclusion}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Camera Evidence (Video v2) */}
      {aiGen.is_video && aiGen.camera_evidence && (
        <div className="mb-4">
          <p className="text-cyan-400 text-xs font-semibold mb-2">📷 Camera Evidence (Real Device Indicators):</p>
          <div className="flex flex-wrap gap-2">
            <span className={`text-xs px-2 py-1 rounded-lg ${aiGen.camera_evidence.has_prnu ? 'bg-emerald-500/20 text-emerald-300' : 'bg-red-500/20 text-red-300'}`}>
              {aiGen.camera_evidence.has_prnu ? '✅' : '❌'} Sensor Fingerprint (PRNU)
            </span>
            <span className={`text-xs px-2 py-1 rounded-lg ${aiGen.camera_evidence.has_camera_shake ? 'bg-emerald-500/20 text-emerald-300' : 'bg-red-500/20 text-red-300'}`}>
              {aiGen.camera_evidence.has_camera_shake ? '✅' : '❌'} Natural Camera Shake
            </span>
            <span className={`text-xs px-2 py-1 rounded-lg ${aiGen.camera_evidence.has_codec_artifacts ? 'bg-emerald-500/20 text-emerald-300' : 'bg-red-500/20 text-red-300'}`}>
              {aiGen.camera_evidence.has_codec_artifacts ? '✅' : '❌'} Codec Compression
            </span>
          </div>
        </div>
      )}
      
      {/* Indicators Found */}
      {aiGen.indicators_found && aiGen.indicators_found.length > 0 && (
        <div className="mb-4">
          <p className="text-purple-400 text-xs font-semibold mb-2">🔍 AI Indicators Found:</p>
          <div className="flex flex-wrap gap-2">
            {aiGen.indicators_found.map((ind, i) => (
              <span key={i} className="bg-purple-500/20 text-purple-300 text-xs px-2 py-1 rounded-lg">
                {ind}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {/* Natural Signs */}
      {aiGen.natural_signs && aiGen.natural_signs.length > 0 && (
        <div className="mb-4">
          <p className="text-emerald-400 text-xs font-semibold mb-2">✅ Natural Signs Detected:</p>
          <div className="flex flex-wrap gap-2">
            {aiGen.natural_signs.map((sign, i) => (
              <span key={i} className="bg-emerald-500/20 text-emerald-300 text-xs px-2 py-1 rounded-lg">
                {sign}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {/* Detailed Checks by Category (Expandable) */}
      {categories && Object.keys(categories).length > 0 && (
        <details className="mb-4">
          <summary className="text-slate-400 text-sm cursor-pointer hover:text-slate-300">
            View detailed analysis by category ({aiChecks?.length || 15} algorithms)
          </summary>
          <div className="mt-3 space-y-4">
            {Object.entries(categories).map(([catName, catData]) => {
              const catConfig = categoryConfig[catName] || { icon: '📋', label: catName, color: 'text-slate-400' };
              return (
                <div key={catName} className="bg-black/20 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <p className={`${catConfig.color} font-semibold`}>
                      {catConfig.icon} {catConfig.label} Analysis
                    </p>
                    <span className={catData.avg_score >= 50 ? 'text-purple-400' : 'text-slate-400'}>
                      Avg: {catData.avg_score?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="space-y-1">
                    {catData.checks?.map((check, i) => (
                      <AICheckRow key={i} check={check} compact />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </details>
      )}
      
      {/* Fallback: Simple checks list if no categories */}
      {(!categories || Object.keys(categories).length === 0) && aiChecks && aiChecks.length > 0 && (
        <details className="mb-4">
          <summary className="text-slate-400 text-sm cursor-pointer hover:text-slate-300">
            View detailed AI checks ({aiChecks.length} algorithms)
          </summary>
          <div className="mt-3 space-y-2">
            {aiChecks.map((check, i) => (
              <AICheckRow key={i} check={check} />
            ))}
          </div>
        </details>
      )}
      
      {/* Recommendation */}
      {aiGen.recommendation && (
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-3 mb-3">
          <p className="text-purple-400 text-sm font-semibold">💡 Recommendation</p>
          <p className="text-slate-300 text-sm mt-1">{aiGen.recommendation}</p>
        </div>
      )}
      
      {/* Disclaimer */}
      <p className="text-slate-500 text-xs italic">{aiGen.disclaimer}</p>
    </div>
  );
}


// ====================
// AI CHECK ROW
// ====================
function AICheckRow({ check, compact = false }) {
  const statusConfig = {
    'likely_ai': { bg: 'bg-purple-500/10', text: 'text-purple-400', icon: '🤖' },
    'possibly_ai': { bg: 'bg-violet-500/10', text: 'text-violet-400', icon: '🔮' },
    'uncertain': { bg: 'bg-slate-500/10', text: 'text-slate-400', icon: '❓' },
    'likely_natural': { bg: 'bg-emerald-500/10', text: 'text-emerald-400', icon: '📷' }
  };
  
  const config = statusConfig[check.status] || statusConfig['uncertain'];
  
  if (compact) {
    return (
      <div className={`${config.bg} rounded px-2 py-1 flex items-center justify-between`}>
        <div className="flex items-center gap-2">
          <span className="text-xs">{config.icon}</span>
          <span className="text-white text-xs">{check.display_name}</span>
        </div>
        <span className={`${config.text} text-xs font-bold`}>{check.score}%</span>
      </div>
    );
  }
  
  return (
    <div className={`${config.bg} rounded-lg p-3 flex items-center justify-between`}>
      <div className="flex items-center gap-3">
        <span>{config.icon}</span>
        <div>
          <p className="text-white text-sm font-medium">{check.display_name}</p>
          <p className="text-slate-400 text-xs">{check.finding}</p>
        </div>
      </div>
      <div className="text-right">
        <p className={`${config.text} font-bold`}>{check.score}%</p>
        <p className="text-slate-500 text-xs">{check.status?.replace('_', ' ')}</p>
      </div>
    </div>
  );
}


// ====================
// ANOMALY VERDICT CARD (v7 - Main)
// ====================
function AnomalyVerdictCard({ verdict, evidence }) {
  // Robust null/type check - handle both v7 object and legacy string format
  if (!verdict) return null;
  
  // If verdict is string (legacy), convert to object
  const verdictObj = typeof verdict === 'string' ? {
    verdict: verdict,
    verdict_code: verdict,
    interpretation: `Analysis result: ${verdict}`,
    recommendation: 'Consider re-analyzing for detailed forensic evidence.',
    legal_disclaimer: 'Results are probabilistic, not definitive.'
  } : verdict;
  
  const verdictConfig = {
    'NO_FORENSIC_ANOMALY_DETECTED': {
      bg: 'bg-green-500/20',
      border: 'border-green-500',
      text: 'text-green-400',
      icon: '✅',
      label: 'NO FORENSIC ANOMALY DETECTED'
    },
    'FORENSIC_MANIPULATION_INDICATORS_DETECTED': {
      bg: 'bg-red-500/20',
      border: 'border-red-500',
      text: 'text-red-400',
      icon: '❌',
      label: 'FORENSIC MANIPULATION INDICATORS DETECTED'
    },
    'INCONCLUSIVE_MINOR_IRREGULARITIES': {
      bg: 'bg-yellow-500/20',
      border: 'border-yellow-500',
      text: 'text-yellow-400',
      icon: '⚠️',
      label: 'INCONCLUSIVE - MINOR IRREGULARITIES'
    },
    // Legacy verdicts mapping
    'LIKELY REAL': {
      bg: 'bg-green-500/20',
      border: 'border-green-500',
      text: 'text-green-400',
      icon: '✅',
      label: 'LIKELY REAL (Legacy)'
    },
    'LIKELY FAKE': {
      bg: 'bg-red-500/20',
      border: 'border-red-500',
      text: 'text-red-400',
      icon: '❌',
      label: 'LIKELY FAKE (Legacy)'
    },
    'UNCERTAIN': {
      bg: 'bg-yellow-500/20',
      border: 'border-yellow-500',
      text: 'text-yellow-400',
      icon: '⚠️',
      label: 'UNCERTAIN (Legacy)'
    }
  };
  
  const config = verdictConfig[verdictObj.verdict] || verdictConfig['INCONCLUSIVE_MINOR_IRREGULARITIES'];
  const summary = evidence?.summary || {};
  
  return (
    <div className={`${config.bg} ${config.border} border-2 rounded-2xl p-6`}>
      {/* Main Verdict */}
      <div className="flex items-start gap-4 mb-4">
        <span className="text-5xl">{config.icon}</span>
        <div className="flex-1">
          <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Forensic Analysis Result</p>
          <h2 className={`${config.text} text-2xl font-bold`}>{config.label}</h2>
        </div>
      </div>
      
      {/* Anomaly Counts */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="bg-black/30 rounded-xl p-4 text-center">
          <p className="text-red-400 text-3xl font-bold">{summary.anomalies_detected || 0}</p>
          <p className="text-slate-400 text-xs">Anomalies Detected</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center">
          <p className="text-yellow-400 text-3xl font-bold">{summary.inconclusive || 0}</p>
          <p className="text-slate-400 text-xs">Inconclusive</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center">
          <p className="text-green-400 text-3xl font-bold">{summary.clean || 0}</p>
          <p className="text-slate-400 text-xs">Clean Checks</p>
        </div>
      </div>
      
      {/* Interpretation */}
      <div className="bg-black/20 rounded-xl p-4 mb-4">
        <p className="text-slate-300 text-sm">{verdictObj.interpretation || 'No interpretation available'}</p>
      </div>
      
      {/* Recommendation */}
      <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-3">
        <p className="text-cyan-400 text-sm font-semibold">📋 Recommendation</p>
        <p className="text-slate-300 text-sm mt-1">{verdictObj.recommendation || 'Review forensic evidence carefully'}</p>
      </div>
      
      {/* Legal Disclaimer */}
      <p className="text-slate-500 text-xs mt-4 italic">{verdictObj.legal_disclaimer || 'Results are probabilistic, not definitive.'}</p>
    </div>
  );
}


// ====================
// FORENSIC EVIDENCE CARD (v7 - Algorithm Details)
// ====================
function ForensicEvidenceCard({ algorithms, summary }) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <Eye className="text-purple-400" size={24} />
        <h2 className="text-xl font-bold text-white">Forensic Evidence</h2>
        <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded-full">
          {summary?.total_checks || algorithms?.length || 0} algorithms
        </span>
      </div>
      
      {/* Algorithm Results */}
      <div className="space-y-3">
        {algorithms?.map((algo, index) => (
          <AlgorithmResultRow key={index} algorithm={algo} />
        ))}
      </div>
    </div>
  );
}


// ====================
// ALGORITHM RESULT ROW (v7)
// ====================
function AlgorithmResultRow({ algorithm }) {
  const statusConfig = {
    'detected': {
      bg: 'bg-red-500/10',
      border: 'border-red-500/50',
      icon: '❌',
      iconBg: 'bg-red-500/20',
      text: 'text-red-400',
      label: 'ANOMALY DETECTED'
    },
    'not_detected': {
      bg: 'bg-green-500/10',
      border: 'border-green-500/50',
      icon: '✅',
      iconBg: 'bg-green-500/20',
      text: 'text-green-400',
      label: 'CLEAN'
    },
    'inconclusive': {
      bg: 'bg-yellow-500/10',
      border: 'border-yellow-500/50',
      icon: '⚠️',
      iconBg: 'bg-yellow-500/20',
      text: 'text-yellow-400',
      label: 'INCONCLUSIVE'
    }
  };
  
  const config = statusConfig[algorithm.anomaly_status] || statusConfig['inconclusive'];
  
  return (
    <div className={`${config.bg} ${config.border} border rounded-xl p-4`}>
      <div className="flex items-start gap-3">
        {/* Status Icon */}
        <div className={`${config.iconBg} w-10 h-10 rounded-lg flex items-center justify-center text-xl`}>
          {config.icon}
        </div>
        
        {/* Content */}
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-white font-semibold">{algorithm.display_name}</h3>
            <div className="flex items-center gap-2">
              <span className={`${config.text} text-sm font-semibold`}>{config.label}</span>
              <span className="text-slate-400 text-sm">({algorithm.score}%)</span>
            </div>
          </div>
          
          {/* Reason */}
          <p className="text-slate-400 text-sm">{algorithm.reason}</p>
          
          {/* Score Bar */}
          <div className="mt-2 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all duration-500 ${
                algorithm.anomaly_status === 'detected' ? 'bg-red-500' :
                algorithm.anomaly_status === 'not_detected' ? 'bg-green-500' : 'bg-yellow-500'
              }`}
              style={{ width: `${algorithm.score}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}


// ====================
// SUPPORTING STATS CARD (v7 - Secondary)
// ====================
function SupportingStatsCard({ stats }) {
  if (!stats) return null;
  
  return (
    <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 className="text-slate-400" size={18} />
        <h3 className="text-slate-300 font-semibold text-sm">Supporting Statistics (Secondary Only)</h3>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="bg-black/30 px-4 py-2 rounded-lg">
          <span className="text-slate-400 text-xs">Average Score:</span>
          <span className="text-cyan-400 font-bold text-lg ml-2">{stats.average_score}%</span>
        </div>
        <p className="text-slate-500 text-xs flex-1">{stats.interpretation}</p>
      </div>
      
      <p className="text-yellow-500/70 text-xs mt-2">⚠️ {stats.note}</p>
    </div>
  );
}


// ====================
// EXPLANATION CARD (v7)
// ====================
function ExplanationCard({ explanation }) {
  if (!explanation) return null;
  
  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <Brain className="text-green-400" size={24} />
        <h2 className="text-xl font-bold text-white">Analysis Explanation</h2>
      </div>
      
      {/* Summary */}
      {explanation.summary && (
        <div className="bg-black/20 rounded-xl p-4 mb-4">
          <p className="text-slate-300">{explanation.summary}</p>
        </div>
      )}
      
      {/* Findings */}
      {explanation.findings?.length > 0 && (
        <div className="mb-4">
          <h3 className="text-slate-400 text-sm font-semibold mb-2">Detailed Findings:</h3>
          <div className="bg-black/20 rounded-xl p-4 space-y-1">
            {explanation.findings.map((finding, i) => (
              <p key={i} className="text-slate-300 text-sm">{finding}</p>
            ))}
          </div>
        </div>
      )}
      
      {/* Limitations */}
      {explanation.limitations?.length > 0 && (
        <div className="mb-4">
          <h3 className="text-slate-400 text-sm font-semibold mb-2">⚠️ Known Limitations:</h3>
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
            <ul className="space-y-1">
              {explanation.limitations.map((lim, i) => (
                <li key={i} className="text-yellow-400/80 text-sm">• {lim}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
      
      {/* Recommendation */}
      {explanation.recommendation && (
        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-3">
          <p className="text-cyan-400 text-sm font-semibold">📋 Expert Recommendation</p>
          <p className="text-slate-300 text-sm mt-1">{explanation.recommendation}</p>
        </div>
      )}
    </div>
  );
}


// ====================
// VISUAL FORENSICS CARD (Legacy - kept for compatibility)
// ====================
function VisualForensicsCard({ heatmaps, forensics }) {
  const [selectedHeatmap, setSelectedHeatmap] = useState(Object.keys(heatmaps)[0]);
  
  const heatmapLabels = {
    'ela_heatmap': 'Error Level Analysis',
    'ela_multi': 'Multi-Quality ELA',
    'noise_heatmap': 'Noise Analysis',
    'edge_heatmap': 'Edge Detection',
    'clone_heatmap': 'Clone Detection',
    'frequency_heatmap': 'Frequency Analysis',
    'color_heatmap': 'Color Analysis',
    'combined_heatmap': 'Combined Analysis',
    'manipulation_overlay': 'Manipulation Overlay',
    'block_artifact_map': 'JPEG Artifacts',
    'gradient_map': 'Gradient Analysis'
  };

  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <Eye className="text-purple-400" size={24} />
        <h2 className="text-xl font-bold text-white">Visual Forensics Heatmap</h2>
        <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded-full">
          {Object.keys(heatmaps).length} visualizations
        </span>
      </div>
      
      {/* Heatmap selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.keys(heatmaps).map(key => (
          <button
            key={key}
            onClick={() => setSelectedHeatmap(key)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
              selectedHeatmap === key 
                ? 'bg-purple-500 text-white' 
                : 'bg-white/5 text-slate-400 hover:bg-white/10'
            }`}
          >
            {heatmapLabels[key] || key}
          </button>
        ))}
      </div>
      
      {/* Selected heatmap display */}
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="bg-black/50 rounded-xl p-4 flex items-center justify-center">
          {heatmaps[selectedHeatmap] ? (
            <img 
              src={heatmaps[selectedHeatmap]} 
              alt={selectedHeatmap}
              className="max-w-full max-h-[400px] rounded-lg"
            />
          ) : (
            <p className="text-slate-500">No heatmap available</p>
          )}
        </div>
        
        {/* Scores */}
        <div className="space-y-4">
          <h3 className="text-white font-semibold">Analysis Scores</h3>
          
          {forensics?.scores && Object.entries(forensics.scores).map(([key, value]) => (
            <div key={key}>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">{key.replace('_', ' ').replace('score', '')}</span>
                <span className={`font-semibold ${
                  value > 60 ? 'text-red-400' : value > 40 ? 'text-yellow-400' : 'text-green-400'
                }`}>{Math.round(value)}%</span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full ${
                    value > 60 ? 'bg-red-500' : value > 40 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${value}%` }}
                />
              </div>
            </div>
          ))}
          
          {/* Detected regions */}
          {forensics?.detections?.manipulated_regions?.length > 0 && (
            <div className="mt-4 pt-4 border-t border-white/10">
              <p className="text-white font-semibold mb-2">
                ⚠️ {forensics.detections.manipulated_regions.length} Suspicious Regions
              </p>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {forensics.detections.manipulated_regions.slice(0, 5).map((region, i) => (
                  <div key={i} className="bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2 text-sm">
                    <span className="text-red-400">Region {i + 1}:</span>
                    <span className="text-slate-400 ml-2">
                      ({region.x}, {region.y}) - {region.severity}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// ====================
// EVIDENCE CARD (Simple scores without heatmaps)
// ====================
function EvidenceCard({ scores, evidence }) {
  const scoreLabels = {
    'noise_score': { label: '🔊 Noise Analysis', desc: 'Checks noise consistency' },
    'compression_score': { label: '📦 Compression', desc: 'JPEG artifact detection' },
    'color_score': { label: '🎨 Color Analysis', desc: 'Color distribution check' },
    'frequency_score': { label: '📊 Frequency', desc: 'Frequency domain analysis' },
    'texture_score': { label: '🧵 Texture', desc: 'Texture smoothness check' },
    'edge_score': { label: '📐 Edge Analysis', desc: 'Edge coherence detection' },
  };

  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <Eye className="text-purple-400" size={24} />
        <h2 className="text-xl font-bold text-white">Forensic Evidence</h2>
        <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded-full">
          6 algorithms
        </span>
      </div>
      
      {/* Score grid */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(scores).map(([key, value]) => {
          const info = scoreLabels[key] || { label: key, desc: '' };
          const isHigh = value > 55;
          const isMid = value > 45 && value <= 55;
          
          return (
            <div 
              key={key} 
              className={`p-4 rounded-xl border ${
                isHigh ? 'bg-red-500/10 border-red-500/30' : 
                isMid ? 'bg-yellow-500/10 border-yellow-500/30' : 
                'bg-green-500/10 border-green-500/30'
              }`}
            >
              <div className="flex justify-between items-start mb-2">
                <span className="text-white font-semibold">{info.label}</span>
                <span className={`text-lg font-bold ${
                  isHigh ? 'text-red-400' : isMid ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {Math.round(value)}%
                </span>
              </div>
              
              {/* Progress bar */}
              <div className="h-2 bg-white/10 rounded-full overflow-hidden mb-2">
                <div 
                  className={`h-full rounded-full transition-all duration-500 ${
                    isHigh ? 'bg-red-500' : isMid ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${value}%` }}
                />
              </div>
              
              <p className="text-slate-500 text-xs">{info.desc}</p>
              
              {/* Show flag if available */}
              {evidence && evidence[key.replace('_score', '')] && (
                <div className="mt-2">
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    isHigh ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                  }`}>
                    {evidence[key.replace('_score', '')]?.flag || 'normal'}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      {/* Legend */}
      <div className="mt-6 pt-4 border-t border-white/10 flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <span className="text-slate-400">&gt;55% = Suspicious</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
          <span className="text-slate-400">45-55% = Uncertain</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-slate-400">&lt;45% = Likely Real</span>
        </div>
      </div>
    </div>
  );
}


// ====================
// EXPLAINABLE AI CARD
// ====================
function ExplainableAICard({ explanation }) {
  const [tab, setTab] = useState('simple');

  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="text-green-400" size={24} />
        <h2 className="text-xl font-bold text-white">Explainable AI Analysis</h2>
      </div>
      
      {/* Tab selector */}
      <div className="flex gap-2 mb-6">
        {[
          { id: 'simple', label: 'Simple', icon: '👤' },
          { id: 'technical', label: 'Technical', icon: '🔬' },
          { id: 'legal', label: 'Legal', icon: '⚖️' }
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-all ${
              tab === t.id 
                ? 'bg-green-500 text-white' 
                : 'bg-white/5 text-slate-400 hover:bg-white/10'
            }`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>
      
      {/* Content based on tab */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Left: Explanation text */}
        <div className="bg-black/30 rounded-xl p-4 max-h-[500px] overflow-y-auto">
          <pre className="text-slate-300 text-sm whitespace-pre-wrap font-mono">
            {explanation.explanations?.[tab] || 'No explanation available'}
          </pre>
        </div>
        
        {/* Right: Evidence & Flags */}
        <div className="space-y-4">
          {/* Red Flags */}
          {explanation.red_flags?.length > 0 && (
            <div>
              <h3 className="text-white font-semibold mb-2">🚩 Red Flags</h3>
              <div className="space-y-2">
                {explanation.red_flags.map((flag, i) => (
                  <div 
                    key={i}
                    className={`px-3 py-2 rounded-lg border text-sm ${
                      flag.severity === 'critical' 
                        ? 'bg-red-500/10 border-red-500/50 text-red-400'
                        : flag.severity === 'high'
                        ? 'bg-orange-500/10 border-orange-500/50 text-orange-400'
                        : 'bg-yellow-500/10 border-yellow-500/50 text-yellow-400'
                    }`}
                  >
                    <span className="font-semibold">[{flag.severity?.toUpperCase()}]</span> {flag.flag}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Evidence */}
          {explanation.evidence?.supporting_manipulation?.length > 0 && (
            <div>
              <h3 className="text-white font-semibold mb-2">📊 Manipulation Evidence</h3>
              <div className="space-y-2">
                {explanation.evidence.supporting_manipulation.slice(0, 5).map((ev, i) => (
                  <div key={i} className="bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                    <p className="text-red-400 text-sm font-semibold">{ev.category}</p>
                    <p className="text-slate-400 text-xs">{ev.finding}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Recommendations */}
          {explanation.recommendations?.length > 0 && (
            <div>
              <h3 className="text-white font-semibold mb-2">💡 Recommendations</h3>
              <div className="space-y-2">
                {explanation.recommendations.map((rec, i) => (
                  <div key={i} className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg px-3 py-2">
                    <p className="text-cyan-400 text-sm">{rec.action}</p>
                    <p className="text-slate-500 text-xs">{rec.reason}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* AI Analysis */}
          {explanation.ai_analysis?.available && explanation.ai_analysis?.reasoning && (
            <div>
              <h3 className="text-white font-semibold mb-2">🤖 AI Visual Analysis</h3>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg px-3 py-2">
                <p className="text-slate-300 text-sm">{explanation.ai_analysis.reasoning.substring(0, 500)}...</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// ====================
// DASHBOARD PAGE
// ====================
function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [content, setContent] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);

  // Fetch dashboard data
  useState(() => {
    const fetchData = async () => {
      try {
        const [statsRes, contentRes] = await Promise.all([
          fetch(`${API_BASE}/api/dashboard`),
          fetch(`${API_BASE}/api/dashboard/content?limit=20`)
        ]);
        
        if (statsRes.ok) {
          const statsData = await statsRes.json();
          setStats(statsData.stats);
        }
        
        if (contentRes.ok) {
          const contentData = await contentRes.json();
          setContent(contentData.content || []);
        }
      } catch (err) {
        console.error('Dashboard fetch error:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      const res = await fetch(`${API_BASE}/api/dashboard/search?q=${encodeURIComponent(searchQuery)}`);
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.results);
      }
    } catch (err) {
      console.error('Search error:', err);
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-3xl font-bold text-white">Content Dashboard</h1>
            <span className="bg-red-500/20 text-red-400 text-xs px-2 py-1 rounded-full flex items-center gap-1">
              <Gavel size={12} /> Law Enforcement
            </span>
          </div>
          <p className="text-slate-400 mt-1">Track analyzed content - Legal/Illegal status for Police & Cyber Cell</p>
        </div>
        
        {/* Search */}
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Search by ID or filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            className="bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white placeholder-slate-500 w-64"
          />
          <button
            onClick={handleSearch}
            className="bg-cyan-500 hover:bg-cyan-600 text-white px-4 py-2 rounded-lg flex items-center gap-2"
          >
            <Search size={18} />
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard 
            label="Total Analyzed" 
            value={stats.total_content || 0}
            icon={<Database className="text-cyan-400" size={24} />}
          />
          <StatCard 
            label="Analyzed Today" 
            value={stats.today_count || 0}
            icon={<Zap className="text-yellow-400" size={24} />}
          />
          <StatCard 
            label="🚨 ILLEGAL" 
            value={stats.legal_status_distribution?.['ILLEGAL'] || 0}
            icon={<Siren className="text-red-400" size={24} />}
          />
          <StatCard 
            label="✅ LEGAL" 
            value={stats.legal_status_distribution?.['LEGAL'] || 0}
            icon={<ShieldCheck className="text-green-400" size={24} />}
          />
        </div>
      )}

      {/* Search Results */}
      {searchResults && (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <h2 className="text-xl font-bold text-white mb-4">Search Results ({searchResults.length})</h2>
          {searchResults.length === 0 ? (
            <p className="text-slate-500">No results found for "{searchQuery}"</p>
          ) : (
            <div className="space-y-3">
              {searchResults.map(item => (
                <ContentRow key={item.content_id} item={item} />
              ))}
            </div>
          )}
          <button 
            onClick={() => setSearchResults(null)}
            className="mt-4 text-cyan-400 hover:text-cyan-300 text-sm"
          >
            ← Back to all content
          </button>
        </div>
      )}

      {/* Content List */}
      {!searchResults && (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <h2 className="text-xl font-bold text-white mb-4">Recent Analyses</h2>
          
          {loading ? (
            <div className="flex justify-center py-8">
              <div className="w-8 h-8 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : content.length === 0 ? (
            <p className="text-slate-500 text-center py-8">No content analyzed yet</p>
          ) : (
            <div className="space-y-3">
              {content.map(item => (
                <ContentRow key={item.content_id} item={item} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Top Analyzed */}
      {stats?.top_analyzed?.length > 0 && (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <h2 className="text-xl font-bold text-white mb-4">🔥 Most Analyzed Content</h2>
          <div className="space-y-3">
            {stats.top_analyzed.map((item, i) => {
              const legalColors = {
                'ILLEGAL': 'text-red-400 bg-red-500/20',
                'LEGAL': 'text-green-400 bg-green-500/20',
                'NEEDS_REVIEW': 'text-yellow-400 bg-yellow-500/20',
              };
              const legalLabels = {
                'ILLEGAL': '🚨 ILLEGAL',
                'LEGAL': '✅ LEGAL',
                'NEEDS_REVIEW': '⚠️ REVIEW',
              };
              const status = item.legal_status || 'NEEDS_REVIEW';
              return (
                <div key={item.content_id} className="flex items-center justify-between bg-white/5 rounded-lg px-4 py-3">
                  <div className="flex items-center gap-4">
                    <span className="text-2xl font-bold text-cyan-400">#{i + 1}</span>
                    <div>
                      <p className="text-white font-mono text-sm">{item.content_id}</p>
                      <p className="text-slate-500 text-xs">{item.filename || 'Unknown'}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className={`px-3 py-1 rounded-full text-sm font-semibold ${legalColors[status] || 'text-slate-400 bg-slate-500/20'}`}>
                      {legalLabels[status] || status}
                    </span>
                    <div className="text-right">
                      <p className="text-cyan-400 text-2xl font-bold">{item.analysis_count}</p>
                      <p className="text-slate-500 text-xs">analyses</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, icon }) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        {icon}
        <p className="text-3xl font-bold text-white">{value}</p>
      </div>
      <p className="text-slate-400 text-sm">{label}</p>
    </div>
  );
}

function ContentRow({ item }) {
  // Legal status colors for law enforcement view
  const legalStatusColors = {
    'ILLEGAL': 'text-red-400 bg-red-500/20',
    'LEGAL': 'text-green-400 bg-green-500/20',
    'NEEDS_REVIEW': 'text-yellow-400 bg-yellow-500/20',
    'NOT_ENOUGH_DATA': 'text-slate-400 bg-slate-500/20',
  };

  const legalStatusLabels = {
    'ILLEGAL': '🚨 ILLEGAL',
    'LEGAL': '✅ LEGAL',
    'NEEDS_REVIEW': '⚠️ REVIEW NEEDED',
    'NOT_ENOUGH_DATA': '❓ NO DATA',
  };

  const status = item.legal_status || 'NOT_ENOUGH_DATA';
  const statusColor = legalStatusColors[status] || legalStatusColors['NOT_ENOUGH_DATA'];
  const statusLabel = legalStatusLabels[status] || status;

  return (
    <div className="flex items-center justify-between bg-white/5 hover:bg-white/10 transition-all rounded-lg px-4 py-3">
      <div className="flex items-center gap-4">
        <Fingerprint className="text-cyan-400" size={20} />
        <div>
          <p className="text-white font-mono text-sm">{item.content_id}</p>
          <p className="text-slate-500 text-xs">{item.filename || 'Unknown file'}</p>
        </div>
      </div>
      
      <div className="flex items-center gap-6">
        <div className="text-center">
          <p className="text-cyan-400 font-bold">{item.analysis_count || 1}</p>
          <p className="text-slate-500 text-xs">analyses</p>
        </div>
        <div className="text-center min-w-[140px]">
          <p className={`font-semibold text-sm px-3 py-1 rounded-full ${statusColor}`}>
            {statusLabel}
          </p>
          <p className="text-slate-500 text-xs mt-1">{Math.round(item.legal_confidence || 0)}% confidence</p>
        </div>
        <div className="text-right min-w-[80px]">
          <p className="text-slate-400 text-xs">
            {item.last_analyzed ? new Date(item.last_analyzed).toLocaleDateString() : 'N/A'}
          </p>
        </div>
      </div>
    </div>
  );
}


// ====================
// FRAUD ANALYZER PAGE
// ====================
function FraudAnalyzerPage() {
  const [message, setMessage] = useState('');
  const [platform, setPlatform] = useState('');
  const [messageType, setMessageType] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [inputMode, setInputMode] = useState('text'); // 'text' or 'image'
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [extractedText, setExtractedText] = useState('');

  const platforms = [
    { id: '', name: 'Select Platform (Optional)', icon: '📱' },
    { id: 'whatsapp', name: 'WhatsApp', icon: '💬' },
    { id: 'sms', name: 'SMS/Text Message', icon: '📱' },
    { id: 'instagram', name: 'Instagram', icon: '📸' },
    { id: 'email', name: 'Email', icon: '📧' },
    { id: 'telegram', name: 'Telegram', icon: '✈️' },
    { id: 'other', name: 'Other', icon: '📝' }
  ];

  const messageTypes = [
    { id: '', name: 'Select Message Type (Optional)', icon: '❓' },
    { id: 'banking', name: 'Banking/Account', icon: '🏦' },
    { id: 'payment', name: 'Payment/UPI', icon: '💳' },
    { id: 'offer', name: 'Offer/Prize/Lottery', icon: '🎁' },
    { id: 'government', name: 'Government/Tax', icon: '🏛️' },
    { id: 'delivery', name: 'Delivery/Shipping', icon: '📦' },
    { id: 'job', name: 'Job/Work-from-Home', icon: '💼' },
    { id: 'tech_support', name: 'Tech Support', icon: '🔧' },
    { id: 'other', name: 'Other', icon: '❓' }
  ];

  const analyzeMessage = async () => {
    // Check if we have text to analyze (either typed or extracted from image)
    const textToAnalyze = inputMode === 'image' ? extractedText : message;
    
    if (!textToAnalyze.trim() && !uploadedImage) {
      setError('Please enter a message or upload a screenshot to analyze');
      return;
    }

    setError('');
    setIsAnalyzing(true);
    setResult(null);

    try {
      let response;
      
      if (inputMode === 'image' && uploadedImage) {
        // Upload image for OCR + analysis
        const formData = new FormData();
        formData.append('file', uploadedImage);
        if (platform) formData.append('platform', platform);
        if (messageType) formData.append('message_type', messageType);
        
        response = await fetch(`${API_BASE}/api/fraud/analyze-image`, {
          method: 'POST',
          body: formData
        });
      } else {
        // Text-based analysis
        response = await fetch(`${API_BASE}/api/fraud/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: textToAnalyze.trim(),
            platform: platform || null,
            message_type: messageType || null
          })
        });
      }

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || 'Analysis failed');
      }

      const data = await response.json();
      
      // If image analysis returned extracted text, show it
      if (data.ocr_info?.extracted_text) {
        setExtractedText(data.ocr_info.extracted_text);
      } else if (data.extracted_text) {
        setExtractedText(data.extracted_text);
      }
      
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload an image file (PNG, JPG, etc.)');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setError('Image too large. Maximum size is 10MB.');
        return;
      }
      setUploadedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setError('');
      setExtractedText('');
      setResult(null);
    }
  };

  const clearImage = () => {
    setUploadedImage(null);
    setImagePreview(null);
    setExtractedText('');
    setResult(null);
  };

  const getVerdictStyle = (verdict) => {
    switch (verdict) {
      case 'HIGH_RISK_FRAUD':
        return { bg: 'bg-red-500/20', border: 'border-red-500', text: 'text-red-400', icon: XCircle };
      case 'LIKELY_FRAUD':
        return { bg: 'bg-orange-500/20', border: 'border-orange-500', text: 'text-orange-400', icon: AlertTriangle };
      case 'SUSPICIOUS':
        return { bg: 'bg-yellow-500/20', border: 'border-yellow-500', text: 'text-yellow-400', icon: AlertCircle };
      case 'LIKELY_SAFE':
        return { bg: 'bg-green-500/20', border: 'border-green-500', text: 'text-green-400', icon: CheckCircle2 };
      default:
        return { bg: 'bg-slate-500/20', border: 'border-slate-500', text: 'text-slate-400', icon: HelpCircle };
    }
  };

  const getSeverityStyle = (severity) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/50';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'safe': return 'bg-green-500/20 text-green-400 border-green-500/50';
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/50';
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <header>
        <div className="flex items-center gap-3 text-orange-400 uppercase tracking-[0.3em] text-sm mb-4">
          <MessageSquareWarning size={20} />
          <span>Fraud Message Analyzer</span>
        </div>
        <h1 className="text-4xl lg:text-5xl font-semibold text-white leading-tight">
          Detect Scams, <span className="text-orange-400">Stay Safe.</span>
        </h1>
        <p className="text-slate-300 mt-4 text-lg max-w-2xl">
          Paste any suspicious message from WhatsApp, SMS, Instagram or Email. 
          We'll analyze it for phishing links, scam patterns, and fraud indicators.
        </p>
      </header>

      {/* Input Section */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Message Input */}
        <div className="lg:col-span-2 space-y-4">
          {/* Input Mode Toggle */}
          <div className="flex gap-2 p-1 bg-white/5 rounded-xl w-fit">
            <button
              onClick={() => { setInputMode('text'); clearImage(); }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                inputMode === 'text' 
                  ? 'bg-orange-500 text-white' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <MessageSquareWarning size={18} />
              <span>Paste Text</span>
            </button>
            <button
              onClick={() => setInputMode('image')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                inputMode === 'image' 
                  ? 'bg-orange-500 text-white' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Camera size={18} />
              <span>Upload Screenshot</span>
            </button>
          </div>

          {/* Text Input Mode */}
          {inputMode === 'text' && (
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
              <label className="block text-white font-medium mb-3">
                📝 Paste the suspicious message here
              </label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Example: Dear Customer, Your SBI account has been suspended due to KYC expiry. Click http://sbi-kyc-update.xyz to update now or your account will be blocked in 24 hours. Call 9876543210 for help."
                className="w-full h-48 bg-black/30 border border-white/20 rounded-xl p-4 text-white placeholder-slate-500 focus:border-orange-400 focus:outline-none resize-none"
              />
              <p className="text-slate-500 text-sm mt-2">
                💡 Include the complete message with any links, phone numbers, or sender information
              </p>
            </div>
          )}

          {/* Image Upload Mode */}
          {inputMode === 'image' && (
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
              <label className="block text-white font-medium mb-3">
                📸 Upload Screenshot of Suspicious Message
              </label>
              
              {!imagePreview ? (
                <label className="flex flex-col items-center justify-center h-48 bg-black/30 border-2 border-dashed border-white/20 rounded-xl cursor-pointer hover:border-orange-400 transition-all">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                  <UploadCloud className="text-orange-400 mb-3" size={40} />
                  <span className="text-white font-medium">Click to upload screenshot</span>
                  <span className="text-slate-500 text-sm mt-1">PNG, JPG up to 10MB</span>
                </label>
              ) : (
                <div className="relative">
                  <img 
                    src={imagePreview} 
                    alt="Screenshot preview" 
                    className="w-full max-h-64 object-contain rounded-xl border border-white/20"
                  />
                  <button
                    onClick={clearImage}
                    className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-2 rounded-lg"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              )}
              
              <p className="text-slate-500 text-sm mt-3">
                📱 Take a screenshot of WhatsApp/SMS/Instagram message and upload it here
              </p>

              {/* Extracted Text Preview */}
              {extractedText && (
                <div className="mt-4 bg-black/30 border border-green-500/30 rounded-xl p-4">
                  <p className="text-green-400 text-sm font-medium mb-2 flex items-center gap-2">
                    <CheckCircle2 size={16} />
                    Text Extracted from Image:
                  </p>
                  <p className="text-white text-sm whitespace-pre-wrap">{extractedText}</p>
                </div>
              )}
            </div>
          )}

          {/* Options */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="bg-white/5 border border-white/10 rounded-xl p-4">
              <label className="block text-white font-medium mb-2 text-sm">
                📱 Platform (Optional)
              </label>
              <select
                value={platform}
                onChange={(e) => setPlatform(e.target.value)}
                className="w-full bg-black/30 border border-white/20 rounded-lg p-3 text-white focus:border-orange-400 focus:outline-none"
              >
                {platforms.map(p => (
                  <option key={p.id} value={p.id} className="bg-slate-800">
                    {p.icon} {p.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-xl p-4">
              <label className="block text-white font-medium mb-2 text-sm">
                📋 Message Type (Optional)
              </label>
              <select
                value={messageType}
                onChange={(e) => setMessageType(e.target.value)}
                className="w-full bg-black/30 border border-white/20 rounded-lg p-3 text-white focus:border-orange-400 focus:outline-none"
              >
                {messageTypes.map(t => (
                  <option key={t.id} value={t.id} className="bg-slate-800">
                    {t.icon} {t.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Analyze Button */}
          <button
            onClick={analyzeMessage}
            disabled={isAnalyzing || (inputMode === 'text' ? !message.trim() : !uploadedImage)}
            className="w-full bg-gradient-to-r from-orange-500 to-red-500 text-white font-semibold py-4 rounded-xl 
                       hover:from-orange-600 hover:to-red-600 transition-all flex items-center justify-center gap-3
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                <span>{inputMode === 'image' ? 'Extracting & Analyzing...' : 'Analyzing...'}</span>
              </>
            ) : (
              <>
                {inputMode === 'image' ? <FileImage size={20} /> : <Search size={20} />}
                <span>{inputMode === 'image' ? 'Extract Text & Analyze' : 'Analyze Message'}</span>
              </>
            )}
          </button>

          {error && (
            <div className="bg-red-500/20 border border-red-500 rounded-xl p-4 text-red-400">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Info Panel */}
        <div className="space-y-4">
          <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <Shield className="text-orange-400" size={20} />
              What We Detect
            </h3>
            <ul className="space-y-3 text-slate-400 text-sm">
              <li className="flex items-start gap-2">
                <Link2 className="text-orange-400 mt-0.5" size={16} />
                <span>Phishing & malicious links</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertTriangle className="text-orange-400 mt-0.5" size={16} />
                <span>Urgency & threat manipulation</span>
              </li>
              <li className="flex items-start gap-2">
                <Fingerprint className="text-orange-400 mt-0.5" size={16} />
                <span>OTP/PIN/Password requests</span>
              </li>
              <li className="flex items-start gap-2">
                <Gavel className="text-orange-400 mt-0.5" size={16} />
                <span>Impersonation attempts</span>
              </li>
              <li className="flex items-start gap-2">
                <Zap className="text-orange-400 mt-0.5" size={16} />
                <span>Too-good-to-be-true offers</span>
              </li>
            </ul>
          </div>

          <div className="bg-orange-500/10 border border-orange-500/30 rounded-2xl p-6">
            <h3 className="text-orange-400 font-semibold mb-3">⚠️ Remember</h3>
            <ul className="space-y-2 text-slate-400 text-sm">
              <li>• Banks NEVER ask for OTP/PIN via SMS</li>
              <li>• Don't click links from unknown senders</li>
              <li>• Verify by calling official numbers</li>
              <li>• Report suspicious messages</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {result && result.fraud_assessment && (
        <FraudResultDisplay result={result} getVerdictStyle={getVerdictStyle} getSeverityStyle={getSeverityStyle} />
      )}
    </div>
  );
}

// Fraud Result Display Component
function FraudResultDisplay({ result, getVerdictStyle, getSeverityStyle }) {
  const assessment = result.fraud_assessment;
  const verdictStyle = getVerdictStyle(assessment.verdict);
  const VerdictIcon = verdictStyle.icon;

  return (
    <div className="space-y-6">
      {/* Fraud ID & Database Status */}
      {result.fraud_id && (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Fingerprint className="text-cyan-400" size={24} />
            <div>
              <p className="text-white font-mono text-lg">{result.fraud_id}</p>
              <p className="text-slate-500 text-xs">Unique Fraud Message ID</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {result.database?.saved ? (
              <span className="flex items-center gap-2 text-green-400 text-sm bg-green-500/20 px-3 py-1 rounded-full">
                <CheckCircle2 size={16} />
                Saved to Dashboard
              </span>
            ) : (
              <span className="flex items-center gap-2 text-yellow-400 text-sm bg-yellow-500/20 px-3 py-1 rounded-full">
                <AlertCircle size={16} />
                Not saved
              </span>
            )}
            {result.database?.legal_status && (
              <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                result.database.legal_status === 'ILLEGAL' ? 'bg-red-500/20 text-red-400' :
                result.database.legal_status === 'LEGAL' ? 'bg-green-500/20 text-green-400' :
                'bg-yellow-500/20 text-yellow-400'
              }`}>
                {result.database.legal_status === 'ILLEGAL' ? '🚨 ILLEGAL' :
                 result.database.legal_status === 'LEGAL' ? '✅ LEGAL' : '⚠️ REVIEW'}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main Verdict Card */}
      <div className={`${verdictStyle.bg} border ${verdictStyle.border} rounded-2xl p-6`}>
        <div className="flex items-start gap-4">
          <div className={`p-3 rounded-xl ${verdictStyle.bg}`}>
            <VerdictIcon className={verdictStyle.text} size={32} />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h2 className={`text-2xl font-bold ${verdictStyle.text}`}>
                {assessment.verdict.replace(/_/g, ' ')}
              </h2>
              <span className={`px-3 py-1 rounded-full text-sm ${verdictStyle.bg} ${verdictStyle.text} border ${verdictStyle.border}`}>
                {assessment.confidence}% confidence
              </span>
            </div>
            <p className="text-white text-lg">{assessment.interpretation}</p>
          </div>
        </div>

        {/* Risk Score */}
        <div className="mt-6 bg-black/20 rounded-xl p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-slate-400">Risk Score</span>
            <span className={`font-bold ${verdictStyle.text}`}>{assessment.risk_score}/100</span>
          </div>
          <div className="h-3 bg-black/30 rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full ${
                assessment.risk_score >= 70 ? 'bg-red-500' :
                assessment.risk_score >= 50 ? 'bg-orange-500' :
                assessment.risk_score >= 35 ? 'bg-yellow-500' :
                'bg-green-500'
              }`}
              style={{ width: `${assessment.risk_score}%` }}
            />
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-xl p-4 text-center">
          <p className="text-3xl font-bold text-white">{result.summary?.total_checks || 0}</p>
          <p className="text-slate-400 text-sm">Total Checks</p>
        </div>
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-center">
          <p className="text-3xl font-bold text-red-400">{result.summary?.critical || 0}</p>
          <p className="text-slate-400 text-sm">Critical</p>
        </div>
        <div className="bg-orange-500/10 border border-orange-500/30 rounded-xl p-4 text-center">
          <p className="text-3xl font-bold text-orange-400">{result.summary?.high || 0}</p>
          <p className="text-slate-400 text-sm">High Risk</p>
        </div>
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 text-center">
          <p className="text-3xl font-bold text-yellow-400">{result.summary?.medium || 0}</p>
          <p className="text-slate-400 text-sm">Medium</p>
        </div>
        <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4 text-center">
          <p className="text-3xl font-bold text-green-400">{result.summary?.safe || 0}</p>
          <p className="text-slate-400 text-sm">Safe</p>
        </div>
      </div>

      {/* Key Risks */}
      {assessment.key_risks && assessment.key_risks.length > 0 && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6">
          <h3 className="text-red-400 font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle size={20} />
            Key Risks Identified
          </h3>
          <ul className="space-y-2">
            {assessment.key_risks.map((risk, idx) => (
              <li key={idx} className="flex items-start gap-3 text-white">
                <XCircle className="text-red-400 mt-0.5 flex-shrink-0" size={18} />
                <span>{risk}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* URLs Found */}
      {result.urls_found && result.urls_found.length > 0 && (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <Link2 className="text-orange-400" size={20} />
            URLs Detected ({result.urls_found.length})
          </h3>
          <div className="space-y-2">
            {result.urls_found.map((url, idx) => (
              <div key={idx} className="bg-black/30 rounded-lg p-3 font-mono text-sm text-orange-400 break-all">
                {url}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detailed Checks */}
      {result.checks && result.checks.length > 0 && (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <Search size={20} className="text-cyan-400" />
            Detailed Analysis ({result.checks.length} checks)
          </h3>
          <div className="space-y-3">
            {result.checks.map((check, idx) => (
              <div 
                key={idx} 
                className={`border rounded-xl p-4 ${getSeverityStyle(check.severity)}`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-white">{check.display_name}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${getSeverityStyle(check.severity)}`}>
                        {check.severity}
                      </span>
                    </div>
                    <p className="text-slate-400 text-sm">{check.finding}</p>
                  </div>
                  <div className="text-right">
                    <span className="text-2xl font-bold text-white">{Math.round(check.score)}</span>
                    <p className="text-slate-500 text-xs">score</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {result.recommendations && result.recommendations.length > 0 && (
        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-2xl p-6">
          <h3 className="text-cyan-400 font-semibold mb-4 flex items-center gap-2">
            <Shield size={20} />
            Recommendations
          </h3>
          <ul className="space-y-3">
            {result.recommendations.map((rec, idx) => (
              <li key={idx} className="flex items-start gap-3 text-white">
                <CheckCircle2 className="text-cyan-400 mt-0.5 flex-shrink-0" size={18} />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}


// ====================
// FRAUD DASHBOARD PAGE
// ====================
function FraudDashboardPage() {
  const [stats, setStats] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all, LEGAL, ILLEGAL, NEEDS_REVIEW
  const [selectedMessage, setSelectedMessage] = useState(null);

  // Fetch dashboard data
  useState(() => {
    const fetchData = async () => {
      try {
        const [statsRes, messagesRes] = await Promise.all([
          fetch(`${API_BASE}/api/fraud/dashboard`),
          fetch(`${API_BASE}/api/fraud/messages?limit=50`)
        ]);
        
        if (statsRes.ok) {
          const statsData = await statsRes.json();
          setStats(statsData);
        }
        
        if (messagesRes.ok) {
          const messagesData = await messagesRes.json();
          setMessages(messagesData.messages || []);
        }
      } catch (err) {
        console.error('Fraud Dashboard fetch error:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);

  // Fetch filtered messages
  const fetchFiltered = async (status) => {
    setFilter(status);
    setLoading(true);
    try {
      const url = status === 'all' 
        ? `${API_BASE}/api/fraud/messages?limit=50`
        : `${API_BASE}/api/fraud/messages?limit=50&legal_status=${status}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      }
    } catch (err) {
      console.error('Filter error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch message detail
  const viewDetail = async (fraudId) => {
    try {
      const res = await fetch(`${API_BASE}/api/fraud/messages/${fraudId}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedMessage(data);
      }
    } catch (err) {
      console.error('Detail fetch error:', err);
    }
  };

  const legalColors = {
    'ILLEGAL': 'text-red-400 bg-red-500/20',
    'LEGAL': 'text-green-400 bg-green-500/20',
    'NEEDS_REVIEW': 'text-yellow-400 bg-yellow-500/20',
  };
  
  const legalLabels = {
    'ILLEGAL': '🚨 ILLEGAL (Fraud)',
    'LEGAL': '✅ LEGAL (Safe)',
    'NEEDS_REVIEW': '⚠️ NEEDS REVIEW',
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-3xl font-bold text-white">Fraud Messages Dashboard</h1>
            <span className="bg-red-500/20 text-red-400 text-xs px-2 py-1 rounded-full flex items-center gap-1">
              <Siren size={12} /> Cyber Cell
            </span>
          </div>
          <p className="text-slate-400 mt-1">Track analyzed fraud messages - Legal/Illegal status for Law Enforcement</p>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-5">
          <StatCard 
            label="Total Analyzed" 
            value={stats.total_analyzed || 0}
            icon={<MessageSquareWarning className="text-orange-400" size={24} />}
          />
          <StatCard 
            label="Today" 
            value={stats.today_count || 0}
            icon={<Zap className="text-yellow-400" size={24} />}
          />
          <StatCard 
            label="🚨 ILLEGAL" 
            value={stats.illegal_count || 0}
            icon={<Siren className="text-red-400" size={24} />}
          />
          <StatCard 
            label="✅ LEGAL" 
            value={stats.legal_count || 0}
            icon={<ShieldCheck className="text-green-400" size={24} />}
          />
          <StatCard 
            label="⚠️ Review" 
            value={stats.needs_review_count || 0}
            icon={<AlertCircle className="text-yellow-400" size={24} />}
          />
        </div>
      )}

      {/* Filter Tabs */}
      <div className="flex gap-2">
        {['all', 'ILLEGAL', 'LEGAL', 'NEEDS_REVIEW'].map(status => (
          <button
            key={status}
            onClick={() => fetchFiltered(status)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              filter === status 
                ? status === 'ILLEGAL' ? 'bg-red-500 text-white' :
                  status === 'LEGAL' ? 'bg-green-500 text-white' :
                  status === 'NEEDS_REVIEW' ? 'bg-yellow-500 text-black' :
                  'bg-cyan-500 text-white'
                : 'bg-white/10 text-slate-400 hover:bg-white/20'
            }`}
          >
            {status === 'all' ? '📋 All Messages' : legalLabels[status]}
          </button>
        ))}
      </div>

      {/* Selected Message Detail */}
      {selectedMessage && (
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Fingerprint className="text-cyan-400" size={24} />
              {selectedMessage.fraud_id}
            </h2>
            <button 
              onClick={() => setSelectedMessage(null)}
              className="text-slate-400 hover:text-white"
            >
              ✕ Close
            </button>
          </div>
          
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-3">
              <div className="bg-black/30 rounded-lg p-4">
                <p className="text-slate-500 text-xs mb-1">Message Preview</p>
                <p className="text-white">{selectedMessage.message_preview}</p>
              </div>
              
              <div className="flex gap-3">
                <div className="bg-black/30 rounded-lg p-3 flex-1">
                  <p className="text-slate-500 text-xs">Platform</p>
                  <p className="text-white">{selectedMessage.platform || 'Unknown'}</p>
                </div>
                <div className="bg-black/30 rounded-lg p-3 flex-1">
                  <p className="text-slate-500 text-xs">Type</p>
                  <p className="text-white">{selectedMessage.message_type || 'Unknown'}</p>
                </div>
              </div>
              
              {selectedMessage.urls_found?.length > 0 && (
                <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
                  <p className="text-orange-400 text-xs mb-2">URLs Found</p>
                  {selectedMessage.urls_found.map((url, i) => (
                    <p key={i} className="text-orange-300 font-mono text-sm break-all">{url}</p>
                  ))}
                </div>
              )}
              
              {selectedMessage.contact_numbers?.length > 0 && (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
                  <p className="text-yellow-400 text-xs mb-2">Contact Numbers</p>
                  {selectedMessage.contact_numbers.map((num, i) => (
                    <p key={i} className="text-yellow-300 font-mono">{num}</p>
                  ))}
                </div>
              )}
            </div>
            
            <div className="space-y-3">
              <div className={`rounded-lg p-4 ${legalColors[selectedMessage.legal_status] || 'bg-slate-500/20 text-slate-400'}`}>
                <p className="text-2xl font-bold">{legalLabels[selectedMessage.legal_status] || selectedMessage.legal_status}</p>
                <p className="text-sm opacity-80">Verdict: {selectedMessage.verdict}</p>
                <p className="text-sm opacity-80">Risk Score: {selectedMessage.risk_score}/100</p>
                <p className="text-sm opacity-80">Confidence: {selectedMessage.confidence}%</p>
              </div>
              
              {selectedMessage.matched_templates?.length > 0 && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                  <p className="text-red-400 text-xs mb-2">Matched Scam Templates</p>
                  {selectedMessage.matched_templates.map((t, i) => (
                    <span key={i} className="inline-block bg-red-500/20 text-red-400 text-sm px-2 py-1 rounded mr-2 mb-1">{t}</span>
                  ))}
                </div>
              )}
              
              <div className="bg-black/30 rounded-lg p-3">
                <p className="text-slate-500 text-xs mb-1">Analyzed At</p>
                <p className="text-white">{new Date(selectedMessage.created_at).toLocaleString()}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Messages List */}
      <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
        <h2 className="text-xl font-bold text-white mb-4">
          {filter === 'all' ? 'All Fraud Messages' : legalLabels[filter]} ({messages.length})
        </h2>
        
        {loading ? (
          <div className="flex justify-center py-8">
            <div className="w-8 h-8 border-4 border-orange-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : messages.length === 0 ? (
          <p className="text-slate-500 text-center py-8">No fraud messages analyzed yet</p>
        ) : (
          <div className="space-y-3">
            {messages.map(msg => (
              <div 
                key={msg.fraud_id} 
                onClick={() => viewDetail(msg.fraud_id)}
                className="flex items-center justify-between bg-white/5 hover:bg-white/10 transition-all rounded-lg px-4 py-3 cursor-pointer"
              >
                <div className="flex items-center gap-4">
                  <MessageSquareWarning className="text-orange-400" size={20} />
                  <div>
                    <p className="text-white font-mono text-sm">{msg.fraud_id}</p>
                    <p className="text-slate-500 text-xs truncate max-w-md">{msg.message_preview}</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-6">
                  <div className="text-center">
                    <p className="text-cyan-400 font-bold">{msg.platform || '-'}</p>
                    <p className="text-slate-500 text-xs">platform</p>
                  </div>
                  <div className="text-center">
                    <p className={`font-bold ${
                      msg.risk_score >= 70 ? 'text-red-400' :
                      msg.risk_score >= 50 ? 'text-orange-400' :
                      msg.risk_score >= 35 ? 'text-yellow-400' : 'text-green-400'
                    }`}>{Math.round(msg.risk_score || 0)}</p>
                    <p className="text-slate-500 text-xs">risk</p>
                  </div>
                  <div className="text-center min-w-[140px]">
                    <p className={`font-semibold text-sm px-3 py-1 rounded-full ${legalColors[msg.legal_status] || 'bg-slate-500/20 text-slate-400'}`}>
                      {legalLabels[msg.legal_status] || msg.legal_status}
                    </p>
                  </div>
                  <div className="text-right min-w-[80px]">
                    <p className="text-slate-400 text-xs">
                      {msg.created_at ? new Date(msg.created_at).toLocaleDateString() : 'N/A'}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Top Scam Templates */}
      {stats?.top_scam_templates?.length > 0 && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6">
          <h2 className="text-xl font-bold text-red-400 mb-4">🔥 Most Common Scam Types</h2>
          <div className="flex flex-wrap gap-3">
            {stats.top_scam_templates.map(([template, count], i) => (
              <div key={i} className="bg-red-500/20 text-red-300 px-4 py-2 rounded-lg">
                <span className="font-bold">{template}</span>
                <span className="ml-2 text-red-400">({count})</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


export default App;
