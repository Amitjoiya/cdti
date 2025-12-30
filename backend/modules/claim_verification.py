"""
Claim Verification System v1.0
==============================
Cross-verifies user-provided claims against forensic evidence.

PHILOSOPHY:
- User claims are "soft claims", NOT ground truth
- Every claim must be independently verified
- Forensic evidence ALWAYS takes precedence
- Contradictions are reported, not accusations

Like a forensic analyst who listens to a witness
but validates everything independently.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


# =====================================================
# ENUMS AND DATA STRUCTURES
# =====================================================

class ClaimType(Enum):
    """Types of claims a user can make"""
    SOURCE = "source"
    AI_GENERATED = "ai_generated"
    AI_ENHANCED = "enhanced"
    WORKFLOW = "workflow"
    CUSTOM = "custom"


class VerificationStatus(Enum):
    """Result of claim verification"""
    SUPPORTED = "supported"          # Evidence supports the claim
    CONTRADICTED = "contradicted"    # Evidence contradicts the claim
    INCONCLUSIVE = "inconclusive"    # Cannot verify either way
    NOT_VERIFIED = "not_verified"    # No relevant evidence available


class SourceType(Enum):
    """Media source options"""
    CAMERA = "camera"
    SOCIAL_MEDIA = "social_media"
    WEB_DOWNLOAD = "web_download"
    MESSAGING_APP = "messaging_app"
    SCREENSHOT = "screenshot"
    UNKNOWN = "unknown"


class WorkflowType(Enum):
    """Known media workflows"""
    CAMERA_DIRECT = "camera_direct"              # Camera → Upload
    CAMERA_EDITED = "camera_edited"              # Camera → Edit → Upload
    AI_GENERATE = "ai_generate"                  # AI Generate
    AI_GENERATE_ENHANCE = "ai_generate_enhance"  # AI Generate → Enhance
    DOWNLOAD_REUPLOAD = "download_reupload"      # Download → Re-upload
    DOWNLOAD_ENHANCE = "download_enhance"        # Download → Enhance → Upload
    UNKNOWN = "unknown"


@dataclass
class UserClaim:
    """Represents a single user claim"""
    claim_type: ClaimType
    claim_value: Any
    claim_text: str  # Human-readable version
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ClaimVerification:
    """Result of verifying a single claim"""
    claim: UserClaim
    status: VerificationStatus
    confidence: float  # 0-100, how confident in verification
    evidence: List[str]  # Supporting/contradicting evidence
    explanation: str  # Human-readable explanation
    forensic_signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserContext:
    """Complete user-provided context"""
    source: Optional[SourceType] = None
    is_ai_generated: Optional[bool] = None  # True/False/None(unsure)
    is_enhanced: Optional[bool] = None
    workflow: Optional[WorkflowType] = None
    additional_notes: Optional[str] = None
    claims: List[UserClaim] = field(default_factory=list)
    
    def to_claims(self) -> List[UserClaim]:
        """Convert context to list of claims"""
        claims = []
        
        if self.source:
            claims.append(UserClaim(
                claim_type=ClaimType.SOURCE,
                claim_value=self.source,
                claim_text=f"Media source: {self.source.value}"
            ))
        
        if self.is_ai_generated is not None:
            claims.append(UserClaim(
                claim_type=ClaimType.AI_GENERATED,
                claim_value=self.is_ai_generated,
                claim_text=f"AI-generated: {'Yes' if self.is_ai_generated else 'No'}"
            ))
        
        if self.is_enhanced is not None:
            claims.append(UserClaim(
                claim_type=ClaimType.AI_ENHANCED,
                claim_value=self.is_enhanced,
                claim_text=f"Enhanced/Edited: {'Yes' if self.is_enhanced else 'No'}"
            ))
        
        if self.workflow:
            claims.append(UserClaim(
                claim_type=ClaimType.WORKFLOW,
                claim_value=self.workflow,
                claim_text=f"Workflow: {self.workflow.value}"
            ))
        
        return claims


# =====================================================
# CLAIM VERIFICATION ENGINE
# =====================================================

class ClaimVerificationEngine:
    """
    Verifies user claims against forensic evidence.
    
    Key Principle:
    - NEVER trust user claims blindly
    - ALWAYS verify against forensic signals
    - Report discrepancies neutrally (not accusatory)
    """
    
    # Thresholds for verification
    STRONG_AI_THRESHOLD = 70  # Above this = strong AI signal
    MODERATE_AI_THRESHOLD = 50
    WEAK_AI_THRESHOLD = 30
    
    STRONG_ENHANCE_THRESHOLD = 65
    MODERATE_ENHANCE_THRESHOLD = 45
    
    def __init__(self):
        """Initialize verification engine"""
        pass
    
    def verify_all_claims(
        self,
        user_context: UserContext,
        forensic_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify all user claims against forensic evidence.
        
        Args:
            user_context: User-provided claims
            forensic_result: Result from visual forensics analysis
            
        Returns:
            Complete verification report
        """
        claims = user_context.to_claims()
        verifications = []
        
        for claim in claims:
            verification = self._verify_single_claim(claim, forensic_result)
            verifications.append(verification)
        
        # Calculate overall consistency
        consistency_score = self._calculate_consistency_score(verifications)
        
        # Generate summary
        summary = self._generate_verification_summary(
            verifications, 
            consistency_score,
            forensic_result
        )
        
        return {
            "verifications": [self._verification_to_dict(v) for v in verifications],
            "consistency_score": consistency_score,
            "overall_status": self._get_overall_status(verifications),
            "summary": summary,
            "user_context": {
                "source": user_context.source.value if user_context.source else None,
                "is_ai_generated": user_context.is_ai_generated,
                "is_enhanced": user_context.is_enhanced,
                "workflow": user_context.workflow.value if user_context.workflow else None,
                "notes": user_context.additional_notes
            },
            "trust_advisory": self._generate_trust_advisory(verifications, consistency_score)
        }
    
    def _verify_single_claim(
        self,
        claim: UserClaim,
        forensic_result: Dict[str, Any]
    ) -> ClaimVerification:
        """Verify a single claim"""
        
        if claim.claim_type == ClaimType.SOURCE:
            return self._verify_source_claim(claim, forensic_result)
        elif claim.claim_type == ClaimType.AI_GENERATED:
            return self._verify_ai_generated_claim(claim, forensic_result)
        elif claim.claim_type == ClaimType.AI_ENHANCED:
            return self._verify_enhanced_claim(claim, forensic_result)
        elif claim.claim_type == ClaimType.WORKFLOW:
            return self._verify_workflow_claim(claim, forensic_result)
        else:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.NOT_VERIFIED,
                confidence=0,
                evidence=["No verification method available"],
                explanation="This claim type cannot be automatically verified."
            )
    
    def _verify_source_claim(
        self,
        claim: UserClaim,
        forensic_result: Dict[str, Any]
    ) -> ClaimVerification:
        """Verify media source claim"""
        source: SourceType = claim.claim_value
        evidence = []
        forensic_signals = {}
        
        # Extract relevant forensic signals
        prediction = forensic_result.get("prediction", {})
        secondary = forensic_result.get("secondary_evidence", {})
        
        ai_confidence = prediction.get("confidence", 0)
        ai_class = prediction.get("class_name", "UNKNOWN")
        compression = secondary.get("compression", {}).get("details", {})
        ela_score = secondary.get("ela", {}).get("score", 50)
        
        forensic_signals["ai_class"] = ai_class
        forensic_signals["ai_confidence"] = ai_confidence
        forensic_signals["compression_format"] = compression.get("format", "unknown")
        
        # Verify based on claimed source
        if source == SourceType.CAMERA:
            # Camera images should be REAL, have JPEG compression, natural ELA
            if ai_class == "REAL" and ai_confidence > 60:
                status = VerificationStatus.SUPPORTED
                confidence = ai_confidence
                evidence.append(f"Image classified as REAL ({ai_confidence:.0f}% confidence)")
                explanation = "Forensic analysis supports camera origin."
            elif ai_class in ["AI_GENERATED", "AI_ENHANCED"]:
                status = VerificationStatus.CONTRADICTED
                confidence = ai_confidence
                evidence.append(f"Image classified as {ai_class} ({ai_confidence:.0f}%)")
                evidence.append("AI patterns detected inconsistent with camera source")
                explanation = "Forensic evidence suggests AI involvement, contradicting camera claim."
            else:
                status = VerificationStatus.INCONCLUSIVE
                confidence = 50
                evidence.append("Analysis inconclusive for source verification")
                explanation = "Cannot definitively verify camera origin."
        
        elif source == SourceType.SOCIAL_MEDIA:
            # Social media often has compression artifacts
            if compression.get("format") == "JPEG" and ela_score > 40:
                status = VerificationStatus.SUPPORTED
                confidence = 70
                evidence.append("Compression patterns consistent with social media")
                explanation = "Compression artifacts support social media origin."
            else:
                status = VerificationStatus.INCONCLUSIVE
                confidence = 50
                evidence.append("Cannot confirm social media origin from artifacts")
                explanation = "Evidence neither confirms nor denies social media source."
        
        elif source == SourceType.WEB_DOWNLOAD:
            # Web downloads can be anything
            status = VerificationStatus.INCONCLUSIVE
            confidence = 40
            evidence.append("Web download source cannot be forensically verified")
            explanation = "Download source cannot be determined from image analysis alone."
        
        else:
            status = VerificationStatus.INCONCLUSIVE
            confidence = 30
            evidence.append("Source claim not verifiable")
            explanation = "This source type cannot be verified with available methods."
        
        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence,
            explanation=explanation,
            forensic_signals=forensic_signals
        )
    
    def _verify_ai_generated_claim(
        self,
        claim: UserClaim,
        forensic_result: Dict[str, Any]
    ) -> ClaimVerification:
        """Verify AI-generated claim"""
        user_says_ai: bool = claim.claim_value
        evidence = []
        forensic_signals = {}
        
        # Get DL prediction
        prediction = forensic_result.get("prediction", {})
        ai_class = prediction.get("class_name", "UNKNOWN")
        ai_confidence = prediction.get("confidence", 0)
        distribution = prediction.get("distribution", {})
        
        # Get frequency analysis
        freq = forensic_result.get("primary_evidence", {}).get("frequency", {})
        freq_anomaly = freq.get("frequency_anomaly_score", 50)
        
        forensic_signals["ai_class"] = ai_class
        forensic_signals["ai_confidence"] = ai_confidence
        forensic_signals["distribution"] = distribution
        forensic_signals["frequency_anomaly"] = freq_anomaly
        
        # Model says AI?
        model_says_ai = ai_class in ["AI_GENERATED", "AI_ENHANCED"]
        model_confidence = ai_confidence if model_says_ai else 100 - ai_confidence
        
        if user_says_ai and model_says_ai:
            # User says AI, model says AI → SUPPORTED
            status = VerificationStatus.SUPPORTED
            confidence = ai_confidence
            evidence.append(f"Model confirms: {ai_class} ({ai_confidence:.0f}%)")
            evidence.append(f"Frequency anomaly score: {freq_anomaly:.0f}")
            explanation = "User claim that image is AI-generated is supported by forensic analysis."
        
        elif user_says_ai and not model_says_ai:
            # User says AI, model says REAL → CONTRADICTED
            status = VerificationStatus.CONTRADICTED
            confidence = 100 - ai_confidence  # Confidence in REAL
            evidence.append(f"Model classifies as REAL ({100-ai_confidence:.0f}% confidence)")
            evidence.append("No significant AI generation patterns detected")
            explanation = "User claims AI-generated, but forensic evidence suggests natural/real image."
        
        elif not user_says_ai and model_says_ai:
            # User says NOT AI, model says AI → CONTRADICTED
            status = VerificationStatus.CONTRADICTED
            confidence = ai_confidence
            evidence.append(f"Model detects {ai_class} ({ai_confidence:.0f}%)")
            evidence.append(f"AI texture patterns detected (freq anomaly: {freq_anomaly:.0f})")
            explanation = "User claims NOT AI-generated, but forensic evidence suggests AI involvement."
        
        else:
            # User says NOT AI, model says REAL → SUPPORTED
            status = VerificationStatus.SUPPORTED
            confidence = 100 - ai_confidence
            evidence.append(f"Model confirms REAL ({100-ai_confidence:.0f}%)")
            explanation = "User claim that image is not AI-generated is supported by analysis."
        
        # Check if confidence is too low for definitive status
        if ai_confidence < self.WEAK_AI_THRESHOLD or (100 - ai_confidence) < self.WEAK_AI_THRESHOLD:
            status = VerificationStatus.INCONCLUSIVE
            confidence = max(30, confidence * 0.7)
            evidence.append("Low confidence in classification - result inconclusive")
            explanation = "Analysis confidence is too low for definitive verification."
        
        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence,
            explanation=explanation,
            forensic_signals=forensic_signals
        )
    
    def _verify_enhanced_claim(
        self,
        claim: UserClaim,
        forensic_result: Dict[str, Any]
    ) -> ClaimVerification:
        """Verify enhancement/editing claim"""
        user_says_enhanced: bool = claim.claim_value
        evidence = []
        forensic_signals = {}
        
        # Get prediction
        prediction = forensic_result.get("prediction", {})
        ai_class = prediction.get("class_name", "UNKNOWN")
        ai_confidence = prediction.get("confidence", 0)
        distribution = prediction.get("distribution", {})
        
        # Get ELA and compression signals
        secondary = forensic_result.get("secondary_evidence", {})
        ela = secondary.get("ela", {})
        ela_score = ela.get("score", 50)
        compression = secondary.get("compression", {})
        noise = secondary.get("noise", {})
        
        forensic_signals["ai_class"] = ai_class
        forensic_signals["enhanced_probability"] = distribution.get("ai_enhanced", 0)
        forensic_signals["ela_score"] = ela_score
        forensic_signals["noise_consistency"] = noise.get("details", {}).get("consistency_ratio", 0)
        
        # Check for enhancement signals
        enhanced_prob = distribution.get("ai_enhanced", 0)
        is_enhanced_class = ai_class == "AI_ENHANCED"
        high_ela = ela_score > self.STRONG_ENHANCE_THRESHOLD
        
        model_says_enhanced = is_enhanced_class or enhanced_prob > 40 or high_ela
        
        if user_says_enhanced and model_says_enhanced:
            status = VerificationStatus.SUPPORTED
            confidence = max(enhanced_prob, ela_score)
            evidence.append(f"Enhancement probability: {enhanced_prob:.0f}%")
            evidence.append(f"ELA score: {ela_score:.0f}")
            if is_enhanced_class:
                evidence.append("Classified as AI_ENHANCED")
            explanation = "User claim of enhancement is supported by forensic signals."
        
        elif user_says_enhanced and not model_says_enhanced:
            status = VerificationStatus.CONTRADICTED
            confidence = 100 - max(enhanced_prob, ela_score)
            evidence.append(f"Low enhancement signals (prob: {enhanced_prob:.0f}%, ELA: {ela_score:.0f})")
            evidence.append("No significant editing artifacts detected")
            explanation = "User claims enhancement, but forensic evidence shows minimal editing signs."
        
        elif not user_says_enhanced and model_says_enhanced:
            status = VerificationStatus.CONTRADICTED
            confidence = max(enhanced_prob, ela_score)
            evidence.append(f"Enhancement signals detected (prob: {enhanced_prob:.0f}%)")
            if high_ela:
                evidence.append(f"High ELA score ({ela_score:.0f}) suggests modifications")
            explanation = "User claims no enhancement, but forensic evidence suggests processing."
        
        else:
            status = VerificationStatus.SUPPORTED
            confidence = 100 - max(enhanced_prob, ela_score)
            evidence.append("No significant enhancement patterns detected")
            explanation = "User claim of no enhancement is consistent with analysis."
        
        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence,
            explanation=explanation,
            forensic_signals=forensic_signals
        )
    
    def _verify_workflow_claim(
        self,
        claim: UserClaim,
        forensic_result: Dict[str, Any]
    ) -> ClaimVerification:
        """Verify workflow claim"""
        workflow: WorkflowType = claim.claim_value
        evidence = []
        forensic_signals = {}
        
        prediction = forensic_result.get("prediction", {})
        ai_class = prediction.get("class_name", "UNKNOWN")
        ai_confidence = prediction.get("confidence", 0)
        distribution = prediction.get("distribution", {})
        
        forensic_signals["ai_class"] = ai_class
        forensic_signals["distribution"] = distribution
        
        # Workflow verification logic
        if workflow == WorkflowType.CAMERA_DIRECT:
            # Should be REAL, no enhancement
            if ai_class == "REAL" and ai_confidence > 60:
                status = VerificationStatus.SUPPORTED
                confidence = ai_confidence
                evidence.append("Classified as REAL - consistent with camera workflow")
            elif ai_class in ["AI_GENERATED", "AI_ENHANCED"]:
                status = VerificationStatus.CONTRADICTED
                confidence = ai_confidence
                evidence.append(f"Detected as {ai_class} - inconsistent with camera-only workflow")
            else:
                status = VerificationStatus.INCONCLUSIVE
                confidence = 50
                evidence.append("Cannot confirm camera-only workflow")
            explanation = f"Workflow verification: Camera direct upload - {status.value}"
        
        elif workflow == WorkflowType.AI_GENERATE:
            if ai_class == "AI_GENERATED" and ai_confidence > 60:
                status = VerificationStatus.SUPPORTED
                confidence = ai_confidence
                evidence.append("AI generation patterns confirmed")
            elif ai_class == "REAL":
                status = VerificationStatus.CONTRADICTED
                confidence = 100 - ai_confidence
                evidence.append("No AI generation patterns found")
            else:
                status = VerificationStatus.INCONCLUSIVE
                confidence = 50
                evidence.append("Mixed signals for AI generation")
            explanation = f"Workflow verification: AI generation - {status.value}"
        
        elif workflow == WorkflowType.AI_GENERATE_ENHANCE:
            # Could be AI_GENERATED or AI_ENHANCED
            ai_total = distribution.get("ai_generated", 0) + distribution.get("ai_enhanced", 0)
            if ai_total > 60:
                status = VerificationStatus.SUPPORTED
                confidence = ai_total
                evidence.append(f"AI involvement detected ({ai_total:.0f}% combined)")
            elif ai_class == "REAL":
                status = VerificationStatus.CONTRADICTED
                confidence = 100 - ai_total
                evidence.append("No AI patterns detected")
            else:
                status = VerificationStatus.INCONCLUSIVE
                confidence = 50
                evidence.append("Insufficient evidence for workflow verification")
            explanation = f"Workflow verification: AI generate + enhance - {status.value}"
        
        else:
            status = VerificationStatus.INCONCLUSIVE
            confidence = 40
            evidence.append("Complex workflow cannot be fully verified")
            explanation = "This workflow type cannot be definitively verified."
        
        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence,
            explanation=explanation,
            forensic_signals=forensic_signals
        )
    
    def _calculate_consistency_score(
        self,
        verifications: List[ClaimVerification]
    ) -> float:
        """Calculate overall consistency between claims and evidence"""
        if not verifications:
            return 100.0  # No claims = no contradiction
        
        scores = []
        for v in verifications:
            if v.status == VerificationStatus.SUPPORTED:
                scores.append(v.confidence)
            elif v.status == VerificationStatus.CONTRADICTED:
                scores.append(100 - v.confidence)  # Inverse
            elif v.status == VerificationStatus.INCONCLUSIVE:
                scores.append(50)  # Neutral
            else:
                continue  # Skip NOT_VERIFIED
        
        if not scores:
            return 50.0
        
        return round(sum(scores) / len(scores), 2)
    
    def _get_overall_status(
        self,
        verifications: List[ClaimVerification]
    ) -> str:
        """Determine overall verification status"""
        if not verifications:
            return "no_claims"
        
        statuses = [v.status for v in verifications]
        
        if all(s == VerificationStatus.SUPPORTED for s in statuses):
            return "all_supported"
        elif any(s == VerificationStatus.CONTRADICTED for s in statuses):
            return "has_contradictions"
        elif all(s == VerificationStatus.INCONCLUSIVE for s in statuses):
            return "all_inconclusive"
        else:
            return "mixed"
    
    def _generate_verification_summary(
        self,
        verifications: List[ClaimVerification],
        consistency_score: float,
        forensic_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate human-readable verification summary"""
        supported = [v for v in verifications if v.status == VerificationStatus.SUPPORTED]
        contradicted = [v for v in verifications if v.status == VerificationStatus.CONTRADICTED]
        inconclusive = [v for v in verifications if v.status == VerificationStatus.INCONCLUSIVE]
        
        summary = {
            "total_claims": len(verifications),
            "supported_count": len(supported),
            "contradicted_count": len(contradicted),
            "inconclusive_count": len(inconclusive),
            "consistency_score": consistency_score,
            
            # Simple explanation
            "simple_explanation": "",
            
            # Technical details
            "technical_notes": [],
            
            # Warnings
            "warnings": []
        }
        
        # Generate simple explanation
        if len(contradicted) == 0 and len(supported) > 0:
            summary["simple_explanation"] = (
                f"All {len(supported)} user-provided claims are consistent with forensic evidence."
            )
        elif len(contradicted) > 0:
            summary["simple_explanation"] = (
                f"⚠️ {len(contradicted)} of {len(verifications)} claims do not align "
                f"with forensic evidence. Review recommended."
            )
            for v in contradicted:
                summary["warnings"].append({
                    "claim": v.claim.claim_text,
                    "issue": v.explanation,
                    "evidence": v.evidence
                })
        else:
            summary["simple_explanation"] = (
                "Claims could not be definitively verified against forensic evidence."
            )
        
        # Technical notes
        for v in verifications:
            summary["technical_notes"].append({
                "claim": v.claim.claim_text,
                "status": v.status.value,
                "confidence": v.confidence,
                "evidence": v.evidence
            })
        
        return summary
    
    def _generate_trust_advisory(
        self,
        verifications: List[ClaimVerification],
        consistency_score: float
    ) -> Dict[str, Any]:
        """Generate advisory on how much to trust user context"""
        contradicted = sum(1 for v in verifications if v.status == VerificationStatus.CONTRADICTED)
        
        if not verifications:
            level = "no_context"
            message = "No user context provided - analysis based solely on forensic evidence."
            recommendation = "Consider requesting context for additional insight."
        elif contradicted > 0:
            level = "low_trust"
            message = (
                f"The provided information does not fully align with forensic evidence. "
                f"{contradicted} claim(s) contradicted."
            )
            recommendation = (
                "User context should be treated with caution. "
                "Forensic evidence takes precedence."
            )
        elif consistency_score > 75:
            level = "high_trust"
            message = "User-provided context is consistent with forensic findings."
            recommendation = "Context can be used to support analysis conclusions."
        elif consistency_score > 50:
            level = "moderate_trust"
            message = "User context is partially verifiable."
            recommendation = "Use context as supplementary information only."
        else:
            level = "low_trust"
            message = "User context could not be adequately verified."
            recommendation = "Rely primarily on forensic evidence."
        
        return {
            "trust_level": level,
            "message": message,
            "recommendation": recommendation,
            "consistency_score": consistency_score
        }
    
    def _verification_to_dict(self, v: ClaimVerification) -> Dict[str, Any]:
        """Convert ClaimVerification to dictionary"""
        return {
            "claim": {
                "type": v.claim.claim_type.value,
                "value": v.claim.claim_value.value if hasattr(v.claim.claim_value, 'value') else v.claim.claim_value,
                "text": v.claim.claim_text
            },
            "status": v.status.value,
            "confidence": round(v.confidence, 2),
            "evidence": v.evidence,
            "explanation": v.explanation,
            "forensic_signals": v.forensic_signals
        }


# =====================================================
# DECISION FUSION WITH CLAIM VERIFICATION
# =====================================================

class DecisionFusion:
    """
    Fuses forensic evidence with claim verification for final verdict.
    
    KEY PRINCIPLE:
    User claims NEVER override forensic evidence.
    They can only:
    - Increase confidence (when consistent)
    - Add warnings (when contradicted)
    """
    
    def __init__(self):
        self.claim_verifier = ClaimVerificationEngine()
    
    def fuse_decision(
        self,
        forensic_result: Dict[str, Any],
        user_context: Optional[UserContext] = None
    ) -> Dict[str, Any]:
        """
        Fuse forensic result with user context for final decision.
        
        Args:
            forensic_result: Complete forensic analysis result
            user_context: Optional user-provided context
            
        Returns:
            Enhanced decision with claim verification
        """
        result = {
            "final_verdict": {},
            "claim_verification": None,
            "confidence_adjustment": {},
            "decision_explanation": {}
        }
        
        # Get base prediction from forensics
        prediction = forensic_result.get("prediction", {})
        base_class = prediction.get("class_name", "UNKNOWN")
        base_confidence = prediction.get("confidence", 0)
        
        # Start with forensic verdict
        final_verdict = base_class
        final_confidence = base_confidence
        
        # Verify claims if user context provided
        if user_context and user_context.to_claims():
            verification = self.claim_verifier.verify_all_claims(
                user_context,
                forensic_result
            )
            result["claim_verification"] = verification
            
            # Adjust confidence based on consistency
            consistency = verification.get("consistency_score", 50)
            overall_status = verification.get("overall_status", "no_claims")
            
            if overall_status == "all_supported":
                # Consistent context boosts confidence
                boost = min(10, (consistency - 50) / 5)
                final_confidence = min(95, base_confidence + boost)
                result["confidence_adjustment"] = {
                    "type": "boost",
                    "reason": "User context consistent with forensic evidence",
                    "amount": boost
                }
            
            elif overall_status == "has_contradictions":
                # Contradictions add uncertainty but don't change verdict
                # Forensic evidence still takes precedence
                result["confidence_adjustment"] = {
                    "type": "warning",
                    "reason": "User context contradicts forensic evidence",
                    "amount": 0  # No change to confidence
                }
                
                # Add warning to verdict
                final_verdict = f"{base_class} (CONTEXT DISCREPANCY)"
            
            else:
                result["confidence_adjustment"] = {
                    "type": "none",
                    "reason": "Context verification inconclusive",
                    "amount": 0
                }
        else:
            result["confidence_adjustment"] = {
                "type": "none",
                "reason": "No user context provided",
                "amount": 0
            }
        
        # Set final verdict
        result["final_verdict"] = {
            "verdict": final_verdict,
            "confidence": round(final_confidence, 2),
            "based_on": "Forensic analysis" + (" + verified user context" if user_context else ""),
            "class_probabilities": prediction.get("distribution", {})
        }
        
        # Generate explanation
        result["decision_explanation"] = self._generate_decision_explanation(
            forensic_result,
            result.get("claim_verification"),
            result["final_verdict"]
        )
        
        return result
    
    def _generate_decision_explanation(
        self,
        forensic_result: Dict[str, Any],
        claim_verification: Optional[Dict],
        final_verdict: Dict
    ) -> Dict[str, Any]:
        """Generate explanation for the decision"""
        explanation = {
            "simple": "",
            "technical": "",
            "factors": []
        }
        
        verdict = final_verdict.get("verdict", "UNKNOWN")
        confidence = final_verdict.get("confidence", 0)
        
        # Simple explanation
        if "REAL" in verdict:
            explanation["simple"] = (
                f"This media appears to be genuine with {confidence:.0f}% confidence. "
                f"No significant AI generation or manipulation patterns were detected."
            )
        elif "AI_GENERATED" in verdict:
            explanation["simple"] = (
                f"This media appears to be AI-generated with {confidence:.0f}% confidence. "
                f"Patterns consistent with AI image generation were detected."
            )
        elif "AI_ENHANCED" in verdict:
            explanation["simple"] = (
                f"This media appears to have been processed by AI with {confidence:.0f}% confidence. "
                f"Signs of AI-based enhancement or editing were detected."
            )
        else:
            explanation["simple"] = (
                f"Analysis is inconclusive ({confidence:.0f}% confidence). "
                f"Manual expert review is recommended."
            )
        
        # Add context discrepancy warning
        if "CONTEXT DISCREPANCY" in verdict:
            explanation["simple"] += (
                " Note: The provided context does not fully align with forensic evidence."
            )
        
        # Factors
        explanation["factors"].append({
            "source": "Deep Learning Analysis",
            "weight": "Primary (60-70%)",
            "finding": forensic_result.get("prediction", {}).get("class_name", "Unknown")
        })
        
        if claim_verification:
            explanation["factors"].append({
                "source": "User Context Verification",
                "weight": "Advisory only",
                "finding": claim_verification.get("overall_status", "not_verified")
            })
        
        # Technical explanation
        explanation["technical"] = (
            f"Classification: {verdict}\n"
            f"Confidence: {confidence:.2f}%\n"
            f"Primary Evidence: Deep Learning + Frequency Fusion\n"
            f"Secondary Evidence: ELA, Noise, Frequency Analysis\n"
        )
        
        if claim_verification:
            explanation["technical"] += (
                f"Claim Consistency: {claim_verification.get('consistency_score', 0):.0f}%\n"
                f"Verification Status: {claim_verification.get('overall_status', 'N/A')}\n"
            )
        
        return explanation


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def create_user_context(
    source: str = None,
    is_ai_generated: bool = None,
    is_enhanced: bool = None,
    workflow: str = None,
    notes: str = None
) -> UserContext:
    """
    Helper function to create UserContext from simple inputs.
    
    Args:
        source: "camera", "social_media", "web_download", "unknown"
        is_ai_generated: True/False/None
        is_enhanced: True/False/None
        workflow: "camera_direct", "ai_generate", "ai_generate_enhance", etc.
        notes: Additional notes
    """
    context = UserContext()
    
    if source:
        source_map = {
            "camera": SourceType.CAMERA,
            "social_media": SourceType.SOCIAL_MEDIA,
            "web_download": SourceType.WEB_DOWNLOAD,
            "messaging": SourceType.MESSAGING_APP,
            "screenshot": SourceType.SCREENSHOT,
            "unknown": SourceType.UNKNOWN
        }
        context.source = source_map.get(source.lower(), SourceType.UNKNOWN)
    
    context.is_ai_generated = is_ai_generated
    context.is_enhanced = is_enhanced
    
    if workflow:
        workflow_map = {
            "camera_direct": WorkflowType.CAMERA_DIRECT,
            "camera_edited": WorkflowType.CAMERA_EDITED,
            "ai_generate": WorkflowType.AI_GENERATE,
            "ai_generate_enhance": WorkflowType.AI_GENERATE_ENHANCE,
            "download_reupload": WorkflowType.DOWNLOAD_REUPLOAD,
            "download_enhance": WorkflowType.DOWNLOAD_ENHANCE,
            "unknown": WorkflowType.UNKNOWN
        }
        context.workflow = workflow_map.get(workflow.lower(), WorkflowType.UNKNOWN)
    
    context.additional_notes = notes
    
    return context


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Claim Verification System v1.0")
    print("=" * 60)
    print("""
User claims are treated as "soft claims", not ground truth.
Every claim is independently verified against forensic evidence.

Available claim types:
- Source: camera, social_media, web_download, unknown
- AI Generated: yes, no, not_sure
- Enhanced: yes, no, not_sure
- Workflow: camera_direct, ai_generate, ai_generate_enhance

Example:
    context = create_user_context(
        source="camera",
        is_ai_generated=False,
        is_enhanced=False
    )
    
    result = decision_fusion.fuse_decision(forensic_result, context)
""")
