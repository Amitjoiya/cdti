"""Quick test for Fraud Analyzer"""
import sys
sys.path.insert(0, '.')
from modules.fraud_analyzer import FraudAnalyzer

analyzer = FraudAnalyzer()

# Test 1: Banking scam
print("\n" + "="*60)
print("TEST 1: Banking Scam Message (SBI)")
print("="*60)

result = analyzer.analyze(
    message="Dear Customer, Your SBI account has been suspended due to KYC expiry. Click http://sbi-kyc-update.xyz to update now or your account will be blocked in 24 hours. Call 9876543210 for help.",
    platform="sms",
    message_type="banking"
)

print(f"Verdict: {result['fraud_assessment']['verdict']}")
print(f"Confidence: {result['fraud_assessment']['confidence']}%")
print(f"Risk Score: {result['fraud_assessment']['risk_score']}/100")
print(f"Risk Level: {result['fraud_assessment']['risk_level']}")
print(f"\nKey Risks:")
for risk in result['fraud_assessment'].get('key_risks', []):
    print(f"  ❌ {risk}")
print(f"\nURLs Found: {result['urls_found']}")

# Test 2: Lottery scam
print("\n" + "="*60)
print("TEST 2: Lottery Scam (WhatsApp)")
print("="*60)

result2 = analyzer.analyze(
    message="Congratulations! You have won Rs 50 Lakh in Amazon Lucky Draw. Click http://amazon-prize.win/claim to claim your prize. Share OTP 4521 to verify. Act NOW before offer expires!",
    platform="whatsapp",
    message_type="offer"
)

print(f"Verdict: {result2['fraud_assessment']['verdict']}")
print(f"Confidence: {result2['fraud_assessment']['confidence']}%")
print(f"Risk Score: {result2['fraud_assessment']['risk_score']}/100")
print(f"\nKey Risks:")
for risk in result2['fraud_assessment'].get('key_risks', []):
    print(f"  ❌ {risk}")

# Test 3: Safe message
print("\n" + "="*60)
print("TEST 3: Safe Message (Normal)")
print("="*60)

result3 = analyzer.analyze(
    message="Your order #12345 has been shipped. Track at https://www.amazon.in/track. Delivery expected by Dec 30.",
    platform="sms",
    message_type="delivery"
)

print(f"Verdict: {result3['fraud_assessment']['verdict']}")
print(f"Confidence: {result3['fraud_assessment']['confidence']}%")
print(f"Risk Score: {result3['fraud_assessment']['risk_score']}/100")
print(f"\nSafe Signs:")
for sign in result3['fraud_assessment'].get('safe_signs', []):
    print(f"  ✅ {sign}")
