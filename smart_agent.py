"""
OSA Smart Agent
Generates conversational, plain-language explanations for scam detection results.
Uses Gemini API when available, falls back to rule-based explanations.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project folder
load_dotenv(Path(__file__).resolve().parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Try to load new google.genai package
GEMINI_AVAILABLE = False
gemini_client    = None

try:
    from google import genai
    if GEMINI_API_KEY:
        gemini_client    = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("Smart Agent: Gemini API ready.")
    else:
        print("Smart Agent: No API key found, using rule-based fallback.")
except ImportError:
    print("Smart Agent: google-genai not installed, using rule-based fallback.")


# ── Scam Type Detector ────────────────────────────────────────────────────────
def detect_scam_type(message: str, triggers: list[dict]) -> str:
    text         = message.lower()
    trigger_keys = [t["key"] for t in triggers]

    if any(w in text for w in ["bank", "mpesa", "m-pesa", "equity", "kcb", "safaricom", "account"]):
        return "bank_impersonation"
    if any(w in text for w in ["won", "winner", "prize", "lottery", "gift card", "congratulations"]):
        return "prize_scam"
    if any(w in text for w in ["job", "work from home", "salary", "hiring", "employment", "position", "applicant"]):
        return "job_scam"
    if any(w in text for w in ["invest", "returns", "profit", "crypto", "bitcoin", "forex", "trading"]):
        return "investment_scam"
    if any(w in text for w in ["verify", "password", "login", "credentials", "sign in", "account details"]):
        return "credential_phishing"
    if any(w in text for w in ["send money", "transfer", "western union", "moneygram", "processing fee"]):
        return "payment_scam"
    if "credential_request" in trigger_keys and "fear_threat" in trigger_keys:
        return "credential_phishing"
    if "payment_request" in trigger_keys:
        return "payment_scam"
    if "reward_gain" in trigger_keys:
        return "prize_scam"

    return "general_scam"


# ── Rule-Based Explanations ───────────────────────────────────────────────────
RULE_BASED_EXPLANATIONS = {
    "bank_impersonation": (
        "This message is impersonating a bank or mobile money service. "
        "It is trying to trick you into giving away your account details or clicking a fake link. "
        "Your real bank or M-Pesa will NEVER ask for your PIN, password, or full account details via SMS or email. "
        "Do not click any links. Call your bank directly using the number on the back of your card or their official website."
    ),
    "prize_scam": (
        "This message is a fake prize or lottery scam. "
        "You did not win anything — this is a trick to get your personal details or money. "
        "Legitimate competitions do not ask winners to pay fees or provide bank details to claim prizes. "
        "Ignore this message and do not respond."
    ),
    "job_scam": (
        "This message is a fake job offer. "
        "Scammers use attractive salaries and easy work-from-home promises to steal your personal information or money. "
        "Legitimate employers will never ask you to pay a registration fee or send your ID and bank details via email. "
        "Research the company independently before responding to any job offer."
    ),
    "investment_scam": (
        "This message is promoting a fraudulent investment scheme. "
        "Promises of guaranteed high returns with no risk are always a scam. "
        "Never invest money based on unsolicited messages. "
        "Consult a licensed financial advisor before making any investment decisions."
    ),
    "credential_phishing": (
        "This message is trying to steal your login credentials or personal information. "
        "It may be pretending to be a trusted service to get your username, password, or ID number. "
        "Never share your login details, PIN, or national ID through a link sent via SMS or email. "
        "Go directly to the official website by typing the address yourself."
    ),
    "payment_scam": (
        "This message is trying to trick you into sending money. "
        "It may claim you need to pay a fee to receive a prize, process a job application, or unlock your account. "
        "These fees are always fake — once you send the money you will never get it back. "
        "Never send money to someone you do not know personally and have not verified."
    ),
    "general_scam": (
        "This message shows signs of being a scam. "
        "Scammers use psychological pressure to make you act quickly without thinking. "
        "Take your time, do not click any links, and do not share any personal information. "
        "If in doubt, contact the organization directly using their official contact details."
    ),
}


def rule_based_explanation(
    message:    str,
    prediction: str,
    probability: float,
    triggers:   list[dict],
) -> str:
    if prediction == "ham":
        return (
            "This message appears safe. "
            "Our analysis found no significant signs of spam or phishing. "
            "Always stay cautious and verify unexpected messages independently."
        )

    scam_type        = detect_scam_type(message, triggers)
    base_explanation = RULE_BASED_EXPLANATIONS.get(
        scam_type, RULE_BASED_EXPLANATIONS["general_scam"]
    )

    if triggers:
        trigger_labels = [t["label"] for t in triggers]
        if len(trigger_labels) == 1:
            trigger_text = trigger_labels[0]
        elif len(trigger_labels) == 2:
            trigger_text = f"{trigger_labels[0]} and {trigger_labels[1]}"
        else:
            trigger_text = ", ".join(trigger_labels[:-1]) + f", and {trigger_labels[-1]}"
        base_explanation = f"Warning signs detected: {trigger_text}.\n\n" + base_explanation

    return base_explanation


# ── Gemini-Powered Explanation ────────────────────────────────────────────────
def gemini_explanation(
    message:    str,
    prediction: str,
    probability: float,
    triggers:   list[dict],
) -> str:
    scam_type      = detect_scam_type(message, triggers)
    trigger_labels = [t["label"] for t in triggers] if triggers else []

    prompt = f"""You are OSA — an Online Scam Awareness assistant helping everyday people in Kenya and East Africa stay safe from scams.

A user has submitted this message for analysis:
---
{message}
---

Our AI model analyzed it with these results:
- Verdict: {"SUSPICIOUS / PHISHING" if prediction == "spam" else "SAFE"}
- Risk probability: {round(probability * 100, 1)}%
- Scam type: {scam_type.replace("_", " ").title()}
- Warning signs: {", ".join(trigger_labels) if trigger_labels else "None"}

Write a SHORT, CLEAR explanation (3-5 sentences) for a non-technical user:
1. What this message is trying to do
2. Why it is dangerous (or safe)
3. What the user should do next

Rules:
- Simple everyday language, no technical jargon
- Be direct and specific to this message
- No bullet points — plain paragraphs only
- Under 100 words
"""

    try:
        response = gemini_client.models.generate_content(
            model    = "gemini-2.0-flash",
            contents = prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Smart Agent: Gemini failed ({e}), using rule-based fallback.")
        return rule_based_explanation(message, prediction, probability, triggers)


# ── Main Entry Point ──────────────────────────────────────────────────────────
def get_smart_agent_explanation(
    message:    str,
    prediction: str,
    probability: float,
    triggers:   list[dict],
) -> dict:
    scam_type = detect_scam_type(message, triggers)

    if GEMINI_AVAILABLE and prediction == "spam":
        explanation = gemini_explanation(message, prediction, probability, triggers)
        engine      = "gemini"
    else:
        explanation = rule_based_explanation(message, prediction, probability, triggers)
        engine      = "rule-based"

    return {
        "explanation": explanation,
        "scam_type":   scam_type.replace("_", " ").title(),
        "engine":      engine,
    }
