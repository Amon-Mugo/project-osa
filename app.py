"""
OSA Email Spam/Phishing Checker – Web App
Run: uvicorn app:app --reload --host 127.0.0.1 --port 8000
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.templating import Jinja2Templates

BASE_DIR        = Path(__file__).resolve().parent
TEMPLATES_DIR   = BASE_DIR / "templates"
STATIC_DIR      = BASE_DIR / "static"
MODEL_PATH      = BASE_DIR / "email_spam_detection.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        "\n[ERROR] Model files not found.\n"
        "Run first:  python train_email_model.py\n"
        f"Expected:   {MODEL_PATH}\n"
        f"            {VECTORIZER_PATH}"
    )

model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

THRESHOLD = 0.30

TRIGGERS = {
    "urgency": {
        "label":       "Urgency",
        "icon":        "⏰",
        "description": "Pressures you to act immediately without thinking",
        "keywords": [
            r"\bact now\b", r"\burgent\b", r"\bimmediately\b", r"\bexpires?\b",
            r"\bexpiring\b", r"\blimited time\b", r"\b24 hours?\b",
            r"\bdeadline\b", r"\btoday only\b", r"\blast chance\b",
            r"\bdo not delay\b", r"\bfinal notice\b", r"\btime is running out\b",
            r"\bhurry\b", r"\bquickly\b", r"\bright now\b", r"\binstantly\b",
        ],
    },
    "reward_gain": {
        "label":       "Reward / Gain",
        "icon":        "🎁",
        "description": "Lures you with prizes, gifts, or money you did not expect",
        "keywords": [
            r"\bfree\b", r"\bwon\b", r"\bwinner\b", r"\bprize\b",
            r"\bgift card\b", r"\bcongratulations\b", r"\bclaim\b",
            r"\breward\b", r"\boffer\b", r"\bdiscount\b", r"\bcashback\b",
            r"\blottery\b", r"\bjackpot\b", r"\bcash prize\b",
            r"\byou have been selected\b", r"\byou are a winner\b",
            r"\bget paid\b", r"\bearn money\b",
        ],
    },
    "credential_request": {
        "label":       "Credential Request",
        "icon":        "🔑",
        "description": "Asks for your personal details, login, or identity information",
        "keywords": [
            r"\bverify\b", r"\bverification\b", r"\bconfirm your\b",
            r"\bpassword\b", r"\bpin\b", r"\blogin\b", r"\blog in\b",
            r"\bsign in\b", r"\baccount details\b", r"\bbank details\b",
            r"\bsend your id\b", r"\bnational id\b", r"\bid number\b",
            r"\bcredentials\b", r"\bsocial security\b", r"\bdate of birth\b",
            r"\bmother.?s maiden\b", r"\bsecurity question\b",
            r"\bupdate your (account|details|information)\b",
        ],
    },
    "fear_threat": {
        "label":       "Fear / Threat",
        "icon":        "⚠️",
        "description": "Uses threats or fear to force you into taking action",
        "keywords": [
            r"\bsuspended\b", r"\bsuspension\b", r"\bterminated\b",
            r"\bblocked\b", r"\bclosed\b", r"\blocked\b",
            r"\bfinal warning\b", r"\blegal action\b", r"\blawsuit\b",
            r"\barrest\b", r"\bpolice\b", r"\bpenalty\b", r"\bfine\b",
            r"\bwarning\b", r"\baccount (will be|has been) (closed|suspended|blocked)\b",
            r"\byour account is at risk\b", r"\bunauthorized access\b",
            r"\bsuspicious activity\b", r"\byou owe\b", r"\bdebt\b",
        ],
    },
    "payment_request": {
        "label":       "Payment Request",
        "icon":        "💸",
        "description": "Asks you to send money or share financial information",
        "keywords": [
            r"\bsend money\b", r"\btransfer\b", r"\bm-?pesa\b", r"\bpayment\b",
            r"\bpay now\b", r"\bpay (us|me|here)\b", r"\bdeposit\b",
            r"\bwire transfer\b", r"\bwestern union\b", r"\bmoneygram\b",
            r"\bcredit card\b", r"\bcard number\b", r"\bcvv\b",
            r"\bbilling\b", r"\binvoice\b", r"\bpay (the )?fee\b",
            r"\bprocessing fee\b", r"\bregistration fee\b",
            r"\bbank account (number|details)\b", r"\bpaypal\b",
        ],
    },
}


def detect_triggers(message: str) -> list[dict]:
    """Return list of triggered psychological patterns found in the message."""
    text    = message.lower()
    found   = []
    for key, trigger in TRIGGERS.items():
        matched_keywords = []
        for pattern in trigger["keywords"]:
            if re.search(pattern, text):
                # Extract readable keyword from pattern for display
                readable = pattern.replace(r"\b", "").replace("?", "").replace("\\", "")
                matched_keywords.append(readable.strip())
        if matched_keywords:
            found.append({
                "key":         key,
                "label":       trigger["label"],
                "icon":        trigger["icon"],
                "description": trigger["description"],
                "matches":     matched_keywords[:3],  # show max 3 matched keywords
            })
    return found


def build_explanation(label: str, triggers: list[dict]) -> str:
    """Generate a plain-English explanation for non-technical users."""
    if label == "ham":
        return "This message appears safe. No significant warning signs were detected."

    if not triggers:
        return (
            "Our model flagged this message as potentially risky based on its "
            "overall pattern, even though no specific warning keywords were found. "
            "Exercise caution."
        )

    trigger_names = [t["label"].lower() for t in triggers]

    if len(trigger_names) == 1:
        combo = trigger_names[0]
    elif len(trigger_names) == 2:
        combo = f"{trigger_names[0]} and {trigger_names[1]}"
    else:
        combo = ", ".join(trigger_names[:-1]) + f", and {trigger_names[-1]}"

    return (
        f"This message is risky because it uses {combo}. "
        "Scammers use these tactics to pressure people into acting without thinking. "
        "Do not click any links, share personal details, or send money."
    )


def classify(message: str) -> dict:
    """Run model + trigger detection and return full result."""
    X        = vectorizer.transform([message])
    proba    = model.predict_proba(X)[0, 1]
    label    = "spam" if proba >= THRESHOLD else "ham"
    triggers = detect_triggers(message) if label == "spam" else []
    explanation = build_explanation(label, triggers)

    return {
        "prediction":       label,
        "spam_probability": float(round(proba, 4)),
        "threshold":        THRESHOLD,
        "triggers":         triggers,
        "explanation":      explanation,
    }



app = FastAPI(title="OSA Email Risk Checker", version="2.0.0")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request":          request,
            "prediction":       None,
            "spam_probability": None,
            "message":          None,
            "threshold":        THRESHOLD,
            "triggers":         [],
            "explanation":      None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, message: str = Form(...)):
    result = classify(message)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "message": message, **result},
    )


# JSON API 
class EmailIn(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_email")
def predict_email(data: EmailIn):
    """
    JSON API for programmatic use.

    Request:
        POST /predict_email
        {"message": "Congratulations! You won a prize..."}

    Response:
        {
            "prediction": "spam",
            "spam_probability": 0.87,
            "threshold": 0.3,
            "triggers": [
                {
                    "key": "urgency",
                    "label": "Urgency",
                    "icon": "⏰",
                    "description": "Pressures you to act immediately without thinking",
                    "matches": ["act now", "expires"]
                }
            ],
            "explanation": "This message is risky because it uses urgency..."
        }
    """
    return classify(data.message)
