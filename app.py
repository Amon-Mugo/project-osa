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
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.templating import Jinja2Templates
from smart_agent import get_smart_agent_explanation

BASE_DIR      = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"
MODEL_DIR     = BASE_DIR / "osa_distilbert_model"

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        "\n[ERROR] DistilBERT model folder not found.\n"
        f"Expected: {MODEL_DIR}\n"
        "Download it from Google Drive and extract it into your project folder."
    )

print("Loading DistilBERT model...")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer  = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
bert_model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
bert_model.to(device)
bert_model.eval()
print(f"Model loaded on: {device}")

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
            r"\bblocked\b", r"\bclosed\b",
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
    text  = message.lower()
    found = []
    for key, trigger in TRIGGERS.items():
        matched_keywords = []
        for pattern in trigger["keywords"]:
            if re.search(pattern, text):
                readable = pattern.replace(r"\b", "").replace("?", "").replace("\\", "")
                matched_keywords.append(readable.strip())
        if matched_keywords:
            found.append({
                "key":         key,
                "label":       trigger["label"],
                "icon":        trigger["icon"],
                "description": trigger["description"],
                "matches":     matched_keywords[:3],
            })
    return found


def classify(message: str) -> dict:
    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        proba   = torch.softmax(outputs.logits, dim=1)[0, 1].item()

    label    = "spam" if proba >= THRESHOLD else "ham"
    triggers = detect_triggers(message) if label == "spam" else []

    agent_result = get_smart_agent_explanation(
        message     = message,
        prediction  = label,
        probability = proba,
        triggers    = triggers,
    )

    return {
        "prediction":       label,
        "spam_probability": float(round(proba, 4)),
        "threshold":        THRESHOLD,
        "triggers":         triggers,
        "explanation":      agent_result["explanation"],
        "scam_type":        agent_result["scam_type"],
        "engine":           agent_result["engine"],
    }


app = FastAPI(title="OSA Email Risk Checker", version="4.0.0")
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
            "scam_type":        None,
            "engine":           None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, message: str = Form(...)):
    result = classify(message)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "message": message, **result},
    )


class EmailIn(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "DistilBERT", "version": "4.0.0"}


@app.post("/predict_email")
def predict_email(data: EmailIn):
    return classify(data.message)
