import os

# Keep inference predictable (and avoid thread/runtime issues on some systems)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

THRESHOLD = 0.30

model = joblib.load("email_spam_detection.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="OSA Email Risk API")


class EmailIn(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_email")
def predict_email(data: EmailIn):
    X = vectorizer.transform([data.message])
    proba = model.predict_proba(X)[0, 1]  # P(spam)
    label = "spam" if proba >= THRESHOLD else "ham"
    return {
        "prediction": label,
        "spam_probability": float(proba),
        "threshold": THRESHOLD,
    }

