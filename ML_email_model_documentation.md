## Email Content ML Component – Documentation

### 1. Purpose

This document describes the Machine Learning component that analyzes **email/SMS content** to detect spam/phishing‑like messages for Project Online Scam Awareness (OSA). It covers:
- Data and preprocessing
- Model architecture and features
- Evaluation metrics and threshold selection
- FastAPI integration
- Planned extensions toward phishing + psychological triggers

---

### 2. Data and Preprocessing

- **Source file**: `spam.csv`
  - Columns: `Category` (`ham` / `spam`), `Message` (text).

- **Cleaning steps** (implemented in `spam email web app.ipynb`):
  - Load:
    - `df = pd.read_csv("spam.csv")`
  - Remove duplicate messages to avoid train/test leakage:
    - `df = df.drop_duplicates(subset=["Message"]).reset_index(drop=True)`
  - Map labels:
    - `df["label"] = df["Category"].map({"ham": 0, "spam": 1})`

- **Train/test split**:
  - Input/output:
    - `X = df["Message"].astype(str)`
    - `y = df["label"].astype(int)`
  - Split:
    - `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
  - `stratify=y` keeps the spam/ham proportion similar in train and test sets.

---

### 3. Feature Engineering and Model

Goal: capture both normal wording and obfuscated scam text (e.g. “ver1fy”, spaced characters).

- **Vectorizer**: `sklearn.pipeline.FeatureUnion` combining:

  1. **Word‑level TF‑IDF**
     - `analyzer="word"`
     - `ngram_range=(1, 2)` (unigrams + bigrams)
     - `min_df=2`
     - `max_df=0.95`
     - `sublinear_tf=True`
     - `strip_accents="unicode"`

  2. **Character‑level TF‑IDF**
     - `analyzer="char_wb"`
     - `ngram_range=(3, 5)`
     - `min_df=2`
     - `sublinear_tf=True`

- **Classifier**: Logistic Regression
  - `LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")`
  - `class_weight="balanced"` helps with spam/ham class imbalance.

- **Training pipeline** (simplified):
  - `X_train_vec = vectorizer.fit_transform(X_train)`
  - `X_test_vec = vectorizer.transform(X_test)`
  - `model.fit(X_train_vec, y_train)`

---

### 4. Evaluation and Threshold Selection

#### 4.1 Metrics

On the held‑out test set (20% split), we compute:
- `Accuracy`
- `Precision` (for spam class)
- `Recall` (for spam class)
- `F1` (for spam class)
- `PR‑AUC` (Average Precision) using `y_proba = model.predict_proba(X_test_vec)[:, 1]`

Example results (with threshold = 0.50):
- Accuracy: **0.9787**
- Spam precision: **0.9344**
- Spam recall: **0.8906**
- Spam F1: **0.9120**
- PR‑AUC: **0.9669**

> Note: PR‑AUC is preferred over accuracy for imbalanced problems like spam/phishing detection.

#### 4.2 Threshold Sweep and Security‑Oriented Operating Point

Rather than using the default 0.50 cutoff, we sweep thresholds from 0.10 to 0.90 and compute precision/recall/F1 for each:

```python
y_proba = model.predict_proba(X_test_vec)[:, 1]  # P(spam)

thresholds = np.linspace(0.1, 0.9, 17)
for thr in thresholds:
    y_thr = (y_proba >= thr).astype(int)
    prec = precision_score(y_test, y_thr)
    rec = recall_score(y_test, y_thr)
    f1 = f1_score(y_test, y_thr)
    ...
```

For a security‑sensitive setting (scam detection), we prefer **higher recall** (fewer missed scams) even if precision drops somewhat. A threshold around:
- **θ = 0.30**

provides a good trade‑off:
- High recall on spam messages
- Still acceptable precision (most alerts are correct)

The final decision rule is:
- If `P(spam) ≥ 0.30` → treat as spam/risky  
- Else → treat as ham/likely safe

---

### 5. FastAPI Integration (Smart Agent Endpoint)

The trained model and vectorizer are exported with `joblib`:

- `email_spam_detection.pkl` – trained `LogisticRegression` model
- `vectorizer.pkl` – fitted `FeatureUnion` TF‑IDF vectorizer

The notebook defines a FastAPI app with a `/predict_email` endpoint:

```python
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import uvicorn
import nest_asyncio

nest_asyncio.apply()

model = joblib.load("email_spam_detection.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

class Emaildet(BaseModel):
    message: str

@app.post("/predict_email")
def predict_email(data: Emaildet):
    msg = vectorizer.transform([data.message])
    proba = model.predict_proba(msg)[0, 1]  # P(spam)

    threshold = 0.30
    is_spam = proba >= threshold
    label = "spam" if is_spam else "ham"

    return {
        "prediction": label,
        "spam_probability": float(proba),
        "threshold": threshold,
    }
```

> For production use, it is recommended to run `uvicorn` from the command line on a `.py` file rather than inside the notebook.

Example (after exporting the notebook to `spam_email_web_app.py`):

```bash
uvicorn spam_email_web_app:app --host 0.0.0.0 --port 8000
```

---

### 6. Planned Extensions (Phishing + Psychological Triggers)

To align with Project OSA’s vision of psychological defense and community intelligence, future iterations will:

1. Switch from SMS spam data to a **phishing email dataset** (full emails: subject + body, plus legitimate emails).
2. Introduce richer labels:
   - `is_phishing` (binary intent)
   - Psychological triggers (multi‑label), e.g.:
     - `urgency`, `authority`, `fear_threat`, `reward_gain`,
       `credential_request`, `payment_request`, `secrecy_isolation`
3. Train a **BERT‑style model** to predict both intent and triggers.
4. Use these outputs in the Smart Agent UI to generate:
   - A refined **phishing risk score**
   - Plain‑language explanations (e.g. “This message is risky because it combines urgency, an authority claim, and a credential request”).

This baseline TF‑IDF + Logistic Regression model, together with the API and thresholding logic, provides a solid foundation to build and test the overall OSA pipeline while the more advanced phishing‑specific models and labels are being developed.

