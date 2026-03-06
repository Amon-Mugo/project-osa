# 🛡️ Project OSA — Online Scam Awareness
### A Multi-Layered AI Defense and Community Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

---

## 📌 Overview

Project OSA is a real-time AI-powered platform designed to help everyday users identify
spam, phishing, and scam messages. It acts as a **"Security Co-pilot"** — not just flagging
dangerous content, but explaining *why* it is dangerous using plain language that anyone
can understand.

The platform is built on three pillars:
1. **Detection** — NLP-based email/SMS risk scoring
2. **Analysis** — Psychological trigger identification and plain-language explanation
3. **Prevention** — Community-driven shared threat intelligence *(in development)*

---

## 🗂️ Project Structure

```
project-osa/
│
├── app.py                          # FastAPI app — web UI + JSON API
├── train_email_model.py            # Model training script (run once)
├── osa_email_api.py                # Standalone JSON API (no UI)
├── spam email web app.ipynb        # Research notebook (experimentation)
├── ML_email_model_documentation.md # Full ML component documentation
│
├── templates/
│   └── index.html                  # Jinja2 web UI template
│
├── static/
│   └── style.css                   # Web UI styling
│
├── .gitignore                      # Excludes datasets, models, cache
└── README.md                       # This file
```

> **Note:** Model files (`*.pkl`) and datasets (`*.csv`) are excluded from the repo
> via `.gitignore` due to file size. See setup instructions below to generate them locally.

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Amon-Mugo/project-osa.git
cd project-osa
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn joblib scikit-learn pandas jinja2 python-multipart
```

### 4. Add the datasets
Place these two files in the project root folder:
- `spam.csv` — SMS spam dataset (columns: `Category`, `Message`)
- `Phishing_Email.csv` — Phishing email dataset (columns: `Email Text`, `Email Type`)

> Contact the ML team if you need access to the datasets.

### 5. Train the model (once)
```bash
python train_email_model.py
```
This generates `email_spam_detection.pkl` and `vectorizer.pkl` in the project folder.

### 6. Run the app
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

---

## 🌐 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI — paste a message and check risk |
| `POST` | `/predict` | Form submission — returns HTML result |
| `POST` | `/predict_email` | JSON API — for programmatic use |
| `GET` | `/health` | Health check |

### JSON API Example

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict_email \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You have won a $1,000 gift card. Claim now!"}'
```

**Response:**
```json
{
  "prediction": "spam",
  "spam_probability": 0.9337,
  "threshold": 0.3,
  "triggers": [
    {
      "key": "reward_gain",
      "label": "Reward / Gain",
      "icon": "🎁",
      "description": "Lures you with prizes, gifts, or money you did not expect",
      "matches": ["congratulations", "won", "claim"]
    },
    {
      "key": "urgency",
      "label": "Urgency",
      "icon": "⏰",
      "description": "Pressures you to act immediately without thinking",
      "matches": ["now"]
    }
  ],
  "explanation": "This message is risky because it uses reward/gain and urgency. Scammers use these tactics to pressure people into acting without thinking. Do not click any links, share personal details, or send money."
}
```

---

## 🤖 ML Model Details

| Component | Details |
|-----------|---------|
| Algorithm | Logistic Regression |
| Vectorizer | FeatureUnion (Word TF-IDF + Character TF-IDF) |
| Training data | spam.csv + Phishing_Email.csv (22,694 rows) |
| Accuracy | 0.9797 |
| PR-AUC | 0.9940 |
| Threshold | 0.30 (tuned for high recall) |

### Psychological Triggers Detected
| Trigger | Description |
|---------|-------------|
| ⏰ Urgency | Pressures immediate action |
| 🎁 Reward / Gain | Lures with prizes or money |
| 🔑 Credential Request | Asks for personal or login details |
| ⚠️ Fear / Threat | Uses threats to force action |
| 💸 Payment Request | Asks to send money or financial info |

> Full documentation: `ML_email_model_documentation.md`

---

## 🗺️ Roadmap

- [x] Logistic Regression baseline model
- [x] Dual TF-IDF vectorizer (word + character level)
- [x] FastAPI web UI + JSON API
- [x] Psychological trigger detection
- [x] Plain-language explanations
- [ ] BERT/transformer model fine-tuning
- [ ] URL detector module
- [ ] Community threat reporting space
- [ ] Real-time threat map dashboard

---

## 👥 Team

| Member | Role |
|--------|------|
| Mariam Charles | Project Manager / Researcher |
| Benson Kivuva, Eugene Kipchirchir | Full-Stack Developers |
| Amon Mugo, Wickliff Momanyi | AI / ML Specialists |
| Paul Mwaura, Willie Karanja | Cybersecurity Lead / QA |

---

## 🌿 Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, working code only |
| `feature/bert-model` | BERT fine-tuning (ML team) |
| `feature/url-detector` | URL analysis module (Cybersecurity team) |
| `feature/community-space` | Community reporting UI (Full-stack team) |

**Workflow for all contributors:**
```bash
# 1. Always pull latest main before starting work
git pull origin main

# 2. Create your feature branch
git checkout -b feature/your-feature-name

# 3. Make changes, then commit
git add .
git commit -m "brief description of what you did"

# 4. Push your branch
git push origin feature/your-feature-name

# 5. Open a Pull Request on GitHub to merge into main
```

---

## 📄 License
MIT License — see LICENSE file for details.
