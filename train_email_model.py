import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion


def load_spam() -> pd.DataFrame:
    """Load spam.csv → standard columns: text, label (0/1)."""
    df = pd.read_csv("spam.csv")
    df = df.drop_duplicates(subset=["Message"]).reset_index(drop=True)
    df["label"] = df["Category"].map({"ham": 0, "spam": 1})
    return df[["Message", "label"]].rename(columns={"Message": "text"})


def load_phishing() -> pd.DataFrame:
    """Load Phishing_Email.csv → standard columns: text, label (0/1)."""
    df = pd.read_csv("Phishing_Email.csv")
    # Drop unnamed index column if present
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    df = df.dropna(subset=["Email Text"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Email Text"]).reset_index(drop=True)
    df["label"] = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})
    return df[["Email Text", "label"]].rename(columns={"Email Text": "text"})


def main() -> None:
    # ── Load & combine both datasets ─────────────────────────────────────────
    print("Loading spam.csv ...")
    spam_df = load_spam()
    print(f"  spam.csv     → {len(spam_df):,} rows  "
          f"(spam: {spam_df['label'].sum():,} | ham: {(spam_df['label']==0).sum():,})")

    print("Loading Phishing_Email.csv ...")
    phish_df = load_phishing()
    print(f"  Phishing_Email.csv → {len(phish_df):,} rows  "
          f"(phishing: {phish_df['label'].sum():,} | safe: {(phish_df['label']==0).sum():,})")

    df = pd.concat([spam_df, phish_df], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df.dropna(subset=["text"])

    print(f"\nCombined dataset → {len(df):,} rows  "
          f"(malicious: {df['label'].sum():,} | safe: {(df['label']==0).sum():,})")

    # ── Split ─────────────────────────────────────────────────────────────────
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Vectorizer ────────────────────────────────────────────────────────────
    vectorizer = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    print("\nFitting vectorizer ...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Training model ...")
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )
    model.fit(X_train_vec, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    print("\n── Evaluation (threshold = 0.50) ──────────────────────────────")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"  PR-AUC    : {average_precision_score(y_test, y_proba):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(model,      "email_spam_detection.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("\n── Saved ───────────────────────────────────────────────────────")
    print("  email_spam_detection.pkl")
    print("  vectorizer.pkl")
    print("\nDone. Restart uvicorn to load the new model.")


if __name__ == "__main__":
    main()
