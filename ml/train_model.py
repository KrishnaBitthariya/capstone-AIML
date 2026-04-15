"""
NIDS ML Training Script
Uses NSL-KDD inspired synthetic dataset (same feature space as NSL-KDD)
Trains: Random Forest (AI model) vs Rule-Based (No AI baseline)
Detects: Normal, DoS, PortScan, BruteForce
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Generate NSL-KDD–style synthetic dataset ─────────────────────────────────
def generate_dataset(n_samples=8000):
    """
    Generates synthetic network traffic with 4 classes.
    Features mirror NSL-KDD key columns so the project is academically valid.
    """
    records = []

    def normal(n):
        return pd.DataFrame({
            'duration':        np.random.exponential(2, n),
            'src_bytes':       np.random.normal(5000, 2000, n).clip(0),
            'dst_bytes':       np.random.normal(4000, 1500, n).clip(0),
            'land':            np.zeros(n),
            'wrong_fragment':  np.zeros(n),
            'urgent':          np.zeros(n),
            'hot':             np.random.randint(0, 5, n).astype(float),
            'num_failed_logins': np.zeros(n),
            'logged_in':       np.ones(n),
            'num_compromised': np.zeros(n),
            'count':           np.random.randint(1, 50, n).astype(float),
            'srv_count':       np.random.randint(1, 50, n).astype(float),
            'serror_rate':     np.random.uniform(0, 0.1, n),
            'rerror_rate':     np.random.uniform(0, 0.1, n),
            'same_srv_rate':   np.random.uniform(0.8, 1.0, n),
            'diff_srv_rate':   np.random.uniform(0, 0.1, n),
            'dst_host_count':  np.random.randint(50, 255, n).astype(float),
            'dst_host_srv_count': np.random.randint(50, 255, n).astype(float),
            'label': 'Normal'
        })

    def dos(n):
        return pd.DataFrame({
            'duration':        np.zeros(n),
            'src_bytes':       np.random.normal(50000, 10000, n).clip(0),  # high traffic
            'dst_bytes':       np.random.normal(100, 50, n).clip(0),
            'land':            np.random.choice([0, 1], n, p=[0.7, 0.3]).astype(float),
            'wrong_fragment':  np.random.randint(0, 3, n).astype(float),
            'urgent':          np.zeros(n),
            'hot':             np.zeros(n),
            'num_failed_logins': np.zeros(n),
            'logged_in':       np.zeros(n),
            'num_compromised': np.zeros(n),
            'count':           np.random.randint(400, 512, n).astype(float),  # flood
            'srv_count':       np.random.randint(400, 512, n).astype(float),
            'serror_rate':     np.random.uniform(0.8, 1.0, n),
            'rerror_rate':     np.random.uniform(0, 0.1, n),
            'same_srv_rate':   np.random.uniform(0.9, 1.0, n),
            'diff_srv_rate':   np.random.uniform(0, 0.05, n),
            'dst_host_count':  np.random.randint(200, 255, n).astype(float),
            'dst_host_srv_count': np.random.randint(200, 255, n).astype(float),
            'label': 'DoS'
        })

    def portscan(n):
        return pd.DataFrame({
            'duration':        np.zeros(n),
            'src_bytes':       np.random.normal(100, 50, n).clip(0),
            'dst_bytes':       np.zeros(n),
            'land':            np.zeros(n),
            'wrong_fragment':  np.zeros(n),
            'urgent':          np.zeros(n),
            'hot':             np.zeros(n),
            'num_failed_logins': np.zeros(n),
            'logged_in':       np.zeros(n),
            'num_compromised': np.zeros(n),
            'count':           np.random.randint(1, 10, n).astype(float),
            'srv_count':       np.random.randint(1, 5, n).astype(float),
            'serror_rate':     np.random.uniform(0, 0.1, n),
            'rerror_rate':     np.random.uniform(0.8, 1.0, n),   # many rejects
            'same_srv_rate':   np.random.uniform(0, 0.1, n),
            'diff_srv_rate':   np.random.uniform(0.9, 1.0, n),   # many diff services
            'dst_host_count':  np.random.randint(1, 20, n).astype(float),
            'dst_host_srv_count': np.random.randint(200, 255, n).astype(float),
            'label': 'PortScan'
        })

    def bruteforce(n):
        return pd.DataFrame({
            'duration':        np.random.exponential(5, n),
            'src_bytes':       np.random.normal(1000, 300, n).clip(0),
            'dst_bytes':       np.random.normal(500, 200, n).clip(0),
            'land':            np.zeros(n),
            'wrong_fragment':  np.zeros(n),
            'urgent':          np.zeros(n),
            'hot':             np.random.randint(5, 20, n).astype(float),
            'num_failed_logins': np.random.randint(3, 10, n).astype(float),  # key signal
            'logged_in':       np.zeros(n),
            'num_compromised': np.random.randint(0, 5, n).astype(float),
            'count':           np.random.randint(10, 100, n).astype(float),
            'srv_count':       np.random.randint(10, 100, n).astype(float),
            'serror_rate':     np.random.uniform(0, 0.3, n),
            'rerror_rate':     np.random.uniform(0.5, 0.9, n),
            'same_srv_rate':   np.random.uniform(0.9, 1.0, n),
            'diff_srv_rate':   np.random.uniform(0, 0.1, n),
            'dst_host_count':  np.random.randint(1, 30, n).astype(float),
            'dst_host_srv_count': np.random.randint(1, 30, n).astype(float),
            'label': 'BruteForce'
        })

    n4 = n_samples // 4
    df = pd.concat([normal(n4), dos(n4), portscan(n4), bruteforce(n4)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ── Main training flow ────────────────────────────────────────────────────────
def train():
    print("📦 Generating NSL-KDD style dataset...")
    df = generate_dataset(8000)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset.csv", index=False)
    print(f"✅ Dataset saved → data/dataset.csv  ({len(df)} rows)")
    print(df['label'].value_counts())

    features = [c for c in df.columns if c != 'label']
    X = df[features]
    y = df['label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\n🤖 Training Random Forest (AI model)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    ai_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ AI Model Accuracy: {ai_accuracy*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Rule-based baseline (No AI) ───────────────────────────────────────────
    def rule_based_predict(X_df):
        preds = []
        for _, row in X_df.iterrows():
            if row['count'] > 300 or row['serror_rate'] > 0.7:
                preds.append('DoS')
            elif row['diff_srv_rate'] > 0.7 and row['rerror_rate'] > 0.6:
                preds.append('PortScan')
            elif row['num_failed_logins'] > 2:
                preds.append('BruteForce')
            else:
                preds.append('Normal')
        return preds

    X_test_df = pd.DataFrame(
        scaler.inverse_transform(X_test), columns=features
    )
    rule_preds = rule_based_predict(X_test_df)
    rule_preds_enc = le.transform(rule_preds)
    rule_accuracy = accuracy_score(y_test, rule_preds_enc)
    print(f"\n📏 Rule-Based (No AI) Accuracy: {rule_accuracy*100:.2f}%")

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("ml", exist_ok=True)
    with open("ml/model.pkl", "wb") as f:
        pickle.dump({
            'model': rf,
            'scaler': scaler,
            'label_encoder': le,
            'features': features,
            'ai_accuracy': ai_accuracy,
            'rule_accuracy': rule_accuracy,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'class_names': le.classes_.tolist()
        }, f)
    print("\n✅ Model saved → ml/model.pkl")
    print("\n🎯 Training complete!")


if __name__ == "__main__":
    train()
