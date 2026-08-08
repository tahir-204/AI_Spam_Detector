"""
train.py
--------
Script untuk melatih model AI Spam Detector.
Model belajar membedakan pesan SPAM dan HAM (bukan spam) dari data teks.

Cara pakai:
    python train.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load dataset
DATA_PATH = "data/dataset.csv"
df = pd.read_csv(DATA_PATH)

print(f"Total data: {len(df)}")
print(df["label"].value_counts())

# 2. Pisahkan fitur (teks) dan label (spam/ham)
X = df["text"]
y = df["label"]

# 3. Split data jadi data latih (train) dan data uji (test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Ubah teks menjadi angka (vectorisasi) pakai TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Latih model Naive Bayes (algoritma klasik untuk klasifikasi teks)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluasi model
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi model: {acc * 100:.2f}%")
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred))

# 7. Simpan model dan vectorizer supaya bisa dipakai lagi tanpa training ulang
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel berhasil disimpan di folder 'models/'")
