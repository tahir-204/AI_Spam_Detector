"""
predict.py
----------
Script untuk mencoba model Spam Detector yang sudah dilatih.
Ketik sebuah pesan, lalu model akan menebak apakah itu SPAM atau HAM.

Cara pakai:
    python predict.py
"""

import joblib
import os

MODEL_PATH = "models/spam_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Pastikan model sudah dilatih terlebih dahulu
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Model belum ditemukan. Jalankan 'python train.py' terlebih dahulu.")
    exit()

# Load model dan vectorizer yang sudah disimpan
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_message(text: str) -> str:
    """Mengembalikan label prediksi: 'spam' atau 'ham'."""
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction


if __name__ == "__main__":
    print("=== AI Spam Detector ===")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        pesan = input("Masukkan pesan: ")
        if pesan.lower() == "exit":
            print("Sampai jumpa!")
            break

        hasil = predict_message(pesan)
        if hasil == "spam":
            print(">> Hasil: SPAM \n")
        else:
            print(">> Hasil: HAM (bukan spam) \n")
