"""
train.py - Tahap BELAJAR (AI Spam Detector)
============================================
Sesi 6 & 8 modul: Embedding + Vector Search.

Ide besarnya: tiap kalimat diubah jadi VEKTOR ANGKA 384 dimensi
(embedding). Kalimat yang maknanya mirip akan punya vektor yang
berdekatan. Di sini kita ubah semua contoh di dataset jadi vektor,
lalu simpan ke file supaya nanti dipakai untuk menebak (predict.py).

Jalankan:  python train.py
"""

import csv
import pickle

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Pustaka belum lengkap. Jalankan dulu:")
    print("   pip install -r requirements.txt")
    raise SystemExit(1)

DATA_PATH = "data/dataset.csv"
MODEL_NAME = "all-MiniLM-L6-v2"   # model kecil, jalan di CPU
OUTPUT = "model_data.pkl"


def baca_dataset(path):
    """Baca CSV jadi dua list: kalimat dan labelnya."""
    teks, label = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for baris in reader:
            teks.append(baris["teks"])
            label.append(baris["label"])
    return teks, label


def main():
    print("Membaca dataset...")
    teks, label = baca_dataset(DATA_PATH)
    print(f"   {len(teks)} contoh dimuat.")

    # Sesi 3-6: memuat model lalu mengubah teks -> vektor (embedding).
    # Saat PERTAMA dijalankan, model (~80MB) diunduh otomatis. Wajar
    # kalau agak lama 1-2 menit, itu bukan error.
    print("Memuat model embedding (unduh otomatis saat pertama kali)...")
    model = SentenceTransformer(MODEL_NAME)

    print("Mengubah semua contoh menjadi vektor...")
    vektor = model.encode(teks, show_progress_bar=True)

    # Simpan vektor + label. Inilah "hasil belajar" AI kita.
    with open(OUTPUT, "wb") as f:
        pickle.dump({"vektor": vektor, "label": label, "teks": teks}, f)

    print(f"\nSelesai! Hasil belajar disimpan di '{OUTPUT}'.")
    print("Lanjut jalankan:  python predict.py")


if __name__ == "__main__":
    main()
