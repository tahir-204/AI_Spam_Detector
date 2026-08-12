"""
predict.py - Tahap MENEBAK (AI Spam Detector)
==============================================
Sesi 8 modul: Vector Search dengan Cosine Similarity.

Cara kerja: kalimat baru diubah jadi vektor, lalu dibandingkan
dengan semua vektor contoh yang sudah dipelajari. Kita cari contoh
yang PALING MIRIP maknanya (nilai cosine similarity tertinggi).
Label dari tetangga termirip itulah jawaban tebakan kita.

Jalankan:  python predict.py
"""

import pickle
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Pustaka belum lengkap. Jalankan dulu:")
    print("   pip install -r requirements.txt")
    raise SystemExit(1)

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DATA = "model_data.pkl"


def cosine_similarity(a, b):
    """Ukur kemiripan arah dua vektor. 1 = sangat mirip, 0 = tidak."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    pembagi = np.linalg.norm(a) * np.linalg.norm(b)
    if pembagi == 0:
        return 0.0
    return float(np.dot(a, b) / pembagi)


def muat_hasil_belajar():
    try:
        with open(MODEL_DATA, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("File hasil belajar tidak ada. Jalankan dulu:")
        print("   python train.py")
        raise SystemExit(1)


def tebak(kalimat, data, model):
    """Cari contoh termirip, kembalikan label + skor kemiripan."""
    vek_baru = model.encode([kalimat])[0]
    skor = [cosine_similarity(vek_baru, v) for v in data["vektor"]]
    idx = int(np.argmax(skor))
    return data["label"][idx], skor[idx], data["teks"][idx]


def main():
    data = muat_hasil_belajar()
    print("Memuat model embedding...")
    model = SentenceTransformer(MODEL_NAME)

    # Kalimat uji coba. Silakan ganti/tambah sesukamu.
    contoh_uji = [
        "Selamat anda menang hadiah undian jutaan rupiah klik sekarang",
        "Besok jangan lupa kumpul tugas kelompok di kampus ya",
        "Transfer dulu untuk klaim bonus saldo gratis anda hari ini",
        "Boleh pinjam catatan kuliahmu yang kemarin?",
    ]

    print("\n===== HASIL DETEKSI SPAM =====")
    for kal in contoh_uji:
        label, skor, mirip = tebak(kal, data, model)
        print(f"\nInput : {kal}")
        print(f"Hasil : {label}  (kemiripan {skor:.2f})")
        print(f"Mirip dengan contoh: \"{mirip}\"")


if __name__ == "__main__":
    main()
