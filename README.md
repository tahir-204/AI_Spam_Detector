# AI Spam Detector

Proyek PAS mata kuliah **AI Computing Platform** — Sistem Informasi, STIKOM Cipta Karya Informatika.

## Identitas

- **Nama** : Abdullah Safi'i Attahiri
- **NIM** : 24120120004
- **Jurusan** : Sistem Informasi

## Apa Ini?

Program AI sederhana yang membedakan pesan **SPAM** dan **BUKAN SPAM**.
Teknik yang dipakai: **Embedding + Cosine Similarity** — sesuai materi
kuliah Sesi 6 (Embedding) dan Sesi 8 (Vector Search).

## Cara Kerja (Singkat)

1. Setiap kalimat diubah menjadi **vektor angka 384 dimensi** (embedding).
   Kalimat yang maknanya mirip punya vektor yang berdekatan.
2. Untuk menebak pesan baru, program mencari contoh yang **paling mirip
   maknanya** menggunakan **Cosine Similarity**.
3. Label dari contoh termirip itulah jawabannya (SPAM / BUKAN).

Beda dengan Machine Learning klasik yang cuma menghitung kata, pendekatan
ini menangkap **makna** kalimat — jadi lebih dekat ke cara kerja AI
generatif (RAG) yang dipelajari di kuliah.

## Struktur File

```
AI_Spam_Detector/
├── data/dataset.csv    # contoh pesan berlabel
├── train.py            # tahap belajar: ubah contoh jadi vektor
├── predict.py          # tahap menebak: cari yang paling mirip
├── requirements.txt    # daftar pustaka
└── README.md           # file ini
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python train.py      # belajar dari dataset
python predict.py    # menebak pesan baru
```

> Saat pertama kali `train.py` dijalankan, model embedding (~80MB)
> diunduh otomatis. Agak lama 1–2 menit itu wajar, bukan error.

## Konsep dari Modul

| Sesi | Konsep | Dipakai di |
|------|--------|-----------|
| 6 | Embedding (teks → vektor) | `train.py` |
| 8 | Cosine Similarity (cari termirip) | `predict.py` |
