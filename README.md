# AI Spam Detector

Program AI sederhana untuk mendeteksi apakah suatu pesan teks termasuk **SPAM** atau **HAM (bukan spam)**, menggunakan metode *Machine Learning* klasik: **TF-IDF Vectorization** dan **Naive Bayes Classifier**.

## Identitas

- Nama   : ABDULLAH SAFI'I ATTAHIRI
- NIM    : 24120120004
- Jurusan: SISTEM INFORMASI
- Semester: 4

## Deskripsi Project

Model AI ini dilatih menggunakan dataset teks berlabel (spam / ham). Model mempelajari pola kata yang sering muncul pada pesan spam (seperti "gratis", "klik link", "menangkan", dll), lalu digunakan untuk memprediksi label pesan baru.

## Struktur Folder

```
spam_detector/
├── data/
│   └── dataset.csv        # Dataset contoh (teks + label)
├── models/                 # Model hasil training disimpan di sini (otomatis dibuat)
├── train.py                # Script untuk melatih model
├── predict.py               # Script untuk mencoba model
├── requirements.txt         # Daftar library yang dibutuhkan
└── README.md
```

## Cara Menjalankan

1. Install library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

2. Latih model:
```bash
python train.py
```

3. Coba prediksi pesan baru:
```bash
python predict.py
```

## Contoh Penggunaan

```
Masukkan pesan: Selamat anda menang hadiah 10 juta klik link ini
>> Hasil: SPAM

Masukkan pesan: Halo bro besok jadi kumpul jam berapa
>> Hasil: HAM (bukan spam)
```

## Teknologi yang Digunakan

- Python 3
- Pandas — untuk mengolah data
- Scikit-learn — untuk vectorisasi teks (TF-IDF) dan algoritma Naive Bayes
- Joblib — untuk menyimpan model yang sudah dilatih
