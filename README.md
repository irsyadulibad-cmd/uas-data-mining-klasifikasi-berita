# Klasifikasi Topik Berita Indonesia Tahun 2024-2025

Project ini dibuat untuk memenuhi tugas Ujian Akhir Semester mata kuliah Data Mining.

## Judul Project

Klasifikasi Topik Berita Indonesia Tahun 2024-2025 Menggunakan TF-IDF dan Algoritma Naive Bayes Berdasarkan Tag Berita.

## Dataset

Dataset yang digunakan adalah Indonesia News Dataset yang berisi data berita Indonesia dari beberapa media, seperti Kompas, Detik, dan Tempo.

Kolom yang digunakan dalam project ini:

- Judul
- Content
- tag1
- source
- Waktu
- Link

Pada penelitian ini, kolom Judul dan Content digabungkan sebagai teks berita, sedangkan tag1 digunakan sebagai label/topik berita.

## Metode

Tahapan penelitian:

1. Pengumpulan dataset
2. Seleksi atribut
3. Preprocessing teks
4. Pembobotan kata menggunakan TF-IDF
5. Pembagian data training dan testing
6. Klasifikasi menggunakan algoritma Multinomial Naive Bayes
7. Evaluasi model
8. Implementasi aplikasi menggunakan Streamlit

## Algoritma

Algoritma yang digunakan:

- TF-IDF Vectorizer
- Multinomial Naive Bayes

## Cara Menjalankan Aplikasi

Install library:

pip install -r requirements.txt

Jalankan aplikasi:

streamlit run app.py

## Output Aplikasi

Pengguna memasukkan teks berita, kemudian aplikasi akan menampilkan prediksi topik berita berdasarkan model yang sudah dilatih.
