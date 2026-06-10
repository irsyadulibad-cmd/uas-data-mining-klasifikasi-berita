import streamlit as st
import joblib
import re
import string
import pandas as pd
import numpy as np

# =========================
# LOAD MODEL
# =========================

model = joblib.load("model_berita.pkl")
tfidf = joblib.load("vectorizer_tfidf.pkl")


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# LABEL MAPPING
# =========================

def label_mapping(label):
    mapping = {
        "kpk": "Hukum / Korupsi",
        "prabowo": "Politik / Pemerintahan",
        "prabowo subianto": "Politik / Pemerintahan",
        "jokowi": "Politik / Pemerintahan",
        "jakarta": "Daerah / Perkotaan",
        "banjir": "Bencana",
        "gempa": "Bencana",
        "kecelakaan": "Peristiwa",
        "info-tempo": "Berita Umum",
        "israel": "Internasional",
        "gaza": "Internasional",
        "palestina": "Internasional",
        "pilkada": "Politik / Pemilu",
        "pemilu": "Politik / Pemilu",
    }

    label_lower = str(label).lower().strip()
    return mapping.get(label_lower, "Topik Berita Umum")


# =========================
# PREDICTION FUNCTION
# =========================

def predict_news(text):
    clean = clean_text(text)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]

    result_df = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        classes = model.classes_

        result_df = pd.DataFrame({
            "Label Dataset": classes,
            "Probabilitas": proba
        })

        result_df = result_df.sort_values(
            by="Probabilitas",
            ascending=False
        ).head(5)

        result_df["Probabilitas (%)"] = result_df["Probabilitas"] * 100
        result_df["Kategori Umum"] = result_df["Label Dataset"].apply(label_mapping)

    return prediction, label_mapping(prediction), clean, result_df


# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="Klasifikasi Topik Berita Indonesia",
    page_icon="📰",
    layout="wide"
)


# =========================
# SIDEBAR
# =========================

st.sidebar.title("📰 Data Mining Berita")
st.sidebar.write("Klasifikasi topik berita Indonesia menggunakan TF-IDF dan Naive Bayes.")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Beranda",
        "Prediksi Topik Berita",
        "Prediksi Banyak Berita",
        "Evaluasi Model",
        "Informasi Dataset",
        "Tentang Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Metode")
st.sidebar.write("1. Text Preprocessing")
st.sidebar.write("2. TF-IDF Vectorizer")
st.sidebar.write("3. Multinomial Naive Bayes")

st.sidebar.markdown("---")
st.sidebar.caption("UAS Data Mining 2026")


# =========================
# PAGE: BERANDA
# =========================

if menu == "Beranda":
    st.title("📰 Klasifikasi Topik Berita Indonesia")
    st.write(
        "Aplikasi ini dibuat untuk memprediksi topik berita Indonesia berdasarkan teks berita "
        "menggunakan metode **TF-IDF** dan algoritma **Multinomial Naive Bayes**."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Metode", "TF-IDF")

    with col2:
        st.metric("Algoritma", "Naive Bayes")

    with col3:
        st.metric("Output", "Topik Berita")

    st.markdown("---")

    st.subheader("Latar Belakang")
    st.write(
        "Perkembangan media online menyebabkan jumlah berita digital meningkat sangat cepat. "
        "Banyaknya berita yang diterbitkan setiap hari membuat proses pengelompokan topik secara manual "
        "menjadi kurang efisien. Oleh karena itu, diperlukan pendekatan data mining berbasis text mining "
        "untuk membantu mengklasifikasikan berita secara otomatis."
    )

    st.subheader("Tujuan Aplikasi")
    st.write(
        "Aplikasi ini bertujuan untuk membantu memprediksi topik berita berdasarkan judul atau isi berita "
        "yang dimasukkan pengguna. Sistem akan melakukan preprocessing teks, mengubah teks menjadi fitur numerik "
        "menggunakan TF-IDF, lalu memprediksi topik menggunakan model Naive Bayes."
    )

    st.info(
        "Catatan: Label prediksi berasal dari kolom tag1 pada dataset, sehingga hasil dapat berupa tag seperti "
        "kpk, jakarta, prabowo, info-tempo, dan tag lainnya."
    )


# =========================
# PAGE: PREDIKSI SATU BERITA
# =========================

elif menu == "Prediksi Topik Berita":
    st.title("🔍 Prediksi Topik Berita")
    st.write("Masukkan judul atau isi berita, lalu klik tombol prediksi.")

    contoh = (
        "Pemerintah Provinsi Jawa Timur memperluas layanan transportasi publik "
        "melalui program Trans Jatim untuk meningkatkan mobilitas masyarakat."
    )

    input_text = st.text_area(
        "Masukkan teks berita:",
        value=contoh,
        height=220
    )

    if st.button("Prediksi Topik Berita"):
        if input_text.strip() == "":
            st.warning("Silakan masukkan teks berita terlebih dahulu.")
        else:
            prediction, kategori_umum, clean, result_df = predict_news(input_text)

            st.markdown("---")
            st.subheader("Hasil Prediksi")

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"Label Dataset: **{prediction}**")

            with col2:
                st.info(f"Kategori Umum: **{kategori_umum}**")

            if result_df is not None:
                st.subheader("5 Probabilitas Tertinggi")

                st.dataframe(
                    result_df[["Label Dataset", "Kategori Umum", "Probabilitas (%)"]],
                    use_container_width=True
                )

                chart_df = result_df.set_index("Label Dataset")["Probabilitas (%)"]
                st.bar_chart(chart_df)

            with st.expander("Lihat Teks Setelah Preprocessing"):
                st.write(clean)


# =========================
# PAGE: PREDIKSI BANYAK BERITA
# =========================

elif menu == "Prediksi Banyak Berita":
    st.title("📁 Prediksi Banyak Berita dari File CSV")
    st.write(
        "Upload file CSV yang berisi kolom teks berita. Aplikasi akan memprediksi topik untuk setiap baris data."
    )

    st.info(
        "Format CSV minimal memiliki satu kolom teks, misalnya: Judul, Content, text, berita, atau title."
    )

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Membaca CSV dengan mode aman
            data = pd.read_csv(
                uploaded_file,
                engine="python",
                on_bad_lines="skip"
            )

            st.success("File berhasil dibaca. Baris yang formatnya rusak otomatis dilewati.")

            st.subheader("Preview Data")
            st.dataframe(data.head(), use_container_width=True)

            st.write(f"Jumlah data terbaca: **{len(data)} baris**")
            st.write(f"Jumlah kolom: **{len(data.columns)} kolom**")

            text_columns = data.columns.tolist()

            selected_column = st.selectbox(
                "Pilih kolom yang berisi teks berita:",
                text_columns
            )

            jumlah_prediksi = st.slider(
                "Jumlah data yang ingin diprediksi",
                min_value=1,
                max_value=min(500, len(data)),
                value=min(20, len(data))
            )

            if st.button("Prediksi Semua Data"):
                hasil = []

                data_prediksi = data.head(jumlah_prediksi)

                for text in data_prediksi[selected_column].astype(str):
                    prediction, kategori_umum, clean, result_df = predict_news(text)

                    hasil.append({
                        "Teks Berita": text,
                        "Label Dataset": prediction,
                        "Kategori Umum": kategori_umum
                    })

                hasil_df = pd.DataFrame(hasil)

                st.subheader("Hasil Prediksi")
                st.dataframe(hasil_df, use_container_width=True)

                csv = hasil_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Hasil Prediksi CSV",
                    data=csv,
                    file_name="hasil_prediksi_berita.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi error saat membaca file: {e}")
            st.warning(
                "Coba gunakan file CSV yang lebih kecil atau pastikan file memiliki format kolom yang rapi."
            )
# =========================
# PAGE: EVALUASI MODEL
# =========================

elif menu == "Evaluasi Model":
    st.title("📊 Evaluasi Model")
    st.write(
        "Halaman ini menjelaskan evaluasi model klasifikasi berita yang digunakan dalam aplikasi."
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", "Naive Bayes")

    with col2:
        st.metric("Fitur", "TF-IDF")

    with col3:
        st.metric("Split Data", "80:20")

    with col4:
        st.metric("Output", "Tag Berita")

    st.markdown("---")

    st.subheader("Metrik Evaluasi")
    st.write(
        "Evaluasi model dilakukan dengan menggunakan beberapa metrik, yaitu akurasi, precision, recall, "
        "f1-score, dan confusion matrix. Metrik tersebut digunakan untuk melihat sejauh mana model mampu "
        "memprediksi topik berita dengan benar."
    )

    st.write(
        "Nilai evaluasi lengkap diperoleh dari proses training dan testing di Google Colab. "
        "Hasil tersebut dapat dilihat pada notebook yang disertakan dalam repository project."
    )

    st.subheader("Interpretasi")
    st.write(
        "Jika nilai akurasi model tinggi, maka model mampu mengklasifikasikan sebagian besar data uji "
        "dengan benar. Namun, jika terdapat kategori dengan jumlah data sedikit, performa model pada kategori "
        "tersebut dapat lebih rendah. Hal ini wajar karena model lebih mudah mempelajari pola dari kategori "
        "yang memiliki jumlah data lebih banyak."
    )

    st.warning(
        "Catatan: Karena label berasal dari tag berita, beberapa label dapat bersifat sangat spesifik, "
        "misalnya nama tokoh, lokasi, atau nama kanal berita."
    )


# =========================
# PAGE: INFORMASI DATASET
# =========================

elif menu == "Informasi Dataset":
    st.title("📚 Informasi Dataset")
    st.write(
        "Dataset yang digunakan adalah Indonesia News Dataset yang berisi data berita Indonesia dari beberapa media."
    )

    st.markdown("---")

    st.subheader("Kolom yang Digunakan")
    dataset_info = pd.DataFrame({
        "Kolom": ["Judul", "Content", "tag1", "source", "Waktu", "Link"],
        "Keterangan": [
            "Judul berita",
            "Isi atau konten berita",
            "Tag utama yang digunakan sebagai label/topik",
            "Sumber media berita",
            "Waktu publikasi berita",
            "URL berita"
        ]
    })

    st.dataframe(dataset_info, use_container_width=True)

    st.subheader("Proses Pembentukan Data Model")
    st.write(
        "Kolom **Judul** dan **Content** digabungkan menjadi satu teks berita. "
        "Kolom **tag1** digunakan sebagai label atau target klasifikasi. "
        "Data kemudian dibersihkan melalui proses preprocessing sebelum diubah menjadi fitur numerik menggunakan TF-IDF."
    )

    st.subheader("Tahapan Preprocessing")
    preprocessing_df = pd.DataFrame({
        "Tahapan": [
            "Lowercase",
            "Menghapus URL",
            "Menghapus angka",
            "Menghapus tanda baca",
            "Menghapus spasi berlebih"
        ],
        "Tujuan": [
            "Menyamakan bentuk huruf",
            "Membersihkan tautan dari teks",
            "Mengurangi noise",
            "Menyederhanakan teks",
            "Merapikan format teks"
        ]
    })

    st.dataframe(preprocessing_df, use_container_width=True)


# =========================
# PAGE: TENTANG PROJECT
# =========================

elif menu == "Tentang Project":
    st.title("ℹ️ Tentang Project")

    st.write(
        "Project ini dibuat untuk memenuhi tugas Ujian Akhir Semester mata kuliah Data Mining."
    )

    st.markdown("---")

    st.subheader("Judul Project")
    st.write(
        "**Klasifikasi Topik Berita Indonesia Tahun 2024–2025 Menggunakan TF-IDF dan Algoritma Naive Bayes Berdasarkan Tag Berita**"
    )

    st.subheader("Rumusan Masalah")
    st.write(
        "Bagaimana menerapkan metode text mining untuk mengklasifikasikan topik berita Indonesia "
        "berdasarkan teks berita dan tag berita?"
    )

    st.subheader("Metode yang Digunakan")
    st.write(
        "Metode yang digunakan dalam project ini adalah preprocessing teks, pembobotan kata dengan TF-IDF, "
        "dan klasifikasi menggunakan algoritma Multinomial Naive Bayes."
    )

    st.subheader("Output")
    st.write(
        "Output dari aplikasi ini adalah prediksi label dataset, kategori umum, serta probabilitas topik tertinggi."
    )

    st.subheader("Keterbatasan")
    st.write(
        "Aplikasi ini masih memiliki keterbatasan karena label yang digunakan berasal dari tag berita, "
        "bukan kategori umum seperti politik, ekonomi, olahraga, atau teknologi. Oleh karena itu, beberapa hasil "
        "prediksi dapat berupa nama tokoh, nama tempat, atau tag khusus dari dataset."
    )

    st.subheader("Pengembangan Selanjutnya")
    st.write(
        "Pengembangan selanjutnya dapat dilakukan dengan menyeimbangkan jumlah data tiap kategori, "
        "mengelompokkan tag menjadi kategori umum, serta membandingkan performa algoritma lain seperti "
        "Support Vector Machine, Logistic Regression, Random Forest, atau deep learning."
    )


# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption(
    "UAS Data Mining - Klasifikasi Topik Berita Indonesia Menggunakan TF-IDF dan Multinomial Naive Bayes"
)
