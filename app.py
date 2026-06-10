import streamlit as st
import joblib
import re
import string
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# =====================================================
# LOAD MODEL DAN FILE PENDUKUNG
# =====================================================

@st.cache_resource
def load_model_files():
    model = joblib.load("model_berita.pkl")
    tfidf = joblib.load("vectorizer_tfidf.pkl")
    return model, tfidf

model, tfidf = load_model_files()


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


metrics = load_json("evaluation_metrics.json", {})
dataset_summary = load_json("dataset_summary.json", {})
classification_report_df = load_csv("classification_report.csv")
confusion_matrix_df = load_csv("confusion_matrix.csv")
tag_distribution_df = load_csv("tag_distribution.csv")
source_distribution_df = load_csv("source_distribution.csv")
monthly_distribution_df = load_csv("monthly_distribution.csv")
top_words_df = load_csv("top_words.csv")
algorithm_comparison_df = load_csv("algorithm_comparison.csv")

# =====================================================
# PREPROCESSING DAN PREDIKSI
# =====================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_mapping(label):
    mapping = {
        "kpk": "Hukum / Korupsi",
        "korupsi": "Hukum / Korupsi",
        "kejaksaan": "Hukum",
        "prabowo": "Politik / Pemerintahan",
        "prabowo subianto": "Politik / Pemerintahan",
        "jokowi": "Politik / Pemerintahan",
        "pilkada": "Politik / Pemilu",
        "pemilu": "Politik / Pemilu",
        "jakarta": "Daerah / Perkotaan",
        "surabaya": "Daerah",
        "jawa timur": "Daerah",
        "banjir": "Bencana",
        "gempa": "Bencana",
        "longsor": "Bencana",
        "kecelakaan": "Peristiwa",
        "info-tempo": "Berita Umum",
        "israel": "Internasional",
        "gaza": "Internasional",
        "palestina": "Internasional",
        "timnas indonesia": "Olahraga",
        "sepak bola": "Olahraga",
        "ekonomi": "Ekonomi",
    }
    label_lower = str(label).lower().strip()
    return mapping.get(label_lower, "Topik Berita Umum")


def interpret_confidence(score):
    if score >= 70:
        return "Tinggi", "Model cukup yakin karena teks memiliki pola kata yang kuat dan dekat dengan label hasil prediksi."
    if score >= 40:
        return "Sedang", "Model menemukan kecenderungan pada label tertentu, tetapi masih ada kemiripan dengan label lain."
    return "Rendah", "Teks kemungkinan terlalu pendek, terlalu umum, atau memiliki kata yang mirip dengan beberapa label."


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
        }).sort_values(by="Probabilitas", ascending=False).head(5)
        result_df["Probabilitas (%)"] = result_df["Probabilitas"] * 100
        result_df["Kategori Umum"] = result_df["Label Dataset"].apply(label_mapping)

    return prediction, label_mapping(prediction), clean, result_df

# =====================================================
# KONFIGURASI STREAMLIT
# =====================================================

st.set_page_config(
    page_title="Klasifikasi Topik Berita Indonesia",
    page_icon="📰",
    layout="wide"
)

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

# =====================================================
# BERANDA
# =====================================================

if menu == "Beranda":
    st.title("📰 Klasifikasi Topik Berita Indonesia")
    st.write(
        "Aplikasi ini digunakan untuk memprediksi topik atau tag berita Indonesia berdasarkan teks berita "
        "menggunakan metode **TF-IDF** dan algoritma **Multinomial Naive Bayes**."
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Algoritma", "Naive Bayes")
    with col2:
        st.metric("Fitur", "TF-IDF")
    with col3:
        st.metric("Jumlah Label", metrics.get("total_classes", dataset_summary.get("total_label", "-")))
    with col4:
        acc = metrics.get("accuracy", None)
        st.metric("Akurasi", f"{acc:.2%}" if acc is not None else "-")

    st.markdown("---")
    st.subheader("Latar Belakang")
    st.write(
        "Perkembangan media online membuat jumlah berita digital meningkat sangat cepat. Berita yang diterbitkan "
        "setiap hari memiliki topik yang beragam, mulai dari politik, hukum, ekonomi, bencana, daerah, hingga isu internasional. "
        "Pengelompokan berita secara manual membutuhkan waktu lama, sehingga diperlukan pendekatan otomatis menggunakan data mining."
    )

    st.subheader("Alur Sistem")
    alur_df = pd.DataFrame({
        "Tahap": ["Input", "Preprocessing", "TF-IDF", "Naive Bayes", "Output"],
        "Penjelasan": [
            "Pengguna memasukkan judul atau isi berita.",
            "Teks dibersihkan dari URL, angka, tanda baca, dan spasi berlebih.",
            "Teks diubah menjadi vektor angka berdasarkan bobot kata.",
            "Model mempelajari pola kata dan memprediksi label/topik.",
            "Sistem menampilkan label, kategori umum, probabilitas, dan analisis."
        ]
    })
    st.dataframe(alur_df, use_container_width=True)

    if monthly_distribution_df is not None and len(monthly_distribution_df) > 0:
        st.subheader("Ringkasan Tren Data Berita")
        st.line_chart(monthly_distribution_df.set_index("bulan")["jumlah"])
        st.write(
            "Grafik ini menunjukkan pergerakan jumlah berita berdasarkan bulan. Lonjakan pada periode tertentu dapat menunjukkan "
            "adanya isu besar atau peningkatan intensitas pemberitaan."
        )

    st.info(
        "Catatan: Label prediksi berasal dari kolom tag1 pada dataset. Hasil dapat berupa nama tokoh, lokasi, isu, kanal berita, "
        "atau tag khusus dari dataset."
    )

# =====================================================
# PREDIKSI SATU BERITA
# =====================================================

elif menu == "Prediksi Topik Berita":
    st.title("🔍 Prediksi Topik Berita")
    st.write("Masukkan judul atau isi berita, lalu klik tombol prediksi.")

    contoh = (
        "Pemerintah Provinsi Jawa Timur memperluas layanan transportasi publik melalui program Trans Jatim "
        "untuk meningkatkan mobilitas masyarakat."
    )

    input_text = st.text_area("Masukkan teks berita:", value=contoh, height=220)

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
                top_score = float(result_df.iloc[0]["Probabilitas (%)"])
                confidence_label, confidence_text = interpret_confidence(top_score)

                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Probabilitas Tertinggi", f"{top_score:.2f}%")
                with col4:
                    st.metric("Tingkat Keyakinan", confidence_label)

                st.subheader("5 Probabilitas Tertinggi")
                st.dataframe(result_df[["Label Dataset", "Kategori Umum", "Probabilitas (%)"]], use_container_width=True)
                st.bar_chart(result_df.set_index("Label Dataset")["Probabilitas (%)"])

                st.subheader("Analisis Hasil Prediksi")
                st.write(confidence_text)
                st.write(
                    "Jika probabilitas antar label berdekatan, artinya teks memiliki kata-kata yang juga sering muncul pada label lain. "
                    "Dalam konteks berita, hal ini wajar karena satu berita dapat memuat tokoh, lokasi, dan isu secara bersamaan."
                )

            with st.expander("Lihat Teks Setelah Preprocessing"):
                st.write(clean)

# =====================================================
# PREDIKSI BANYAK BERITA
# =====================================================

elif menu == "Prediksi Banyak Berita":
    st.title("📁 Prediksi Banyak Berita dari File CSV")
    st.write("Upload file CSV yang berisi kolom teks berita. Aplikasi akan memprediksi topik untuk setiap baris data.")
    st.info("Format CSV minimal memiliki satu kolom teks, misalnya: Judul, Content, text, berita, atau title.")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")
            st.success("File berhasil dibaca. Jika ada baris rusak, sistem akan melewatinya otomatis.")

            st.subheader("Preview Data")
            st.dataframe(data.head(), use_container_width=True)
            st.write(f"Jumlah data terbaca: **{len(data)} baris**")
            st.write(f"Jumlah kolom: **{len(data.columns)} kolom**")

            selected_column = st.selectbox("Pilih kolom yang berisi teks berita:", data.columns.tolist())
            jumlah_prediksi = st.slider("Jumlah data yang ingin diprediksi", min_value=1, max_value=min(500, len(data)), value=min(20, len(data)))

            if st.button("Prediksi Semua Data"):
                hasil = []
                data_prediksi = data.head(jumlah_prediksi)

                progress = st.progress(0)
                for i, text in enumerate(data_prediksi[selected_column].astype(str)):
                    prediction, kategori_umum, clean, result_df = predict_news(text)
                    confidence = float(result_df.iloc[0]["Probabilitas (%)"]) if result_df is not None else 0
                    hasil.append({
                        "Teks Berita": text,
                        "Label Dataset": prediction,
                        "Kategori Umum": kategori_umum,
                        "Probabilitas Tertinggi (%)": confidence
                    })
                    progress.progress((i + 1) / len(data_prediksi))

                hasil_df = pd.DataFrame(hasil)
                st.subheader("Hasil Prediksi")
                st.dataframe(hasil_df, use_container_width=True)

                st.subheader("Distribusi Hasil Prediksi")
                pred_count = hasil_df["Label Dataset"].value_counts()
                st.bar_chart(pred_count)

                dominant_label = pred_count.index[0]
                dominant_count = pred_count.iloc[0]
                st.subheader("Analisis Prediksi Banyak Data")
                st.write(
                    f"Dari **{len(hasil_df)}** data yang diprediksi, label paling dominan adalah **{dominant_label}** "
                    f"dengan jumlah **{dominant_count} berita**. Distribusi ini menunjukkan kecenderungan topik pada file yang diunggah."
                )

                csv = hasil_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Hasil Prediksi CSV", csv, "hasil_prediksi_berita.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi error saat membaca file: {e}")
            st.warning("Coba gunakan file CSV yang lebih kecil atau pastikan format kolomnya rapi.")

# =====================================================
# EVALUASI MODEL
# =====================================================

elif menu == "Evaluasi Model":
    st.title("📊 Evaluasi Model")
    st.write("Halaman ini menampilkan hasil evaluasi model klasifikasi berita menggunakan data testing.")
    st.markdown("---")

    accuracy = metrics.get("accuracy", None)
    macro_precision = metrics.get("macro_precision", None)
    macro_recall = metrics.get("macro_recall", None)
    macro_f1 = metrics.get("macro_f1", None)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}" if accuracy is not None else "-")
    with col2:
        st.metric("Macro Precision", f"{macro_precision:.2%}" if macro_precision is not None else "-")
    with col3:
        st.metric("Macro Recall", f"{macro_recall:.2%}" if macro_recall is not None else "-")
    with col4:
        st.metric("Macro F1-Score", f"{macro_f1:.2%}" if macro_f1 is not None else "-")

    st.markdown("---")
    st.subheader("Grafik Metrik Evaluasi")
    if metrics:
        metric_chart = pd.DataFrame({
            "Metrik": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "Weighted F1-Score"],
            "Nilai": [
                metrics.get("accuracy", 0) * 100,
                metrics.get("macro_precision", 0) * 100,
                metrics.get("macro_recall", 0) * 100,
                metrics.get("macro_f1", 0) * 100,
                metrics.get("weighted_f1", 0) * 100
            ]
        })
        st.bar_chart(metric_chart.set_index("Metrik"))
        st.write(
            "Accuracy menunjukkan persentase prediksi benar. Precision menunjukkan ketepatan prediksi model, "
            "recall menunjukkan kemampuan model menemukan data pada kelas tertentu, dan F1-score menunjukkan keseimbangan "
            "antara precision dan recall. Macro average memberi bobot sama pada semua label, sedangkan weighted average mempertimbangkan jumlah data tiap label."
        )
    else:
        st.warning("File evaluation_metrics.json belum tersedia. Jalankan kode pembuat file evaluasi di Google Colab.")

    st.markdown("---")
    st.subheader("Perbandingan Algoritma")
    if algorithm_comparison_df is not None:
        st.dataframe(algorithm_comparison_df, use_container_width=True)
        if "accuracy" in algorithm_comparison_df.columns:
            temp = algorithm_comparison_df.copy()
            temp["accuracy_percent"] = temp["accuracy"] * 100
            st.bar_chart(temp.set_index("model")["accuracy_percent"])
        st.write(
            "Perbandingan algoritma digunakan untuk melihat apakah Naive Bayes masih menjadi pilihan terbaik dibandingkan "
            "model lain seperti Logistic Regression dan Linear SVM. Jika model lain memiliki F1-score lebih tinggi, maka model tersebut "
            "dapat dipertimbangkan untuk pengembangan berikutnya."
        )
    else:
        st.info("File algorithm_comparison.csv belum tersedia. Menu ini tetap bisa digunakan tanpa perbandingan algoritma.")

    st.markdown("---")
    st.subheader("Classification Report")
    if classification_report_df is not None:
        report_show = classification_report_df.copy()
        if "Unnamed: 0" in report_show.columns:
            report_show = report_show.rename(columns={"Unnamed: 0": "label"})
        st.dataframe(report_show, use_container_width=True)

        report_plot = report_show.copy()
        if "label" in report_plot.columns:
            report_plot = report_plot[~report_plot["label"].isin(["accuracy", "macro avg", "weighted avg"])]
            if "f1-score" in report_plot.columns:
                report_plot["f1-score"] = pd.to_numeric(report_plot["f1-score"], errors="coerce")
                report_plot = report_plot.dropna(subset=["f1-score"])
                top_f1 = report_plot.sort_values("f1-score", ascending=False).head(10)
                low_f1 = report_plot.sort_values("f1-score", ascending=True).head(10)

                st.subheader("Top 10 Label dengan F1-Score Tertinggi")
                st.bar_chart(top_f1.set_index("label")["f1-score"])
                st.write("Label dengan F1-score tinggi biasanya memiliki pola kata yang khas dan jumlah data yang cukup.")

                st.subheader("10 Label dengan F1-Score Terendah")
                st.bar_chart(low_f1.set_index("label")["f1-score"])
                st.write(
                    "Label dengan F1-score rendah perlu diperhatikan karena bisa disebabkan data terlalu sedikit, label terlalu mirip, "
                    "atau teks berita yang kurang spesifik."
                )
    else:
        st.warning("File classification_report.csv belum tersedia.")

    st.markdown("---")
    st.subheader("Confusion Matrix")
    if confusion_matrix_df is not None:
        cm_display = confusion_matrix_df.copy()
        if "Unnamed: 0" in cm_display.columns:
            cm_display = cm_display.set_index("Unnamed: 0")

        max_classes = st.slider("Jumlah label yang ditampilkan", 5, min(30, len(cm_display)), min(10, len(cm_display)))
        cm_small = cm_display.iloc[:max_classes, :max_classes]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(cm_small.values)
        ax.set_xticks(np.arange(len(cm_small.columns)))
        ax.set_yticks(np.arange(len(cm_small.index)))
        ax.set_xticklabels(cm_small.columns, rotation=45, ha="right")
        ax.set_yticklabels(cm_small.index)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im)
        st.pyplot(fig)
        st.write(
            "Nilai diagonal menunjukkan prediksi benar. Nilai di luar diagonal menunjukkan kesalahan prediksi. "
            "Jika banyak nilai berkumpul pada diagonal, model bekerja cukup baik."
        )
    else:
        st.warning("File confusion_matrix.csv belum tersedia.")

    st.markdown("---")
    st.subheader("Analisis Evaluasi Model")
    st.write(
        "Model Naive Bayes dengan TF-IDF cocok untuk klasifikasi teks karena dapat memanfaatkan bobot kemunculan kata. "
        "Namun, performa model sangat dipengaruhi oleh kualitas label dan keseimbangan data. Label yang terlalu spesifik, terlalu sedikit, "
        "atau memiliki kemiripan istilah dengan label lain dapat menurunkan nilai precision, recall, dan F1-score."
    )

# =====================================================
# INFORMASI DATASET
# =====================================================

elif menu == "Informasi Dataset":
    st.title("📚 Informasi Dataset")
    st.write("Halaman ini menampilkan ringkasan dataset yang digunakan dalam proses training model.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data Awal", dataset_summary.get("total_data_awal", "-"))
    with col2:
        st.metric("Total Data Model", dataset_summary.get("total_data_model", "-"))
    with col3:
        st.metric("Total Label", dataset_summary.get("total_label", "-"))

    st.markdown("---")
    st.subheader("Kolom yang Digunakan")
    dataset_info = pd.DataFrame({
        "Kolom": ["Judul", "Content", "tag1", "source", "Waktu", "Link"],
        "Keterangan": ["Judul berita", "Isi atau konten berita", "Tag utama sebagai label/topik", "Sumber media berita", "Waktu publikasi", "URL berita"]
    })
    st.dataframe(dataset_info, use_container_width=True)

    st.markdown("---")
    st.subheader("Distribusi Topik/Tag Terbanyak")
    if tag_distribution_df is not None:
        top_n = st.slider("Jumlah tag yang ditampilkan", 5, min(30, len(tag_distribution_df)), min(10, len(tag_distribution_df)))
        top_tags = tag_distribution_df.head(top_n)
        st.bar_chart(top_tags.set_index("tag")["jumlah"])
        dominant_tag = top_tags.iloc[0]["tag"]
        dominant_count = top_tags.iloc[0]["jumlah"]
        st.write(
            f"Tag paling dominan adalah **{dominant_tag}** dengan **{dominant_count} data**. Tag yang dominan cenderung lebih mudah "
            "dipelajari oleh model, sedangkan tag dengan data sedikit berpotensi menghasilkan prediksi yang kurang stabil."
        )
    else:
        st.warning("File tag_distribution.csv belum tersedia.")

    st.markdown("---")
    st.subheader("Distribusi Sumber Berita")
    if source_distribution_df is not None:
        top_sources = source_distribution_df.head(10)
        st.bar_chart(top_sources.set_index("source")["jumlah"])
        st.write(
            "Grafik ini menunjukkan media atau sumber berita paling banyak dalam dataset. Distribusi sumber penting untuk melihat "
            "keberagaman data dan kemungkinan dominasi gaya penulisan dari media tertentu."
        )
    else:
        st.warning("File source_distribution.csv belum tersedia.")

    st.markdown("---")
    st.subheader("Tren Jumlah Berita Berdasarkan Waktu")
    if monthly_distribution_df is not None:
        st.line_chart(monthly_distribution_df.set_index("bulan")["jumlah"])
        st.write(
            "Grafik tren waktu menunjukkan perubahan jumlah berita berdasarkan bulan. Lonjakan jumlah berita dapat menunjukkan adanya "
            "isu besar atau peningkatan aktivitas pemberitaan pada periode tertentu."
        )
    else:
        st.warning("File monthly_distribution.csv belum tersedia.")

    st.markdown("---")
    st.subheader("Kata Paling Sering Muncul")
    if top_words_df is not None:
        st.bar_chart(top_words_df.head(20).set_index("kata")["jumlah"])
        st.write(
            "Kata yang sering muncul dapat menggambarkan fokus isu dalam dataset. Namun, kata yang terlalu umum sebaiknya dipertimbangkan "
            "sebagai stopword tambahan pada penelitian berikutnya."
        )
    else:
        st.info("File top_words.csv belum tersedia. Fitur ini opsional.")

    st.markdown("---")
    st.subheader("Tahapan Preprocessing")
    preprocessing_df = pd.DataFrame({
        "Tahapan": ["Lowercase", "Menghapus URL", "Menghapus angka", "Menghapus tanda baca", "Menghapus spasi berlebih", "TF-IDF"],
        "Tujuan": ["Menyamakan huruf", "Membersihkan tautan", "Mengurangi noise", "Menyederhanakan teks", "Merapikan teks", "Mengubah teks menjadi fitur numerik"]
    })
    st.dataframe(preprocessing_df, use_container_width=True)

# =====================================================
# TENTANG PROJECT
# =====================================================

elif menu == "Tentang Project":
    st.title("ℹ️ Tentang Project")
    st.write("Project ini dibuat untuk memenuhi tugas Ujian Akhir Semester mata kuliah Data Mining.")
    st.markdown("---")

    st.subheader("Judul Project")
    st.write("**Klasifikasi Topik Berita Indonesia Tahun 2024–2025 Menggunakan TF-IDF dan Algoritma Naive Bayes Berdasarkan Tag Berita**")

    st.subheader("Rumusan Masalah")
    st.write("Bagaimana menerapkan text mining untuk mengklasifikasikan topik berita Indonesia berdasarkan teks berita dan tag berita?")

    st.subheader("Metode yang Digunakan")
    st.write(
        "Metode yang digunakan meliputi preprocessing teks, pembobotan kata menggunakan TF-IDF, dan klasifikasi menggunakan "
        "algoritma Multinomial Naive Bayes."
    )

    st.subheader("Kelebihan Aplikasi")
    st.write(
        "- Dapat memprediksi topik berita dari input teks.\n"
        "- Menampilkan probabilitas lima label tertinggi.\n"
        "- Mendukung prediksi banyak berita dari CSV.\n"
        "- Menyediakan grafik distribusi, evaluasi model, dan analisis hasil.\n"
        "- Dapat digunakan sebagai prototipe dashboard klasifikasi berita."
    )

    st.subheader("Keterbatasan")
    st.write(
        "- Label berasal dari tag berita, bukan kategori umum resmi.\n"
        "- Beberapa label dapat berupa nama tokoh, lokasi, atau kanal berita.\n"
        "- Model dapat kurang akurat pada label dengan jumlah data sedikit.\n"
        "- Model masih menggunakan algoritma klasik dan belum memahami konteks sedalam model bahasa modern."
    )

    st.subheader("Pengembangan Selanjutnya")
    dev_df = pd.DataFrame({
        "Pengembangan": [
            "Pengelompokan tag menjadi kategori umum",
            "Penyeimbangan data tiap kelas",
            "Perbandingan algoritma",
            "Penggunaan IndoBERT atau deep learning",
            "Analisis sentimen berita",
            "Dashboard monitoring berita real-time",
            "Filter sumber, tanggal, dan topik",
            "Ekspor laporan otomatis"
        ],
        "Manfaat": [
            "Hasil prediksi lebih mudah dipahami pengguna umum.",
            "Model lebih adil terhadap label kecil dan tidak hanya dominan pada label besar.",
            "Menemukan model terbaik berdasarkan accuracy, precision, recall, dan F1-score.",
            "Meningkatkan pemahaman konteks bahasa Indonesia.",
            "Mengetahui kecenderungan nada berita: positif, negatif, atau netral.",
            "Aplikasi dapat memantau isu terbaru secara otomatis.",
            "Pengguna dapat menganalisis berita secara lebih spesifik.",
            "Hasil analisis dapat langsung diunduh untuk kebutuhan laporan."
        ]
    })
    st.dataframe(dev_df, use_container_width=True)

    st.write(
        "Dengan pengembangan tersebut, aplikasi tidak hanya berfungsi sebagai alat klasifikasi teks, tetapi juga dapat dikembangkan "
        "menjadi dashboard analisis berita yang lebih lengkap, informatif, dan relevan untuk kebutuhan monitoring isu."
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("UAS Data Mining - Klasifikasi Topik Berita Indonesia Menggunakan TF-IDF dan Multinomial Naive Bayes")
