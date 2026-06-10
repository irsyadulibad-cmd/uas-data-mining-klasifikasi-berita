import streamlit as st
import joblib
import re
import string
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL
# =========================

model = joblib.load("model_berita.pkl")
tfidf = joblib.load("vectorizer_tfidf.pkl")


# =========================
# HELPER LOAD FILE
# =========================

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


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
        "Aplikasi ini digunakan untuk memprediksi topik atau tag berita Indonesia "
        "berdasarkan teks berita menggunakan metode **TF-IDF** dan algoritma "
        "**Multinomial Naive Bayes**."
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Algoritma", "Naive Bayes")

    with col2:
        st.metric("Fitur", "TF-IDF")

    with col3:
        st.metric("Jumlah Label", metrics.get("total_classes", "-"))

    with col4:
        acc = metrics.get("accuracy", None)
        st.metric("Akurasi", f"{acc:.2%}" if acc is not None else "-")

    st.markdown("---")

    st.subheader("Latar Belakang")
    st.write(
        "Perkembangan media online membuat jumlah berita digital meningkat sangat cepat. "
        "Berita yang diterbitkan setiap hari memiliki topik yang beragam, mulai dari politik, "
        "hukum, ekonomi, bencana, daerah, hingga isu internasional. Pengelompokan berita secara manual "
        "membutuhkan waktu yang lama, sehingga diperlukan pendekatan otomatis menggunakan data mining."
    )

    st.subheader("Tujuan Aplikasi")
    st.write(
        "Aplikasi ini bertujuan untuk membantu mengklasifikasikan topik berita berdasarkan teks. "
        "Pengguna dapat memasukkan judul atau isi berita, kemudian sistem akan melakukan preprocessing, "
        "transformasi TF-IDF, dan prediksi menggunakan model Naive Bayes."
    )

    st.info(
        "Catatan: Label prediksi berasal dari kolom tag1 pada dataset. Karena itu, hasil prediksi "
        "bisa berupa nama tokoh, lokasi, isu, kanal berita, atau tag khusus dari dataset."
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

                st.subheader("Analisis Hasil Prediksi")
                top_score = result_df.iloc[0]["Probabilitas (%)"]

                if top_score >= 70:
                    st.write(
                        "Model memiliki tingkat keyakinan yang tinggi terhadap hasil prediksi. "
                        "Hal ini menunjukkan bahwa teks input memiliki pola kata yang cukup kuat "
                        "dan sesuai dengan label yang diprediksi."
                    )
                elif top_score >= 40:
                    st.write(
                        "Model memiliki tingkat keyakinan sedang. Artinya, teks berita memiliki kecenderungan "
                        "ke label tertentu, tetapi masih terdapat kemungkinan kemiripan dengan label lain."
                    )
                else:
                    st.write(
                        "Model memiliki tingkat keyakinan rendah. Hal ini dapat terjadi karena teks terlalu pendek, "
                        "topik terlalu umum, atau terdapat kemiripan kata dengan beberapa label lain."
                    )

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
            data = pd.read_csv(
                uploaded_file,
                engine="python",
                on_bad_lines="skip"
            )

            st.success("File berhasil dibaca. Jika ada baris rusak, sistem akan melewatinya otomatis.")

            st.subheader("Preview Data")
            st.dataframe(data.head(), use_container_width=True)

            st.write(f"Jumlah data terbaca: **{len(data)} baris**")
            st.write(f"Jumlah kolom: **{len(data.columns)} kolom**")

            selected_column = st.selectbox(
                "Pilih kolom yang berisi teks berita:",
                data.columns.tolist()
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

                st.subheader("Distribusi Hasil Prediksi")
                pred_count = hasil_df["Label Dataset"].value_counts()
                st.bar_chart(pred_count)

                st.subheader("Analisis Prediksi Banyak Data")
                dominant_label = pred_count.index[0]
                dominant_count = pred_count.iloc[0]

                st.write(
                    f"Dari {len(hasil_df)} data yang diprediksi, label paling dominan adalah "
                    f"**{dominant_label}** dengan jumlah **{dominant_count} berita**. "
                    "Distribusi ini menunjukkan kecenderungan topik berita pada file yang diunggah."
                )

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
                "Coba gunakan file CSV yang lebih kecil atau pastikan format kolomnya rapi."
            )


# =========================
# PAGE: EVALUASI MODEL
# =========================

elif menu == "Evaluasi Model":
    st.title("📊 Evaluasi Model")
    st.write(
        "Halaman ini menampilkan hasil evaluasi model klasifikasi berita menggunakan data testing."
    )

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
            "Metrik": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"],
            "Nilai": [
                metrics.get("accuracy", 0) * 100,
                metrics.get("macro_precision", 0) * 100,
                metrics.get("macro_recall", 0) * 100,
                metrics.get("macro_f1", 0) * 100
            ]
        })

        st.bar_chart(metric_chart.set_index("Metrik"))

        st.write(
            "Grafik di atas menunjukkan performa umum model. Accuracy menggambarkan jumlah prediksi benar "
            "dibandingkan seluruh data uji. Precision menunjukkan ketepatan prediksi model, recall menunjukkan "
            "kemampuan model menemukan data pada setiap kelas, sedangkan F1-score merupakan keseimbangan antara "
            "precision dan recall."
        )
    else:
        st.warning("File evaluation_metrics.json belum tersedia.")

    st.markdown("---")

    st.subheader("Classification Report")

    if classification_report_df is not None:
        st.dataframe(classification_report_df, use_container_width=True)

        report_plot = classification_report_df.copy()

        if "Unnamed: 0" in report_plot.columns:
            report_plot = report_plot.rename(columns={"Unnamed: 0": "label"})

        report_plot = report_plot[
            ~report_plot["label"].isin(["accuracy", "macro avg", "weighted avg"])
        ]

        if "f1-score" in report_plot.columns:
            report_plot["f1-score"] = pd.to_numeric(report_plot["f1-score"], errors="coerce")
            report_plot = report_plot.dropna(subset=["f1-score"])
            report_plot = report_plot.sort_values("f1-score", ascending=False).head(10)

            st.subheader("Top 10 Label dengan F1-Score Tertinggi")
            st.bar_chart(report_plot.set_index("label")["f1-score"])

            st.write(
                "Label dengan F1-score tinggi menunjukkan bahwa model mampu mengenali pola kata pada label tersebut "
                "dengan baik. Biasanya hal ini terjadi karena jumlah data pada label tersebut cukup banyak dan kata-kata "
                "yang muncul memiliki ciri khas yang jelas."
            )

    else:
        st.warning("File classification_report.csv belum tersedia.")

    st.markdown("---")

    st.subheader("Confusion Matrix")

    if confusion_matrix_df is not None:
        cm_display = confusion_matrix_df.copy()

        if "Unnamed: 0" in cm_display.columns:
            cm_display = cm_display.set_index("Unnamed: 0")

        max_classes = st.slider(
            "Jumlah label yang ditampilkan pada confusion matrix",
            min_value=5,
            max_value=min(30, len(cm_display)),
            value=min(10, len(cm_display))
        )

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
            "Confusion matrix menunjukkan perbandingan antara label aktual dan label hasil prediksi. "
            "Nilai diagonal menunjukkan jumlah prediksi yang benar, sedangkan nilai di luar diagonal menunjukkan "
            "kesalahan prediksi. Jika banyak nilai berada pada diagonal, maka model bekerja dengan baik."
        )

    else:
        st.warning("File confusion_matrix.csv belum tersedia.")

    st.markdown("---")

    st.subheader("Analisis Evaluasi Model")
    st.write(
        "Berdasarkan proses evaluasi, model Naive Bayes dengan TF-IDF dapat digunakan untuk klasifikasi "
        "teks berita karena mampu mempelajari pola kemunculan kata pada setiap tag. Namun, performa model "
        "sangat dipengaruhi oleh kualitas label dan keseimbangan jumlah data. Jika terdapat label yang terlalu "
        "spesifik atau jumlah datanya sedikit, model cenderung lebih sulit mengenali label tersebut."
    )


# =========================
# PAGE: INFORMASI DATASET
# =========================

elif menu == "Informasi Dataset":
    st.title("📚 Informasi Dataset")
    st.write(
        "Halaman ini menampilkan ringkasan dataset yang digunakan dalam proses training model."
    )

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

    st.markdown("---")

    st.subheader("Distribusi Topik/Tag Terbanyak")

    if tag_distribution_df is not None:
        top_n = st.slider(
            "Jumlah topik/tag yang ditampilkan",
            min_value=5,
            max_value=min(30, len(tag_distribution_df)),
            value=min(10, len(tag_distribution_df))
        )

        top_tags = tag_distribution_df.head(top_n)
        st.bar_chart(top_tags.set_index("tag")["jumlah"])

        st.write(
            "Grafik ini menunjukkan tag yang paling banyak muncul dalam dataset. Tag dengan jumlah data tinggi "
            "lebih mudah dipelajari oleh model karena memiliki lebih banyak contoh teks."
        )
    else:
        st.warning("File tag_distribution.csv belum tersedia.")

    st.markdown("---")

    st.subheader("Distribusi Sumber Berita")

    if source_distribution_df is not None:
        top_sources = source_distribution_df.head(10)
        st.bar_chart(top_sources.set_index("source")["jumlah"])

        st.write(
            "Grafik ini menunjukkan media atau sumber berita yang paling banyak muncul dalam dataset. "
            "Distribusi sumber penting untuk melihat keberagaman data yang digunakan."
        )
    else:
        st.warning("File source_distribution.csv belum tersedia.")

    st.markdown("---")

    st.subheader("Tren Jumlah Berita Berdasarkan Waktu")

    if monthly_distribution_df is not None:
        st.line_chart(monthly_distribution_df.set_index("bulan")["jumlah"])

        st.write(
            "Grafik tren waktu menunjukkan perubahan jumlah berita berdasarkan bulan. "
            "Jika terdapat lonjakan jumlah berita pada bulan tertentu, hal itu dapat menunjukkan adanya isu besar "
            "atau peningkatan aktivitas pemberitaan."
        )
    else:
        st.warning("File monthly_distribution.csv belum tersedia.")

    st.markdown("---")

    st.subheader("Tahapan Preprocessing")

    preprocessing_df = pd.DataFrame({
        "Tahapan": [
            "Lowercase",
            "Menghapus URL",
            "Menghapus angka",
            "Menghapus tanda baca",
            "Menghapus spasi berlebih",
            "TF-IDF"
        ],
        "Tujuan": [
            "Menyamakan bentuk huruf",
            "Membersihkan tautan dari teks",
            "Mengurangi noise",
            "Menyederhanakan teks",
            "Merapikan format teks",
            "Mengubah teks menjadi fitur numerik"
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
        "Metode yang digunakan dalam project ini meliputi preprocessing teks, pembobotan kata menggunakan TF-IDF, "
        "dan klasifikasi menggunakan algoritma Multinomial Naive Bayes."
    )

    st.subheader("Alur Sistem")
    alur_df = pd.DataFrame({
        "Tahap": [
            "Input teks berita",
            "Preprocessing",
            "TF-IDF",
            "Naive Bayes",
            "Output prediksi"
        ],
        "Penjelasan": [
            "Pengguna memasukkan judul atau isi berita",
            "Teks dibersihkan dari URL, angka, tanda baca, dan spasi berlebih",
            "Teks diubah menjadi representasi numerik",
            "Model memprediksi label berdasarkan pola kata",
            "Sistem menampilkan label, kategori umum, dan probabilitas"
        ]
    })

    st.dataframe(alur_df, use_container_width=True)

    st.subheader("Kelebihan")
    st.write(
        "- Aplikasi sederhana dan mudah digunakan.\n"
        "- Model dapat memprediksi topik berita secara otomatis.\n"
        "- Menampilkan probabilitas prediksi.\n"
        "- Mendukung prediksi satu berita dan banyak berita dari CSV.\n"
        "- Memiliki halaman evaluasi dan informasi dataset."
    )

    st.subheader("Keterbatasan")
    st.write(
        "- Label berasal dari tag berita, bukan kategori umum resmi.\n"
        "- Beberapa label dapat berupa nama tokoh, lokasi, atau kanal berita.\n"
        "- Model dapat kurang akurat pada label dengan jumlah data sedikit.\n"
        "- Model masih menggunakan algoritma klasik, belum menggunakan deep learning."
    )

    st.subheader("Pengembangan Selanjutnya")
    st.write(
        "Pengembangan dapat dilakukan dengan mengelompokkan tag menjadi kategori umum seperti politik, ekonomi, "
        "hukum, olahraga, teknologi, bencana, dan internasional. Selain itu, model dapat dibandingkan dengan "
        "algoritma lain seperti Support Vector Machine, Logistic Regression, Random Forest, atau IndoBERT untuk "
        "meningkatkan akurasi klasifikasi teks berbahasa Indonesia."
    )


# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption(
    "UAS Data Mining - Klasifikasi Topik Berita Indonesia Menggunakan TF-IDF dan Multinomial Naive Bayes"
)
