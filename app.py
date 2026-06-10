import streamlit as st
import joblib
import re
import string
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Klasifikasi Topik Berita Indonesia",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# MODERN CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 30%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 55%, #1e293b 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 14px;
    padding: 9px 12px;
    margin-bottom: 4px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    transition: .2s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(59,130,246,0.20);
    border: 1px solid rgba(147,197,253,0.35);
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1240px;
}

.hero-card {
    padding: 34px 36px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(15,23,42,0.94), rgba(30,64,175,0.92)),
        radial-gradient(circle at top right, rgba(56,189,248,.30), transparent 35%);
    color: white;
    box-shadow: 0 24px 70px rgba(15, 23, 42, .20);
    border: 1px solid rgba(255,255,255,.12);
    margin-bottom: 26px;
}

.hero-title {
    font-size: 44px;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 17px;
    color: #dbeafe;
    max-width: 850px;
    line-height: 1.65;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}

.badge {
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.16);
    color: #f8fafc;
    font-size: 13px;
    font-weight: 600;
}

.glass-card {
    padding: 24px;
    border-radius: 24px;
    background: rgba(255,255,255,.78);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(148,163,184,.24);
    box-shadow: 0 18px 48px rgba(15, 23, 42, .08);
    margin-bottom: 20px;
}

.soft-card {
    padding: 22px;
    border-radius: 22px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    box-shadow: 0 12px 32px rgba(15,23,42,.06);
    margin-bottom: 18px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #0f172a;
    margin-bottom: 8px;
}

.section-caption {
    color: #64748b;
    font-size: 15px;
    line-height: 1.65;
    margin-bottom: 18px;
}

.metric-card {
    padding: 20px;
    border-radius: 22px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    box-shadow: 0 14px 35px rgba(15,23,42,.07);
}

.metric-label {
    font-size: 13px;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .5px;
}

.metric-value {
    font-size: 26px;
    color: #0f172a;
    font-weight: 800;
    margin-top: 6px;
}

.result-box {
    padding: 24px;
    border-radius: 24px;
    background: linear-gradient(135deg, #dcfce7 0%, #ecfeff 100%);
    border: 1px solid #bbf7d0;
    box-shadow: 0 16px 42px rgba(22, 163, 74, .10);
    margin-top: 18px;
}

.result-label {
    color: #166534;
    font-size: 14px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .5px;
}

.result-value {
    color: #052e16;
    font-size: 28px;
    font-weight: 800;
    margin-top: 4px;
}

.warning-modern {
    padding: 18px 20px;
    border-radius: 20px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
    margin-bottom: 18px;
}

.info-modern {
    padding: 18px 20px;
    border-radius: 20px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e3a8a;
    margin-bottom: 18px;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 18px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 12px 30px rgba(15,23,42,.06);
}

div[data-testid="stFileUploader"] {
    padding: 16px;
    border-radius: 22px;
    background: rgba(255,255,255,.74);
    border: 1px dashed #94a3b8;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 14px !important;
    border: 0 !important;
    background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
    color: white !important;
    padding: 0.72rem 1.1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 14px 32px rgba(37,99,235,.24) !important;
    transition: .2s ease !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 44px rgba(37,99,235,.30) !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 12px 35px rgba(15,23,42,.06);
}

hr {
    margin: 1.6rem 0;
    border-color: rgba(148,163,184,.30);
}

.footer-modern {
    margin-top: 32px;
    padding: 18px;
    text-align: center;
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("model_berita.pkl")
    tfidf = joblib.load("vectorizer_tfidf.pkl")
    return model, tfidf

model, tfidf = load_model()


# =========================================================
# LOAD SUPPORT FILES
# =========================================================

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


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


# =========================================================
# HELPER FUNCTIONS
# =========================================================

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
        "prabowo": "Politik / Pemerintahan",
        "prabowo subianto": "Politik / Pemerintahan",
        "jokowi": "Politik / Pemerintahan",
        "jakarta": "Daerah / Perkotaan",
        "surabaya": "Daerah / Perkotaan",
        "banjir": "Bencana",
        "gempa": "Bencana",
        "kecelakaan": "Peristiwa",
        "info-tempo": "Berita Umum",
        "israel": "Internasional",
        "gaza": "Internasional",
        "palestina": "Internasional",
        "pilkada": "Politik / Pemilu",
        "pemilu": "Politik / Pemilu",
        "bola": "Olahraga",
        "timnas": "Olahraga",
        "ekonomi": "Ekonomi",
    }

    label_lower = str(label).lower().strip()
    return mapping.get(label_lower, "Topik Berita Umum")


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


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def hero(title, subtitle, badges=None):
    badges = badges or []
    badge_html = "".join([f'<span class="badge">{b}</span>' for b in badges])
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div class="hero-badges">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title, caption=""):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-caption">{caption}</div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📰 News Mining")
    st.caption("Klasifikasi topik berita Indonesia")
    st.markdown("---")

    menu = st.radio(
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

    st.markdown("---")
    st.markdown("### Metode")
    st.write("• Text Preprocessing")
    st.write("• TF-IDF Vectorizer")
    st.write("• Multinomial Naive Bayes")

    st.markdown("---")
    st.caption("UAS Data Mining 2026")


# =========================================================
# PAGE: BERANDA
# =========================================================

if menu == "Beranda":
    acc = metrics.get("accuracy", None)
    acc_text = f"{acc:.2%}" if acc is not None else "-"

    hero(
        "Klasifikasi Topik Berita Indonesia",
        "Aplikasi data mining untuk memprediksi topik/tag berita berdasarkan teks berita menggunakan TF-IDF dan Multinomial Naive Bayes. Dilengkapi prediksi satu berita, prediksi banyak berita, grafik evaluasi, dan analisis dataset.",
        ["TF-IDF", "Naive Bayes", "Streamlit", "Text Mining"]
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Algoritma", "Naive Bayes")
    with col2:
        metric_card("Fitur", "TF-IDF")
    with col3:
        metric_card("Jumlah Label", metrics.get("total_classes", "-"))
    with col4:
        metric_card("Akurasi", acc_text)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.25, 1])

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section(
            "Latar Belakang",
            "Jumlah berita digital yang terus meningkat membutuhkan sistem otomatis untuk membantu mengenali topik berita secara cepat."
        )
        st.write(
            "Aplikasi ini memanfaatkan pendekatan text mining. Teks berita dibersihkan, diubah menjadi fitur numerik menggunakan TF-IDF, lalu diklasifikasikan menggunakan Multinomial Naive Bayes."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Alur Sistem")
        alur_df = pd.DataFrame({
            "Tahap": ["Input", "Preprocessing", "TF-IDF", "Naive Bayes", "Output"],
            "Fungsi": ["Teks berita", "Pembersihan teks", "Ekstraksi fitur", "Klasifikasi", "Topik/tag berita"]
        })
        st.dataframe(alur_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if tag_distribution_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Topik Paling Dominan dalam Dataset", "Ringkasan tag terbanyak yang menjadi dasar pembelajaran model.")
        top_tags = tag_distribution_df.head(10)
        st.bar_chart(top_tags.set_index("tag")["jumlah"])
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: PREDIKSI TOPIK BERITA
# =========================================================

elif menu == "Prediksi Topik Berita":
    hero(
        "Prediksi Topik Berita",
        "Masukkan judul atau isi berita. Sistem akan membersihkan teks, melakukan transformasi TF-IDF, lalu memprediksi tag/topik berita.",
        ["Input teks", "Probabilitas", "Analisis hasil"]
    )

    contoh = (
        "Pemerintah Provinsi Jawa Timur memperluas layanan transportasi publik "
        "melalui program Trans Jatim untuk meningkatkan mobilitas masyarakat."
    )

    col1, col2 = st.columns([1.45, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        input_text = st.text_area(
            "Masukkan teks berita:",
            value=contoh,
            height=240
        )
        prediksi = st.button("🔍 Prediksi Topik Berita")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Tips Input", "Gunakan judul dan isi berita agar hasil prediksi lebih stabil.")
        st.write("Contoh topik yang dapat dikenali model mengikuti label/tag pada dataset, seperti KPK, Jakarta, Prabowo, banjir, dan tag lainnya.")
        st.markdown('<div class="info-modern">Label prediksi berasal dari kolom tag1 pada dataset, bukan kategori umum resmi.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if prediksi:
        if input_text.strip() == "":
            st.warning("Silakan masukkan teks berita terlebih dahulu.")
        else:
            prediction, kategori_umum, clean, result_df = predict_news(input_text)

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Hasil Prediksi</div>
                    <div class="result-value">{prediction}</div>
                    <div style="margin-top:8px;color:#166534;font-weight:600;">Kategori umum: {kategori_umum}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if result_df is not None:
                col_a, col_b = st.columns([1.05, 1])

                with col_a:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    section("5 Probabilitas Tertinggi")
                    st.dataframe(
                        result_df[["Label Dataset", "Kategori Umum", "Probabilitas (%)"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_b:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    section("Grafik Probabilitas")
                    st.bar_chart(result_df.set_index("Label Dataset")["Probabilitas (%)"])
                    st.markdown("</div>", unsafe_allow_html=True)

                top_score = result_df.iloc[0]["Probabilitas (%)"]
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                section("Analisis Hasil")
                if top_score >= 70:
                    st.write("Model memiliki keyakinan tinggi. Teks input memiliki pola kata yang kuat dan sesuai dengan label prediksi.")
                elif top_score >= 40:
                    st.write("Model memiliki keyakinan sedang. Teks cenderung mengarah ke satu topik, tetapi masih memiliki kemiripan dengan beberapa label lain.")
                else:
                    st.write("Model memiliki keyakinan rendah. Kemungkinan teks terlalu pendek, terlalu umum, atau beririsan dengan beberapa topik.")
                with st.expander("Lihat teks setelah preprocessing"):
                    st.write(clean)
                st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: PREDIKSI BANYAK BERITA
# =========================================================

elif menu == "Prediksi Banyak Berita":
    hero(
        "Prediksi Banyak Berita dari CSV",
        "Unggah file CSV berisi kolom teks berita. Aplikasi akan memprediksi topik untuk setiap baris dan menyediakan hasil yang dapat diunduh.",
        ["Upload CSV", "Batch prediction", "Download hasil"]
    )

    st.markdown('<div class="info-modern">Format CSV minimal memiliki satu kolom teks, misalnya Judul, Content, text, berita, atau title. Untuk demo, gunakan file kecil 5–100 baris agar cepat.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")
            st.success("File berhasil dibaca. Jika ada baris rusak, sistem akan melewatinya otomatis.")

            col1, col2 = st.columns([1, 1])
            with col1:
                metric_card("Jumlah Baris", len(data))
            with col2:
                metric_card("Jumlah Kolom", len(data.columns))

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Preview Data")
            st.dataframe(data.head(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

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

            if st.button("🚀 Prediksi Semua Data"):
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

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                section("Hasil Prediksi")
                st.dataframe(hasil_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

                pred_count = hasil_df["Label Dataset"].value_counts()

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                section("Distribusi Hasil Prediksi")
                st.bar_chart(pred_count)
                dominant_label = pred_count.index[0]
                dominant_count = pred_count.iloc[0]
                st.write(
                    f"Label paling dominan adalah **{dominant_label}** dengan jumlah **{dominant_count} berita** dari total **{len(hasil_df)}** data."
                )
                st.markdown("</div>", unsafe_allow_html=True)

                csv = hasil_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Hasil Prediksi CSV",
                    data=csv,
                    file_name="hasil_prediksi_berita.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi error saat membaca file: {e}")
            st.warning("Coba gunakan file CSV yang lebih kecil atau pastikan format kolomnya rapi.")


# =========================================================
# PAGE: EVALUASI MODEL
# =========================================================

elif menu == "Evaluasi Model":
    hero(
        "Evaluasi Model",
        "Halaman ini menampilkan performa model berdasarkan data testing, termasuk metrik utama, classification report, confusion matrix, dan analisis.",
        ["Accuracy", "Precision", "Recall", "F1-score", "Confusion matrix"]
    )

    accuracy = metrics.get("accuracy", None)
    macro_precision = metrics.get("macro_precision", None)
    macro_recall = metrics.get("macro_recall", None)
    macro_f1 = metrics.get("macro_f1", None)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Accuracy", f"{accuracy:.2%}" if accuracy is not None else "-")
    with col2:
        metric_card("Macro Precision", f"{macro_precision:.2%}" if macro_precision is not None else "-")
    with col3:
        metric_card("Macro Recall", f"{macro_recall:.2%}" if macro_recall is not None else "-")
    with col4:
        metric_card("Macro F1-score", f"{macro_f1:.2%}" if macro_f1 is not None else "-")

    if metrics:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Grafik Metrik Evaluasi", "Perbandingan metrik utama dalam persentase.")
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
        st.write("Accuracy menunjukkan proporsi prediksi benar. Precision menunjukkan ketepatan prediksi, recall menunjukkan kemampuan menemukan data pada setiap kelas, dan F1-score adalah keseimbangan precision dan recall.")
        st.markdown("</div>", unsafe_allow_html=True)

    if classification_report_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Classification Report")
        st.dataframe(classification_report_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        report_plot = classification_report_df.copy()
        if "Unnamed: 0" in report_plot.columns:
            report_plot = report_plot.rename(columns={"Unnamed: 0": "label"})

        if "label" in report_plot.columns and "f1-score" in report_plot.columns:
            report_plot = report_plot[
                ~report_plot["label"].isin(["accuracy", "macro avg", "weighted avg"])
            ]
            report_plot["f1-score"] = pd.to_numeric(report_plot["f1-score"], errors="coerce")
            report_plot = report_plot.dropna(subset=["f1-score"])

            col_top, col_bottom = st.columns(2)

            with col_top:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                section("Top 10 F1-score")
                top_f1 = report_plot.sort_values("f1-score", ascending=False).head(10)
                st.bar_chart(top_f1.set_index("label")["f1-score"])
                st.markdown("</div>", unsafe_allow_html=True)

            with col_bottom:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                section("Bottom 10 F1-score")
                bottom_f1 = report_plot.sort_values("f1-score", ascending=True).head(10)
                st.bar_chart(bottom_f1.set_index("label")["f1-score"])
                st.markdown("</div>", unsafe_allow_html=True)

    if confusion_matrix_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Confusion Matrix", "Nilai diagonal menunjukkan prediksi benar, sedangkan nilai di luar diagonal menunjukkan kesalahan prediksi.")
        cm_display = confusion_matrix_df.copy()

        if "Unnamed: 0" in cm_display.columns:
            cm_display = cm_display.set_index("Unnamed: 0")

        max_classes = st.slider(
            "Jumlah label yang ditampilkan",
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
        st.markdown("</div>", unsafe_allow_html=True)

    if algorithm_comparison_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Perbandingan Algoritma", "Perbandingan performa beberapa algoritma apabila file hasil perbandingan tersedia.")
        st.dataframe(algorithm_comparison_df, use_container_width=True, hide_index=True)
        if "accuracy" in algorithm_comparison_df.columns:
            st.bar_chart(algorithm_comparison_df.set_index("algorithm")["accuracy"])
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: INFORMASI DATASET
# =========================================================

elif menu == "Informasi Dataset":
    hero(
        "Informasi Dataset",
        "Ringkasan dataset, distribusi tag, sumber berita, tren waktu, dan kata paling sering muncul.",
        ["Dataset summary", "Distribusi tag", "Tren waktu", "Top words"]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Total Data Awal", dataset_summary.get("total_data_awal", "-"))
    with col2:
        metric_card("Total Data Model", dataset_summary.get("total_data_model", "-"))
    with col3:
        metric_card("Total Label", dataset_summary.get("total_label", "-"))

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section("Kolom yang Digunakan")
    dataset_info = pd.DataFrame({
        "Kolom": ["Judul", "Content", "tag1", "source", "Waktu", "Link"],
        "Keterangan": [
            "Judul berita",
            "Isi atau konten berita",
            "Tag utama sebagai label/topik",
            "Sumber media berita",
            "Waktu publikasi berita",
            "URL berita"
        ]
    })
    st.dataframe(dataset_info, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if tag_distribution_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Distribusi Topik/Tag Terbanyak")
        top_n = st.slider("Jumlah tag ditampilkan", min_value=5, max_value=min(30, len(tag_distribution_df)), value=min(10, len(tag_distribution_df)))
        top_tags = tag_distribution_df.head(top_n)
        st.bar_chart(top_tags.set_index("tag")["jumlah"])
        st.write("Tag dengan data lebih banyak cenderung lebih mudah dipelajari oleh model.")
        st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if source_distribution_df is not None:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Distribusi Sumber Berita")
            top_sources = source_distribution_df.head(10)
            st.bar_chart(top_sources.set_index("source")["jumlah"])
            st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        if top_words_df is not None:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Kata Paling Sering Muncul")
            st.bar_chart(top_words_df.head(15).set_index("word")["count"])
            st.markdown("</div>", unsafe_allow_html=True)

    if monthly_distribution_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Tren Jumlah Berita per Bulan")
        st.line_chart(monthly_distribution_df.set_index("bulan")["jumlah"])
        st.write("Lonjakan jumlah berita pada bulan tertentu dapat menunjukkan peningkatan aktivitas pemberitaan atau munculnya isu besar.")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: TENTANG PROJECT
# =========================================================

elif menu == "Tentang Project":
    hero(
        "Tentang Project",
        "Project UAS Data Mining berbasis text mining untuk klasifikasi topik berita Indonesia.",
        ["UAS Data Mining", "Machine Learning", "News analytics"]
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section("Judul Project")
    st.write("**Klasifikasi Topik Berita Indonesia Tahun 2024–2025 Menggunakan TF-IDF dan Algoritma Naive Bayes Berdasarkan Tag Berita**")

    section("Rumusan Masalah")
    st.write("Bagaimana menerapkan text mining untuk mengklasifikasikan topik berita Indonesia berdasarkan teks berita dan tag berita?")

    section("Metode yang Digunakan")
    st.write("Metode meliputi preprocessing teks, pembobotan kata menggunakan TF-IDF, dan klasifikasi menggunakan Multinomial Naive Bayes.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section("Kelebihan Aplikasi")
    st.write("""
- Tampilan aplikasi lebih modern dan interaktif.
- Mendukung prediksi satu berita dan prediksi banyak berita dari CSV.
- Menampilkan probabilitas prediksi.
- Menyediakan dashboard evaluasi model.
- Menampilkan analisis dataset, grafik distribusi, dan tren berita.
""")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section("Keterbatasan")
    st.write("""
- Label masih berasal dari tag berita, bukan kategori umum resmi.
- Beberapa label dapat berupa nama tokoh, lokasi, atau kanal berita.
- Model dapat kurang akurat pada label dengan jumlah data sedikit.
- Model belum menggunakan deep learning seperti IndoBERT.
""")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section("Pengembangan Selanjutnya")
    st.write("""
Pengembangan dapat dilakukan dengan mengelompokkan tag menjadi kategori umum seperti politik, ekonomi, hukum, olahraga, teknologi, bencana, daerah, dan internasional. Model juga dapat dibandingkan dengan SVM, Logistic Regression, Random Forest, XGBoost, atau model deep learning seperti IndoBERT.

Aplikasi juga dapat dikembangkan menjadi dashboard monitoring berita real-time dengan integrasi RSS, Google News, API berita, analisis sentimen, word cloud, filter sumber media, filter waktu, dan ringkasan otomatis.
""")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-modern">
        UAS Data Mining · Klasifikasi Topik Berita Indonesia · TF-IDF + Multinomial Naive Bayes
    </div>
    """,
    unsafe_allow_html=True
)
