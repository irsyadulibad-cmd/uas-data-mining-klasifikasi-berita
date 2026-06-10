import os
import re
import json
import string
from datetime import datetime
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional libraries
try:
    import feedparser
except Exception:
    feedparser = None

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Dashboard Klasifikasi & Analisis Berita Indonesia",
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
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.13), transparent 30%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 55%, #1e293b 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

.block-container {
    padding-top: 2.1rem;
    padding-bottom: 4rem;
    max-width: 1260px;
}

.hero-card {
    padding: 34px 36px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,64,175,0.93)),
        radial-gradient(circle at top right, rgba(56,189,248,.30), transparent 35%);
    color: white;
    box-shadow: 0 24px 70px rgba(15, 23, 42, .20);
    border: 1px solid rgba(255,255,255,.12);
    margin-bottom: 26px;
}

.hero-title {
    font-size: 42px;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 16.5px;
    color: #dbeafe;
    max-width: 940px;
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
    background: rgba(255,255,255,.82);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(148,163,184,.24);
    box-shadow: 0 18px 48px rgba(15, 23, 42, .08);
    margin-bottom: 20px;
}

.metric-card {
    padding: 20px;
    border-radius: 22px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    box-shadow: 0 14px 35px rgba(15,23,42,.07);
    min-height: 112px;
}

.metric-label {
    font-size: 12.5px;
    color: #64748b;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .6px;
}

.metric-value {
    font-size: 25px;
    color: #0f172a;
    font-weight: 800;
    margin-top: 7px;
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

.info-modern {
    padding: 18px 20px;
    border-radius: 20px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e3a8a;
    margin-bottom: 18px;
}

.warning-modern {
    padding: 18px 20px;
    border-radius: 20px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
    margin-bottom: 18px;
}

.success-modern {
    padding: 18px 20px;
    border-radius: 20px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    margin-bottom: 18px;
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

.dev-card {
    padding: 22px;
    border-radius: 22px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    box-shadow: 0 14px 38px rgba(15,23,42,.06);
    height: 100%;
    margin-bottom: 18px;
}

.dev-number {
    width: 40px;
    height: 40px;
    border-radius: 13px;
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    margin-bottom: 14px;
}

.dev-title {
    font-size: 18px;
    color: #0f172a;
    font-weight: 800;
    margin-bottom: 8px;
}

.dev-text {
    color: #475569;
    line-height: 1.65;
    font-size: 14.5px;
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

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    border-radius: 18px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 12px 30px rgba(15,23,42,.06);
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 12px 35px rgba(15,23,42,.06);
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
# UTILITIES
# =========================================================

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


def dev_card(number, title, text):
    st.markdown(
        f"""
        <div class="dev-card">
            <div class="dev-number">{number}</div>
            <div class="dev-title">{title}</div>
            <div class="dev-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(
                path,
                engine="python",
                on_bad_lines="skip"
            )
        except Exception:
            return None
    return None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text, top_n=10):
    stopwords = set("""
    yang dan di ke dari untuk dengan pada adalah itu ini dalam sebagai karena atau juga akan telah tidak ada
    lebih oleh saat agar menjadi bisa dapat serta namun yaitu yakni para tersebut mereka kami kita saya kamu dia
    sebuah seorang hingga setelah sebelum kepada tentang menurut ujar kata katanya serta ini itu
    """.split())
    clean = clean_text(text)
    words = [w for w in clean.split() if len(w) > 3 and w not in stopwords]
    return Counter(words).most_common(top_n)


def simple_summary(text, max_sentences=3):
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    if len(sentences) <= max_sentences:
        return " ".join(sentences) if sentences else str(text)[:600]

    keywords = dict(extract_keywords(text, top_n=25))

    scored = []
    for sent in sentences:
        score = 0
        sent_clean = clean_text(sent)
        for word, freq in keywords.items():
            if word in sent_clean:
                score += freq
        scored.append((score, sent))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
    selected = [s for _, s in top]
    return " ".join(selected)


# =========================================================
# CATEGORY MAPPING
# =========================================================

CATEGORY_RULES = {
    "Hukum": ["kpk", "korupsi", "kejaksaan", "pengadilan", "hakim", "jaksa", "polisi", "tersangka", "pidana", "suap"],
    "Politik": ["prabowo", "jokowi", "pilkada", "pemilu", "dpr", "partai", "presiden", "menteri", "kampanye", "kabinet"],
    "Bencana": ["banjir", "gempa", "longsor", "kebakaran", "erupsi", "gunung", "cuaca", "hujan", "bnpb"],
    "Daerah": ["jakarta", "surabaya", "bandung", "jawa timur", "jatim", "kabupaten", "kota", "desa", "pemprov", "pemkab"],
    "Internasional": ["israel", "palestina", "gaza", "amerika", "china", "rusia", "ukraina", "iran", "pbb", "asing"],
    "Olahraga": ["bola", "timnas", "liga", "pemain", "pelatih", "pertandingan", "gol", "sepak", "pssi"],
    "Ekonomi": ["ekonomi", "rupiah", "investasi", "inflasi", "harga", "pasar", "bank", "saham", "bisnis", "pajak"],
    "Teknologi": ["teknologi", "digital", "ai", "internet", "aplikasi", "startup", "data", "siber", "kominfo"],
    "Kesehatan": ["kesehatan", "dokter", "rumah sakit", "pasien", "penyakit", "vaksin", "obat", "bpjs"],
    "Pendidikan": ["sekolah", "kampus", "mahasiswa", "guru", "pendidikan", "siswa", "kuliah", "universitas"]
}


def map_to_general_category(label_or_text):
    value = clean_text(label_or_text)
    scores = {}
    for category, keywords in CATEGORY_RULES.items():
        score = sum(1 for k in keywords if k in value)
        if score > 0:
            scores[category] = score
    if not scores:
        return "Berita Umum"
    return max(scores, key=scores.get)


# =========================================================
# SENTIMENT SIMPLE LEXICON
# =========================================================

POSITIVE_WORDS = set("""
baik bagus meningkat sukses berhasil aman lancar menang kuat positif dukung tumbuh maju stabil pulih
apresiasi optimal efektif efisien unggul prestasi terbaik inovasi manfaat membantu
""".split())

NEGATIVE_WORDS = set("""
buruk gagal turun krisis konflik korupsi suap banjir gempa tewas luka negatif rugi macet
ditangkap tersangka masalah bahaya ancaman protes kritik kisruh bentrok mahal lemah
""".split())


def sentiment_analysis(text):
    words = clean_text(text).split()
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    score = pos - neg

    if score > 0:
        label = "Positif"
    elif score < 0:
        label = "Negatif"
    else:
        label = "Netral"

    return label, pos, neg, score


# =========================================================
# LOAD MODEL AND DATA FILES
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("model_berita.pkl")
    vectorizer = joblib.load("vectorizer_tfidf.pkl")
    return model, vectorizer


model = None
vectorizer = None
model_available = True

try:
    model, vectorizer = load_model()
except Exception:
    model_available = False


metrics = load_json("evaluation_metrics.json", {})
dataset_summary = load_json("dataset_summary.json", {})
classification_report_df = load_csv("classification_report.csv")
confusion_matrix_df = load_csv("confusion_matrix.csv")
tag_distribution_df = load_csv("tag_distribution.csv")
source_distribution_df = load_csv("source_distribution.csv")
monthly_distribution_df = load_csv("monthly_distribution.csv")
top_words_df = load_csv("top_words.csv")
algorithm_comparison_df = load_csv("algorithm_comparison.csv")
dataset_clean_df = load_csv("dataset_clean.csv")


def predict_news(text):
    if not model_available:
        label_dataset = map_to_general_category(text)
        kategori_umum = label_dataset
        clean = clean_text(text)
        result_df = pd.DataFrame({
            "Label Dataset": [label_dataset],
            "Kategori Umum": [kategori_umum],
            "Probabilitas (%)": [100.0]
        })
        return label_dataset, kategori_umum, clean, result_df

    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    kategori_umum = map_to_general_category(str(prediction) + " " + text)

    result_df = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        classes = model.classes_

        result_df = pd.DataFrame({
            "Label Dataset": classes,
            "Probabilitas": proba
        })

        result_df = result_df.sort_values("Probabilitas", ascending=False).head(5)
        result_df["Probabilitas (%)"] = result_df["Probabilitas"] * 100
        result_df["Kategori Umum"] = result_df["Label Dataset"].apply(lambda x: map_to_general_category(str(x) + " " + text))

    return prediction, kategori_umum, clean, result_df


# =========================================================
# SIDEBAR - MENU DIKELOMPOKKAN DAN LEBIH RINGKAS
# =========================================================

with st.sidebar:
    st.markdown("## 📰 News Intelligence")
    st.caption("Klasifikasi, sentimen, ringkasan, dan monitoring berita")
    st.markdown("---")

    kelompok_menu = st.radio(
        "Kelompok Menu",
        [
            "📊 Dataset & Model",
            "✍️ Input Pengguna",
            "🌐 Real-Time",
        ]
    )

    st.markdown("---")

    if kelompok_menu == "📊 Dataset & Model":
        st.markdown("### 📊 Dataset & Model")
        st.caption("Menggunakan dataset sample, hasil training, dan file evaluasi model.")

        menu = st.radio(
            "Pilih Menu",
            [
                "Beranda",
                "Informasi Dataset",
                "Dashboard Monitoring",
                "Filter Analisis",
                "Word Cloud & Keyword",
                "Evaluasi Model",
                "Pengembangan",
            ]
        )

        with st.expander("📁 Sumber Data"):
            st.write("dataset_clean_sample.csv")
            st.write("evaluation_metrics.json")
            st.write("classification_report.csv")
            st.write("confusion_matrix.csv")
            st.write("tag_distribution.csv")
            st.write("source_distribution.csv")
            st.write("monthly_distribution.csv")
            st.write("top_words.csv")
            st.write("algorithm_comparison.csv")

    elif kelompok_menu == "✍️ Input Pengguna":
        st.markdown("### ✍️ Input Pengguna")
        st.caption("Menggunakan teks atau file CSV yang dimasukkan pengguna.")

        menu = st.radio(
            "Pilih Menu",
            [
                "Prediksi Topik",
                "Prediksi Banyak Berita",
                "Analisis Sentimen",
                "Ringkasan Otomatis",
            ]
        )

        with st.expander("📁 Sumber Data"):
            st.write("Teks berita manual")
            st.write("File CSV upload")
            st.write("Input pengguna di aplikasi")
            st.write("Model hasil training")

    else:
        st.markdown("### 🌐 Real-Time")
        st.caption("Mengambil data terbaru dari RSS/Google News secara langsung.")

        menu = st.radio(
            "Pilih Menu",
            [
                "Berita Real-Time",
            ]
        )

        with st.expander("📁 Sumber Data"):
            st.write("RSS Google News")
            st.write("RSS media online")
            st.write("Data berita terbaru")
            st.write("Model hasil training")

    st.markdown("---")

    with st.expander("⚙️ Model yang Digunakan", expanded=False):
        st.write("TF-IDF")
        st.write("Naive Bayes / best model")
        st.write("Mapping kategori umum")
        st.write("Sentimen lexicon")
        st.write("Ringkasan extractive")

    if model_available:
        st.success("Model aktif")
    else:
        st.warning("Model tidak ditemukan, mode rule-based aktif")
# =========================================================
# BERANDA
# =========================================================

if menu == "Beranda":
    acc = metrics.get("accuracy", None)
    acc_text = f"{acc:.2%}" if acc is not None else "-"

    hero(
        "Dashboard Klasifikasi & Analisis Berita Indonesia",
        "Aplikasi ini dikembangkan dari klasifikasi topik berita menjadi dashboard analisis berita yang lebih lengkap: kategori umum, penyeimbangan data, perbandingan algoritma, sentimen, keyword, filter analisis, monitoring, real-time RSS, dan ringkasan otomatis.",
        ["Text Mining", "TF-IDF", "Sentiment", "Keyword", "Monitoring", "RSS"]
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Model", "Aktif" if model_available else "Rule-based")
    with col2:
        metric_card("Akurasi", acc_text)
    with col3:
        metric_card("Total Label", metrics.get("total_classes", "-"))
    with col4:
        metric_card("Total Data", dataset_summary.get("total_data_model", "-"))

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Fitur yang Sudah Dikembangkan")
        fitur_df = pd.DataFrame({
            "No": list(range(1, 11)),
            "Fitur": [
                "Mengelompokkan tag menjadi kategori umum",
                "Menyeimbangkan jumlah data setiap kelas",
                "Membandingkan beberapa algoritma",
                "Menyiapkan pengembangan model IndoBERT",
                "Analisis sentimen berita",
                "Word cloud dan kata kunci dominan",
                "Filter analisis",
                "Dashboard monitoring berita",
                "Integrasi berita real-time RSS",
                "Ringkasan otomatis berita"
            ],
            "Status": [
                "Aktif",
                "Aktif di Colab training",
                "Aktif di Colab training",
                "Opsional/siap dikembangkan",
                "Aktif",
                "Aktif",
                "Aktif",
                "Aktif",
                "Aktif RSS",
                "Aktif extractive"
            ]
        })
        st.dataframe(fitur_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Alur Sistem")
        alur_df = pd.DataFrame({
            "Tahap": ["Input", "Preprocessing", "TF-IDF", "Model", "Analisis", "Output"],
            "Keterangan": [
                "Teks/CSV/RSS",
                "Cleaning teks",
                "Ekstraksi fitur",
                "Prediksi topik",
                "Sentimen, keyword, ringkasan",
                "Dashboard dan file hasil"
            ]
        })
        st.dataframe(alur_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if tag_distribution_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Distribusi Tag Terbanyak")
        st.bar_chart(tag_distribution_df.head(12).set_index("tag")["jumlah"])
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDIKSI TOPIK
# =========================================================

elif menu == "Prediksi Topik":
    hero(
        "Prediksi Topik Berita",
        "Masukkan teks berita. Sistem akan memprediksi label dataset, kategori umum, sentimen, keyword utama, dan ringkasan singkat.",
        ["Topik", "Kategori umum", "Sentimen", "Keyword", "Ringkasan"]
    )

    contoh = "KPK memeriksa sejumlah pejabat terkait dugaan korupsi pengadaan barang dan jasa di lingkungan pemerintah daerah."

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        text = st.text_area("Masukkan judul atau isi berita:", value=contoh, height=240)
        run = st.button("🔍 Analisis Berita")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Output Analisis")
        st.write("Aplikasi akan menampilkan prediksi topik, kategori umum, sentimen, kata kunci, dan ringkasan otomatis.")
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        pred, cat, clean, prob_df = predict_news(text)
        sent_label, pos, neg, score = sentiment_analysis(text)
        keywords = extract_keywords(text, top_n=10)
        summary = simple_summary(text)

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Hasil Prediksi</div>
                <div class="result-value">{pred}</div>
                <div style="margin-top:8px;color:#166534;font-weight:600;">Kategori umum: {cat} · Sentimen: {sent_label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Probabilitas Topik")
            if prob_df is not None:
                st.dataframe(prob_df[["Label Dataset", "Kategori Umum", "Probabilitas (%)"]], use_container_width=True, hide_index=True)
                st.bar_chart(prob_df.set_index("Label Dataset")["Probabilitas (%)"])
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Kata Kunci Dominan")
            key_df = pd.DataFrame(keywords, columns=["Kata", "Frekuensi"])
            st.dataframe(key_df, use_container_width=True, hide_index=True)
            if len(key_df) > 0:
                st.bar_chart(key_df.set_index("Kata")["Frekuensi"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Ringkasan Otomatis")
        st.write(summary)
        with st.expander("Lihat teks setelah preprocessing"):
            st.write(clean)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDIKSI BANYAK BERITA
# =========================================================

elif menu == "Prediksi Banyak Berita":
    hero(
        "Prediksi Banyak Berita",
        "Upload CSV berisi teks berita. Sistem akan memprediksi topik, kategori umum, sentimen, dan keyword untuk banyak data sekaligus.",
        ["Upload CSV", "Batch prediction", "Download CSV"]
    )

    st.markdown('<div class="info-modern">Gunakan file CSV kecil untuk demo. Pilih kolom yang berisi teks berita, misalnya Judul, Content, text, berita, atau title.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")
            st.success("File berhasil dibaca.")

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Preview Data")
            st.dataframe(data.head(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            selected_col = st.selectbox("Pilih kolom teks:", data.columns.tolist())
            limit = st.slider("Jumlah data diprediksi", 1, min(500, len(data)), min(20, len(data)))

            if st.button("🚀 Prediksi Banyak Berita"):
                rows = []
                for text in data[selected_col].astype(str).head(limit):
                    pred, cat, clean, prob_df = predict_news(text)
                    sent_label, pos, neg, score = sentiment_analysis(text)
                    keywords = ", ".join([w for w, c in extract_keywords(text, top_n=5)])
                    rows.append({
                        "Teks": text,
                        "Label Dataset": pred,
                        "Kategori Umum": cat,
                        "Sentimen": sent_label,
                        "Keyword": keywords,
                        "Ringkasan": simple_summary(text, max_sentences=2)
                    })

                out = pd.DataFrame(rows)

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                section("Hasil Prediksi")
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    section("Distribusi Kategori Umum")
                    st.bar_chart(out["Kategori Umum"].value_counts())
                    st.markdown("</div>", unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    section("Distribusi Sentimen")
                    st.bar_chart(out["Sentimen"].value_counts())
                    st.markdown("</div>", unsafe_allow_html=True)

                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Hasil CSV", data=csv, file_name="hasil_analisis_berita.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Terjadi error: {e}")


# =========================================================
# ANALISIS SENTIMEN
# =========================================================

elif menu == "Analisis Sentimen":
    hero(
        "Analisis Sentimen Berita",
        "Fitur ini menilai kecenderungan sentimen berita menggunakan pendekatan lexicon sederhana: positif, negatif, atau netral.",
        ["Positif", "Negatif", "Netral"]
    )

    text = st.text_area(
        "Masukkan teks berita:",
        value="Program transportasi publik baru dinilai berhasil meningkatkan pelayanan masyarakat meskipun masih mendapat kritik terkait keterlambatan.",
        height=220
    )

    if st.button("Analisis Sentimen"):
        label, pos, neg, score = sentiment_analysis(text)
        keywords = extract_keywords(text, 10)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Sentimen", label)
        with col2:
            metric_card("Kata Positif", pos)
        with col3:
            metric_card("Kata Negatif", neg)
        with col4:
            metric_card("Skor", score)

        chart_df = pd.DataFrame({"Jenis": ["Positif", "Negatif"], "Jumlah": [pos, neg]})
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Grafik Sentimen")
        st.bar_chart(chart_df.set_index("Jenis")["Jumlah"])
        st.write("Analisis ini bersifat sederhana dan berbasis daftar kata. Untuk hasil lebih akurat, dapat dikembangkan dengan model machine learning khusus sentimen.")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# WORD CLOUD & KEYWORD
# =========================================================

elif menu == "Word Cloud & Keyword":
    hero(
        "Word Cloud & Kata Kunci Dominan",
        "Fitur ini menampilkan kata yang paling sering muncul dalam teks berita atau dataset yang tersedia.",
        ["Keyword", "Frekuensi kata", "Word cloud"]
    )

    if dataset_clean_df is not None and "text" in dataset_clean_df.columns:
        source_option = st.radio("Sumber teks:", ["Input Manual", "Dataset Clean"])
    else:
        source_option = "Input Manual"

    if source_option == "Dataset Clean":
        sample_n = st.slider("Jumlah data digunakan", 100, min(5000, len(dataset_clean_df)), min(1000, len(dataset_clean_df)))
        text = " ".join(dataset_clean_df["text"].astype(str).head(sample_n).tolist())
    else:
        text = st.text_area(
            "Masukkan kumpulan teks berita:",
            value="KPK memeriksa pejabat terkait dugaan korupsi. Pemerintah menyiapkan kebijakan ekonomi baru. Banjir melanda beberapa wilayah setelah hujan deras.",
            height=220
        )

    top_n = st.slider("Jumlah kata kunci", 5, 30, 15)

    if st.button("Tampilkan Keyword"):
        keywords = extract_keywords(text, top_n)
        key_df = pd.DataFrame(keywords, columns=["Kata", "Frekuensi"])

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Tabel Kata Kunci")
        st.dataframe(key_df, use_container_width=True, hide_index=True)
        st.bar_chart(key_df.set_index("Kata")["Frekuensi"])
        st.markdown("</div>", unsafe_allow_html=True)

        if WordCloud is not None and len(key_df) > 0:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Word Cloud")
            wc = WordCloud(width=1000, height=420, background_color="white").generate(" ".join([w for w, c in keywords for _ in range(c)]))
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("WordCloud belum aktif. Pastikan package wordcloud ada di requirements.txt.")


# =========================================================
# FILTER ANALISIS
# =========================================================

elif menu == "Filter Analisis":
    hero(
        "Filter Analisis Berita",
        "Gunakan filter sumber media, waktu, kategori, dan kata kunci untuk menganalisis dataset berita secara lebih spesifik.",
        ["Filter tanggal", "Filter sumber", "Filter keyword"]
    )

    if dataset_clean_df is None:
        st.warning("File dataset_clean.csv belum tersedia. Jalankan kode Colab pengembangan total untuk membuat file ini.")
    else:
        data = dataset_clean_df.copy()

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Filter Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            source_col = "source" if "source" in data.columns else None
            if source_col:
                sources = ["Semua"] + sorted(data[source_col].dropna().astype(str).unique().tolist())[:100]
                selected_source = st.selectbox("Sumber media", sources)
            else:
                selected_source = "Semua"

        with col2:
            cat_col = "general_category" if "general_category" in data.columns else "category" if "category" in data.columns else None
            if cat_col:
                cats = ["Semua"] + sorted(data[cat_col].dropna().astype(str).unique().tolist())
                selected_cat = st.selectbox("Kategori", cats)
            else:
                selected_cat = "Semua"

        with col3:
            keyword = st.text_input("Kata kunci", "")

        filtered = data.copy()
        if source_col and selected_source != "Semua":
            filtered = filtered[filtered[source_col].astype(str) == selected_source]
        if cat_col and selected_cat != "Semua":
            filtered = filtered[filtered[cat_col].astype(str) == selected_cat]
        if keyword.strip():
            filtered = filtered[filtered["text"].astype(str).str.contains(keyword, case=False, na=False)]

        st.write(f"Jumlah data hasil filter: **{len(filtered)}**")
        st.dataframe(filtered.head(100), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if len(filtered) > 0 and cat_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Distribusi Kategori Hasil Filter")
            st.bar_chart(filtered[cat_col].value_counts().head(15))
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# DASHBOARD MONITORING
# =========================================================

elif menu == "Dashboard Monitoring":
    hero(
        "Dashboard Monitoring Berita",
        "Dashboard ini menampilkan ringkasan jumlah berita, topik dominan, sumber media aktif, tren waktu, keyword, dan sentimen.",
        ["Monitoring", "Topik", "Media", "Trend"]
    )

    data = dataset_clean_df

    if data is None:
        st.warning("File dataset_clean.csv belum tersedia. Jalankan kode Colab pengembangan total untuk membuat file monitoring.")
    else:
        total = len(data)
        total_source = data["source"].nunique() if "source" in data.columns else "-"
        total_cat = data["general_category"].nunique() if "general_category" in data.columns else data["category"].nunique() if "category" in data.columns else "-"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Total Berita", total)
        with col2:
            metric_card("Total Sumber", total_source)
        with col3:
            metric_card("Total Kategori", total_cat)
        with col4:
            metric_card("Mode", "Dataset")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Topik/Kategori Dominan")
            cat_col = "general_category" if "general_category" in data.columns else "category" if "category" in data.columns else None
            if cat_col:
                st.bar_chart(data[cat_col].value_counts().head(10))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Sumber Media Aktif")
            if "source" in data.columns:
                st.bar_chart(data["source"].value_counts().head(10))
            st.markdown("</div>", unsafe_allow_html=True)

        if "Waktu" in data.columns:
            temp = data.copy()
            temp["Waktu"] = pd.to_datetime(temp["Waktu"], errors="coerce")
            temp = temp.dropna(subset=["Waktu"])
            if len(temp) > 0:
                temp["bulan"] = temp["Waktu"].dt.to_period("M").astype(str)
                monthly = temp["bulan"].value_counts().sort_index()
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                section("Tren Berita per Bulan")
                st.line_chart(monthly)
                st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# BERITA REAL-TIME
# =========================================================

elif menu == "Berita Real-Time":
    hero(
        "Integrasi Berita Real-Time",
        "Fitur ini mengambil berita terbaru dari RSS, kemudian melakukan prediksi topik, sentimen, keyword, dan ringkasan otomatis.",
        ["RSS", "Real-time", "Auto analysis"]
    )

    st.markdown('<div class="warning-modern">Fitur ini membutuhkan koneksi internet dari server Streamlit. Jika server memblokir akses RSS tertentu, gunakan RSS lain.</div>', unsafe_allow_html=True)

    rss_url = st.text_input(
        "Masukkan URL RSS",
        value="https://news.google.com/rss/search?q=Jawa%20Timur&hl=id&gl=ID&ceid=ID:id"
    )

    limit = st.slider("Jumlah berita real-time", 3, 30, 10)

    if st.button("Ambil & Analisis RSS"):
        if feedparser is None:
            st.error("Package feedparser belum tersedia. Tambahkan feedparser ke requirements.txt.")
        else:
            feed = feedparser.parse(rss_url)
            entries = feed.entries[:limit]

            rows = []
            for e in entries:
                title = getattr(e, "title", "")
                summary = getattr(e, "summary", "")
                link = getattr(e, "link", "")
                text = f"{title} {summary}"

                pred, cat, clean, prob_df = predict_news(text)
                sent_label, pos, neg, score = sentiment_analysis(text)

                rows.append({
                    "Judul": title,
                    "Link": link,
                    "Label Dataset": pred,
                    "Kategori Umum": cat,
                    "Sentimen": sent_label,
                    "Keyword": ", ".join([w for w, c in extract_keywords(text, 5)]),
                    "Ringkasan": simple_summary(text, 2)
                })

            result = pd.DataFrame(rows)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Hasil Analisis RSS")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if len(result) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    section("Distribusi Kategori")
                    st.bar_chart(result["Kategori Umum"].value_counts())
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    section("Distribusi Sentimen")
                    st.bar_chart(result["Sentimen"].value_counts())
                    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# RINGKASAN OTOMATIS
# =========================================================

elif menu == "Ringkasan Otomatis":
    hero(
        "Ringkasan Otomatis Berita",
        "Masukkan teks berita panjang. Sistem akan menampilkan ringkasan, topik, keyword, dan sentimen.",
        ["Summarization", "Keyword", "Sentiment", "Topic"]
    )

    text = st.text_area(
        "Masukkan teks berita panjang:",
        value="Pemerintah Provinsi Jawa Timur memperluas layanan transportasi publik melalui program Trans Jatim. Program ini bertujuan meningkatkan mobilitas masyarakat dan mengurangi kemacetan. Sejumlah pihak mengapresiasi langkah ini karena dinilai membantu masyarakat. Namun, sebagian pengguna masih mengeluhkan keterlambatan jadwal pada jam sibuk.",
        height=280
    )

    max_sent = st.slider("Jumlah kalimat ringkasan", 1, 5, 3)

    if st.button("Buat Ringkasan"):
        pred, cat, clean, prob_df = predict_news(text)
        sent_label, pos, neg, score = sentiment_analysis(text)
        summary = simple_summary(text, max_sentences=max_sent)
        keywords = extract_keywords(text, 10)

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Topik", cat)
        with col2:
            metric_card("Sentimen", sent_label)
        with col3:
            metric_card("Label", pred)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Ringkasan")
        st.write(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Kata Kunci")
        key_df = pd.DataFrame(keywords, columns=["Kata", "Frekuensi"])
        st.dataframe(key_df, use_container_width=True, hide_index=True)
        st.bar_chart(key_df.set_index("Kata")["Frekuensi"])
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# EVALUASI MODEL
# =========================================================

elif menu == "Evaluasi Model":
    hero(
        "Evaluasi Model & Perbandingan Algoritma",
        "Menampilkan akurasi, precision, recall, F1-score, confusion matrix, dan perbandingan algoritma.",
        ["Accuracy", "Precision", "Recall", "F1-score", "Algorithms"]
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
        section("Grafik Metrik Evaluasi")
        metric_chart = pd.DataFrame({
            "Metrik": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
            "Nilai": [
                metrics.get("accuracy", 0) * 100,
                metrics.get("macro_precision", 0) * 100,
                metrics.get("macro_recall", 0) * 100,
                metrics.get("macro_f1", 0) * 100
            ]
        })
        st.bar_chart(metric_chart.set_index("Metrik")["Nilai"])
        st.markdown("</div>", unsafe_allow_html=True)

    if algorithm_comparison_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Perbandingan Algoritma")
        st.dataframe(algorithm_comparison_df, use_container_width=True, hide_index=True)
        if "algorithm" in algorithm_comparison_df.columns and "accuracy" in algorithm_comparison_df.columns:
            st.bar_chart(algorithm_comparison_df.set_index("algorithm")["accuracy"])
        st.markdown("</div>", unsafe_allow_html=True)

    if classification_report_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Classification Report")
        st.dataframe(classification_report_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if confusion_matrix_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Confusion Matrix")
        cm_display = confusion_matrix_df.copy()
        if "Unnamed: 0" in cm_display.columns:
            cm_display = cm_display.set_index("Unnamed: 0")
        max_classes = st.slider("Jumlah label ditampilkan", 5, min(30, len(cm_display)), min(10, len(cm_display)))
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

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    section("Catatan IndoBERT")
    st.write(
        "Penggunaan IndoBERT direkomendasikan sebagai pengembangan lanjutan karena mampu memahami konteks bahasa Indonesia lebih baik. "
        "Namun, IndoBERT membutuhkan resource komputasi lebih besar sehingga lebih tepat dilatih di Google Colab GPU, bukan langsung di Streamlit Cloud."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# INFORMASI DATASET
# =========================================================

elif menu == "Informasi Dataset":
    hero(
        "Informasi Dataset",
        "Ringkasan dataset, distribusi tag, sumber berita, tren waktu, dan kata paling sering muncul.",
        ["Dataset", "Tag", "Source", "Trend"]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Total Data Awal", dataset_summary.get("total_data_awal", "-"))
    with col2:
        metric_card("Total Data Model", dataset_summary.get("total_data_model", "-"))
    with col3:
        metric_card("Total Label", dataset_summary.get("total_label", "-"))

    if tag_distribution_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Distribusi Tag")
        st.bar_chart(tag_distribution_df.head(20).set_index("tag")["jumlah"])
        st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if source_distribution_df is not None:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Distribusi Sumber")
            st.bar_chart(source_distribution_df.head(10).set_index("source")["jumlah"])
            st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        if top_words_df is not None:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section("Top Words")
            st.bar_chart(top_words_df.head(15).set_index("word")["count"])
            st.markdown("</div>", unsafe_allow_html=True)

    if monthly_distribution_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section("Tren Berita per Bulan")
        st.line_chart(monthly_distribution_df.set_index("bulan")["jumlah"])
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PENGEMBANGAN
# =========================================================

elif menu == "Pengembangan":
    hero(
        "Pengembangan yang Sudah Diimplementasikan",
        "Sepuluh pengembangan utama telah dimasukkan ke aplikasi agar lebih lengkap sebagai dashboard analisis berita.",
        ["10 fitur", "Dashboard", "Monitoring", "Future work"]
    )

    c1, c2 = st.columns(2)
    with c1:
        dev_card("1", "Kategori Umum", "Tag spesifik dipetakan ke kategori umum seperti Hukum, Politik, Bencana, Daerah, Internasional, Olahraga, Ekonomi, Teknologi, Kesehatan, dan Pendidikan.")
    with c2:
        dev_card("2", "Balancing Data", "Penyeimbangan data dilakukan di tahap training Colab dengan membatasi jumlah data per kelas agar model tidak bias terhadap tag dominan.")

    c3, c4 = st.columns(2)
    with c3:
        dev_card("3", "Perbandingan Algoritma", "Kode Colab membandingkan Naive Bayes, Logistic Regression, Linear SVM, dan Random Forest, lalu menyimpan metrik ke algorithm_comparison.csv.")
    with c4:
        dev_card("4", "IndoBERT", "Aplikasi menyiapkan ruang pengembangan IndoBERT. Pelatihan direkomendasikan di Colab GPU karena membutuhkan resource lebih besar.")

    c5, c6 = st.columns(2)
    with c5:
        dev_card("5", "Analisis Sentimen", "Aplikasi mendeteksi sentimen positif, negatif, dan netral menggunakan pendekatan lexicon sederhana.")
    with c6:
        dev_card("6", "Word Cloud & Keyword", "Aplikasi menampilkan kata kunci dominan dan word cloud jika package wordcloud tersedia.")

    c7, c8 = st.columns(2)
    with c7:
        dev_card("7", "Filter Analisis", "Data dapat difilter berdasarkan sumber media, kategori, dan kata kunci.")
    with c8:
        dev_card("8", "Dashboard Monitoring", "Dashboard menampilkan total berita, topik dominan, sumber aktif, dan tren waktu.")

    c9, c10 = st.columns(2)
    with c9:
        dev_card("9", "Berita Real-Time", "Fitur RSS mengambil berita terbaru dan langsung menganalisis topik, sentimen, keyword, dan ringkasan.")
    with c10:
        dev_card("10", "Ringkasan Otomatis", "Sistem membuat ringkasan extractive dari teks berita panjang dan menampilkan topik serta keyword.")


st.markdown(
    """
    <div class="footer-modern">
        UAS Data Mining · Dashboard Analisis Berita Indonesia · TF-IDF + Machine Learning + Monitoring
    </div>
    """,
    unsafe_allow_html=True
)
