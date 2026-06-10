import streamlit as st
import joblib
import re
import string
import pandas as pd

# =========================
# LOAD MODEL
# =========================

model = joblib.load('model_berita.pkl')
tfidf = joblib.load('vectorizer_tfidf.pkl')

# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title='Klasifikasi Topik Berita Indonesia',
    page_icon='📰',
    layout='wide'
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title('📌 Informasi Project')
st.sidebar.write(
    'Aplikasi ini dibuat untuk memenuhi tugas Ujian Akhir Semester '
    'mata kuliah Data Mining.'
)

st.sidebar.markdown('---')

st.sidebar.subheader('Metode')
st.sidebar.write('1. Text Preprocessing')
st.sidebar.write('2. TF-IDF Vectorizer')
st.sidebar.write('3. Multinomial Naive Bayes')

st.sidebar.markdown('---')

st.sidebar.subheader('Dataset')
st.sidebar.write(
    'Dataset yang digunakan adalah Indonesia News Dataset '
    'yang berisi berita dari media Indonesia.'
)

st.sidebar.markdown('---')

st.sidebar.caption('UAS Data Mining 2026')

# =========================
# MAIN PAGE
# =========================

st.title('📰 Klasifikasi Topik Berita Indonesia')
st.write(
    'Aplikasi ini digunakan untuk memprediksi topik/tag berita berdasarkan teks berita '
    'menggunakan metode **TF-IDF** dan algoritma **Multinomial Naive Bayes**.'
)

st.info(
    'Catatan: Label prediksi berasal dari kolom tag1 pada dataset, sehingga hasil prediksi '
    'dapat berupa tag seperti jakarta, kpk, prabowo, info-tempo, dan tag berita lainnya.'
)

st.markdown('---')

# =========================
# INPUT AREA
# =========================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Masukkan Teks Berita')

    contoh = (
        'Pemerintah Provinsi Jawa Timur memperluas layanan transportasi publik '
        'melalui program Trans Jatim untuk meningkatkan mobilitas masyarakat.'
    )

    input_text = st.text_area(
        'Masukkan judul atau isi berita:',
        value=contoh,
        height=220
    )

    prediksi_button = st.button('🔍 Prediksi Topik Berita')

with col2:
    st.subheader('Contoh Input')
    st.write(
        'Gunakan judul atau isi berita. Semakin panjang teks yang dimasukkan, '
        'biasanya hasil prediksi akan lebih baik.'
    )

    st.markdown('**Contoh:**')
    st.write(
        'KPK melakukan pemeriksaan terhadap sejumlah pejabat terkait dugaan korupsi '
        'pengadaan barang dan jasa.'
    )

# =========================
# PREDICTION
# =========================

if prediksi_button:
    if input_text.strip() == '':
        st.warning('Silakan masukkan teks berita terlebih dahulu.')
    else:
        clean = clean_text(input_text)
        vector = tfidf.transform([clean])
        prediction = model.predict(vector)[0]

        st.markdown('---')
        st.subheader('Hasil Prediksi')

        st.success(f'Prediksi Topik Berita: **{prediction}**')

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vector)[0]
            classes = model.classes_

            result_df = pd.DataFrame({
                'Topik': classes,
                'Probabilitas': proba
            })

            result_df = result_df.sort_values(
                by='Probabilitas',
                ascending=False
            ).head(5)

            result_df['Probabilitas (%)'] = result_df['Probabilitas'] * 100

            st.subheader('5 Probabilitas Tertinggi')

            st.dataframe(
                result_df[['Topik', 'Probabilitas (%)']],
                use_container_width=True
            )

            st.bar_chart(
                result_df.set_index('Topik')['Probabilitas (%)']
            )

        st.markdown('---')

        with st.expander('Lihat Teks Setelah Preprocessing'):
            st.write(clean)

# =========================
# FOOTER
# =========================

st.markdown('---')
st.caption(
    'Project UAS Data Mining - Klasifikasi Topik Berita Indonesia '
    'Menggunakan TF-IDF dan Multinomial Naive Bayes'
)
