
import streamlit as st
import joblib
import re
import string

# Load model dan vectorizer
model = joblib.load('model_berita.pkl')
tfidf = joblib.load('vectorizer_tfidf.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(
    page_title='Klasifikasi Topik Berita Indonesia',
    page_icon='📰',
    layout='centered'
)

st.title('📰 Klasifikasi Topik Berita Indonesia')
st.write(
    'Aplikasi ini menggunakan metode TF-IDF dan algoritma Naive Bayes '
    'untuk memprediksi topik berita berdasarkan teks yang dimasukkan.'
)

st.markdown('---')

input_text = st.text_area(
    'Masukkan judul atau isi berita:',
    height=200,
    placeholder='Contoh: Pemerintah Provinsi Jawa Timur memperluas layanan transportasi publik melalui program Trans Jatim.'
)

if st.button('Prediksi Topik Berita'):
    if input_text.strip() == '':
        st.warning('Silakan masukkan teks berita terlebih dahulu.')
    else:
        clean = clean_text(input_text)
        vector = tfidf.transform([clean])
        prediction = model.predict(vector)[0]

        st.success(f'Prediksi Topik Berita: {prediction}')

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vector)[0]
            classes = model.classes_

            result = sorted(
                zip(classes, proba),
                key=lambda x: x[1],
                reverse=True
            )

            st.subheader('Probabilitas Prediksi')
            for label, score in result[:5]:
                st.write(f'{label}: {score:.2%}')

st.markdown('---')
st.caption('UAS Data Mining - Klasifikasi Topik Berita Indonesia Tahun 2024-2025')
