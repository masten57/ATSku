from flask import Flask, request, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import PyPDF2
import os
from werkzeug.utils import secure_filename
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

classifier = joblib.load('model_svm.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

resume_dataset_path = 'resume.csv'
resume_dataset = pd.read_csv(resume_dataset_path)

uploads_dir = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''

        # Tentukan halaman awal dan akhir sesuai dengan kebutuhan
        start_page = 1  # Ganti dengan halaman awal yang sesuai
        end_page = len(reader.pages)  # Ganti dengan halaman terakhir yang sesuai

        for page_num in range(start_page - 1, end_page):
            page = reader.pages[page_num]
            text += page.extract_text()

    # Tambahkan preprocessing di sini
    text = clean_project_text(text)

    # Return teks dan tutup file
    return text

def clean_project_text(output_text):
    # Pisahkan setiap baris menjadi list
    lines = output_text.split('\n')

    # Hapus baris kosong dari list
    lines = [line.strip() for line in lines if line.strip()]

    # Hapus teks "Keahlian, Penghargaan & Pengalaman Lain"
    lines = lines[2:]

    cleaned_text = ' '.join(lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.lower()

    # Hapus stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word not in stop_words)

    # Hapus simbol
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)

    # Tambahkan preprocessing regex sesuai kebutuhan
    cleaned_text = re.sub(r'pattern_regex', 'replacement', cleaned_text)

    return cleaned_text

def calculate_similarity_scores(user_text, category, dataset):
    # Filter dataset berdasarkan kategori yang dipilih
    category_dataset = dataset[dataset['Category'] == category]

    # Gabungkan teks CV pengguna dengan dataset kategori
    combined_texts = category_dataset['cleaned_resume'].tolist() + [user_text]

    # Hitung TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Hitung cosine similarity antara teks CV pengguna dan semua resume dalam dataset kategori
    user_tfidf_vector = tfidf_matrix[-1]  # Vektor TF-IDF CV pengguna
    similarity_scores = cosine_similarity(user_tfidf_vector, tfidf_matrix[:-1])[0]

    # Mengembalikan rata-rata skor kesamaan
    average_similarity_score = sum(similarity_scores) / len(similarity_scores)
    return average_similarity_score

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/klasifikasi", methods=['GET', 'POST'])
def klasifikasi():
    if request.method == 'POST':
        # Dapatkan file CV yang diunggah
        file = request.files['resume']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(uploads_dir, filename)
            file.save(filepath)
            
            pdf_text = extract_text_from_pdf(filepath)
            os.remove(filepath)  # Hapus file setelah digunakan
            
            # Prediksi kategori menggunakan model yang telah dilatih
            vectorized_text = vectorizer.transform([pdf_text])
            category = classifier.predict(vectorized_text)
            
            return render_template('fitur_klasifikasi.html', category=category, pdf_text=pdf_text)
    
    # Jika bukan POST, tampilkan halaman dengan formulir
    return render_template('fitur_klasifikasi.html')

@app.route('/similarity-dataset', methods=['GET','POST'])
def similarity_dataset():
    if request.method == 'POST':
        # Dapatkan file CV yang diunggah
        user_input_pdf = request.files['resume']

        # Pastikan input kategori ada dalam form dan ambil nilainya
        selected_category = request.form.get('category', 'DefaultCategory')

        # Proses file CV yang diunggah
        filepath = os.path.join(uploads_dir, secure_filename(user_input_pdf.filename))
        user_input_pdf.save(filepath)
        pdf_text = extract_text_from_pdf(filepath)

        # Hitung skor kesamaan (anda perlu mengimplementasikan logika ini)
        similarity_score = calculate_similarity_scores(pdf_text, selected_category, resume_dataset)

        # Hapus file setelah digunakan
        os.remove(filepath)

        # Kirim hasil ke template untuk ditampilkan
        return render_template('similarity_score_dataset.html', category=selected_category, 
                               pdf_text=pdf_text, score1=similarity_score)
    return render_template('similarity_score_dataset.html')

@app.route('/similarity-duacv', methods=['GET','POST'])
def similarity_duacv():
    if request.method == 'POST':
        # Dapatkan kedua file CV
        resume1 = request.files['resume1']
        resume2 = request.files['resume2']

        # Proses dan simpan sementara file
        filepath1 = os.path.join(uploads_dir, secure_filename(resume1.filename))
        filepath2 = os.path.join(uploads_dir, secure_filename(resume2.filename))
        resume1.save(filepath1)
        resume2.save(filepath2)

        # Ekstrak teks dari kedua CV
        text1 = extract_text_from_pdf(filepath1)
        text2 = extract_text_from_pdf(filepath2)

        # Hitung skor kesamaan
        similarity_score = calculate_similarity(text1, text2)

        # Hapus file setelah digunakan
        os.remove(filepath1)
        os.remove(filepath2)

        # Kirim hasil ke template untuk ditampilkan
        return render_template('similarity_score_duacv.html', score2=similarity_score)
    return render_template('similarity_score_duacv.html')

if __name__ == '__main__':
    app.run(debug=True)