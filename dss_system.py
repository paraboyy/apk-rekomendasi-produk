from flask import Flask, render_template, request
import pandas as pd
import joblib

# 1. Inisialisasi Flask
app = Flask(__name__)

# 2. Load Dataset dan Model
df = pd.read_csv('Dataset/preprocessing_data.csv')  # Ganti dengan path dataset Anda
model = joblib.load('hasil/xgboost_model.pkl')  # Load model yang telah dilatih

# Pastikan kolom 'age' dalam dataframe bertipe numerik
# df['age'] = pd.to_numeric(df['age'], errors='coerce')

# 3. Fungsi untuk Memberikan Rekomendasi Berdasarkan Usia
def recommend_products_by_age(age, num_recommendations=5):
    # Memeriksa apakah usia ada dalam dataset
    if age not in df['age'].values:
        return None
    
    # Filter data berdasarkan usia
    filtered_data = df[df['age'] == age].copy() 
    
    # Membuat fitur untuk prediksi
    features = filtered_data[['gender', 'age', 'category', 'quantity', 'price', 'payment_method', 'month', 'year', 'shopping_mall']]
    
    # Prediksi kategori produk yang akan direkomendasikan
    predicted_categories = model.predict(features)
    
    # Mengambil produk yang sudah diprediksi untuk usia tersebut
    filtered_data['predicted_category'] = predicted_categories
    
    # Menampilkan beberapa rekomendasi produk (kategori) yang paling sering muncul
    recommendations = filtered_data['predicted_category'].value_counts().head(num_recommendations)
    return recommendations

# 4. Route untuk Halaman Utama
@app.route('/')
def home():
    return render_template('index.html')

# 5. Route untuk Menangani Form Input dan Menampilkan Rekomendasi
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        age = int(request.form['age'])  # Mendapatkan input usia dari form
        recommendations = recommend_products_by_age(age)
        
        if recommendations is None:
            return render_template('index.html', error="Usia tidak ditemukan dalam dataset.")
        
        return render_template('index.html', recommendations=recommendations, age=age)
    
    except ValueError:
        return render_template('index.html', error="Masukkan usia yang valid!")

# 6. Menjalankan Aplikasi
if __name__ == '__main__':
    app.run(debug=True)
