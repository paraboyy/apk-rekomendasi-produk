import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# load_data('Dataset/Sales Transaction v.4a.csv')

# === 1. Load Dataset ===
def load_data(file_path):
    # data = pd.read_parquet(file_path)
    data = pd.read_csv(file_path) 
    return data

# === 2. Preprocessing ===
def preprocess_data(data):
    # Encode categorical features
    label_encoder = LabelEncoder()
    data['ProductName'] = label_encoder.fit_transform(data['ProductName'])
    data['Country'] = label_encoder.fit_transform(data['Country'])
    
    data['total_price'] = data['Price'] * data['Quantity']
    
    # Fitur dan target
    X = data[['ProductName', 'Price', 'Quantity', 'CustomerNo', 'Country']]
    y = data['ProductName']  # Prediksi produk
    
    # Normalisasi fitur numerik jika perlu
    scaler = StandardScaler()
    X[['Price', 'Quantity']] = scaler.fit_transform(X[['Price', 'Quantity']])
    
    return X, y, label_encoder, scaler

# === 3. Training Model ===
def train_model(X, y):
    # Memeriksa distribusi kelas
    class_counts = y.value_counts()
    if class_counts.min() == 1:
        # Menangani kelas yang hanya memiliki satu sampel
        classes_to_remove = class_counts[class_counts == 1].index
        X = X[~y.isin(classes_to_remove)]
        y = y[~y.isin(classes_to_remove)]
    
    # Splitting dataset dengan stratifikasi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply LabelEncoder to both y_train and y_test to ensure consistent label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42, tree_method='gpu_hist')
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, label_encoder

# === 4. Rekomendasi Produk ===
def recommend_products(model, input_data, label_encoder, scaler):
    # Transform input_data agar sesuai dengan format yang digunakan dalam pelatihan
    input_data_transformed = np.array(input_data).reshape(1, -1)
    
    # Normalisasi input_data yang numerik
    input_data_transformed[:, 1:3] = scaler.transform(input_data_transformed[:, 1:3])
    
    # Prediksi produk
    prediction = model.predict(input_data_transformed)
    recommended_product = label_encoder.inverse_transform(prediction)
    return recommended_product[0]

# === 5. Streamlit Interface ===
def main():
    st.title("Sistem Rekomendasi Produk")
    
    # Upload file dataset
    uploaded_file = st.file_uploader("Upload Dataset Anda", type=["csv"])
    
    if uploaded_file:
        data = load_data(uploaded_file)
        st.write("Dataset:")
        st.write(data.head())
        
        # Preprocessing
        # streamlit run app.py 
        X, y, label_encoder, scaler = preprocess_data(data)
        model, label_encoder = train_model(X, y)  # Pastikan label_encoder dikembalikan
        
        # Input user untuk rekomendasi
        st.subheader("Masukkan Detail untuk Rekomendasi")
        product_name = st.selectbox("Nama Produk", data['ProductName'].unique())
        price = st.number_input("Harga Produk", min_value=0.0)
        quantity = st.number_input("Jumlah", min_value=1)
        id_customer = st.number_input("ID Customer", min_value=1)
        country = st.selectbox("Negara", data['Country'].unique())
        
        if st.button("Dapatkan Rekomendasi"):
            input_data = [product_name, price, quantity, id_customer, country]
            recommended_product = recommend_products(model, input_data, label_encoder, scaler)
            st.success(f"Produk yang Direkomendasikan: {recommended_product}")

if __name__ == "__main__":
    main()
