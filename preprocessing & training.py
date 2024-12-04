import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# 1. Load Dataset
df = pd.read_csv('Dataset/customer_shopping_data.csv')  # Ganti dengan path dataset Anda

# 2. Preprocessing Data
# Mengubah data kategori menjadi numerik menggunakan LabelEncoder
label_encoder = LabelEncoder()

# Kolom yang perlu diubah
df['gender'] = label_encoder.fit_transform(df['gender'])
df['category'] = label_encoder.fit_transform(df['category'])
df['payment_method'] = label_encoder.fit_transform(df['payment_method'])
df['shopping_mall'] = label_encoder.fit_transform(df['shopping_mall'])

# Mengonversi invoice_date menjadi format datetime dan mengekstrak fitur seperti bulan dan tahun
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)
df['month'] = df['invoice_date'].dt.month
df['year'] = df['invoice_date'].dt.year

# Menghapus data yang tidak lengkap atau NaN
df = df.dropna()

# 3. Menyimpan Dataset yang Sudah Diproses ke File Baru
df.to_csv('Dataset/preprocessing_data.csv', index=False)
print("Data telah disimpan di file 'processed_data.csv'.")

# 4. Fitur dan Target
# Misalnya kita ingin memprediksi produk yang dibeli (category) berdasarkan fitur lainnya
X = df[['gender', 'age', 'category', 'quantity', 'price', 'payment_method', 'month', 'year', 'shopping_mall']]
y = df['category']  # Kita anggap target adalah kategori produk yang dibeli

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Pelatihan Model XGBoost
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 7. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Menyimpan model untuk digunakan nanti dalam DSS
joblib.dump(model, 'hasil/xgboost_model.pkl')

# 8. Visualisasi Hasil Akurasi
# Membuat gambar untuk hasil akurasi
plt.figure(figsize=(6, 6))
plt.barh(['Akurasi Model'], [accuracy * 100], color='skyblue')
plt.xlabel('Akurasi (%)')
plt.title('Hasil Akurasi Model XGBoost')
plt.xlim(0, 100)
plt.savefig('hasil/accuracy_result.png')  # Menyimpan gambar hasil akurasi
plt.show()
