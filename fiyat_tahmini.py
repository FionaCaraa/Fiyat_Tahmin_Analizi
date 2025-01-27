import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Örnek yapay veri seti (araba özellikleri ve fiyatları)
# Özellikler: Yaş, Kilometre, Motor Gücü, Marka (kategori) 
data = {
    'Age': [5, 3, 8, 10, 4, 6, 7, 2, 5, 9],  # Arabanın yaşı
    'Mileage': [60000, 50000, 100000, 150000, 40000, 70000, 80000, 30000, 65000, 110000],  # Kilometre
    'Engine': [1.6, 2.0, 2.5, 2.2, 1.8, 1.9, 2.0, 1.5, 2.3, 2.0],  # Motor gücü (litre cinsinden)
    'Brand': ['Ford', 'Toyota', 'BMW', 'Mercedes', 'Audi', 'Honda', 'Ford', 'Toyota', 'BMW', 'Audi'],  # Marka
    'Price': [15000, 20000, 12000, 18000, 25000, 16000, 14000, 22000, 13000, 21000]  # Fiyat
}

# Veri çerçevesi oluşturma
df = pd.DataFrame(data)

# Kategorik veriyi sayısal verilere dönüştürme (Marka)
df = pd.get_dummies(df, columns=['Brand'], drop_first=True)

# Özellikler ve etiketler
X = df.drop('Price', axis=1)  # Özellikler (bağımsız değişkenler)
y = df['Price']  # Etiket (bağımlı değişken)

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma (Linear Regression)
model = LinearRegression()

# Modeli eğitme
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Tahminlerin doğruluğunu değerlendirme (ortalama mutlak hata)
mae = mean_absolute_error(y_test, y_pred)
print(f"Ortalama Mutlak Hata: {mae}")

# İlk test örneği için tahmin ve gerçek fiyatı gösterme
print(f"Gerçek Fiyat: {y_test.iloc[0]}")
print(f"Tahmin Edilen Fiyat: {y_pred[0]}")
