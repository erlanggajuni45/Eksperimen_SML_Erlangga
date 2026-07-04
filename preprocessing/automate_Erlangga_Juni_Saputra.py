import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing(input_path, output_dir):
    # 1. Memuat Dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset mentah tidak ditemukan di {input_path}")
        
    df = pd.read_csv(input_path)
    
    # 2. Mengatasi nilai 0 tersembunyi pada kolom medis sesuai EDA
    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
        
    # 3. Memisahkan Fitur dan Target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Standarisasi Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Menyimpan Hasil Preprocessing
    os.makedirs(output_dir, exist_ok=True)
    
    train_processed = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_processed['Outcome'] = y_train.values
    
    test_processed = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_processed['Outcome'] = y_test.values
    
    train_processed.to_csv(os.path.join(output_dir, "diabetes_train.csv"), index=False)
    test_processed.to_csv(os.path.join(output_dir, "diabetes_test.csv"), index=False)
    
    print(f"Otomatisasi Berhasil! File siap latih disimpan di: {output_dir}")

if __name__ == "__main__":
    run_preprocessing(
        input_path="diabetes.csv", 
        output_dir="preprocessing/diabetes_dataset"
    )