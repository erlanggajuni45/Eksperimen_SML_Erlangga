import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, RocCurveDisplay
import mlflow
import mlflow.sklearn

import dagshub
dagshub.init(repo_owner="erlanggajuni45", repo_name="Eksperimen_SML_Erlangga", mlflow=True)

def train_and_log():
    # Path dataset hasil preprocessing
    train_path = "preprocessing/diabetes_dataset/diabetes_train.csv"
    test_path = "preprocessing/diabetes_dataset/diabetes_test.csv"
    
    if not os.path.exists(train_path):
        # Fallback jika dijalankan di folder berbeda
        train_path = "../preprocessing/diabetes_dataset/diabetes_train.csv"
        test_path = "../preprocessing/diabetes_dataset/diabetes_test.csv"

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.drop(columns=['Outcome'])
    y_train = train_data['Outcome']
    X_test = test_data.drop(columns=['Outcome'])
    y_test = test_data['Outcome']
    
    # Set nama eksperimen di MLflow
    mlflow.set_experiment("Diabetes_Classification_Erlangga")
    
    # Variasi Hyperparameter untuk Tuning manual
    c_values = [0.01, 0.1, 1.0]
    
    for c in c_values:
        with mlflow.start_run(run_name=f"LogisticRegression_C_{c}"):
            # Inisialisasi dan latih model
            model = LogisticRegression(C=c, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            # Prediksi
            y_pred = model.predict(X_test)
            
            # Hitung Metrik
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            
            # --- MANUAL LOGGING ---
            # 1. Log Hyperparameter
            mlflow.log_param("C_value", c)
            mlflow.log_param("model_type", "LogisticRegression")
            
            # 2. Log Metrik Utama
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            
            # 3. Log Artefak Tambahan 1: Confusion Matrix Plot
            plt.figure(figsize=(5,4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (C={c})')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            cm_path = f"confusion_matrix_C_{c}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path) # Upload artefak gambar ke DagsHub
            
            # 4. Log Artefak Tambahan 2: ROC Curve Plot
            plt.figure(figsize=(5,4))
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            plt.title(f'ROC Curve (C={c})')
            
            roc_path = f"roc_curve_C_{c}.png"
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(roc_path) # Upload artefak gambar ke DagsHub
            
            # 5. Log Model sklearn itu sendiri
            mlflow.sklearn.log_model(model, "model")
            
            # Hapus file plot lokal agar bersih
            if os.path.exists(cm_path): os.remove(cm_path)
            if os.path.exists(roc_path): os.remove(roc_path)
            
            print(f"Run selesai untuk C={c}. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_and_log()