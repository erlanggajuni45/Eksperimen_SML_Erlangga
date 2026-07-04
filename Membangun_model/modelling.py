import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub

# Inisialisasi koneksi DagsHub Tracking 
dagshub.init(repo_owner="erlanggajuni45", repo_name="Eksperimen_SML_Erlangga", mlflow=True)

def train_baseline():
    # Path dataset hasil preprocessing
    train_path = "preprocessing/diabetes_dataset/diabetes_train.csv"
    test_path = "preprocessing/diabetes_dataset/diabetes_test.csv"
    
    # Fallback path jika dieksekusi dari dalam folder Membangun_model
    if not os.path.exists(train_path):
        train_path = "../preprocessing/diabetes_dataset/diabetes_train.csv"
        test_path = "../preprocessing/diabetes_dataset/diabetes_test.csv"

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.drop(columns=['Outcome'])
    y_train = train_data['Outcome']
    X_test = test_data.drop(columns=['Outcome'])
    y_test = test_data['Outcome']
    
    # Set Eksperimen MLflow
    mlflow.set_experiment("Diabetes_Baseline_Erlangga")
    
    # Aktifkan Autolog untuk merekam parameter bawaan scikit-learn secara otomatis
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Baseline_LogisticRegression"):
        # Model dengan parameter default
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi sederhana
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Baseline Model Training Selesai. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_baseline()