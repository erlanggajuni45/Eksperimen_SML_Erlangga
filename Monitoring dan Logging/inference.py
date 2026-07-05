import requests
import json

url = "http://localhost:5001/invocations"

data = {
    "dataframe_records": [
        {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
    ]
}

headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print("Status Code:", response.status_code)
    print("Hasil Prediksi Model:", response.json())
except Exception as e:
    print("Gagal melakukan inference:", e)