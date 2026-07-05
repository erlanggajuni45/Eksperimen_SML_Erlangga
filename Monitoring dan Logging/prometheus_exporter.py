from prometheus_client import start_http_server, Counter, Histogram
import time
import random

REQUEST_COUNT = Counter(
    'model_request_total',
    'Total request prediksi yang diproses oleh model'
)

PREDICTION_RESULT = Counter(
    'model_prediction_result_total',
    'Total hasil prediksi model (0 = Negatif, 1 = Positif)',
    ['result']  
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Waktu yang dibutuhkan model untuk melakukan satu prediksi'
)

def simulate_inference():
    """
    Fungsi ini menyimulasikan model yang sedang memproses data.
    Di dunia nyata, ini adalah tempat di mana requests.post() ke Flask/MLflow berada.
    """
    REQUEST_COUNT.inc()
    
    with REQUEST_LATENCY.time():
        time.sleep(random.uniform(0.1, 0.5))
        
        prediction = random.choice([0, 1])
        
        PREDICTION_RESULT.labels(result=str(prediction)).inc()
        
        print(f"Prediksi berhasil diproses. Hasil: {prediction}")


if __name__ == '__main__':
    start_http_server(8000)
    print("🚀 Prometheus Exporter Model berjalan di http://localhost:8000/metrics")
    
    try:
        while True:
            simulate_inference()
            time.sleep(3) # Jeda 3 detik antar request
    except KeyboardInterrupt:
        print("\nExporter dihentikan.")