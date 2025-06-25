from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Configurează CORS pentru toate rutele, dar în special pentru /predict
CORS(app, resources={r"/predict": {"origins": "*"}})

# Încarcă modelul antrenat
model = joblib.load("model.pkl")

# Numele caracteristicilor folosite la antrenare
feature_names = [
    'ecg_mean', 'ecg_std', 'ecg_min', 'ecg_max',
    'emg_mean', 'emg_std', 'emg_min', 'emg_max',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max',
    'hum_mean', 'hum_std', 'hum_min', 'hum_max'
]

def extract_features_from_array(data):
    def stats(arr):
        arr = np.array(arr)
        return [
            np.mean(arr),
            np.std(arr),
            np.min(arr),
            np.max(arr)
        ]
    try:
        ecg = stats(data['ecg_data'])
        emg = stats(data['emg_data'])
        temp = stats(data['temp_data'])
        hum = stats(data['humidity_data'])
        return ecg + emg + temp + hum
    except Exception as e:
        print(f"Eroare în datele trimise: {e}")
        return [0] * 16

@app.route("/predict", methods=["POST"])
@cross_origin()  # permite CORS explicit pentru această rută
def predict():
    input_data = request.get_json()
    features = extract_features_from_array(input_data)
    df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(df)[0]
    return jsonify({ "predictie": prediction })

if __name__ == "__main__":
    app.run(port=8000)
