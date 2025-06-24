import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

# === 1. Deschidere conexiune la baza de date SQLite ===
conn = sqlite3.connect("../Back/database/licenta.db")  # ajustează dacă e altă locație
cursor = conn.cursor()

# === 2. Citire date din tabelul ai_samples ===
query = "SELECT label, ecg_data, emg_data, temp_data, humidity_data FROM ai_samples"
df = pd.read_sql_query(query, conn)

# === 3. Prelucrare și extragere trăsături ===
def extract_features(row):
    def stats(arr):
        arr = np.array(arr)
        return [
            np.mean(arr),
            np.std(arr),
            np.min(arr),
            np.max(arr)
        ]
    
    try:
        ecg = stats(eval(row['ecg_data']))
        emg = stats(eval(row['emg_data']))
        temp = stats(eval(row['temp_data']))
        hum = stats(eval(row['humidity_data']))
        return ecg + emg + temp + hum
    except Exception as e:
        print(f"⚠️ Eroare la rândul: {row['label']}")
        print(f"Motiv: {e}\n")
        return None  # semnalăm eroare clar

# === 4. Aplicare funcție + filtrare doar mostre valide ===
feature_list = []
label_list = []

for _, row in df.iterrows():
    features_row = extract_features(row)
    if features_row and any(val != 0 for val in features_row):  # păstrăm doar cele bune
        feature_list.append(features_row)
        label_list.append("sincer" if row['label'] == "control" else row['label'])  # conversie directă

# === 5. Transformare în DataFrame + verificare ===
features = pd.DataFrame(feature_list, columns=[
    "ecg_mean", "ecg_std", "ecg_min", "ecg_max",
    "emg_mean", "emg_std", "emg_min", "emg_max",
    "temp_mean", "temp_std", "temp_min", "temp_max",
    "hum_mean", "hum_std", "hum_min", "hum_max"
])
y = pd.Series(label_list)

print(f"✅ Mostre valide: {features.shape[0]} / {df.shape[0]}")

# === 6. Împărțire în train/test ===
if features.shape[0] < 2:
    print("❌ Prea puține date valide pentru antrenare. Adaugă mai multe mostre în ai_samples.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# === 7. Antrenare model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 8. Evaluare și salvare ===
y_pred = model.predict(X_test)
print("\n📊 Raport evaluare:\n")
print(classification_report(y_test, y_pred))

dump(model, "model.pkl")
print("✅ Model salvat în: model.pkl")
