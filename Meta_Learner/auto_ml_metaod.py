import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, average_precision_score, f1_score

# PyOD-Modelle
from pyod.models.loda import LODA
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.cof import COF

# MetaOD
from metaod.models.utility import prepare_trained_model
import os
from zipfile import ZipFile
def prepare_trained_model_local(filename='trained_models.zip', save_path='trained_models'):
    """
    Nutzt lokale pretrained Modelle. Kein Internet nötig.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nicht gefunden! Bitte manuell herunterladen.")
    
    # Extrahiere ZIP
    with ZipFile(filename, 'r') as zip:
        print('Extracting trained models now...')
        zip.extractall(save_path)
        print('Finish extracting models')

from metaod.models.predict_metaod import select_model
from metaod.models.gen_meta_features import generate_meta_features

# -------------------------
# 1. Daten laden und vorbereiten
# -------------------------
df = pd.read_csv("data/SmA-Four-Tank-Batch-Process_V2.csv", delimiter=";")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Anomalien markieren
df["Anomalie"] = (df["DeviationID ValueY"] != 1).astype(int)

# Optional: Prozessschritt filtern
df = df[df["CuStepNo ValueY"] == 8]

# Features und Label trennen
drop_cols = ["timestamp", "DeviationID ValueY"]
label_col = "Anomalie"

X_all = df.drop(columns=drop_cols + [label_col])
y_all = df[label_col]

# Train/Test Split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_all, test_size=0.1, random_state=42, stratify=y_all
)

# Nur Good-Daten für Training verwenden
X_train = X_train_full[y_train_full == 0].values

# Testdaten
X_test = X_test.values
y_test = y_test.values
y_train = y_train_full.values

# -------------------------
# 2. Feature-Skalierung
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------------
# 3. MetaOD vorbereiten
# -------------------------
# prepare_trained_model()
prepare_trained_model_local()
selected_models = select_model(X_train, n_selection=100)

meta_features, meta_names = generate_meta_features(X_train)
print("Meta-Features:")
for name, value in zip(meta_names, meta_features):
    print("-", name, value)

print("\nTop empfohlene Modelle:")
for i, model in enumerate(selected_models):
    print(i, model)

# -------------------------
# 4. Dynamisches Top-Modell erstellen und trainieren
# -------------------------

# Mapping Modellname → Klasse
model_class_map = {
    "ABOD": ABOD,
    "LOF": LOF,
    "KNN": KNN,
    "LODA": LODA,
    "IForest": IForest,
    "COF": COF,
    "OCSVM": OCSVM
}

# Mapping Modellname → Parameter-Namen
param_name_map = {
    "ABOD": ["n_neighbors"],
    "LOF": ["n_neighbors", "metric"],
    "KNN": ["n_neighbors", "method"],
    "LODA": ["n_bins", "n_random_cuts"],
    "IForest": ["n_estimators", "contamination"],
    "COF": ["n_neighbors"],
    "OCSVM": ["kernel", "nu"]
}

# Top-Modell auswählen
best_model_str = selected_models[0]
print("Top Modell:", best_model_str)

# Modellname extrahieren
match_name = re.match(r"([a-zA-Z]+)", best_model_str)
if not match_name:
    raise ValueError("Konnte Modellname nicht erkennen")
model_name = match_name.group(1)

# Parameter extrahieren
param_match = re.search(r"\((.*)\)", best_model_str)
if param_match:
    # Parameter in Klammern, splitten und typ konvertieren
    param_str = param_match.group(1)
    param_list = []
    for p in param_str.split(","):
        p = p.strip()
        if p.startswith("'") or p.startswith('"'):
            param_list.append(p.strip("'\""))
        else:
            if '.' in p:
                param_list.append(float(p))
            else:
                param_list.append(int(p))
else:
    # Nur eine Zahl nach dem Modellname
    numbers = re.findall(r"\d+\.?\d*", best_model_str)
    param_list = [int(numbers[0])] if numbers else []

# Parameter-Dict erstellen
param_names = param_name_map.get(model_name, [])
params = dict(zip(param_names, param_list))

# Modell instanziieren
model_class = model_class_map[model_name]
model = model_class(**params)
print(f"Instanziiertes Modell: {model}")

# -------------------------
# 5. Training und Evaluation
# -------------------------
model.fit(X_train)

# Vorhersage
preds_test = model.predict(X_test)
scores_test = model.decision_function(X_test)

# Evaluation
ap = average_precision_score(y_test, preds_test)
accuracy = accuracy_score(y_test, preds_test)
recall = recall_score(y_test, preds_test)
f1 = f1_score(y_test, preds_test)

print(f"\nEvaluation für {model_name}:")
print("Parameters:", params)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-Score:", f1)
print("Average Precision:", ap)
