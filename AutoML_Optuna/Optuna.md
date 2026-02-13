# AutoML mit Optuna

---

## 1) Datenformat & Spalten

Erwartet wird eine CSV mit u. a. folgenden Spalten:

- timestamp (wird zu datetime geparst und zur Sortierung genutzt)
- CuStepNo (Prozessschritt)
- DeviationID (Szenario/Abweichung; wird zur Definition von Normal/Anomal genutzt)
- mehrere numerische Sensor-/Aktuator-Signale als Features

Im Code werden die Spaltennamen bereinigt (strip) und ggf. umbenannt:
- "CuStepNo ValueY" → "CuStepNo"
- "DeviationID ValueY" → "DeviationID"

---

## 2) Labeling (nur für Evaluation)

Es wird ein binäres Label y erzeugt:

- y = 0 für Normalbetrieb („good data“)
- y = 1 für Anomalie/Abweichung

Wichtig: In diesem Datensatz ist DeviationID == 0 fast leer. Daher muss Normalbetrieb über eine oder mehrere *NORMAL_IDS* definiert werden (z. B. DeviationID == 1).

Beispiel:
- Normal: DeviationID ∈ NORMAL_IDS
- Anomal: DeviationID ∉ NORMAL_IDS

Dieses Label wird *nicht* zum Training genutzt, sondern nur für F1/Accuracy/Recall usw.

---

## 3) Feature-Selektion & Skalierung

### Feature-Selektion
feature_cols enthält alle numerischen Spalten außer:
- timestamp
- CuStepNo
- DeviationID
- y

### RobustScaler
Für IF/OCSVM/SOM wird RobustScaler verwendet:
- zentriert über den *Median*
- skaliert über den *Interquartilsabstand (IQR)*
- robust gegenüber Ausreißern/Noise (typisch für Prozessdaten)

*Wichtig gegen Data Leakage:*  
Scaler wird pro Trainingssplit gefittet (und im step-wise Setup häufig pro Step), dann auf VAL/TEST angewendet.

---

## 4) Semi-supervised Split (Train / Val / Test)

Konzept:
- *Train*: nur Normaldaten (damit das Modell „Normalverhalten“ lernt)
- *Val/Test*: enthalten Normal + Anomalien (damit F1 sinnvoll berechnet werden kann)

Der Split passiert zeit-/zufallsbasiert (je nach Notebook-Version). In der step-wise Variante wird außerdem pro Step geprüft, ob genügend Normaldaten vorhanden sind.

---

## 5) Step-wise Modellierung (Batch-Prozess)

Da Batch-Prozesse in unterschiedlichen Schritten unterschiedliche Signalverteilungen haben, wird step-wise gearbeitet:

Für jeden CuStepNo:
1. filtere Trainingsdaten dieses Steps (nur Normaldaten)
2. fit scaler (optional pro Step)
3. trainiere Modell für diesen Step
4. berechne Scores für VAL/TEST innerhalb dieses Steps

Damit existiert pro Step *ein eigenes Modell je Verfahren* (z. B. Step 1: Z-Score/IF/OCSVM/SOM separat).

---

## 6) Modelle & Score-Definition

Alle Modelle liefern einen kontinuierlichen *Anomalie-Score* (höher = anomaler).  
Die finale Vorhersage pred ∈ {0,1} entsteht später über einen Threshold.

### 6.1 Z-Score (univariat)
- pro Feature: Z = (x - μ) / σ
- Score pro Sample: max(|Z|) über alle Features
- train: μ, σ aus Normaldaten (pro Step)

### 6.2 Isolation Forest (multivariat)
- train: IsolationForest auf skalierten Normaldaten
- Score: -decision_function(X) (invertiert, damit höher = anomal)

Wichtige Hyperparameter:
- n_estimators, max_samples, max_features, bootstrap, contamination

### 6.3 One-Class SVM
- train: OCSVM auf skalierten Normaldaten
- Score: -decision_function(X) (höher = anomal)

Wichtige Hyperparameter:
- nu, kernel, gamma

### 6.4 SOM (Self-Organizing Map)
- train: SOM-Codebook W per iterativem Update (SGD-artig)
- Score: mittlere Distanz zu den k nächsten Codebook-Vektoren

Wichtige Hyperparameter:
- Gridgröße m×n, k, n_iter, alpha, sigma

---

## 7) Threshold-Logik (datengetrieben)

Es gibt keinen festen Threshold, da Score-Skalen modellabhängig sind.  
Stattdessen:

1. berechne Scores auf VAL (scores_val)
2. erzeuge Threshold-Kandidaten aus hohen Quantilen:
   - z. B. Quantile von 0.80 bis 0.999
3. teste jeden Threshold:
   - pred = (scores_val > thr)
   - berechne F1 auf VAL
4. wähle Threshold mit maximalem F1

Der ausgewählte Threshold wird anschließend unverändert auf TEST angewendet.

---

## 8) Optuna-Optimierung (GridSearch-like)

Für jedes Modell wird ein eigener Suchraum definiert.  
Optuna wird mit GridSampler so verwendet, dass alle Kombinationen systematisch getestet werden (ähnlich GridSearch).

Pro Trial:
- Hyperparameter vorschlagen
- step-wise trainieren + Scores berechnen
- Threshold auf VAL über Quantile optimieren
- Zielfunktion = F1_VAL

Nebenwerte werden im Trial gespeichert:
- F1_TEST
- BestThreshold
- AnomalyRate_VAL, AnomalyRate_TEST

Am Ende liefert das Study:
- beste Parameter (best.params)
- bestes F1_VAL (Optimierungsziel)
- F1_TEST als Generalisierungscheck

---

## 9) Outputs / Reporting

Typische Outputs:
- Tabelle: bestes Modell / beste Parameter / F1_VAL / F1_TEST / Threshold / Anomalieraten
- Plot: F1 vor vs. nach Optuna pro Step (zeigt Effekt der Parameter- und Threshold-Optimierung)

Interpretation:
- F1 ist Hauptmetrik (balanciert Precision/Recall)
- AnomalyRate ist ein Praxischeck (zu hoch → zu viele Fehlalarme; zu niedrig → verpasst Anomalien)