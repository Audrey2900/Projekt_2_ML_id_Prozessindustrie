# MetaOD – Automatische Modellauswahl für Outlier Detection

**MetaOD** ist ein meta-learning-basierter Ansatz zur automatischen Auswahl von Outlier-Detection-Modellen.  
Es nutzt historische Datensätze und Modell-Performances, um für neue, unbekannte Datensätze direkt das am besten geeignete Modell vorherzusagen.

- Verwendet **Meta-Features**, um Ähnlichkeiten zwischen neuen und historischen Datensätzen zu erkennen.
- Optimiert die **Rangordnung der Modelle** (Top-1) über **Smoothed DCG**, nicht die absoluten Performance-Werte.
- Offline-Training der Meta-Learner ermöglicht eine **schnelle Online-Modellauswahl ohne Labels**.

---

## Installation

**Conda-Umgebung erstellen:**

```bash
conda create -n metaod_new python=3.7
conda activate metaod_new
```

### Requirements (Python-Pakete)

```
joblib>=0.14.1          # Serialisierung von Modellen
liac-arff               # Lesen von ARFF-Datasets
numpy>=1.18.1           # Numerische Berechnungen
scipy>=0.20             # Wissenschaftliche Berechnungen
scikit_learn==0.22.1    # Klassische ML-Modelle und Tools
pandas>=0.20            # Datenmanipulation
pyod>=0.8               # Outlier Detection Bibliothek
```

### Lokales Pretrained-Modell vorbereiten

Wenn kein Internet verfügbar ist, können die Pretrained-Modelle lokal genutzt werden:

```python
from zipfile import ZipFile
import os

def prepare_trained_model_local(filename='trained_models.zip', save_path='trained_models'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nicht gefunden! Bitte manuell herunterladen.")
    
    with ZipFile(filename, 'r') as zip:
        print('Extracting trained models now...')
        zip.extractall(save_path)
        print('Finish extracting models')
```

- `trained_models.zip` enthält historische OD-Modelle.  
- Nach Extraktion kann `select_model()` die **Top-Modelle** für neue Datensätze vorhersagen.

## Funktionsweise Code

### Schrittweise Unterteilung Code

#### 1. Daten vorbereiten
- CSV laden, Zeitstempel konvertieren, Anomalien markieren  
- Features und Labels trennen  
- Train/Test Split  
- Nur normale Daten für OD-Training nutzen  

#### 2. Feature-Skalierung
- StandardScaler auf Trainingsdaten fitten und Testdaten transformieren  

#### 3. MetaOD vorbereiten
- Lokale Pretrained-Modelle extrahieren (`prepare_trained_model_local`)  
- `select_model()` zur Modellvorauswahl  
- Meta-Features mit `generate_meta_features()` berechnen  

#### 4. Top-Modell auswählen & trainieren
- Modellname und Parameter extrahieren  
- Entsprechende PyOD-Klasse instanziieren  
- Training auf den Trainingsdaten  

#### 5. Evaluation
- Vorhersage auf Testdaten  
- Metriken: Accuracy, Recall, F1-Score, Average Precision


### Kurzdiagramm

Historische Datasets + Modellperformances
↓ (offline Training)
Meta-Learner + Embeddings (U,V) lernen
↓
Neue Daten (Meta-Features)
↓
Embeddings für neuen Datensatz berechnen
↓
Modell-Performance vorhersagen
↓
Bestes Modell auswählen & trainieren

## Referenzen

- [Zhao et al., *Automating Outlier Detection via Meta-Learning*, 2021](https://arxiv.org/pdf/2009.10606)
- [MetaOD GitHub Code](https://github.com/yzhao062/MetaOD/tree/master)