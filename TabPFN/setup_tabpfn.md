# TabPFN Setup für Anomaly Detection

Diese Anleitung beschreibt **alle notwendigen Schritte**, um **TabPFN für Anomaly Detection** zu verwenden – inklusive:

- Python-Umgebung erstellen  
- Installation von **PyTorch mit CUDA**
- Installation von **TabPFN + Extensions**
- **Hugging Face Account erstellen**
- Erstmalige Authentifizierung
- Testlauf

---

## 1️⃣ Voraussetzungen

### Systemanforderungen

- Python **3.9+**
- NVIDIA GPU (empfohlen)
- CUDA-fähiger Treiber installiert
- Internetverbindung (für Model-Download)

---

## 2️⃣ Virtuelle Umgebung erstellen

### Mit `conda`

```bash
conda create -n tabpfn_env python=3.11
conda activate tabpfn_env
```

---

## 3️⃣ PyTorch mit CUDA installieren (WICHTIG)

TabPFN läuft performant nur mit **GPU + CUDA**.

Gehe auf:

👉 https://pytorch.org/get-started/locally/

Wähle:
- Stable
- Pip
- Python
- CUDA-Version passend zu deinem System (z. B. 12.1)

Beispiel (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### ✅ CUDA Installation prüfen

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Wenn `True` erscheint → GPU wird korrekt erkannt.

---

## 4️⃣ TabPFN installieren

```bash
pip install tabpfn
pip install tabpfn-extensions
```

Optional (falls noch nicht vorhanden):

```bash
pip install pandas numpy
```

---

## 5️⃣ Hugging Face Account erstellen (ERFORDERLICH)

TabPFN lädt die Modelle von Hugging Face herunter.

### Schritt 1: Account erstellen

👉 https://huggingface.co/join

Account erstellen und E-Mail bestätigen.

---

### Schritt 2: TabPFN Modell dem Account hinzufügen

1. Gehe zu:  
   👉 https://huggingface.co/Prior-Labs/tabpfn_2_5
2. Fülle die Informationen aus und klicke auf **Agree to license terms and send request to access repo.** (Teilt Email und Username mit Autoren)
3. Wähle:
   - Role: **Read**
4. Token kopieren


### Schritt 3: Access Token erzeugen

1. Gehe zu:  
   👉 https://huggingface.co/settings/tokens  
2. Klicke auf **Create new token**
3. Wähle:
   - **Read access to contents of all public gated repos you can access** unter **Repositories**
4. Token erstellen und kopieren

---

## 6️⃣ Hugging Face CLI installieren & authentifizieren

```bash
pip install huggingface_hub
```

Dann:

```bash
hf auth login
```

Token einfügen.

Ergebnis sollte sein:

```
Login successful
```

---

## 7️⃣ Beispiel: TabPFN für Anomaly Detection verwenden

TabPFN kann über das **Unsupervised Extension Modul** genutzt werden.

```python
import pandas as pd
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

# Beispieldaten
df = pd.read_csv("data.csv")

model = TabPFNUnsupervisedModel(device="cuda")

model.fit(df)

anomaly_scores = model.predict(df)

print(anomaly_scores[:10])
```

---

## 8️⃣ Empfohlene requirements.txt

```txt
pandas>=2.3.3
numpy>=2.4.1

--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.10.0+cu126
torchvision==0.25.0+cu126

tabpfn>=6.3.1
tabpfn-extensions>=0.2.2

huggingface-hub>=1.3.4
```

---

# ✅ Fertig!

Du kannst jetzt:

- TabPFN für **Classification**
- TabPFN für **Regression**
- `TabPFNUnsupervisedModel` für **Anomaly Detection**

verwenden.
