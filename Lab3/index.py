# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "pandas>=3.0.1",
#     "scikit-learn>=1.8.0",
# ]
# ///

# ==========================================
# ETAP 0: ŚRODOWISKO I BIBLIOTEKI
# ==========================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

plt.rcParams["figure.figsize"] = (8, 5)

# ==========================================
# ETAP 1: PRZYGOTOWANIE DANYCH TEKSTOWYCH
# ==========================================

try:
    df = pd.read_csv("variant_03_er.csv")
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku 'variant_03_er.csv'. Upewnij się, że znajduje się w tym samym folderze co skrypt.")
    exit()

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'\d+', '', s) # Usunięcie liczb
    s = re.sub(r'[^\w\s]', ' ', s) # Usunięcie interpunkcji/znaków specjalnych
    s = re.sub(r'\s+', ' ', s).strip() # Usunięcie nadmiarowych spacji
    return s

df["clean"] = df["text"].apply(clean_text)
df["tokens"] = df["clean"].str.split()

print("--- ETAP 1: DANE PO OCZYSZCZENIU ---")
print(df[["clean", "tokens"]].head())

# ==========================================
# ETAP 2: ROZPOZNAWANIE JEDNOSTEK (NER)
# ==========================================
def simple_dict_ner(texts):
    lex = {
        "DISEASE":["trauma", "fracture", "pneumonia", "fever", "cough", "chest pain", "stemi", "bradycardia", "bite"],
        "DRUG":["ceftriaxone", "tetanus", "amoxicillin"],
        "PROCEDURE":["ct", "x ray", "consult", "ecg", "pci"]
    }
    results =[]
    for s in texts:
        low = s.lower()
        ents =[]
        for label, words in lex.items():
            for w in words:
                if re.search(r'\b' + re.escape(w) + r'\b', low):
                    ents.append((w, label))
        results.append(ents)
    return results

df["entities"] = simple_dict_ner(df["clean"].tolist())

print("\n--- ETAP 2: WYKRYTE JEDNOSTKI MEDYCZNE (NER) ---")
print(df[["clean", "entities"]].head())

all_ents = [ent[0] for ents in df["entities"] for ent in ents]
ent_counts = Counter(all_ents)

# ==========================================
# ETAP 3: KLASYFIKACJA DOKUMENTÓW
# ==========================================
X = df["clean"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
clf = LogisticRegression(max_iter=1000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

print("\n--- ETAP 3: WYNIKI KLASYFIKACJI ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nRaport klasyfikacji:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# ETAP 4: WIZUALIZACJA I INTERPRETACJA
# ==========================================
print("\n--- ETAP 4: WIZUALIZACJA ---")
def highlight_entities(text, entities):
    out = text
    entities_sorted = sorted(entities, key=lambda x: len(x[0]), reverse=True)
    for ent, label in entities_sorted:
        
        pattern = r'\b' + re.escape(ent) + r'\b'
        out = re.sub(pattern, f"[{ent.upper()}<{label}>]", out)
    return out

print("Fragmenty tekstu z oznaczonymi jednostkami medycznymi:\n")
for i in range(len(df)):
    t = df.loc[i, "clean"]
    ents = df.loc[i, "entities"]
    print(f"- {highlight_entities(t, ents)}")

plt.figure(figsize=(10, 5))
plt.bar(*zip(*ent_counts.most_common(10)), color='skyblue')
plt.title("Najczęściej występujące jednostki medyczne (NER)")
plt.xlabel("Jednostka")
plt.ylabel("Częstość")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Macierz pomyłek - klasyfikacja ER")
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y), rotation=45)
plt.yticks(tick_marks, np.unique(y))

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > (cm.max() / 2) else "black")

plt.ylabel("Rzeczywiste")
plt.xlabel("Przewidywane")
plt.tight_layout()
plt.show()
