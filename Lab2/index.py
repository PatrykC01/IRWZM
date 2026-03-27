# /// script
# requires-python = ">=3.12"
# dependencies =[
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.3",
#     "pandas>=3.0.1",
#     "scikit-learn>=1.8.0",
#     "tensorflow>=2.21.0",
#     "medmnist>=3.0.1",
# ]
# ///
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, mean_squared_error, r2_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import medmnist
from medmnist import PneumoniaMNIST

import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 0. WCZYTANIE DANYCH KLINICZNYCH
# =====================================================================
print("--- WCZYTYWANIE DANYCH KLINICZNYCH ---")
file_path = "Myocardial infarction complications Database.csv"
df = pd.read_csv(file_path)

# =====================================================================
# ZADANIE 1: Zbuduj model regresji logistycznej do klasyfikacji pacjentów
# =====================================================================
print("\n=== ZADANIE 1: KLASYFIKACJA (Regresja logistyczna) ===")

df['target_clf'] = (df['LET_IS'] > 0).astype(int)
clf_features =['AGE', 'SEX', 'S_AD_ORIT', 'D_AD_ORIT', 'K_BLOOD', 'NA_BLOOD', 'L_BLOOD', 'ROE']

X_clf = df[clf_features]
y_clf = df['target_clf']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf)

preprocessor_clf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

clf_pipeline = Pipeline(steps=[
    ("prep", preprocessor_clf),
    ("classifier", LogisticRegression(max_iter=1000, class_weight='balanced'))
])
clf_pipeline.fit(X_train_c, y_train_c)

y_pred_c = clf_pipeline.predict(X_test_c)
y_proba_c = clf_pipeline.predict_proba(X_test_c)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test_c, y_pred_c):.3f}")
print(f"Precision: {precision_score(y_test_c, y_pred_c, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test_c, y_pred_c):.3f}")
print(f"F1-score:  {f1_score(y_test_c, y_pred_c):.3f}")

fpr, tpr, _ = roc_curve(y_test_c, y_proba_c)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test_c, y_proba_c):.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Zadanie 1: Krzywa ROC - Klasyfikacja (Komplikacje)")
plt.legend()
plt.show()

# =====================================================================
# ZADANIE 2: Zastosuj regresję liniową do przewidywania danych klinicznych
# =====================================================================
print("\n=== ZADANIE 2: REGRESJA LINIOWA ===")

df_reg = df.dropna(subset=['S_AD_ORIT'])
reg_features =['AGE', 'SEX', 'D_AD_ORIT', 'K_BLOOD', 'NA_BLOOD', 'L_BLOOD', 'ROE']

X_reg = df_reg[reg_features]
y_reg = df_reg['S_AD_ORIT']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

reg_pipeline = Pipeline(steps=[
    ("prep", preprocessor_clf),
    ("regressor", LinearRegression())
])
reg_pipeline.fit(X_train_r, y_train_r)

y_pred_r = reg_pipeline.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)
print(f"RMSE: {rmse:.2f}")
print(f"R^2:  {r2:.3f}")

plt.figure(figsize=(6,4))
plt.scatter(y_test_r, y_pred_r, alpha=0.5, color='blue')
plt.plot([y_test_r.min(), y_test_r.max()],[y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Rzeczywiste ciśnienie skurczowe (S_AD_ORIT)")
plt.ylabel("Przewidywane ciśnienie skurczowe")
plt.title("Zadanie 2: Rzeczywiste vs Przewidywane (Regresja)")
plt.show()

# =====================================================================
# ZADANIE 3: Prosta sieć CNN rozpoznająca obrazy medyczne (RTG płuc)
# Cel: Klasyfikacja binarna (Zdrowe płuca vs Zapalenie płuc)
# =====================================================================
print("\n=== ZADANIE 3: SIECI NEURONOWE CNN (PneumoniaMNIST: Zdrowe vs Chore) ===")


print("Pobieranie zbioru PneumoniaMNIST (zdjęcia 28x28 pikseli)...")
train_dataset = PneumoniaMNIST(split='train', download=True)
val_dataset = PneumoniaMNIST(split='val', download=True)
test_dataset = PneumoniaMNIST(split='test', download=True)

X_train_i, y_train_i = train_dataset.imgs, train_dataset.labels
X_val_i, y_val_i = val_dataset.imgs, val_dataset.labels
X_test_i, y_test_i = test_dataset.imgs, test_dataset.labels

X_train_i = X_train_i.astype("float32") / 255.0
X_val_i = X_val_i.astype("float32") / 255.0
X_test_i = X_test_i.astype("float32") / 255.0

X_train_i = np.expand_dims(X_train_i, axis=-1)
X_val_i = np.expand_dims(X_val_i, axis=-1)
X_test_i = np.expand_dims(X_test_i, axis=-1)

model = keras.Sequential([
    layers.Conv2D(16, (3,3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid") 
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("\nTrenowanie modelu CNN na zdjęciach RTG...")
history = model.fit(X_train_i, y_train_i, 
                    validation_data=(X_val_i, y_val_i),
                    epochs=10, 
                    batch_size=32, 
                    verbose=1)

test_loss, test_acc = model.evaluate(X_test_i, y_test_i, verbose=0)
print(f"\nCNN (PneumoniaMNIST) - Dokładność na danych testowych: {test_acc:.3f}")
print(f"CNN (PneumoniaMNIST) - Wartość funkcji straty na testowych: {test_loss:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history.history['accuracy'], label='Trening')
ax[0].plot(history.history['val_accuracy'], label='Walidacja')
ax[0].set_title('Krzywa uczenia: Dokładność (Accuracy)')
ax[0].set_xlabel('Epoka')
ax[0].set_ylabel('Dokładność')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Trening')
ax[1].plot(history.history['val_loss'], label='Walidacja')
ax[1].set_title('Krzywa uczenia: Funkcja straty (Loss)')
ax[1].set_xlabel('Epoka')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.show()
