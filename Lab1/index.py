# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.2",
#     "pandas>=3.0.1",
#     "scikit-learn>=1.8.0",
#     "seaborn>=0.13.2",
# ]
# ///
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid")

# ==========================================
# 1. IMPORT DANYCH I ANALIZA STATYSTYCZNA
# ==========================================
print("--- ETAP 1: Import danych ---")
file_path = "Myocardial infarction complications Database.csv"
df = pd.read_csv(file_path)

print(f"Kształt zbioru danych: {df.shape}")
print(df.head())
print("\nPodstawowe statystyki opisowe zmiennych numerycznych:")
print(df.describe())

df = df.copy()

df['target'] = (df['LET_IS'] > 0).astype(int)

# ==========================================
# 2. UZUPEŁNIANIE BRAKÓW DANYCH I ZALEŻNOŚCI
# ==========================================
print("\n--- ETAP 2: Braki danych i korelacje ---")

missing_data = df.isna().sum().sort_values(ascending=False)
print("Największe braki danych w kolumnach:")
print(missing_data[missing_data > 0].head(10))

numeric_vars =['AGE', 'S_AD_KBRIG', 'D_AD_KBRIG', 'K_BLOOD', 'NA_BLOOD', 'L_BLOOD', 'ROE', 'target']
numeric_vars =[col for col in numeric_vars if col in df.columns]

plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_vars].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Macierz korelacji dla wybranych parametrów klinicznych")
plt.tight_layout()
plt.show()

# ==========================================
# 3. WIZUALIZACJA CZĘSTOŚCI KATEGORII
# ==========================================
print("\n--- ETAP 3: Wizualizacja częstości występowania kategorii ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x='SEX', ax=axes[0], hue='SEX', palette='Set2', legend=False)
axes[0].set_title("Rozkład płci pacjentów w populacji")
axes[0].set_xlabel("Płeć (Zakodowana)")
axes[0].set_ylabel("Liczba pacjentów")

sns.countplot(data=df, x='target', ax=axes[1], hue='target', palette='pastel', legend=False)
axes[1].set_title("Częstość występowania powikłań śmiertelnych (Target)")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Brak zgonu (0)', 'Zgon (1)'])
axes[1].set_ylabel("Liczba pacjentów")

plt.tight_layout()
plt.show()

# ==========================================
# 4 & 5. MODEL PREDYKCYJNY I PORÓWNANIE NORMALIZACJI
# ==========================================
print("\n--- ETAP 4 & 5: Modelowanie i porównanie metod normalizacji ---")

X = df.drop(columns=['ID', 'LET_IS', 'target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

imputer = SimpleImputer(strategy='median')

pipe_std = Pipeline([
    ('imputer', imputer),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
])

pipe_minmax = Pipeline([
    ('imputer', imputer),
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
])

print("Trenowanie modelu ze StandardScaler...")
pipe_std.fit(X_train, y_train)

print("Trenowanie modelu z MinMaxScaler...")
pipe_minmax.fit(X_train, y_train)

y_pred_std = pipe_std.predict(X_test)
y_proba_std = pipe_std.predict_proba(X_test)[:, 1]

y_pred_minmax = pipe_minmax.predict(X_test)
y_proba_minmax = pipe_minmax.predict_proba(X_test)[:, 1]

acc_std = accuracy_score(y_test, y_pred_std)
auc_std = roc_auc_score(y_test, y_proba_std)

acc_minmax = accuracy_score(y_test, y_pred_minmax)
auc_minmax = roc_auc_score(y_test, y_proba_minmax)

print("\n--- WYNIKI ---")
print(f"StandardScaler -> Accuracy: {acc_std:.4f} | ROC AUC: {auc_std:.4f}")
print(f"MinMaxScaler   -> Accuracy: {acc_minmax:.4f} | ROC AUC: {auc_minmax:.4f}")

# Wyrysowanie krzywych ROC
fpr_std, tpr_std, _ = roc_curve(y_test, y_proba_std)
fpr_minmax, tpr_minmax, _ = roc_curve(y_test, y_proba_minmax)

plt.figure(figsize=(8, 6))
plt.plot(fpr_std, tpr_std, label=f'StandardScaler (AUC = {auc_std:.3f})', color='blue')
plt.plot(fpr_minmax, tpr_minmax, label=f'MinMaxScaler (AUC = {auc_minmax:.3f})', color='green', linestyle='-.')
plt.plot([0, 1],[0, 1], 'k--', label='Model losowy')
plt.xlabel('Odsetek fałszywie pozytywnych (FPR)')
plt.ylabel('Odsetek prawdziwie pozytywnych (TPR)')
plt.title('Porównanie krzywych ROC dla różnych metod normalizacji')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

print("\nSzczegółowy raport klasyfikacji dla modelu (StandardScaler):")
print(classification_report(y_test, y_pred_std))
