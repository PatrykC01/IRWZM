# /// script
# dependencies =[
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
# ]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, average_precision_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

np.random.seed(42)
sns.set_theme(style="whitegrid")

print("1. Wczytywanie i przygotowanie danych...")

df = pd.read_csv('cases_clinical_for_lab12.csv')
target_col = 'high_risk_cvd'

X = df.drop(columns=[target_col])
y = df[target_col]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols =[c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print("2. Budowa i trening modeli (LR vs RF)...")

lr_model = Pipeline(steps=[
    ("pre", preprocess),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

rf_model = Pipeline(steps=[
    ("pre", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"))
])


lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# --- ZADANIE 1: Porównanie ROC AUC i PR AUC ---
print("\n=== PORÓWNANIE METRYK GLOBALNYCH ===")
proba_lr = lr_model.predict_proba(X_test)[:, 1]
proba_rf = rf_model.predict_proba(X_test)[:, 1]

roc_auc_lr = roc_auc_score(y_test, proba_lr)
pr_auc_lr = average_precision_score(y_test, proba_lr)

roc_auc_rf = roc_auc_score(y_test, proba_rf)
pr_auc_rf = average_precision_score(y_test, proba_rf)

print(f"Logistic Regression -> ROC AUC: {roc_auc_lr:.3f}, PR AUC: {pr_auc_lr:.3f}")
print(f"Random Forest       -> ROC AUC: {roc_auc_rf:.3f}, PR AUC: {pr_auc_rf:.3f}")

# --- ZADANIE 2: Dobór progów (Recall >= 0.85) ---
print("\n=== DOBÓR PROGU DLA RECALL = 0.85 ===")

def find_threshold_for_recall(y_true, probas, target_recall=0.85):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas)
    valid_idx = [i for i, r in enumerate(recalls[:-1]) if r >= target_recall]
    return thresholds[valid_idx[-1]] if valid_idx else 0.5

# Standardowy próg 0.5 dla obu modeli
pred_lr_05 = (proba_lr >= 0.5).astype(int)
pred_rf_05 = (proba_rf >= 0.5).astype(int)

tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_test, pred_lr_05).ravel()
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, pred_rf_05).ravel()

print("[Próg = 0.50]")
print(f" LR -> Recall: {tp_lr/(tp_lr+fn_lr):.3f} | FP: {fp_lr} | FN: {fn_lr}")
print(f" RF -> Recall: {tp_rf/(tp_rf+fn_rf):.3f} | FP: {fp_rf} | FN: {fn_rf}")

# Szukanie progu dla Recall >= 0.85
thr_lr = find_threshold_for_recall(y_test, proba_lr, 0.85)
thr_rf = find_threshold_for_recall(y_test, proba_rf, 0.85)

pred_lr_adj = (proba_lr >= thr_lr).astype(int)
pred_rf_adj = (proba_rf >= thr_rf).astype(int)

tn_lr_a, fp_lr_a, fn_lr_a, tp_lr_a = confusion_matrix(y_test, pred_lr_adj).ravel()
tn_rf_a, fp_rf_a, fn_rf_a, tp_rf_a = confusion_matrix(y_test, pred_rf_adj).ravel()

print("\n[Próg dostosowany (Docelowy Recall >= 0.85)]")
print(f" LR (próg={thr_lr:.3f}) -> Recall: {tp_lr_a/(tp_lr_a+fn_lr_a):.3f} | FP: {fp_lr_a} | FN: {fn_lr_a}")
print(f" RF (próg={thr_rf:.3f}) -> Recall: {tp_rf_a/(tp_rf_a+fn_rf_a):.3f} | FP: {fp_rf_a} | FN: {fn_rf_a}")


# --- ZADANIE 3: Permutation Importance dla RF ---
print("\n=== PERMUTATION IMPORTANCE (Random Forest) ===")
perm = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')

imp_df = pd.DataFrame({
    'Cecha': X.columns,
    'Znaczenie (Mean)': perm.importances_mean,
    'Odchylenie standardowe': perm.importances_std
}).sort_values('Znaczenie (Mean)', ascending=False)

print(imp_df.head(6).to_string(index=False))

plt.figure(figsize=(8,5))
sns.barplot(data=imp_df, x='Znaczenie (Mean)', y='Cecha', color='b')
plt.title('Permutation Importance - Random Forest')
plt.tight_layout()
plt.show()

# --- ANALIZA PRZYPADKÓW FP i FN DLA MODELU RF (PRZY PROGU 0.5) ---
print("\n=== STUDIA PRZYPADKÓW (False Positives i False Negatives) dla RF ===")

results = X_test.copy()
results['y_true'] = y_test.values
results['proba_rf'] = proba_rf
results['pred_rf'] = pred_rf_05

results['Error'] = 'OK'
results.loc[(results['y_true'] == 0) & (results['pred_rf'] == 1), 'Error'] = 'FP'
results.loc[(results['y_true'] == 1) & (results['pred_rf'] == 0), 'Error'] = 'FN'

fp_cases = results[results['Error'] == 'FP'].sort_values(by='proba_rf', ascending=False)
fn_cases = results[results['Error'] == 'FN'].sort_values(by='proba_rf', ascending=True)

print("\n--- Najpewniejsze False Positives (Model uważa, że to ryzyko CVD, a nie jest) ---")
print(fp_cases.head(3).to_string())

print("\n--- Najpewniejsze False Negatives (Model uważa, że ryzyka nie ma, a jest) ---")
print(fn_cases.head(3).to_string())
print("\nZakończono działanie skryptu.")
