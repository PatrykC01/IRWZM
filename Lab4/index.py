# /// script
# requires-python = ">=3.10"
# dependencies =[
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "matplotlib",
#     "shap",
#     "lime"
# ]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import lime.lime_tabular
import warnings

warnings.filterwarnings("ignore")

def main():
    print("--- 1. WCZYTYWANIE I PRZYGOTOWANIE DANYCH ---")
    df = pd.read_csv("Myocardial infarction complications Database.csv", na_values=["", "?", "NA"])
    df = df.dropna(subset=['LET_IS'])
    
    y = (df['LET_IS'] > 0).astype(int)
    
    num_features =['AGE', 'S_AD_ORIT', 'D_AD_ORIT', 'K_BLOOD', 'NA_BLOOD', 'L_BLOOD']
    cat_features = ['SEX', 'INF_ANAM']
    features = num_features + cat_features
    
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("--- 2. BUDOWA I TRENOWANIE MODELU ---")
    num_transformer = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    model = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    print(f"ACC: {accuracy_score(y_test, pred):.4f} | AUC: {roc_auc_score(y_test, proba):.4f}\n")
    
    print("--- 3. KRZYWA ROC I MACIERZ POMYŁEK ---")
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, proba):.3f}', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Krzywa ROC - Predykcja Zgonu")
    plt.legend(loc="lower right")
    plt.show()

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Przeżycie", "Zgon"])
    disp.plot(cmap="Blues")
    plt.title("Macierz pomyłek")
    plt.show()
    
    print("--- 4. PERMUTATION IMPORTANCE (PI) ---")
    pi = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    pi_df = pd.DataFrame({"Cecha": X_test.columns, "Waznosc": pi.importances_mean})
    pi_df = pi_df.sort_values("Waznosc", ascending=True)
    
    plt.figure(figsize=(8, 5))
    plt.barh(pi_df["Cecha"], pi_df["Waznosc"], color='skyblue')
    plt.xlabel("Spadek AUC/ACC po permutacji cechy")
    plt.title("Permutation Importance - Ważność cech")
    plt.tight_layout()
    plt.show()

    print("--- 5. ZALEŻNOŚCI GLOBALNE (PDP) I LOKALNE (ICE) ---")
    fig, ax = plt.subplots(figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(model, X_test, features=[num_features[0], num_features[1]], kind="average", ax=ax)
    plt.suptitle("Partial Dependence Plot (PDP)")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(model, X_test, features=['AGE'], kind="individual", subsample=100, ax=ax)
    plt.suptitle("ICE Plot dla wieku (AGE)")
    plt.tight_layout()
    plt.show()

    print("--- 6. SHAP - WYJAŚNIENIA LOKALNE I GLOBALNE ---")
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['oh']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_features)
    all_feature_names = num_features + list(encoded_cat_names)
    
    explainer = shap.TreeExplainer(model.named_steps['clf'])
    shap_explanation = explainer(X_test_trans)
    
    if len(shap_explanation.shape) == 3:
        shap_explanation_pos = shap_explanation[:, :, 1]
    else:
        shap_explanation_pos = shap_explanation
        
    shap_explanation_pos.feature_names = all_feature_names

    shap.plots.beeswarm(shap_explanation_pos, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()
    
    idx = 0
    shap.plots.waterfall(shap_explanation_pos[idx], show=False)
    plt.title(f"SHAP Waterfall Plot - Pacjent {idx}")
    plt.tight_layout()
    plt.show()

    print("--- 7. LIME - WYJAŚNIENIE LOKALNE DLA POJEDYNCZEGO PACJENTA ---")
    
    num_imputer = preprocessor.transformers_[0][1].named_steps['imp']
    X_train_num_imp = num_imputer.transform(X_train[num_features])
    X_test_num_imp  = num_imputer.transform(X_test[num_features])

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_num_imp,
        training_labels=y_train.values,
        feature_names=num_features,
        class_names=["Przeżył", "Zgon"],
        discretize_continuous=True,
        discretizer='entropy', 
        mode="classification",
        random_state=42
    )

    def make_predict_fn_for_patient(cat_values):
        def predict_fn(x_num):
            df_temp = pd.DataFrame(x_num, columns=num_features)
            for feat in cat_features:
                df_temp[feat] = cat_values[feat]
            return model.predict_proba(df_temp)
        return predict_fn

    cat_vals_idx = X_test.iloc[idx][cat_features]
    predict_fn = make_predict_fn_for_patient(cat_vals_idx)

    exp = lime_explainer.explain_instance(
        data_row=X_test_num_imp[idx],
        predict_fn=predict_fn,
        num_features=5,
        labels=(1,)
    )

    print(f"\nWyjaśnienie LIME dla pacjenta {idx} (Wpływ na ryzyko Zgonu):")
    for feat, val in exp.as_list(label=1):
        print(f"{feat} => {val:.4f}")
        
    fig = exp.as_pyplot_figure()
    plt.title(f"Wykres wag LIME - Pacjent {idx}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
