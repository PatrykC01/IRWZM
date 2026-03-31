# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "joblib>=1.5.3",
#     "matplotlib>=3.10.8",
#     "pandas>=2.3.3",
#     "scikit-learn>=1.8.0",
#     "streamlit>=1.55.0",
# ]
# ///
import sys
import streamlit as st
import streamlit.runtime


if not streamlit.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv =["streamlit", "run", __file__]
    sys.exit(stcli.main())


import os
from datetime import datetime

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

DATA_PATH = "health_measurements.csv"
MODEL_PATH = "risk_model.joblib"

st.set_page_config(page_title="Monitor zdrowia + ML", layout="centered")
st.title("📱 Monitor zdrowia + analiza ML (Wariant 3)")


def ensure_data_file():
    if not os.path.exists(DATA_PATH):
        df = pd.DataFrame(columns=[
            "timestamp", "age", "bmi", "glucose", "systolic_bp", "diastolic_bp", "note"
        ])
        df.to_csv(DATA_PATH, index=False)

def load_data():
    ensure_data_file()
    df = pd.read_csv(DATA_PATH)
   
    if "note" not in df.columns:
        df["note"] = ""
    return df

def append_measurement(row: dict):
    df = load_data()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

def make_demo_label(df: pd.DataFrame) -> pd.Series:
    """
    Etykieta do celów dydaktycznych (nie jest diagnozą!):
    1 jeśli SBP>=140 lub DBP>=90, inaczej 0.
    """
    return ((df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)).astype(int)

def train_model(df: pd.DataFrame):

    if len(df) < 20:
        raise ValueError("Za mało danych do trenowania (min. 20 pomiarów). Dodaj więcej wpisów.")

    df_ml = df.copy()
    df_ml["timestamp"] = pd.to_datetime(df_ml["timestamp"], errors="coerce")
    df_ml = df_ml.sort_values("timestamp")

    # WARIANT 3: Tworzenie cech pochodnych (średnia krocząca z 7 pomiarów)
    df_ml["systolic_bp_ma7"] = df_ml["systolic_bp"].rolling(window=7, min_periods=1).mean()
    df_ml["diastolic_bp_ma7"] = df_ml["diastolic_bp"].rolling(window=7, min_periods=1).mean()

    y = make_demo_label(df_ml)
    # WARIANT 3: Dodanie cech pochodnych do wejścia modelu
    X = df_ml[["age", "bmi", "glucose", "systolic_bp", "diastolic_bp", "systolic_bp_ma7", "diastolic_bp_ma7"]].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    num_cols = list(X.columns)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )

    clf = Pipeline(steps=[
        ("pre", pre),
        ("model", LogisticRegression(max_iter=2000))
    ])

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, digits=3, zero_division=0)
    }

    joblib.dump({"model": clf, "metrics": metrics}, MODEL_PATH)
    return clf, metrics

def load_model():
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        return obj["model"], obj["metrics"]
    return None, None


# =========================
# ETAP 1: Zbieranie danych
# =========================
st.header("Etap 1 — Zbieranie danych zdrowotnych")

with st.form("health_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Wiek [lata]", min_value=18, max_value=110, value=40, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
        glucose = st.number_input("Glukoza [mg/dl]", min_value=40, max_value=300, value=95, step=1)
    with col2:
        systolic_bp = st.number_input("Ciśnienie skurczowe SBP [mmHg]", min_value=70, max_value=260, value=120, step=1)
        diastolic_bp = st.number_input("Ciśnienie rozkurczowe DBP [mmHg]", min_value=40, max_value=150, value=80, step=1)
    
    # WARIANT 3: Dodanie pola "note"
    note = st.text_input("Krótka notatka (np. po biegu, stresujący dzień) - opcjonalnie")

    submitted = st.form_submit_button("💾 Zapisz pomiar")

if submitted:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "age": int(age),
        "bmi": float(bmi),
        "glucose": int(glucose),
        "systolic_bp": int(systolic_bp),
        "diastolic_bp": int(diastolic_bp),
        "note": note,
    }
    append_measurement(row)
    st.success("Zapisano pomiar do pliku health_measurements.csv")

df = load_data()
st.caption(f"Liczba zapisanych pomiarów: {len(df)}")

st.dataframe(df[["timestamp", "systolic_bp", "diastolic_bp", "note"]].tail(10), use_container_width=True)

# =====================================
# ETAP 2: Analiza i wizualizacja danych
# =====================================
st.header("Etap 2 — Analiza i wizualizacja (Trendy i Średnie Kroczące)")

if len(df) == 0:
    st.info("Dodaj co najmniej jeden pomiar, aby zobaczyć analizę.")
else:
    st.subheader("Wykres trendu ze średnią kroczącą (Wariant 3)")
    plot_cols = st.multiselect(
        "Wybierz parametry do wykresu (wybranie SBP/DBP pokaże również średnią):",
        options=["bmi", "glucose", "systolic_bp", "diastolic_bp"],
        default=["systolic_bp", "diastolic_bp"]
    )

    if plot_cols:
        df_plot = df.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
        df_plot = df_plot.dropna(subset=["timestamp"]).sort_values("timestamp")

        # WARIANT 3: Obliczenie średniej kroczącej w wizualizacji
        df_plot["systolic_bp_ma7"] = df_plot["systolic_bp"].rolling(window=7, min_periods=1).mean()
        df_plot["diastolic_bp_ma7"] = df_plot["diastolic_bp"].rolling(window=7, min_periods=1).mean()
        
        df_plot = df_plot.tail(50)

        fig = plt.figure(figsize=(9, 5))
        for c in plot_cols:
            
            plt.plot(df_plot["timestamp"], df_plot[c], marker='o', label=c)
            # WARIANT 3: Nakładanie średnich kroczących (linia przerywana) dla ciśnienia
            if c == "systolic_bp":
                plt.plot(df_plot["timestamp"], df_plot["systolic_bp_ma7"], linestyle='--', color='blue', alpha=0.6, label="SBP (średnia z 7)")
            if c == "diastolic_bp":
                plt.plot(df_plot["timestamp"], df_plot["diastolic_bp_ma7"], linestyle='--', color='orange', alpha=0.6, label="DBP (średnia z 7)")

        plt.xlabel("Czas")
        plt.ylabel("Wartość")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)


# ==============================
# ETAP 3: Model uczenia maszynowego
# ==============================
st.header("Etap 3 — Budowa modelu ML (z cechami pochodnymi)")

model, metrics = load_model()

colA, colB = st.columns([1, 2])
with colA:
    if st.button("🧠 Wytrenuj / odśwież model"):
        try:
            model, metrics = train_model(df)
            st.success("Model wytrenowano! Uwzględniono średnie kroczące.")
        except Exception as e:
            st.error(str(e))

with colB:
    if metrics:
        st.subheader("Metryki (na części testowej)")
        st.write(f"Accuracy: **{metrics['accuracy']:.3f}**")
        if metrics["roc_auc"] is not None:
            st.write(f"ROC AUC: **{metrics['roc_auc']:.3f}**")
    else:
        st.info("Model nie jest jeszcze wytrenowany.")


# ===================================
# ETAP 4: Integracja modelu z aplikacją
# ===================================
st.header("Etap 4 — Predykcja w aplikacji (Ocena Trendu)")

if model is None:
    st.warning("Najpierw wytrenuj model w Etapie 3.")
elif len(df) == 0:
    st.warning("Brak pomiarów do predykcji.")
else:
    st.subheader("Predykcja ryzyka dla ostatniego pomiaru z CSV")
    
    df_pred = df.copy()
    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"], errors="coerce")
    df_pred = df_pred.sort_values("timestamp")
    
    df_pred["systolic_bp_ma7"] = df_pred["systolic_bp"].rolling(window=7, min_periods=1).mean()
    df_pred["diastolic_bp_ma7"] = df_pred["diastolic_bp"].rolling(window=7, min_periods=1).mean()
    
    current_row = df_pred.iloc[-1]
    X_current = pd.DataFrame([current_row[["age", "bmi", "glucose", "systolic_bp", "diastolic_bp", "systolic_bp_ma7", "diastolic_bp_ma7"]]])
    
    proba_current = float(model.predict_proba(X_current)[0, 1])
    pred_current = int(proba_current >= 0.5)
    
    st.write(f"Prawdopodobieństwo podwyższonego ryzyka: **{proba_current:.3f}**")
    
    if pred_current == 1:
        st.error("Wynik: **podwyższone ryzyko (demo)**")
    else:
        st.success("Wynik: **niskie ryzyko (demo)**")

    # WARIANT 3: Określenie trendu poprzez porównanie predykcji poprzedniej vs obecnej
    if len(df_pred) >= 2:
        prev_row = df_pred.iloc[-2]
        X_prev = pd.DataFrame([prev_row[["age", "bmi", "glucose", "systolic_bp", "diastolic_bp", "systolic_bp_ma7", "diastolic_bp_ma7"]]])
        proba_prev = float(model.predict_proba(X_prev)[0, 1])
        
        diff = proba_current - proba_prev
        st.write(f"Poprzednie prawdopodobieństwo: **{proba_prev:.3f}**")
        
        if diff > 0.05:
            st.warning("📈 **Trend rosnący:** Ryzyko istotnie wzrosło w porównaniu do poprzedniego pomiaru.")
        elif diff < -0.05:
            st.info("📉 **Trend malejący:** Ryzyko zmalało. Dobra robota!")
        else:
            st.info("➡️ **Trend stabilny:** Ryzyko utrzymuje się na zbliżonym poziomie.")
    else:
        st.info("To Twój pierwszy pomiar. Dodaj kolejny, aby zobaczyć trend ryzyka.")

st.divider()
st.caption("Pliki lokalne: health_measurements.csv (historia), risk_model.joblib (model).")

