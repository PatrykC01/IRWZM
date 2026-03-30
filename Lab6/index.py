# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
# ]
# ///

import csv
import numpy as np

# ==========================================
# FUNKCJE POMOCNICZE (ZBIORY ROZMYTE)
# ==========================================
def trimf(x, a, b, c):
    """Trójkątna funkcja przynależności."""
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a)
    y[x == b] = 1.0
    idx = (b < x) & (x < c)
    y[idx] = (c - x[idx]) / (c - b)
    return np.clip(y, 0, 1)

def trapmf(x, a, b, c, d):
    """Trapezowa funkcja przynależności."""
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a)
    idx = (b <= x) & (x <= c)
    y[idx] = 1.0
    idx = (c < x) & (x < d)
    y[idx] = (d - x[idx]) / (d - c)
    return np.clip(y, 0, 1)

# ==========================================
# UNIWERSA I ZBIORY ROZMYTE DLA WARIANTU 3
# ==========================================
RISK = np.linspace(0, 100, 101) 

risk_low    = trimf(RISK, 0, 20, 40)
risk_medium = trimf(RISK, 30, 50, 70)
risk_high   = trimf(RISK, 60, 80, 100)

def age_old(x): return trapmf([x], 55, 65, 120, 120)[0]

def sbp_low(x):    return trapmf([x], 0, 0, 110, 125)[0]
def sbp_border(x): return trimf([x], 120, 135, 150)[0]
def sbp_high(x):   return trapmf([x], 135, 145, 300, 300)[0]


# ==========================================
# 1. SYSTEM KLASYCZNY
# ==========================================
def classical_inference(patient):
    """Klasyczna reguła: wiek >= 65 AND SBP >= 140 -> Wysokie ryzyko."""
    age = float(patient["age"])
    sbp = float(patient["systolic_bp"])
    
    if age >= 65 and sbp >= 140:
        return "Wysokie ryzyko (Reguła spełniona)"
    return "Niskie/Umiarkowane (Reguła niespełniona)"


# ==========================================
# 2 & 3. SYSTEM ROZMYTY Z WYJAŚNIENIEM (XAI)
# ==========================================
def explainable_cv_risk(patient):
    age = float(patient["age"])
    sbp = float(patient["systolic_bp"])

    mu = {
        "AGE_old": age_old(age),
        "AGE_not_old": 1.0 - age_old(age),
        "SBP_low": sbp_low(sbp),
        "SBP_border": sbp_border(sbp),
        "SBP_high": sbp_high(sbp),
    }

    AND = min
    
    rules =[
        {
            "id": "R1",
            "text": "Jeśli SBP jest WYSOKIE, to ryzyko jest WYSOKIE.",
            "strength": mu["SBP_high"],
            "out_set": risk_high,
            "why": f"SBP_high={mu['SBP_high']:.2f}"
        },
        {
            "id": "R2",
            "text": "Jeśli wiek jest PODESZŁY I SBP jest GRANICZNE, to ryzyko jest WYSOKIE.",
            "strength": AND(mu["AGE_old"], mu["SBP_border"]),
            "out_set": risk_high,
            "why": f"AGE_old={mu['AGE_old']:.2f} AND SBP_border={mu['SBP_border']:.2f}"
        },
        {
            "id": "R3",
            "text": "Jeśli wiek NIE JEST PODESZŁY I SBP jest GRANICZNE, to ryzyko jest ŚREDNIE.",
            "strength": AND(mu["AGE_not_old"], mu["SBP_border"]),
            "out_set": risk_medium,
            "why": f"AGE_not_old={mu['AGE_not_old']:.2f} AND SBP_border={mu['SBP_border']:.2f}"
        },
        {
            "id": "R4",
            "text": "Jeśli SBP jest NISKIE, to ryzyko jest NISKIE.",
            "strength": mu["SBP_low"],
            "out_set": risk_low,
            "why": f"SBP_low={mu['SBP_low']:.2f}"
        }
    ]

    clipped =[]
    for r in rules:
        clipped_set = np.minimum(r["strength"], r["out_set"])
        clipped.append(clipped_set)
        r["clipped_area"] = float(clipped_set.sum())

    aggregated = np.maximum.reduce(clipped) if clipped else np.zeros_like(RISK)

    if aggregated.sum() == 0:
        crisp = 0.0
    else:
        crisp = float((RISK * aggregated).sum() / aggregated.sum())

    total_area = sum(r["clipped_area"] for r in rules) + 1e-12
    for r in rules:
        r["contribution_pct"] = 100.0 * r["clipped_area"] / total_area

    rules_sorted = sorted(rules, key=lambda r: r["contribution_pct"], reverse=True)
    
    age_impact_desc = ""
    if mu["AGE_old"] > 0 and mu["SBP_border"] > 0:
        age_impact_desc = f"Wiek pacjenta ({age} lat) aktywował zbiór 'Wiek podeszły' w {mu['AGE_old']*100:.0f}%. W połączeniu z granicznym ciśnieniem (SBP {sbp}), wiek ZNACZĄCO PODWYŻSZYŁ końcowe ryzyko (Reguła R2)."
    elif mu["AGE_not_old"] > 0 and mu["SBP_border"] > 0:
        age_impact_desc = f"Młodszy wiek pacjenta ({age} lat) uchronił go przed wysokim ryzykiem. Mimo granicznego ciśnienia, brak podeszłego wieku zachował ryzyko na ŚREDNIM poziomie (Reguła R3)."
    elif mu["SBP_high"] == 1.0:
        age_impact_desc = f"Ciśnienie jest tak wysokie (SBP {sbp}), że wiek przestał być dominującym czynnikiem. Ryzyko jest wysokie niezależnie od wieku (Reguła R1)."
    else:
        age_impact_desc = "Wiek nie miał kluczowego wpływu na wynik (brak aktywacji reguł zależnych od wieku w tym przypadku)."

    return {
        "crisp": crisp,
        "label": "Wysokie" if crisp >= 60 else ("Średnie" if crisp >= 35 else "Niskie"),
        "top_rules": [r for r in rules_sorted if r["strength"] > 0],
        "age_impact": age_impact_desc
    }


def main():
    print("Wczytywanie danych z 'pacjenci_demo_system_ekspertowy.csv'...\n")
    
    patients_to_analyze =["P02", "P20", "P15", "P04"]
    
    try:
        with open('pacjenci_demo_system_ekspertowy.csv', mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row["patient_id"] in patients_to_analyze:
                    pid = row["patient_id"]
                    age = float(row["age"])
                    sbp = float(row["systolic_bp"])
                    
                    print(f"{'='*60}")
                    print(f"PACJENT: {pid} | Wiek: {age} | SBP: {sbp}")
                    print(f"{'='*60}")
                    
                    # 1. System klasyczny
                    class_res = classical_inference(row)
                    print(f"[1] Klasyczna ewaluacja binarna:")
                    print(f"    Decyzja: {class_res}\n")
                    
                    # 2 & 3. System rozmyty (XAI)
                    fuzzy_res = explainable_cv_risk(row)
                    print(f"[2] Ewaluacja logiką rozmytą (Fuzzy):")
                    print(f"    Wynik liczbowy (Crisp Risk): {fuzzy_res['crisp']:.1f}/100")
                    print(f"    Etykieta ryzyka: {fuzzy_res['label']}\n")
                    
                    print(f"[3] WYJAŚNIENIE DECYZJI (XAI):")
                    print("    Aktywowane reguły (posortowane po wpływie na wynik):")
                    for r in fuzzy_res["top_rules"]:
                        print(f"      -> [{r['id']}] (Wkład: {r['contribution_pct']:.1f}%, Siła: {r['strength']:.2f})")
                        print(f"         Treść: {r['text']}")
                        print(f"         Dlaczego: {r['why']}")
                    
                    print(f"\n    *** WPŁYW WIEKU NA KOŃCOWĄ DECYZJĘ ***")
                    print(f"    {fuzzy_res['age_impact']}\n")
                    
    except FileNotFoundError:
        print("BŁĄD: Nie znaleziono pliku 'pacjenci_demo_system_ekspertowy.csv'.")

if __name__ == "__main__":
    main()
