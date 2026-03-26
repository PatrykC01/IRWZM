# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cryptography>=46.0.6",
#     "numpy>=2.4.3",
#     "pandas>=3.0.1",
#     "scikit-learn>=1.8.0",
# ]
# ///
import pandas as pd
import numpy as np
import hashlib
import uuid
import time
import os
from cryptography.fernet import Fernet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

CSV_FILE = "Myocardial infarction complications Database.csv"
ENC_FILE = "database_encrypted.enc"

def main():
    print("Rozpoczęcie laboratorium: Bezpieczeństwo danych medycznych\n")
    
    # =========================================================================
    # ETAP 1: PRZYGOTOWANIE DANYCH
    # =========================================================================
    print("="*50)
    print("ETAP 1: Przygotowanie i analiza danych")
    if not os.path.exists(CSV_FILE):
        print(f"BŁĄD: Brak pliku {CSV_FILE} w katalogu roboczym!")
        return

    columns_to_use =['ID', 'AGE', 'SEX', 'S_AD_KBRIG', 'D_AD_KBRIG', 'ALT_BLOOD', 'AST_BLOOD', 'LET_IS']
    df = pd.read_csv(CSV_FILE, usecols=columns_to_use)
    
    for col in columns_to_use:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median() if col != 'ID' else 0)

    # Przekształcamy LET_IS na zmienną binarną (0 - przeżycie, >0 - zgon/powikłania)
    df['DEATH'] = (df['LET_IS'] > 0).astype(int)
    print(f"Zaimportowano {len(df)} rekordów.")
    print("Zidentyfikowane dane wrażliwe: ID (identyfikator), AGE (może ułatwić identyfikację).")


    # =========================================================================
    # ZADANIE 1: PORÓWNANIE ALGORYTMÓW HASZUJĄCYCH (Etap 2)
    # =========================================================================
    print("\n" + "="*50)
    print("ZADANIE 1: Porównanie algorytmów haszujących")
    
    sample_data = df.to_csv(index=False).encode('utf-8')
    
    algos = {
        'MD5': hashlib.md5, 
        'SHA-256': hashlib.sha256, 
        'SHA-3-256': hashlib.sha3_256
    }
    
    for name, func in algos.items():
        t0 = time.time()
        h = func(sample_data).hexdigest()
        t1 = time.time()
        print(f"{name:>10}: {h} (Czas: {(t1-t0)*1000:.3f} ms)")
        
    print("\nOcena odporności na kolizje:")
    print("- MD5: Przestarzały, podatny na ataki kolizyjne. Nie nadaje się do danych medycznych.")
    print("- SHA-256: Uznawany za bezpieczny, standard branżowy.")
    print("- SHA-3: Najnowszy standard odporny na tzw. length extension attacks.")


    # =========================================================================
    # ZADANIE 2: PSEUDONIMIZACJA I WPŁYW NA MODEL (Etap 3)
    # =========================================================================
    print("\n" + "="*50)
    print("ZADANIE 2: Pseudonimizacja, anonimizacja i model ML")
    
    # Pseudonimizacja: podmiana ID na UUID
    df['pseudo_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Anonimizacja: generalizacja wieku
    def age_to_group(age):
        if age < 40: return 0      # "<40"
        elif age < 60: return 1    # "40-59"
        elif age < 80: return 2    # "60-79"
        else: return 3             # ">=80"
    df['AGE_GROUP'] = df['AGE'].apply(age_to_group)
    
    df_anonymized = df.drop(columns=['ID'])
    
    features_raw =['AGE', 'SEX', 'S_AD_KBRIG', 'D_AD_KBRIG', 'ALT_BLOOD', 'AST_BLOOD']
    features_anon =['AGE_GROUP', 'SEX', 'S_AD_KBRIG', 'D_AD_KBRIG', 'ALT_BLOOD', 'AST_BLOOD']
    
    def train_and_eval(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    acc_raw = train_and_eval(df[features_raw], df['DEATH'])
    acc_anon = train_and_eval(df_anonymized[features_anon], df_anonymized['DEATH'])
    
    print(f"Dokładność modelu na surowych danych (z AGE): {acc_raw:.4f}")
    print(f"Dokładność modelu na zanonimizowanych (AGE_GROUP): {acc_anon:.4f}")
    print("Wniosek: Generalizacja usunęła trochę szczegółów, co może minimalnie wpłynąć na model, ale drastycznie zwiększa prywatność pacjentów.")


    # =========================================================================
    # ZADANIE 3: MANIPULACJA DANYMI I DETEKCJA (Etap 5)
    # =========================================================================
    print("\n" + "="*50)
    print("ZADANIE 3: Detekcja naruszeń integralności")
    
    original_hash = hashlib.sha256(df_anonymized.to_csv(index=False).encode('utf-8')).hexdigest()
    print(f"Oryginalny hash SHA-256: {original_hash}")
    
    # Symulacja manipulacji (Atak: Data Tampering)
    df_tampered = df_anonymized.copy()
    # Haker modyfikuje wynik kliniczny dla pierwszego pacjenta (np. dla ubezpieczenia)
    df_tampered.loc[0, 'DEATH'] = 1 - df_tampered.loc[0, 'DEATH'] 
    
    tampered_hash = hashlib.sha256(df_tampered.to_csv(index=False).encode('utf-8')).hexdigest()
    print(f"Hash po manipulacji 1 bitem danych: {tampered_hash}")
    
    if original_hash != tampered_hash:
        print("-> ALERT: ODKRYTO MODYFIKACJĘ DANYCH! Sumy kontrolne są różne.")


    # =========================================================================
    # ZADANIE 4: KONTROLA DOSTĘPU (RBAC) (Etap 4)
    # =========================================================================
    print("\n" + "="*50)
    print("ZADANIE 4: System kontroli dostępu RBAC")
    
    ROLES = {
        "administrator": df_anonymized.columns.tolist(), # Pełny dostęp
        "lekarz":["pseudo_id", "AGE_GROUP", "SEX", "S_AD_KBRIG", "D_AD_KBRIG", "DEATH"],
        "analityk":["AGE_GROUP", "SEX", "S_AD_KBRIG", "D_AD_KBRIG"] # Brak pseudo_id i celu
    }
    
    audit_logs =[]

    def request_access(user, role, dataset):
        timestamp = datetime.now().isoformat(timespec="seconds")
        if role not in ROLES:
            audit_logs.append(f"[{timestamp}] Odmowa: Nieznana rola '{role}' dla użytkownika '{user}'")
            return None
        
        audit_logs.append(f"[{timestamp}] Zezwolenie: Użytkownik '{user}' (Rola: {role}) uzyskał dostęp.")
        return dataset[ROLES[role]].copy()

    # Symulacja dostępów
    import datetime
    from datetime import datetime
    view_admin = request_access("admin_anna", "administrator", df_anonymized)
    view_doc = request_access("dr_jan", "lekarz", df_anonymized)
    view_analyst = request_access("analityk_tomasz", "analityk", df_anonymized)
    view_hacker = request_access("hacker_eve", "stazysta", df_anonymized) # Nieistniejąca rola
    
    print(f"Kolumny widoczne dla Lekarza: {list(view_doc.columns)}")
    print(f"Kolumny widoczne dla Analityka: {list(view_analyst.columns)}")


    # =========================================================================
    # ZADANIE 5: RYZYKA W SYSTEMACH SI (Etap 5)
    # =========================================================================
    print("\n" + "="*50)
    print("ZADANIE 5: Analiza ryzyk w systemach SI")
    print("1. Data Poisoning (Zatruwanie danych): Atakujący modyfikuje dane treningowe (jak w Zad 3), aby pogorszyć dokładność modelu lub wbudować 'backdoor'.")
    print("2. Data Tampering (Manipulacja danymi): Zmiana surowych danych w bazie w celu sfałszowania np. diagnozy lekarskiej przed predykcją.")
    print("3. Adversarial Examples: Delikatna zmiana parametrów wejściowych pacjenta (niewidoczna gołym okiem), która sprawia, że model SI zwraca nieprawidłową decyzję.")


    # =========================================================================
    # ZADANIE 6: SZYFROWANIE CAŁEGO PLIKU I KLUCZE (Etap 2)
    # =========================================================================
    print("\n" + "="*50)
    print("ZADANIE 6: Szyfrowanie całego pliku (AES/Fernet)")
    
    key = Fernet.generate_key()
    cipher = Fernet(key)
    
    anon_file_temp = "temp_anon.csv"
    df_anonymized.to_csv(anon_file_temp, index=False)
    
    with open(anon_file_temp, "rb") as f:
        raw_bytes = f.read()
        
    encrypted_bytes = cipher.encrypt(raw_bytes)
    
    with open(ENC_FILE, "wb") as f:
        f.write(encrypted_bytes)
        
    print(f"Wygenerowano klucz szyfrujący: {key.decode('utf-8')[:15]}...")
    print(f"Zaszyfrowano bazę danych. Rozmiar przed: {len(raw_bytes)} B, Po zaszyfrowaniu: {len(encrypted_bytes)} B")
    os.remove(anon_file_temp) 


    # =========================================================================
    # ZADANIE 7: AUDYT LOGÓW DOSTĘPU (Etap 4)
    # =========================================================================
    print("\n" + "="*50)
    print("ZADANIE 7: Audyt logów i polityki bezpieczeństwa")
    for log in audit_logs:
        print(log)
        
    print("\nPropozycje ulepszeń polityk bezpieczeństwa (Wniosek z audytu):")
    print("1. Wykryto nieautoryzowaną próbę dostępu (rola 'stazysta'). Należy zaimplementować system ostrzegania (alerting) przy próbach ominięcia uprawnień.")
    print("2. Brak weryfikacji MFA - logi nie wskazują kontekstu logowania (np. IP, próby autoryzacji 2FA).")
    print("3. Polityka najmniejszych przywilejów (PoLP) działa prawidłowo - analityk ma obcięte ID pacjentów i etykiety kliniczne.")


if __name__ == "__main__":
    main()
