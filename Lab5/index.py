# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas>=3.0.1",
# ]
# ///

import hashlib
import pandas as pd
import os

def compute_file_hash(filepath: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_data_integrity(filepath: str, expected_hash: str) -> bool:
    current_hash = compute_file_hash(filepath, "sha256")
    return current_hash == expected_hash

def main():
    original_file = "Myocardial infarction complications Database.csv"
    trusted_file = "trusted_data.csv"
    manipulated_file = "manipulated_data.csv"

    if not os.path.exists(original_file):
        print(f"Brak pliku {original_file}. Upewnij się, że jest w tym samym folderze.")
        return

    # 1. PRZYGOTOWANIE ZAUFANEGO PLIKU
    df = pd.read_csv(original_file)
    df.to_csv(trusted_file, index=False)

    trusted_hash = compute_file_hash(trusted_file)
    print(f"[ETAP 1] Wyliczono hash zaufanego pliku (SHA-256):")
    print(f"         {trusted_hash}\n")

    # 2. SYMULACJA MANIPULACJI (ATAKU)
    print("[ETAP 2] Symulacja ataku na integralność danych...")
    df_manipulated = df.copy()
    
    # Symulujemy atak: zaniżamy wiek pierwszego pacjenta (ID 1) z 77 na 20 lat.
    target_row_idx = 0
    old_age = df_manipulated.loc[target_row_idx, 'AGE']
    df_manipulated.loc[target_row_idx, 'AGE'] = 20
    new_age = df_manipulated.loc[target_row_idx, 'AGE']
    
    print(f"         Zmieniono wiek pacjenta w wierszu 0 z {old_age} na {new_age}.")
    
    df_manipulated.to_csv(manipulated_file, index=False)
    print(f"         Zapisano sfałszowany zbiór jako '{manipulated_file}'.\n")

    # 3. DETEKCJA NARUSZEŃ (WERYFIKACJA INTEGRALNOŚCI)
    print("[ETAP 3] Detekcja naruszeń przed uruchomieniem sztucznej inteligencji")
    
    # Test 1: Próba wczytania poprawnego pliku
    print(f"   -> Sprawdzam plik: {trusted_file}")
    if verify_data_integrity(trusted_file, trusted_hash):
        print("      WYNIK: [OK] Integralność zachowana. Można bezpiecznie trenować model.")
    else:
        print("      WYNIK: [BŁĄD] Plik został zmieniony!")

    # Test 2: Próba wczytania zmanipulowanego pliku
    print(f"\n   -> Sprawdzam plik: {manipulated_file}")
    if verify_data_integrity(manipulated_file, trusted_hash):
        print("      WYNIK: [OK] Integralność zachowana.")
    else:
        manipulated_hash = compute_file_hash(manipulated_file)
        print("      WYNIK: [BŁĄD KRYTYCZNY] Wykryto manipulację danymi!")
        print("             Dane nie pokrywają się ze skrótem referencyjnym.")
        print(f"             Oczekiwano : {trusted_hash}")
        print(f"             Otrzymano  : {manipulated_hash}")
        print("      AKCJA: Zatrzymano rurociąg ML (Pipeline aborted) w celu ochrony predykcji.")

if __name__ == "__main__":
    main()
