# 3_predict_file.py
import os
import sys
import pickle
import numpy as np
from utils_features import EXTRACTORS  # Importujemy, żeby nie duplikować kodu!

MODELS_ROOT_DIR = "models"

def predict_single_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"\n{'='*60}")
    print(f" FORENSIC ANALYSIS REPORT")
    print(f" Target File: {os.path.basename(file_path)}")
    print(f"{'='*60}\n")

    # Pobieramy listę dostępnych metod (podkatalogów w models/)
    if not os.path.exists(MODELS_ROOT_DIR):
        print("Error: Models directory not found. Train models first.")
        return

    methods = sorted([d for d in os.listdir(MODELS_ROOT_DIR) if os.path.isdir(os.path.join(MODELS_ROOT_DIR, d))])
    
    if not methods:
        print("No trained methods found.")
        return

    # Cache features to avoid re-calculating for different methods if feature logic is identical
    # (W tym projekcie cechy zależą tylko od ekstraktora, a nie od metody stego, 
    # więc obliczamy je raz dla danego pliku audio)
    feature_cache = {}
    print("Extracting features...", end=" ", flush=True)
    for feat_name, extractor in EXTRACTORS.items():
        # reshape(1, -1) jest konieczny dla sklearn (oczekuje tablicy 2D dla jednej próbki)
        feature_cache[feat_name] = extractor(file_path).reshape(1, -1)
    print("Done.\n")

    # Iteracja po metodach steganograficznych
    for method in methods:
        method_path = os.path.join(MODELS_ROOT_DIR, method)
        print(f"Checking against method: [{method}]")
        
        votes = []
        details = []

        # Sprawdzamy 4 klasyfikatory (RME, DHS, WMC, DMC) dla tej metody
        for feat_name in ["rme", "dhs", "wmc", "dmc"]:
            model_file = os.path.join(method_path, f"model_{feat_name}.pkl")
            
            if not os.path.exists(model_file):
                details.append(f"{feat_name.upper()}: N/A")
                continue

            try:
                with open(model_file, "rb") as f:
                    clf = pickle.load(f)
                
                # Używamy feature_cache - dane są automatycznie skalowane przez Pipeline wewnątrz modelu!
                feat_vector = feature_cache[feat_name]
                prediction = clf.predict(feat_vector)[0]
                
                label = "STEGO" if prediction == 1 else "Cover"
                votes.append(prediction)
                
                # Dodajemy kolorowanie dla czytelności (jeśli terminal obsługuje ANSI)
                res_str = f"\033[91mSTEGO\033[0m" if prediction == 1 else "\033[92mCover\033[0m"
                details.append(f"{feat_name.upper()}: {res_str}")

            except Exception as e:
                details.append(f"{feat_name.upper()}: Err")

        # Decyzja końcowa dla danej metody (Głosowanie większościowe lub "Any")
        # W steganalizie bezpieczniej jest alarmować, jeśli >1 cecha wskazuje stego
        stego_count = sum(votes)
        total_models = len(votes)
        
        if total_models > 0:
            confidence = (stego_count / total_models) * 100
            verdict = "SUSPICIOUS" if stego_count >= 2 else "Clean"
            color_verdict = f"\033[91m{verdict}\033[0m" if verdict == "SUSPICIOUS" else f"\033[92m{verdict}\033[0m"
            
            print(f"  -> Verdict: {color_verdict} (Detection rate: {stego_count}/{total_models})")
            print(f"  -> Details: {', '.join(details)}")
        else:
            print("  -> No models found for this method.")
        
        print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 3_predict_file.py <path_to_wav>")
        sys.exit(1)

    target_wav = sys.argv[1]
    predict_single_file(target_wav)