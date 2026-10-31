# 5_generate_roc_plots_v2.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, auc

# Importy klasyfikatorów niezbędne do deserializacji (pickle.load)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
# ZMIANA: Importujemy FRBB, bo inaczej pickle nie odczyta plików modeli FRBB
try:
    from frbb import FastRealBoostBins
except ImportError:
    print(" [OSTRZEŻENIE] Nie znaleziono modułu 'frbb'. Jeśli próbujesz ładować modele FRBB, wystąpi błąd.")

# --- KONFIGURACJA ---
FEATURES_DIR = "features"
MODELS_ROOT_DIR = "models"

# ZMIANA: Wybierz, dla którego klasyfikatora chcesz wygenerować wykresy
# Opcje: "SVM" lub "FRBB" (zgodnie z nazwami folderów stworzonymi w skrypcie treningowym)
CLASSIFIER_TYPE = "SVM" 

# Automatyczne ustawienie ścieżek w zależności od wyboru
MODELS_DIR = os.path.join(MODELS_ROOT_DIR, CLASSIFIER_TYPE)
PLOTS_DIR = os.path.join("thesis_plots_roc", CLASSIFIER_TYPE)
METRICS_FILE = f"roc_metrics_{CLASSIFIER_TYPE}.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Estetyka wykresów
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Mapowanie nazw (możesz tu dodać nowe metody, jeśli się pojawią)
NAME_MAPPING = {
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1_L_1_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC V1",
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2_L_1_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC V2",
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3_L_8_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC V3",
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1_L_1_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC Xue V1",
    "DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2_L_1_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_20_ADAPTIVE_True": "PC Xue V2)",
    "DIFFERENTIAL_SVD_XUE_L_0_A_0.7_B_0.7_VOICED_rVAD_FREQ_L_1500_FREQ_H_2500_FL_20": "Xue et al",
    "MDCT_SVD_WEN_L_A_0.7_B_0.7_VOICED_FREQ_L_1500_FREQ_H_2500_FL_40": "Wen et al"
}

def get_clean_name(raw_name):
    return NAME_MAPPING.get(raw_name, raw_name[:15] + "...")

def load_data(method_path, feature_type):
    """Ładuje cechy i etykiety dla danej metody i typu cechy."""
    file_path = os.path.join(method_path, f"features_{feature_type}.pkl")
    if not os.path.exists(file_path):
        return None, None, None
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"], np.array(data["filenames"])

def find_model_file(method_dir_name, feature_type):
    """
    Szuka pliku modelu w podkatalogu models/[TYPE]/[NAZWA_METODY]/.
    """
    target_dir = os.path.join(MODELS_DIR, method_dir_name)
    
    if not os.path.exists(target_dir):
        return None
        
    files = os.listdir(target_dir)
    
    # Szukamy pliku .pkl zawierającego nazwę cechy
    for f in files:
        if f.lower().endswith(".pkl") and feature_type.lower() in f.lower():
            return os.path.join(target_dir, f)
            
    return None

csv_results = []

def plot_roc_for_feature(feature_type):
    print(f"--- [{CLASSIFIER_TYPE}] Generowanie ROC dla: {feature_type.upper()} ---")
    
    methods = sorted([d for d in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, d))])
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random Guess', alpha=0.8)

    processed_count = 0

    for method in methods:
        method_path = os.path.join(FEATURES_DIR, method)
        
        # 1. Ładujemy dane
        X, y, filenames = load_data(method_path, feature_type)
        if X is None:
            continue

        # 2. Odtwarzamy podział
        groups = [f.split('|')[0] for f in filenames]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, test_idx = next(gss.split(X, y, groups))

        X_test = X[test_idx]
        y_test = y[test_idx]

        # 3. Szukamy modelu
        model_path = find_model_file(method, feature_type)

        if not model_path:
            # Cicha informacja, żeby nie zaśmiecać konsoli, jeśli jakiegoś modelu brakuje
            # print(f" [INFO] Brak modelu dla {get_clean_name(method)} ({feature_type})")
            continue

        # 4. Ładujemy model
        try:
            with open(model_path, "rb") as f:
                clf = pickle.load(f)
        except Exception as e:
            print(f" [BŁĄD] Nie można załadować {model_path}: {e}")
            continue

        # 5. Predykcja (Score)
        try:
            # Próba uzyskania "soft score" (prawdopodobieństwa lub odległości) dla ładnej krzywej
            if hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_test)
            elif hasattr(clf, "predict_proba"):
                # Pobieramy prawdopodobieństwo klasy pozytywnej (1)
                y_score = clf.predict_proba(X_test)[:, 1]
            else:
                # Fallback dla modeli, które mają tylko hard predict (np. proste implementacje boostingu)
                # Wtedy ROC będzie wyglądać "kanciaście" (tylko 3 punkty)
                y_score = clf.predict(X_test)
        except Exception as e:
            print(f" [BŁĄD] Predykcja nieudana dla {model_path}: {e}")
            continue
        
        # 6. Rysowanie
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        processed_count += 1

        clean_name = get_clean_name(method)
        label = f"{clean_name} (AUC = {roc_auc:.2f})"
        plt.plot(fpr, tpr, lw=2, label=label)

        # 7. Metryki
        idx_05 = np.abs(fpr - 0.05).argmin()
        tpr_at_fpr05 = tpr[idx_05]
        
        idx_10 = np.abs(fpr - 0.10).argmin()
        tpr_at_fpr10 = tpr[idx_10]

        csv_results.append({
            "Method": clean_name,
            "Classifier": CLASSIFIER_TYPE,
            "Feature_Set": feature_type.upper(),
            "AUC": round(roc_auc, 4),
            "TPR_at_FPR_5%": round(tpr_at_fpr05, 4),
            "TPR_at_FPR_10%": round(tpr_at_fpr10, 4)
        })

    if processed_count == 0:
        print(f" [!] Brak wyników dla {feature_type}. Upewnij się, że modele istnieją w {MODELS_DIR}.")
        plt.close()
        return

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title(f'ROC ({CLASSIFIER_TYPE}) - {feature_type.upper()}', fontsize=16, pad=20)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(PLOTS_DIR, f"roc_{CLASSIFIER_TYPE}_{feature_type}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" -> Zapisano: {save_path}")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        print(f"BŁĄD: Katalog modeli '{MODELS_DIR}' nie istnieje.")
        print(f"Upewnij się, że uruchomiłeś skrypt treningowy i masz folder: {MODELS_ROOT_DIR}/{CLASSIFIER_TYPE}")
    else:
        print(f"Generowanie wykresów dla klasyfikatora: {CLASSIFIER_TYPE}")
        print(f"Katalog modeli: {MODELS_DIR}")
        print(f"Katalog wyjściowy wykresów: {PLOTS_DIR}\n")

        for feat in ["rme", "dhs", "wmc", "dmc"]:
            plot_roc_for_feature(feat)
        
        if csv_results:
            df = pd.DataFrame(csv_results)
            df = df.sort_values(by=["Feature_Set", "AUC"], ascending=[True, False])
            df.to_csv(METRICS_FILE, index=False)
            print(f"\nGotowe! Wyniki zapisano do pliku: {METRICS_FILE}")