import os
import pickle
import json
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix

# Twoje importy
from frbb import FastRealBoostBins
from utils import gpu_props, dict_to_str

# --- KONFIGURACJA ---
FEATURES_ROOT_DIR = "features"
MODELS_ROOT_DIR = "models"
RESULTS_PATH = "results_comparison.json"

# Konfiguracja katalogów wyjściowych
SVM_DIR = os.path.join(MODELS_ROOT_DIR, "SVM")
FRBB_DIR = os.path.join(MODELS_ROOT_DIR, "FRBB")

SEED = 42

def calculate_metrics(y_true, y_pred, execution_time):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    p_fa = fp / (fp + tn) if (fp + tn) else 0
    p_md = fn / (fn + tp) if (fn + tp) else 0
    p_e = 0.5 * (p_fa + p_md)  # Min. Average Decision Error
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        "accuracy": round(acc, 4),
        "detection_error": round(p_e, 4),
        "training_time": round(execution_time, 4),
        "confusion_matrix": cm.tolist()
    }

def train_and_compare_method(method_name):
    """Trenuje i porównuje SVM oraz FRBB dla danej metody."""
    
    method_feature_dir = os.path.join(FEATURES_ROOT_DIR, method_name)
    
    # Ścieżki do zapisu modeli dla konkretnej metody
    svm_method_dir = os.path.join(SVM_DIR, method_name)
    frbb_method_dir = os.path.join(FRBB_DIR, method_name)
    
    os.makedirs(svm_method_dir, exist_ok=True)
    os.makedirs(frbb_method_dir, exist_ok=True)
    
    feature_groups = ["rme", "dhs", "wmc", "dmc"]
    method_results = {}

    print(f"\n=== Processing Method: {method_name} ===")

    for group in feature_groups:
        feat_file = os.path.join(method_feature_dir, f"features_{group}.pkl")
        
        if not os.path.exists(feat_file):
            print(f"  [SKIP] {group} not found")
            continue

        # 1. Ładowanie danych (RAZ dla obu modeli)
        with open(feat_file, "rb") as f:
            data = pickle.load(f)
        X, y = data["X"], data["y"]
        filenames = np.array(data["filenames"])
        groups = [f.split('|')[0] for f in filenames]

        # 2. Podział Train/Test (WSPÓLNY dla obu modeli)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  Feature: {group.upper()} | Train: {len(X_train)}, Test: {len(X_test)}")

        # -------------------------------
        # MODEL 1: SVM
        # -------------------------------
        # SVM wymaga skalowania danych
        clf_svm = make_pipeline(
            StandardScaler(), 
            SVC(kernel="rbf", gamma="scale", C=10.0, probability=False) # probability=False jest szybsze
        )
        
        start_time = time.time()
        clf_svm.fit(X_train, y_train)
        svm_time = time.time() - start_time
        
        # Predykcja i Metryki SVM
        preditc_time = 0
        start_time_2 = time.time()
        y_pred_svm = clf_svm.predict(X_test)
        preditc_time_svm = time.time() - start_time_2
        metrics_svm = calculate_metrics(y_test, y_pred_svm, svm_time)
        
        # Zapis modelu SVM
        svm_path = os.path.join(svm_method_dir, f"model_{method_name}_{group}.pkl")
        with open(svm_path, "wb") as f:
            pickle.dump(clf_svm, f)

        # -------------------------------
        # MODEL 2: FRBB
        # -------------------------------
        # FRBB zazwyczaj nie wymaga skalowania (drzewa), ale można podać surowe X
        clf_frbb = FastRealBoostBins(T=16) # Możesz zmienić T na wyższe
        
        start_time = time.time()
        clf_frbb.fit(X_train, y_train)
        frbb_time = time.time() - start_time
        
        # Predykcja i Metryki FRBB
        start_time_2 = time.time()
        y_pred_frbb = clf_frbb.predict(X_test)
        preditc_time += time.time() - start_time_2
        metrics_frbb = calculate_metrics(y_test, y_pred_frbb, frbb_time)
        
        # Zapis modelu FRBB
        frbb_path = os.path.join(frbb_method_dir, f"model_{method_name}_{group}.pkl")
        with open(frbb_path, "wb") as f:
            pickle.dump(clf_frbb, f)

        # -------------------------------
        # Porównanie w konsoli
        # -------------------------------
        print(f"    > SVM  -> Acc: {metrics_svm['accuracy']:.4f} (Time: {svm_time:.2f}s, Predict: time: {preditc_time_svm:.2f}s)")
        print(f"    > FRBB -> Acc: {metrics_frbb['accuracy']:.4f} (Time: {frbb_time:.2f}s, Predict: time: {preditc_time:.2f}s)")

        # Zapis do słownika wyników
        method_results[group] = {
            "SVM": metrics_svm,
            "FRBB": metrics_frbb
        }

    return method_results

if __name__ == "__main__":
    try:
        print(dict_to_str(gpu_props()))
    except:
        pass

    if not os.path.exists(FEATURES_ROOT_DIR):
        print("Features directory not found.")
        exit(1)

    all_results = {}
    methods = sorted([d for d in os.listdir(FEATURES_ROOT_DIR) if os.path.isdir(os.path.join(FEATURES_ROOT_DIR, d))])

    for method in methods:
        all_results[method] = train_and_compare_method(method)

    # Zapis wyników do JSON
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nComparison complete. Results saved to {RESULTS_PATH}")
    print(f"Models saved in:\n - {SVM_DIR}\n - {FRBB_DIR}")