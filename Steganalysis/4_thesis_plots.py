import os
import pickle
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import FRBB dla pewności (do ładowania modeli)
try:
    from frbb import FastRealBoostBins
except ImportError:
    pass 

# --- KONFIGURACJA ---
# Wybierz, które wyniki analizujemy
CLASSIFIER_TYPE = "SVM"  # lub "FRBB"

FEATURES_DIR = "features"
# Dostosowanie do nowej struktury z poprzedniego kroku
MODELS_DIR = os.path.join("models", CLASSIFIER_TYPE) 
# Plik wyników z benchmarku (dopasuj nazwę do tego, który wygenerowałeś)
RESULTS_FILE = "results_benchmark.json" # lub "results_frbb.json" / "results_comparison.json"

PLOTS_DIR = os.path.join("thesis_plots", CLASSIFIER_TYPE)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Estetyka
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Mapowanie nazw
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

def load_features(method_name, feature_type):
    path = os.path.join(FEATURES_DIR, method_name, f"features_{feature_type}.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"]

def find_model_file(method_dir_name, feature_type):
    """Pomocnicza funkcja do znajdowania modelu niezależnie od dokładnej nazwy pliku"""
    target_dir = os.path.join(MODELS_DIR, method_dir_name)
    if not os.path.exists(target_dir): return None
    for f in os.listdir(target_dir):
        if f.lower().endswith(".pkl") and feature_type.lower() in f.lower():
            return os.path.join(target_dir, f)
    return None

# --- 1. NOWA WIZUALIZACJA: HEATMAPA DOKŁADNOŚCI ---
def plot_accuracy_heatmap():
    """Tworzy czytelną tabelę/mapę ciepła z wynikami Accuracy."""
    if not os.path.exists(RESULTS_FILE):
        print(f" [!] Brak pliku {RESULTS_FILE}. Pomiń.")
        return

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    # Parsowanie JSON do DataFrame
    data = []
    for method, metrics in results.items():
        # Obsługa różnych struktur JSON (zależnie czy użyłeś comparison.json czy benchmark.json)
        # Ten kod zakłada prostszą strukturę lub próbuje się dostosować
        for feat_name, content in metrics.items():
            acc = 0
            if "accuracy" in content: # Struktura prosta
                acc = content["accuracy"]
            elif CLASSIFIER_TYPE in content and "accuracy" in content[CLASSIFIER_TYPE]: # Struktura comparison
                acc = content[CLASSIFIER_TYPE]["accuracy"]
            
            if acc > 0:
                data.append({
                    "Method": get_clean_name(method),
                    "Feature Set": feat_name.upper(),
                    "Accuracy": acc
                })
    
    if not data:
        print(" [!] Brak danych do wykresu Heatmap.")
        return

    df = pd.DataFrame(data)
    
    # Tworzenie macierzy: Wiersze=Metody, Kolumny=Cechy
    pivot_table = df.pivot(index="Method", columns="Feature Set", values="Accuracy")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2%", cmap="viridis", 
                linewidths=.5, cbar_kws={'label': 'Accuracy'})
    
    plt.title(f"Heatmapa Dokładności ({CLASSIFIER_TYPE})", fontsize=16, pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "accuracy_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" -> Zapisano: {save_path}")

# --- 2. NOWA WIZUALIZACJA: POINT PLOT (Zamiast Bar Plot) ---
def plot_accuracy_pointplot():
    """Tworzy wykres punktowy - znacznie lżejszy wizualnie od słupkowego."""
    if not os.path.exists(RESULTS_FILE): return

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    data = []
    for method, metrics in results.items():
        for feat_name, content in metrics.items():
            acc = 0
            if "accuracy" in content:
                acc = content["accuracy"]
            elif CLASSIFIER_TYPE in content:
                acc = content[CLASSIFIER_TYPE]["accuracy"]
            
            if acc > 0:
                data.append({
                    "Method": get_clean_name(method),
                    "Feature Set": feat_name.upper(),
                    "Accuracy": acc
                })
    
    df = pd.DataFrame(data)
    if df.empty: return

    plt.figure(figsize=(12, 7))
    
    # Pointplot: X=Accuracy, Y=Method (poziomy jest czytelniejszy dla długich nazw)
    sns.pointplot(data=df, x="Accuracy", y="Method", hue="Feature Set", 
                  join=False, dodge=0.4, palette="deep", scale=1.2)
    
    # Dodanie linii siatki dla łatwiejszego odczytu
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle=':', label="Losowość")

    plt.title(f"Porównanie Metod - Point Plot ({CLASSIFIER_TYPE})", fontsize=16)
    plt.xlabel("Accuracy")
    plt.ylabel("")
    plt.xlim(0.4, 1.0)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Feature Set")
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, "accuracy_pointplot.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" -> Zapisano: {save_path}")

# --- 3. MACIERZ PRZENOSZALNOŚCI (Zaktualizowana) ---
def plot_transferability_heatmap(feature_type="rme"):
    if not os.path.exists(MODELS_DIR):
        print(f" [!] Folder modeli {MODELS_DIR} nie istnieje.")
        return

    methods = sorted([d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))])
    if not methods: return

    clean_labels = [get_clean_name(m) for m in methods]
    n = len(methods)
    matrix = np.zeros((n, n))

    print(f"Generowanie macierzy transferowalności dla {feature_type.upper()}...")
    
    # Pętla: Trenowano na (i), Testowano na (j)
    for i, train_method in enumerate(tqdm(methods)):
        # Szukanie modelu za pomocą nowej funkcji
        model_path = find_model_file(train_method, feature_type)
        
        if not model_path:
            continue
            
        try:
            with open(model_path, "rb") as f:
                clf = pickle.load(f)
        except:
            continue

        for j, test_method in enumerate(methods):
            # Tu ładujemy cechy testowe
            X_test, y_test = load_features(test_method, feature_type)
            if X_test is None: continue
            
            # Predykcja
            try:
                acc = clf.score(X_test, y_test)
                matrix[i, j] = acc
            except:
                pass # Jeśli model nie pasuje do danych (np. wymiary)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", 
                xticklabels=clean_labels, yticklabels=clean_labels, 
                vmin=0.4, vmax=1.0, cbar_kws={'label': 'Accuracy'})
    
    plt.title(f"Transferability Matrix - {feature_type.upper()} ({CLASSIFIER_TYPE})", pad=20)
    plt.ylabel("Trained on...")
    plt.xlabel("Tested on...")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, f"transferability_{feature_type}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" -> Zapisano: {save_path}")

# --- 4. T-SNE (Bez zmian, tylko integracja) ---
def plot_feature_space_tsne(feature_type="rme", max_samples=200):
    print(f"Generowanie t-SNE dla {feature_type.upper()}...")
    methods = sorted([d for d in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, d))])
    
    X_all, labels = [], []
    
    # 1. Cover (tylko raz, np. z pierwszej metody, bo cover jest ten sam lub podobny)
    X_cover, y_cover = load_features(methods[0], feature_type)
    if X_cover is not None:
        idxs = np.where(y_cover == 0)[0]
        if len(idxs) > max_samples: idxs = np.random.choice(idxs, max_samples, replace=False)
        X_all.append(X_cover[idxs])
        labels.extend(["Cover"] * len(idxs))

    # 2. Stego (dla każdej metody)
    for method in methods:
        X_stego, y_stego = load_features(method, feature_type)
        if X_stego is None: continue
        
        idxs = np.where(y_stego == 1)[0]
        if len(idxs) > max_samples: idxs = np.random.choice(idxs, max_samples, replace=False)
        
        X_all.append(X_stego[idxs])
        labels.extend([get_clean_name(method)] * len(idxs))

    if not X_all: return

    X_all = np.vstack(X_all)
    # Skalowanie ważne dla t-SNE
    X_scaled = StandardScaler().fit_transform(X_all)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random')
    X_embedded = tsne.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        "Dim 1": X_embedded[:, 0], "Dim 2": X_embedded[:, 1], "Class": labels
    })

    plt.figure(figsize=(11, 11)) # Kwadratowy
    sns.scatterplot(
        data=df_plot, x="Dim 1", y="Dim 2", hue="Class", 
        style="Class", palette="tab10", s=80, alpha=0.8
    )
    
    plt.title(f"Przestrzeń cech t-SNE: {feature_type.upper()}", fontsize=16)
    plt.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=3, frameon=False)
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, f"tsne_{feature_type}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" -> Zapisano: {save_path}")

if __name__ == "__main__":
    # Upewnij się, że nazwa pliku z wynikami jest poprawna
    if not os.path.exists(RESULTS_FILE):
        print(f"UWAGA: Nie znaleziono pliku {RESULTS_FILE}. Sprawdź nazwę.")
    else:
        # 1. Heatmapa dokładności (Zastępuje histogram)
        plot_accuracy_heatmap()
        
        # 2. Point Plot (Alternatywa dla histogramu)
        plot_accuracy_pointplot()

    # 3. Macierz przenoszalności (Dla wybranej cechy, np. RME)
    plot_transferability_heatmap(feature_type="rme")
    
    # 4. t-SNE (Wizualizacja przestrzeni)
    plot_feature_space_tsne(feature_type="rme")
    
    print(f"\nGotowe! Wszystkie wykresy w folderze: {PLOTS_DIR}")