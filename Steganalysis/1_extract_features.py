# 1_extract_features.py
import os
import pickle
import numpy as np
from tqdm import tqdm
from utils_features import EXTRACTORS  # Importujemy funkcje z utils_features.py

CARRIER_DIR = "Carriers"
STEGO_ROOT_DIR = "Stegobjects"
FEATURES_ROOT_DIR = "features"


def process_method(method_name, extractors):
    stego_method_dir = os.path.join(STEGO_ROOT_DIR, method_name)
    output_method_dir = os.path.join(FEATURES_ROOT_DIR, method_name)
    os.makedirs(output_method_dir, exist_ok=True)

    print(f"\n[Processing Method] {method_name}")

    # Przygotowanie kontenerów na dane dla każdego ekstraktora
    data_store = {key: {"X": [], "y": [], "filenames": []} for key in extractors.keys()}

    # Pobieramy listę plików z folderu Carriers (zakładamy, że nazwy są identyczne)
    carrier_files = sorted([f for f in os.listdir(CARRIER_DIR) if f.endswith(".wav")])

    for fname in tqdm(carrier_files, desc=f"Extracting {method_name}"):
        cover_path = os.path.join(CARRIER_DIR, fname)
        stego_path = os.path.join(stego_method_dir , 'Stegobjects', fname)

        # Sprawdzenie czy odpowiednik stego istnieje
        if not os.path.exists(stego_path):
            continue  # Pomijamy jeśli brak pliku stego

        # Ekstrakcja dla wszystkich typów cech
        for feat_name, extractor_func in extractors.items():
            # Cover (Label 0)
            feat_cover = extractor_func(cover_path)
            data_store[feat_name]["X"].append(feat_cover)
            data_store[feat_name]["y"].append(0)
            data_store[feat_name]["filenames"].append(f"{fname}|cover")

            # Stego (Label 1)
            feat_stego = extractor_func(stego_path)
            data_store[feat_name]["X"].append(feat_stego)
            data_store[feat_name]["y"].append(1)
            data_store[feat_name]["filenames"].append(f"{fname}|stego")

    # # Zapis wyników do plików pkl w folderze features/METODA/
    # for feat_name, content in data_store.items():
    #     out_path = os.path.join(output_method_dir, f"features_{feat_name}.pkl")
    #     with open(out_path, "wb") as f:
    #         pickle.dump(
    #             {
    #                 "X": np.array(content["X"]),
    #                 "y": np.array(content["y"]),
    #                 "filenames": content["filenames"],
    #             },
    #             f,
    #         )


if __name__ == "__main__":
    if not os.path.exists(STEGO_ROOT_DIR):
        print(f"Directory {STEGO_ROOT_DIR} not found!")
        exit(1)

    # Iteracja po wszystkich podkatalogach w Stegobjects
    methods = sorted(
        [
            d
            for d in os.listdir(STEGO_ROOT_DIR)
            if os.path.isdir(os.path.join(STEGO_ROOT_DIR, d))
        ]
    )

    if not methods:
        print("No steganography methods found inside Stegobjects directory.")

    for method in methods:
        process_method(method, EXTRACTORS)

    print("\nExtraction complete for all methods.")
