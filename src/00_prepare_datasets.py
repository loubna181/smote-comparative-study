import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

OUTPUT_DIR  = "datasets_prepared"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "datasets_final.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS_CONFIG = {
    "Abalone":      {"minority_classes": [1]},
    "Car":          {"minority_classes": [3]},
    "CTG":          {"minority_classes": [2]},
    "Ecoli":        {"minority_classes": [5]},
    "Glass":        {"minority_classes": [5]},
    "Libra":        {"minority_classes": [14]},
    "Lymphography": {"minority_classes": [2]},
    "OCR":          {"minority_classes": [6]},
    "Pageblocks":   {"minority_classes": [4]},
    "Phoneme":      {"minority_classes": [1]},
    "Robot":        {"minority_classes": [3]},
    "Satimage":     {"minority_classes": [4]},
    "Satlandsdat":  {"minority_classes": [6]},
    "Segment":      {"minority_classes": [6]},
    "Vehicle":      {"minority_classes": [3]},
    "Wine":         {"minority_classes": [1]},
    "Yeast":        {"minority_classes": [9]},
}

def binarize(X, y, minority_classes):
    y_bin = np.where(np.isin(y, minority_classes), 1, 0)
    return X, y_bin

def load_dataset(name):
    path = os.path.join("datasets_raw", name + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path, header=None)
    X  = df.iloc[:, :-1].values.astype(float)
    y  = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y

def main():
    datasets = {}
    for name, cfg in DATASETS_CONFIG.items():
        try:
            X, y = load_dataset(name)
            X, y = binarize(X, y, cfg["minority_classes"])
            X    = StandardScaler().fit_transform(X)
            n_min = np.sum(y == 1)
            n_maj = np.sum(y == 0)
            ir    = round(n_maj / n_min, 1)
            datasets[name] = {"X": X, "y": y}
            print(f"  {name:15s}  {X.shape[0]:5d} samples  IR={ir}")
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(datasets, f)
    print(f"Saved {len(datasets)} datasets -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()