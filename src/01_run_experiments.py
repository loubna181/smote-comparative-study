import os, pickle, warnings
import numpy as np
from itertools import product
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import smote_variants as sv

warnings.filterwarnings("ignore")

RESULTS_DIR  = os.path.join("results", "pkl")
DATA_FILE    = os.path.join("datasets_prepared", "datasets_final.pkl")
N_FOLDS      = 5
RANDOM_STATE = 42
os.makedirs(RESULTS_DIR, exist_ok=True)

METHODS = {
    "SMOTE": {
        "class": sv.SMOTE,
        "grid":  {"proportion": [0.5, 1.0, 1.5], "n_neighbors": [3, 5, 7]},
    },
    "Safe_Level_SMOTE": {
        "class": sv.Safe_Level_SMOTE,
        "grid":  {"proportion": [0.5, 1.0, 1.5], "n_neighbors": [3, 5, 7, 9]},
    },
    "ADASYN": {
        "class": sv.ADASYN,
        "grid":  {"n_neighbors": [3, 5, 7], "beta": [0.5, 1.0, 1.5],
                  "d_th": [0.7, 0.9]},
    },
    "LVQ_SMOTE": {
        "class": sv.LVQ_SMOTE,
        "grid":  {"proportion": [0.5, 1.0, 1.5], "n_neighbors": [3, 5, 7],
                  "n_clusters": [5, 10, 15]},
    },
    "ProWSyn": {
        "class": sv.ProWSyn,
        "grid":  {"proportion": [0.5, 1.0, 1.5], "n_neighbors": [3, 5, 7],
                  "L": [3, 5, 7], "theta": [0.5, 1.0, 2.0]},
    },
    "SMOTE_IPF": {
        "class": sv.SMOTE_IPF,
        "grid":  {"proportion": [0.5, 1.0, 1.5], "n_neighbors": [3, 5, 7],
                  "n_folds": [3, 5], "k": [3, 5],
                  "p": [0.01], "voting": ["majority"]},
    },
    "MWMOTE": {
        "class": sv.MWMOTE,
        "grid":  {"proportion": [0.5, 1.0, 1.5],
                  "k1": [3, 7], "k2": [3, 7], "k3": [3, 7],
                  "M": [5, 10, 15], "cf_th": [5.0], "cmax": [2.0]},
    },
}

def gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        return float(np.sqrt(sens * spec))
    return 0.0

def get_combos(grid):
    keys = list(grid.keys())
    return [dict(zip(keys, v)) for v in product(*grid.values())]

def run_method(name, cfg, datasets):
    cls    = cfg["class"]
    combos = get_combos(cfg["grid"])
    clf    = RandomForestClassifier(n_estimators=100,
                                    random_state=RANDOM_STATE)
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
    results = {}
    for ds_name, data in datasets.items():
        X, y = data["X"], data["y"]
        best = {"f1": -1}
        all_r = []
        for params in tqdm(combos, desc=f"{name}/{ds_name}", leave=False):
            fold_m = []
            try:
                for tr, te in skf.split(X, y):
                    Xr, yr = cls(**params,
                                 random_state=RANDOM_STATE).sample(
                        X[tr], y[tr])
                    clf.fit(Xr, yr)
                    yp = clf.predict(X[te])
                    yb = clf.predict_proba(X[te])[:, 1]
                    fold_m.append({
                        "f1":    f1_score(y[te], yp, zero_division=0),
                        "auc":   roc_auc_score(y[te], yb),
                        "gmean": gmean(y[te], yp),
                    })
                avg = {k: float(np.mean([m[k] for m in fold_m]))
                       for k in ["f1", "auc", "gmean"]}
                avg["params"] = params
                all_r.append(avg)
                if avg["f1"] > best["f1"]:
                    best = avg.copy()
            except Exception:
                pass
        results[ds_name] = {"all_results": all_r, "best": best}
    return results

def main():
    with open(DATA_FILE, "rb") as f:
        datasets = pickle.load(f)
    print(f"Loaded {len(datasets)} datasets")
    for name, cfg in METHODS.items():
        out = os.path.join(RESULTS_DIR, f"{name}_results.pkl")
        if os.path.exists(out):
            print(f"  SKIP {name} (exists)")
            continue
        print(f"Running {name}...")
        r = run_method(name, cfg, datasets)
        with open(out, "wb") as f:
            pickle.dump(r, f)
        avg = np.mean([v["best"]["f1"] for v in r.values()
                       if v["best"]["f1"] > 0])
        print(f"  {name}: avg F1 = {avg:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()