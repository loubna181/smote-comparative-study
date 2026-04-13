import os, pickle
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

RESULTS_DIR = os.path.join("results", "pkl")
OUT_DIR     = "results"
METHODS = ["SMOTE", "MWMOTE", "ProWSyn", "Safe_Level_SMOTE",
           "SMOTE_IPF", "ADASYN", "LVQ_SMOTE"]

def load_all():
    out = {}
    for m in METHODS:
        p = os.path.join(RESULTS_DIR, f"{m}_results.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[m] = pickle.load(f)
    return out

def build_df(all_r):
    rows = []
    datasets = list(next(iter(all_r.values())).keys())
    for m, r in all_r.items():
        for ds in datasets:
            b = r[ds]["best"]
            rows.append({"method": m, "dataset": ds,
                         "f1":    round(b.get("f1",    0), 4),
                         "auc":   round(b.get("auc",   0), 4),
                         "gmean": round(b.get("gmean", 0), 4)})
    return pd.DataFrame(rows)

def main():
    all_r = load_all()
    if not all_r:
        print("No results found. Run 01_run_experiments.py first.")
        return
    df = build_df(all_r)
    df.to_csv(os.path.join(OUT_DIR, "comparaison_finale.csv"), index=False)
    print("=== Mean F1 per method ===")
    print(df.groupby("method")["f1"].mean().sort_values(
          ascending=False).round(4))
    print("\n=== Wins per method ===")
    winners = df.loc[df.groupby("dataset")["f1"].idxmax(), "method"]
    print(winners.value_counts())
    print("\n=== Friedman test ===")
    groups = [df[df["method"] == m]["f1"].values for m in METHODS
              if m in df["method"].values]
    stat, p = friedmanchisquare(*groups)
    print(f"  stat={stat:.3f}, p={p:.4f}")
    print("\n=== Wilcoxon vs SMOTE ===")
    smote_f1 = df[df["method"] == "SMOTE"]["f1"].values
    rows = []
    for m in METHODS:
        if m == "SMOTE":
            continue
        m_f1 = df[df["method"] == m]["f1"].values
        try:
            _, pv = wilcoxon(smote_f1, m_f1)
        except Exception:
            pv = 1.0
        delta = float(np.mean(m_f1) - np.mean(smote_f1))
        rows.append({"method": m, "delta_f1": round(delta, 4),
                     "p_value": round(pv, 4),
                     "significant": "YES" if pv < 0.05 else "no"})
    wd = pd.DataFrame(rows)
    wd.to_csv(os.path.join(OUT_DIR, "wilcoxon_final.csv"), index=False)
    print(wd.to_string(index=False))

if __name__ == "__main__":
    main()