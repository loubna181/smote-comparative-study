import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

COMPARE_FILE = os.path.join("results", "comparaison_finale.csv")
FIGURES_DIR  = os.path.join("results", "figures")
CSV_DIR      = os.path.join("results", "csv_excel")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CSV_DIR,     exist_ok=True)

METHODS = ["SMOTE", "MWMOTE", "ProWSyn", "Safe_Level_SMOTE",
           "SMOTE_IPF", "ADASYN", "LVQ_SMOTE"]
COLORS  = ["#2E75B6", "#C00000", "#548235", "#7030A0",
           "#E36C09", "#17375E", "#00B0F0"]
DPI = 240

def load():
    return pd.read_csv(COMPARE_FILE)

def fig1_boxplots(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, lbl) in zip(axes, [("f1","F1-Score"),
                                      ("auc","AUC-ROC"),
                                      ("gmean","G-mean")]):
        data = [df[df["method"] == m][col].values for m in METHODS]
        bp   = ax.boxplot(data, patch_artist=True)
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(METHODS)+1))
        ax.set_xticklabels([m.replace("_"," ") for m in METHODS],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(lbl, fontsize=12)
        ax.grid(axis="y", alpha=0.4)
    plt.suptitle("Performance distributions per method",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_boxplots.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()
    df.to_csv(os.path.join(CSV_DIR, "fig1_boxplots.csv"), index=False)
    print("  fig1_boxplots.png")

def fig2_heatmap(df):
    pivot = df.pivot(index="method", columns="dataset", values="f1")
    pivot = pivot.reindex(METHODS)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.4, cbar_kws={"label": "F1-Score"})
    ax.set_title("F1-Score by method and dataset", fontsize=13, pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_heatmap.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()
    pivot.to_csv(os.path.join(CSV_DIR, "fig2_heatmap.csv"))
    print("  fig2_heatmap.png")

def fig3_wins(df):
    winners = df.loc[df.groupby("dataset")["f1"].idxmax()]
    counts  = winners["method"].value_counts().reindex(METHODS, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color=COLORS, alpha=0.85)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xlabel("Number of wins", fontsize=11)
    ax.set_title("Wins per method (17 datasets)", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_wins.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()
    counts.to_csv(os.path.join(CSV_DIR, "fig3_wins.csv"))
    print("  fig3_wins.png")

def fig4_radar(df):
    means  = df.groupby("method")[["f1","auc","gmean"]].mean()
    means  = means.reindex(METHODS)
    labels = ["F1-Score", "AUC-ROC", "G-mean"]
    N      = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.7, 1.0)
    for m, c in zip(METHODS, COLORS):
        vals = means.loc[m, ["f1","auc","gmean"]].tolist()
        vals = vals + [vals[0]]
        ax.plot(angles, vals, "o-", lw=2, color=c,
                label=m.replace("_"," "))
        ax.fill(angles, vals, alpha=0.05, color=c)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Average performance radar", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_radar.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()
    means.to_csv(os.path.join(CSV_DIR, "fig4_radar.csv"))
    print("  fig4_radar.png")

def fig5_barres(df):
    means = df.groupby("method")[["f1","auc","gmean"]].mean()
    means = means.reindex(METHODS)
    x     = np.arange(len(METHODS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (col, lbl) in enumerate([("f1","F1"),
                                     ("auc","AUC"),
                                     ("gmean","G-mean")]):
        ax.bar(x + i*width, means[col], width, label=lbl, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_"," ") for m in METHODS],
                       rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("F1, AUC and G-mean per method", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_barres.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()
    means.to_csv(os.path.join(CSV_DIR, "fig5_barres.csv"))
    print("  fig5_barres.png")

def main():
    if not os.path.exists(COMPARE_FILE):
        print("Run 02_statistical_analysis.py first.")
        return
    df = load()
    print("Generating figures:")
    fig1_boxplots(df)
    fig2_heatmap(df)
    fig3_wins(df)
    fig4_radar(df)
    fig5_barres(df)
    print("Done.")

if __name__ == "__main__":
    main()