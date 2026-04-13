# SMOTE Comparative Study

Systematic comparison of 7 oversampling methods on 17 UCI datasets.

## Methods

| Method | Combinations |
|--------|-------------|
| SMOTE (benchmark) | 6 |
| Safe_Level_SMOTE | 12 |
| ADASYN | 18 |
| LVQ_SMOTE | 27 |
| ProWSyn | 81 |
| SMOTE_IPF | 144 |
| MWMOTE | 288 |

Total: 576 combinations x 17 datasets x 5 folds = 48,960 runs

## Key results

- LVQ_SMOTE: best global F1 (0.7461)
- MWMOTE: most wins (7/17 datasets)
- ADASYN: only method significantly worse than SMOTE (p=0.031)
- Friedman test: stat=14.75, p=0.022

## Installation

    pip install -r requirements.txt

## Pipeline

    python src/00_prepare_datasets.py
    python src/01_run_experiments.py
    python src/02_statistical_analysis.py
    python src/03_generate_figures.py
    python src/04_generate_report.py

## Reference

Kovacs, G. (2019). smote-variants. Neurocomputing, 366, 352-354.