# Unsupervised Fraud Detection with Isolation Forest

Research code for the **Women in Mathematics Directed Research Program (WiM DRP)**, University of Waterloo, Fall 2026.

This project tests whether **unsupervised anomaly detection** can identify fraudulent e-commerce transactions **without using fraud labels during training**.

---

## Research Question

> Can an Isolation Forest flag fraudulent transactions using only the structure of the data, with labels withheld until evaluation?

Fraud labels in practice are scarce, delayed and incomplete, so a label-free detector could help surface new or unseen fraud patterns.

---

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle)

| Item | Detail |
|---|---|
| Files used | `train_transaction.csv`, `train_identity.csv` (left-joined on `TransactionID`) |
| Size | 590,540 transactions × 434 columns |
| Fraud rate | 3.5% (20,663 fraud vs 569,877 legitimate) |

The raw data is **not included** in this repository. Download it from the [Kaggle competition page](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place the CSVs in a `Data/` folder:

```
Data/
├── train_transaction.csv
└── train_identity.csv
```

---

## Repository Structure

```
.
├── README.md
├── .gitignore
├── isolationforest_fraud.ipynb      # Exploratory data analysis + baseline model
└── ...                              # Experiment notebooks / results
```

---

## Methodology

1. **Exploratory Data Analysis**
   - Fraud prevalence, transaction amount distribution (log scale)
   - Fraud rates across categorical features (`ProductCD`, `card4`, `card6`)
   - Missing-value patterns for fraud vs non-fraud
   - Correlation structure of the anonymized `V1`–`V339` features

2. **Baseline Model**
   - scikit-learn `IsolationForest` with default parameters (`random_state = 100`)
   - Features: numeric transaction columns, `isFraud` withheld, missing values filled with −999

3. **Sensitivity Experiments** (one parameter changed at a time)
   - `contamination`: 0.01, 0.02, 0.035, 0.05, 0.1
   - `n_estimators`: 10 → 1,000 trees
   - `max_samples`: 128, 512, full dataset
   - `max_features`: 0.1 → 1.0

4. **Evaluation** (labels used only after scoring)
   - ROC-AUC, PR-AUC (random baseline = 0.035), top-1% ranking overlap, runtime

---

## Key Findings

| Area | Finding |
|---|---|
| **EDA** | Fraud amounts are more spread out than legitimate ones; ProductCD C (11.7%), Discover (7.7%) and credit cards (6.7%) have above-average fraud rates; fraud rows are consistently **less** likely to have missing data |
| **Baseline** | ROC-AUC **0.751**, PR-AUC **0.094** (~2.7× random). The top 10 anomalies were a cluster of near-identical legitimate transactions, none fraud. |
| **contamination** | Only moves the decision threshold: flagged count scales linearly (~5,905 per +0.01); ROC-AUC and PR-AUC unchanged |
| **n_estimators** | Ranking stabilizes at ~50–100 trees (top-1% overlap 91–95%); more trees improve consistency, not accuracy |
| **max_samples** | Small sub-samples (128–512) match the default; the full dataset performs worse (ROC-AUC 0.714, PR-AUC 0.082) and runs ~4–5× slower |
| **max_features** | Small PR-AUC gain at 0.10 (0.100 vs 0.094); ROC-AUC unchanged |

**Takeaway:** Isolation Forest captures real fraud signal without labels, but its most extreme anomalies are rare legitimate patterns. Parameter tuning gives only marginal gains, which suggests feature representation matters more than model settings.

---

## How to Run

```bash
git clone https://github.com/millxcc7/WIM-Direct-Research-MLFraudDetection.git
cd WIM-Direct-Research-MLFraudDetection

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Add the Kaggle CSVs to `Data/`, open the notebook in Jupyter or VS Code, and run all cells.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

---

## References

1. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* IEEE International Conference on Data Mining (ICDM).
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
3. IEEE-CIS & Vesta Corporation (2019). *IEEE-CIS Fraud Detection* [Dataset]. Kaggle.

---

## Author

**Milly**: Mathematics & Business Administration (Double Degree), University of Waterloo
