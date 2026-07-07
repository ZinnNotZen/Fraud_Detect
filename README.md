# Fraud_Detect
Using ML to find if an transaction is fraudulent or not

## Overview
This project builds a fraud detection model using the IEEE-CIS Fraud Detection dataset (Vesta Corporation e-commerce transaction data, via Kaggle). The goal was to predict whether a transaction is fraudulent, with a focus on realistic data handling — messy real-world data, severe class imbalance, and a threshold choice grounded in a business trade-off rather than a default cutoff.

The project combines **SQL** (for joining and preparing the raw data) and **Python** (for cleaning, feature engineering, modeling, and evaluation).

## Data
- **Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
- **Files used:** `train_transaction.csv`, `train_identity.csv` (joined on `TransactionID`)
- ~590,540 transactions, ~3.5% labeled as fraud

## Project Structure
- `01_sql/` — SQL notebook: joining transaction and identity tables (DuckDB), initial row/consistency checks
- `02_cleaning/` — Python notebook: missing value analysis, null-vs-fraud correlation checks, feature engineering (presence flags, time-based features from `TransactionDT`)
- `03_modeling/` — Python notebook: baseline XGBoost model, feature selection (importance-based sweep), threshold tuning, final evaluation

## Methodology Summary

### Data Preparation
- Joined `train_transaction` and `train_identity` via a SQL LEFT JOIN on `TransactionID` (using DuckDB), preserving all transactions regardless of whether identity data was captured.
- Investigated missing data rather than dropping it by default: checked whether missingness itself correlated with fraud. Several `id_` columns and `dist2` showed a consistent ~2x higher fraud rate when the field *was* present versus missing — so instead of dropping these high-null columns outright, I engineered binary "was this field present" flags to preserve that signal.
- Engineered time-based features from `TransactionDT` (seconds-since-reference, not a real timestamp) to extract transaction hour and day.

### Modeling & Feature Selection
- Trained a baseline XGBoost classifier on the full cleaned feature set (383 columns).
- Used feature importance to test smaller feature subsets (50, 100, 120 features). The **120-feature model slightly outperformed the full 383-feature model** on both ROC AUC and PR AUC, suggesting the lowest-importance features were adding noise rather than useful signal.
- Final model uses the top 120 features by importance.

### Threshold Selection
XGBoost outputs a fraud probability, not a hard label — the cutoff used to call something "fraud" is a design decision. I evaluated F1-score across thresholds from 0.05 to 0.50:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.05 | 0.281 | 0.803 | 0.416 |
| 0.20 | 0.679 | 0.610 | 0.643 |
| 0.25 | 0.739 | 0.574 | 0.646 |
| 0.50 (default) | 0.892 | 0.448 | 0.596 |

F1 technically peaked at 0.25, but I selected **threshold 0.20** as the final model, reasoning that a missed fraud case (false negative) typically carries a higher real-world cost than a false alarm (false positive) — so recall was intentionally weighted above a pure F1-maximizing choice. I stopped short of the most aggressive recall-maximizing thresholds (e.g., 0.05) because the resulting false-positive rate would likely overwhelm a real review process and create a poor customer experience.

## Final Model Results (Threshold = 0.20)
- **Fraud caught:** 2,521 of 4,133 actual fraud cases (61.0%)
- **Fraud missed:** 1,612 cases (39.0%)
- **Legitimate transactions incorrectly flagged:** 1,192 of 113,975 (1.05%)

This is a substantial improvement in fraud recall over the default 0.5 threshold (45% → 61%), while keeping the false-positive rate on legitimate transactions low (~1%).

Note on metrics: accuracy (98%) and weighted-average F1 (0.98) look strong but are dominated by the large non-fraud class and don't reflect fraud-detection performance well. Macro-average F1 (0.82) and the fraud-class metrics specifically (precision 0.68, recall 0.61, F1 0.64) are the more meaningful numbers for this use case.

## Recommendation: Tiered Security Levels
Rather than a single fixed threshold for every customer, a real deployment could offer **tiered, opt-in fraud protection**:
- **Standard tier (threshold ≈ 0.20):** balances fraud protection with minimal disruption — suitable for most users.
- **Enhanced tier (threshold ≈ 0.05):** for customers who prioritize maximum fraud protection and are willing to accept more frequent verification (e.g., a text confirmation for borderline transactions). At this threshold, fraud recall rises to 80.3%, though roughly 7.5% of legitimate transactions would trigger a review step.

A single global threshold forces the same trade-off on every customer, when in practice risk tolerance varies. Letting customers opt into a stricter protection tier (in exchange for more frequent step-up verification) would let the system serve both preferences rather than picking one trade-off for everyone.

*(Implementing customer-selectable thresholds in a live product would require additional infrastructure — consent flows, a preference-setting mechanism, and likely compliance/liability considerations — beyond what the model alone determines.)*

## Tools Used
- **DuckDB** — SQL joins and initial data exploration directly on CSV files
- **pandas** — data cleaning and feature engineering
- **XGBoost** — modeling
- **scikit-learn** — train/test splitting, evaluation metrics

## Limitations & Future Work
- Model was evaluated on a held-out validation split from the training data only (Kaggle's true test set labels are not public).
- Many features (`V1`–`V339`, `C1`–`C14`, `D1`–`D15`) are anonymized by Vesta, limiting interpretability of individual feature contributions beyond importance ranking.
- Future work could include testing alternative models (LightGBM, logistic regression baseline for comparison), more thorough hyperparameter tuning, and exploring the tiered-threshold idea with a cost-based (dollar-value) framing rather than precision/recall alone.
