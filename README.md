# Credit Card Fraud Detection — XGBoost ML Model

A machine learning pipeline for real-time credit card fraud detection using XGBoost on the Kaggle Credit Card Fraud dataset (284,807 transactions, 0.17% fraud rate).

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Base XGBoost | 0.92 | 0.81 | 0.86 | — |
| XGBoost + SMOTE | 0.73 | 0.89 | 0.80 | — |
| **XGBoost + Class Weight** | **0.88** | **0.85** | **0.86** | **0.9652** |
| Stacking (XGB + LightGBM) | 0.90 | 0.80 | 0.84 | 0.9736 |

**Final model:** XGBoost with `scale_pos_weight` — selected for highest recall (0.85), critical in banking where missing real fraud is worse than false alarms.

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions — 492 fraud (0.17%)
- **Features:** V1–V28 (PCA-anonymized), Amount, Time, Class

## Approach

1. **EDA** — Analyzed class imbalance and feature correlations with fraud label
2. **Feature Engineering** — StandardScaler applied to Amount and Time
3. **Class Imbalance** — Compared SMOTE vs class_weight; class_weight selected for better precision-recall balance
4. **Model Training** — XGBoost with `scale_pos_weight=577` to penalize missed fraud cases
5. **Threshold Tuning** — Precision-Recall curve analysis for optimal threshold
6. **Ensemble** — Stacking with LightGBM as comparison
7. **Evaluation** — Classification report + ROC-AUC curve

## Key Finding

V14 is the most influential feature with 62.6% importance score — consistent with correlation analysis showing V14 as the strongest fraud signal in the dataset.

## Stack

- Python 3.9
- XGBoost, LightGBM, scikit-learn
- imbalanced-learn (SMOTE)
- pandas, numpy, matplotlib, seaborn
- joblib (model serialization)

## Usage

```python
import joblib
import pandas as pd

model = joblib.load('model.pkl')
# Input: DataFrame with V1-V28, Amount_scaled, Time_scaled
fraud_probability = model.predict_proba(X)[:, 1]
prediction = model.predict(X)
```

## Part of

This model is used as the ML backbone of the [Real-Time Fraud Detection Pipeline](https://github.com/erdemcetinn/fraud-detection-pipeline) — an AWS microservices system with EKS, SQS, SNS, RDS, and CI/CD.
