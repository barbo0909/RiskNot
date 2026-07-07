# RiskNot Final Report

## Project Overview

RiskNot is an explainable credit-risk machine learning project built with the Taiwan Default of Credit Card Clients dataset. The objective is to predict whether a credit card client will default next month and convert the model output into an interpretable risk score.

The target variable was renamed from `default.payment.next.month` to `default`.

- `0`: No default next month
- `1`: Default next month

The project uses a tabular machine learning workflow: data cleaning, feature engineering, model comparison, threshold optimization, test evaluation, SHAP explainability, and risk scoring.

## Dataset Summary

The dataset contains 30,000 client records from Taiwan, covering credit card behavior between April 2005 and September 2005.

Main feature groups:

- Demographic variables: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- Credit limit: `LIMIT_BAL`
- Repayment status variables: `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`
- Bill statement variables: `BILL_AMT1` to `BILL_AMT6`
- Previous payment variables: `PAY_AMT1` to `PAY_AMT6`

The dataset has no missing values in the checked core columns. The default rate is:

- No default: 77.88%
- Default: 22.12%

This shows that the dataset is imbalanced. Because of this imbalance, model evaluation focused on recall, PR-AUC, F1-score, and false negatives instead of accuracy alone.

## EDA Insights

The target distribution shows that most clients did not default next month. Only about 22.12% of clients belong to the default class.

The `LIMIT_BAL` distribution indicates that defaulting clients are more concentrated in lower credit-limit ranges. Non-default clients appear more often in higher credit-limit ranges. This suggests that higher credit limits may be associated with lower default risk, but this should not be interpreted as a causal relationship.

Overall, the exploratory analysis supports the idea that repayment behavior, credit utilization, and payment amounts are important signals for default prediction.

## Feature Engineering

Several behavior-based credit-risk features were created from the raw dataset. These engineered features help the model learn repayment patterns, utilization behavior, and payment strength.

Important engineered features include:

- `max_delay`
- `avg_delay`
- `recent_delay`
- `num_delayed_months`
- `severe_delay_flag`
- `repeated_delay_flag`
- `delay_trend`
- `credit_utilization_ratio`
- `recent_utilization_ratio`
- `payment_to_bill_ratio`
- `recent_payment_to_bill_ratio`
- `low_payment_flag`
- `high_utilization_flag`
- `recent_nonpayment_flag`

Safe division was used for ratio features to avoid division by zero and invalid infinite values.

## Model Comparison

The following models were trained and compared:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

Validation results:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.7620 | 0.5387 | 0.4149 | 0.6472 | 0.5057 |
| Random Forest | 0.7602 | 0.5331 | 0.5905 | 0.4261 | 0.4950 |
| XGBoost | 0.7390 | 0.5113 | 0.4515 | 0.5337 | 0.4892 |
| Logistic Regression | 0.7591 | 0.5095 | 0.3866 | 0.6905 | 0.4957 |

LightGBM achieved the strongest validation performance, especially by PR-AUC, and was selected as the final model.

## Threshold Optimization

Different probability thresholds were compared instead of blindly using the default 0.50 threshold.

| Threshold | Precision | Recall | F1 | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.3477 | 0.7437 | 0.4739 | 1388 | 255 |
| 0.35 | 0.3812 | 0.6935 | 0.4920 | 1120 | 305 |
| 0.40 | 0.4149 | 0.6472 | 0.5057 | 908 | 351 |
| 0.45 | 0.4521 | 0.6070 | 0.5182 | 732 | 391 |
| 0.50 | 0.4846 | 0.5678 | 0.5229 | 601 | 430 |
| 0.60 | 0.5439 | 0.4854 | 0.5130 | 405 | 512 |

The selected threshold is `0.30`.

This threshold was selected to prioritize recall for the default class. In credit risk, a false negative means that a truly risky client is classified as safe. This can be more costly than a false positive. The trade-off is that more non-default clients are flagged as risky.

## Test Set Evaluation

Final test results using the selected threshold of `0.30`:

- ROC-AUC: 0.7693
- PR-AUC: 0.5399
- Brier score: 0.1628
- Accuracy: 0.65

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 - No Default | 0.90 | 0.62 | 0.73 | 3504 |
| 1 - Default | 0.36 | 0.75 | 0.49 | 996 |

Confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2164 | 1340 |
| Actual 1 | 245 | 751 |

The model correctly identified 751 default cases and missed 245 default cases. This means it captured approximately 75% of actual defaults on the test set. The cost of this conservative threshold is a higher number of false positives.

## SHAP Explainability

SHAP analysis was used to explain the final LightGBM model.

The most important features were:

- `max_delay`
- `PAY_0`
- `min_payment_amount`
- `recent_utilization_ratio`
- `credit_utilization_ratio`
- `LIMIT_BAL`
- `bill_amount_std`
- `bill_trend`
- `PAY_AMT3`
- `PAY_AMT2`

Main risk-increasing factors:

- Higher repayment delay
- Recent payment delay
- Higher credit utilization ratio
- Lower payment behavior
- Lower minimum payment amount

Main risk-decreasing factors:

- Higher credit limit
- Stronger payment behavior
- Lower credit utilization
- No recent repayment delay

The SHAP results are aligned with credit-risk intuition. The model relies heavily on repayment history, recent payment status, credit utilization, and payment strength.

## Risk Scoring

The model probability was converted into a simple 0-100 risk score:

```text
risk_score = round(default_probability * 100)
```

Risk segments:

- 0-30: Low Risk
- 31-60: Medium Risk
- 61-100: High Risk

Example prediction:

```text
default_probability: 0.0705
risk_score: 7
risk_segment: Low Risk
```

This customer has an estimated default probability of about 7%. Since the probability is below the selected threshold of 0.30, the customer is classified as low risk / no default.

## Final Conclusion

The RiskNot notebook successfully builds an explainable credit-risk prediction pipeline. LightGBM was selected as the final model because it achieved the best validation PR-AUC and competitive ROC-AUC.

The final model is designed as a conservative screening system. By selecting a threshold of 0.30, the model prioritizes catching more defaulting clients and reducing false negatives. This is appropriate for a credit-risk use case where missing a risky client may be more costly than flagging a safe client for review.

The most important model drivers are repayment delay, recent repayment status, credit utilization, payment behavior, and credit limit. These findings are consistent with business intuition.

## Limitations

This project is for educational and portfolio purposes.

Important limitations:

- The dataset is from Taiwan and from 2005.
- The model should not be used for real credit decisions without updated, local, validated financial data.
- Demographic variables such as `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` require careful fairness, legal, and compliance review.
- SHAP explanations show model behavior, not guaranteed causal relationships.
- A real deployment would require deeper fairness testing, monitoring, calibration, governance, and regulatory review.

## Future Work

Potential improvements:

- Add full hyperparameter tuning for LightGBM.
- Add probability calibration and compare calibrated vs uncalibrated probabilities.
- Add fairness diagnostics by `SEX`, `EDUCATION`, `MARRIAGE`, and age groups.
- Build a Streamlit dashboard for single-customer and batch predictions.
- Convert the notebook into a modular production-style repository with reusable `src/` scripts.
