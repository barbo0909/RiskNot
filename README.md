# RiskNot: Explainable Credit Default Risk Modeling

RiskNot is an end-to-end machine learning portfolio project for predicting next-month credit-card default risk using the Taiwan Default of Credit Card Clients dataset.

The project trains tabular classification models, tunes a final LightGBM model with cross-validation, optimizes the classification threshold for credit-risk trade-offs, explains model behavior with SHAP, and converts model probability into a readable 0-100 risk score.

## Business Problem

Credit-risk teams need to identify clients who may default before the next billing cycle. In this project, the model predicts:

- Default probability
- Risk score from 0 to 100
- Risk segment: Low Risk, Medium Risk, or High Risk
- Main model drivers using SHAP explainability

## Dataset

Dataset: Taiwan Default of Credit Card Clients.

The dataset contains 30,000 client records from Taiwan, using demographic, credit, bill statement, repayment status, and payment amount variables from April 2005 to September 2005.

Target column:

```text
default.payment.next.month -> default
```

Target meaning:

- `0`: no default next month
- `1`: default next month

## Project Pipeline

1. Load CSV or zipped CSV.
2. Remove `ID`.
3. Rename target to `default`.
4. Clean categorical values.
5. Engineer repayment, utilization, bill, and payment behavior features.
6. Split data into train, validation, and test sets.
7. Fit preprocessing only on training data.
8. Train Logistic Regression, Random Forest, LightGBM, and XGBoost when available.
9. Tune LightGBM with `RandomizedSearchCV` using stratified cross-validation and PR-AUC scoring.
10. Select the final model based on validation PR-AUC.
11. Optimize threshold based on credit-risk trade-offs.
12. Evaluate on the test set.
13. Explain predictions with SHAP.
14. Save model artifacts with joblib.

## Feature Engineering

Important engineered features include:

- `max_delay`
- `avg_delay`
- `recent_delay`
- `num_delayed_months`
- `severe_delay_flag`
- `repeated_delay_flag`
- `credit_utilization_ratio`
- `recent_utilization_ratio`
- `payment_to_bill_ratio`
- `recent_payment_to_bill_ratio`
- `low_payment_flag`
- `high_utilization_flag`
- `recent_nonpayment_flag`

Safe division is used for ratio features to avoid invalid infinite or NaN values.

## Model Results

Validation comparison:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.7620 | 0.5387 | 0.4149 | 0.6472 | 0.5057 |
| Random Forest | 0.7602 | 0.5331 | 0.5905 | 0.4261 | 0.4950 |
| XGBoost | 0.7390 | 0.5113 | 0.4515 | 0.5337 | 0.4892 |
| Logistic Regression | 0.7591 | 0.5095 | 0.3866 | 0.6905 | 0.4957 |

LightGBM was selected as the final model because it achieved the strongest validation PR-AUC.

## Threshold Choice

The selected threshold is `0.30`.

This threshold prioritizes recall for the default class. In credit risk, false negatives are costly because they represent risky clients classified as safe.

Test results at threshold `0.30`:

- ROC-AUC: 0.7693
- PR-AUC: 0.5399
- Brier score: 0.1628
- Accuracy: 0.65
- Default recall: 0.75
- Default precision: 0.36

Confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2164 | 1340 |
| Actual 1 | 245 | 751 |

## SHAP Explainability

SHAP analysis shows that the most important model drivers are:

- `max_delay`
- `PAY_0`
- `min_payment_amount`
- `recent_utilization_ratio`
- `credit_utilization_ratio`
- `LIMIT_BAL`

Risk-increasing patterns include repayment delays, high utilization, and weak payment behavior. Risk-decreasing patterns include higher credit limits, stronger payment behavior, and lower utilization.

## Risk Scoring

The risk score is calculated as:

```text
risk_score = round(default_probability * 100)
```

Segments:

- 0-30: Low Risk
- 31-60: Medium Risk
- 61-100: High Risk

Example:

```text
default_probability: 0.0705
risk_score: 7
risk_segment: Low Risk
```

## Repository Structure

```text
RiskNot/
├── Data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── app/
├── api/
├── frontend/
├── models/
├── Reports/
│   ├── figures/
│   └── final_report.md
├── notebooks/
│   └── RiskNot_Colab.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train from the modular script:

```bash
python -m src.train
```

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Run the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

To persist predictions in Supabase, create the table from `supabase_schema.sql` and set backend-only environment variables:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

The React frontend does not receive the service role key. It sends predictions to FastAPI, and FastAPI stores the assessment result in Supabase.

Run the React + Vite frontend:

```bash
cd frontend
npm install
npm run dev
```

For Colab study and heavy training, open `notebooks/RiskNot_Colab.ipynb` and run the notebook top to bottom. The notebook saves the final artifact to both the local runtime `models/` folder and Google Drive when mounted:

```text
/content/drive/MyDrive/RiskNot/models/risknot_artifacts.joblib
```

## Dashboard Screenshot Placeholder

Screenshots can be added later under:

```text
Reports/figures/
```

## Limitations

- The dataset is from Taiwan and from 2005.
- The model is for educational and portfolio purposes.
- It should not be used for real credit decisions without updated, local, validated financial data.
- Demographic variables require careful fairness, legal, and compliance review.
- SHAP explanations show model behavior, not guaranteed causal relationships.

## Future Improvements

- Add deeper LightGBM hyperparameter tuning.
- Add probability calibration and compare before/after calibration.
- Add full fairness diagnostics by `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE_GROUP`.
- Expand the Streamlit dashboard with SHAP local explanations.
- Add automated tests for data cleaning, feature engineering, and prediction.
