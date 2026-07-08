# RiskNot

RiskNot is an end-to-end explainable credit-risk project that predicts next-month credit card default risk from the Taiwan Default of Credit Card Clients dataset.

The project includes a modular machine learning pipeline, a FastAPI scoring service, a React/Vite user interface, SHAP local explanations, Supabase persistence, and a transparent credit-policy overlay for severe delinquency cases.

## What It Predicts

For a single customer, RiskNot returns:

- Default probability
- Risk score from 0 to 100
- Risk segment: Low Risk, Medium Risk, or High Risk
- Local SHAP factors that raise or lower the predicted risk

## Dataset

Dataset: Taiwan Default of Credit Card Clients.

The dataset contains 30,000 credit card client records from Taiwan, covering demographic information, credit limit, repayment status, bill statements, and payment amounts from April 2005 to September 2005.

Target:

```text
default.payment.next.month -> default
```

Target meaning:

- `0`: no default next month
- `1`: default next month

## Current Architecture

```text
RiskNot/
├── api/
│   └── main.py                  # FastAPI prediction and persistence API
├── Data/
│   ├── raw/                     # optional raw data location
│   ├── processed/               # generated processed data
│   └── UCI_Credit_Card.csv      # local dataset copy
├── frontend/
│   ├── src/
│   │   ├── main.jsx             # React UI
│   │   └── styles.css
│   ├── package.json
│   └── package-lock.json
├── models/
│   └── risknot_artifacts.joblib # trained model artifact
├── notebooks/
│   └── RiskNot_Colab.ipynb      # training/study notebook
├── Reports/
│   ├── figures/
│   └── final_report.md
├── src/
│   ├── config.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── predict.py
├── .env.example
├── requirements.txt
├── setup_risknot.bat
├── start_risknot.bat
├── supabase_schema.sql
└── README.md
```

## Machine Learning Pipeline

The training pipeline follows this flow:

```text
Raw CSV
-> cleaning
-> feature engineering
-> stratified train/validation/test split
-> preprocessing fitted on train only
-> model comparison
-> threshold selection
-> test evaluation
-> artifact export
```

Models compared:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost when available

The current production artifact uses LightGBM.

## Feature Engineering

RiskNot creates behavior-based features from the six-month repayment, billing, and payment history.

Examples:

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

Safe division is used for ratio features to avoid division by zero, NaN, and infinite values.

## Model Performance

Validation results from the portfolio notebook:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.7677 | 0.5378 | 0.4179 | 0.6422 | 0.5063 |
| Random Forest | 0.7611 | 0.5349 | 0.5866 | 0.4221 | 0.4909 |
| Logistic Regression | 0.7657 | 0.5212 | 0.3917 | 0.7015 | 0.5027 |
| XGBoost | 0.7423 | 0.5104 | 0.4806 | 0.5367 | 0.5071 |

The selected operating threshold is `0.30`, chosen to prioritize recall for default risk screening.

## Risk Scoring

Risk score:

```text
risk_score = round(default_probability * 100)
```

Segments:

- `0-30`: Low Risk
- `31-60`: Medium Risk
- `61-100`: High Risk

## Credit Policy Overlay

The primary model is LightGBM, a tree-based model. Tree models can sometimes behave non-monotonically on rare or extreme values. For example, a severe repayment delay should not appear safer than a smaller delay in a credit-risk interface.

To make the scoring layer operationally sensible, RiskNot applies a transparent credit-policy overlay after the model prediction. The final displayed probability is:

```text
max(model_probability, policy_floor)
```

The API keeps this transparent by returning:

- `model_probability`: raw model probability
- `default_probability`: final displayed probability
- `policy_adjusted`: whether the overlay changed the output

## Explainable AI

RiskNot uses SHAP local explanations for each prediction.

After scoring, the React UI shows:

- `Raises risk`: strongest positive SHAP contributors
- `Lowers risk`: strongest negative SHAP contributors

These explanations describe model behavior, not guaranteed causal relationships.

## Supabase Persistence

Predictions are saved to Supabase in:

```text
public.risk_assessments
```

Create the table by running `supabase_schema.sql` in the Supabase SQL Editor.

Environment variables are backend-only:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

Copy `.env.example` to `.env` and fill in real values. Do not commit `.env`.

## Run Locally On Windows

Install Python and Node dependencies once:

```powershell
cd "C:\Users\barba\OneDrive\Masaüstü\RiskNot"
.\setup_risknot.bat
```

Start the API and frontend:

```powershell
.\start_risknot.bat
```

Open:

```text
http://localhost:5173
```

Backend health check:

```text
http://localhost:8000/health
```

## Manual Run

FastAPI backend:

```powershell
C:\Users\barba\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn api.main:app
```

React frontend:

```powershell
cd frontend
"C:\Program Files\nodejs\npm.cmd" install
"C:\Program Files\nodejs\npm.cmd" run dev
```

## Training

Use the notebook for study and training:

```text
notebooks/RiskNot_Colab.ipynb
```

The notebook exports:

```text
models/risknot_artifacts.joblib
```

The modular training script is also available:

```powershell
python -m src.train
```

## Limitations

- The dataset is from Taiwan and from 2005.
- This project is for educational and portfolio purposes.
- It should not be used for real credit decisions without updated, local, validated financial data.
- Demographic variables require careful fairness, legal, and compliance review.
- SHAP explanations show model behavior, not guaranteed causal relationships.
- The credit-policy overlay is a transparent product guardrail, not a substitute for real underwriting policy.

## Future Improvements

- Add CatBoost and calibrated ensemble comparison.
- Train a monotonic-constrained LightGBM model.
- Add probability calibration with Brier score comparison.
- Add automated tests for cleaning, feature engineering, and prediction.
- Add deployment configuration for a hosted backend and frontend.
