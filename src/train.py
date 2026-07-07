"""Train RiskNot models and save the final artifact."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from src.config import DEFAULT_ARTIFACT_PATH, MODELS_DIR, RANDOM_STATE, TARGET_COLUMN
from src.data_cleaning import clean_credit_data, load_credit_data
from src.evaluate import classification_metrics, threshold_table
from src.feature_engineering import add_features
from src.preprocessing import build_preprocessor, get_feature_groups


def build_models(columns: list[str], y_train: pd.Series) -> dict[str, Pipeline]:
    """Build candidate model pipelines."""
    pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(columns, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(columns, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(columns, scale_numeric=False)),
                (
                    "model",
                    LGBMClassifier(
                        random_state=RANDOM_STATE,
                        scale_pos_weight=pos_weight,
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                    ),
                ),
            ]
        )

    if XGBOOST_AVAILABLE:
        models["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(columns, scale_numeric=False)),
                (
                    "model",
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        n_estimators=300,
                        eval_metric="logloss",
                        scale_pos_weight=pos_weight,
                        tree_method="hist",
                    ),
                ),
            ]
        )

    return models


def train_and_save(
    artifact_path=DEFAULT_ARTIFACT_PATH,
    selected_threshold: float = 0.30,
    tune_lightgbm: bool = True,
    n_iter: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train candidate models, select by validation PR-AUC, and save artifacts."""
    raw = load_credit_data()
    df = add_features(clean_credit_data(raw))
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    models = build_models(X_train.columns.tolist(), y_train)
    rows = []
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        row = {"model": name}
        row.update(classification_metrics(y_val, val_proba, threshold=0.40))
        rows.append(row)
        fitted_models[name] = pipeline

    if tune_lightgbm and LIGHTGBM_AVAILABLE:
        pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        tuned_lgbm_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train.columns.tolist(), scale_numeric=False)),
                (
                    "model",
                    LGBMClassifier(
                        random_state=RANDOM_STATE,
                        scale_pos_weight=pos_weight,
                        objective="binary",
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
        param_distributions = {
            "model__n_estimators": [300, 500, 700, 900, 1200, 1500],
            "model__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
            "model__num_leaves": [15, 31, 45, 63, 90, 127],
            "model__max_depth": [-1, 3, 4, 5, 6, 8, 10, 12],
            "model__min_child_samples": [10, 20, 30, 50, 80, 120],
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            "model__reg_lambda": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "model__min_split_gain": [0.0, 0.01, 0.05, 0.1],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            tuned_lgbm_pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=2,
            refit=True,
        )
        search.fit(X_train, y_train)
        tuned_model = search.best_estimator_
        tuned_val_proba = tuned_model.predict_proba(X_val)[:, 1]
        tuned_row = {"model": "lightgbm_tuned"}
        tuned_row.update(classification_metrics(y_val, tuned_val_proba, threshold=0.40))
        rows.append(tuned_row)
        fitted_models["lightgbm_tuned"] = tuned_model

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    final_name = results.iloc[0]["model"]
    final_model = fitted_models[final_name]
    val_proba = final_model.predict_proba(X_val)[:, 1]
    thresholds = threshold_table(y_val, val_proba, [0.30, 0.35, 0.40, 0.45, 0.50, 0.60])

    categorical, numeric = get_feature_groups(X_train.columns.tolist())
    artifacts = {
        "model": final_model,
        "final_model_name": final_name,
        "selected_threshold": selected_threshold,
        "feature_columns": X_train.columns.tolist(),
        "categorical_features": categorical,
        "numeric_features": numeric,
        "validation_results": results,
        "threshold_table": thresholds,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, artifact_path)
    return results, thresholds


if __name__ == "__main__":
    model_results, threshold_results = train_and_save()
    print(model_results)
    print(threshold_results)
