from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from src.data import TARGET_COL, feature_columns


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def candidate_models(random_state: int = 42) -> dict:
    models = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1, min_samples_leaf=2),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }


def train_compare_models(df: pd.DataFrame, include_molecular_descriptors: bool = True, test_size: float = 0.2):
    data = df.copy()
    numeric_features, categorical_features = feature_columns(data)

    if not include_molecular_descriptors:
        mol_cols = ["mol_weight", "logp", "tpsa", "hbd", "hba", "aromatic_rings", "rotatable_bonds", "formal_charge"]
        numeric_features = [c for c in numeric_features if c not in mol_cols]

    X = data[numeric_features + categorical_features]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    rows = []
    fitted = {}
    for name, estimator in candidate_models().items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = evaluate_regression(y_test, pred)
        rows.append({"model": name, **metrics})
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("RMSE")
    best_name = results.iloc[0]["model"]
    return fitted[best_name], results, (X_test, y_test)


def save_model(model, path: str | Path = "models/best_model.joblib") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path = "models/best_model.joblib"):
    return joblib.load(path)
