"""
Throo NYC demand-level estimate based on surge and available driver supply.
Run `python ml_pipeline.py` to train and save artifacts/demand_model.joblib.

The model classifies the supplied conditions; it does not forecast a future
time period.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("throo_nyc_demand.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_FILENAME = "demand_model.joblib"
MODEL_PATH = ARTIFACT_DIR / MODEL_FILENAME

DEMAND_MAP = {"Low": 0, "Medium": 1, "High": 2}
DEMAND_NAMES = {v: k for k, v in DEMAND_MAP.items()}
DEMAND_NAME_LOOKUP = {name.lower(): code for name, code in DEMAND_MAP.items()}

NUMERIC_COLS = ["surge", "available_driver_count"]
INPUT_COLS = NUMERIC_COLS
FEATURE_COLS = NUMERIC_COLS


def get_model_path(artifact_dir: Path = ARTIFACT_DIR) -> Path:
    return Path(artifact_dir) / MODEL_FILENAME


def normalize_demand_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        lowered = value.lower()
        if lowered in DEMAND_NAME_LOOKUP:
            return DEMAND_NAME_LOOKUP[lowered]
        try:
            numeric = int(float(value))
        except ValueError:
            return None
        return numeric if numeric in set(DEMAND_MAP.values()) else None
    if isinstance(value, (int, float)):
        value = int(value)
        if value in set(DEMAND_MAP.values()):
            return value
    return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in INPUT_COLS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    if "demand" in cleaned.columns:
        cleaned["demand"] = cleaned["demand"].map(normalize_demand_label)
    return cleaned.drop_duplicates().dropna(subset=INPUT_COLS + ["demand"])


def train(data_path: Path = DATA_PATH, artifact_dir: Path = ARTIFACT_DIR) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(artifact_dir)
    df = clean_data(pd.read_csv(data_path))

    X = df[FEATURE_COLS]
    y = pd.to_numeric(df["demand"], errors="coerce").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # Model selection uses cross-validation on the training set only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
    }
    cv_results = {}
    for name, (model, scaled) in candidates.items():
        Xtr = X_train_s if scaled else X_train
        scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="f1_weighted")
        cv_results[name] = float(scores.mean())
        print(f"  {name}: CV F1={scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_results, key=cv_results.get)
    model, scaled = candidates[best_name]
    best_params = None
    if best_name == "Random Forest":
        gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                          {"max_depth": [3, 6, None], "min_samples_leaf": [5, 20]},
                          cv=cv, scoring="f1_weighted", n_jobs=-1).fit(X_train, y_train)
        model, best_params = gs.best_estimator_, gs.best_params_
    model.fit(X_train_s if scaled else X_train, y_train)

    # The test set is used once, for the final report.
    y_pred = model.predict(X_test_s if scaled else X_test)
    test_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred,
                                   target_names=["Low", "Medium", "High"], digits=4)

    joblib.dump({"model": model, "scaler": scaler, "needs_scaling": scaled,
                 "feature_cols": FEATURE_COLS}, model_path)
    metadata = {"best_model": best_name, "best_params": best_params,
                "cv_f1_weighted": cv_results, "test_f1_weighted": float(test_f1),
                "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
                "features": FEATURE_COLS, "classification_report": report}
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


class DemandPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.needs_scaling = bool(bundle.get("needs_scaling", False))
        self.feature_cols = list(bundle.get("feature_cols", FEATURE_COLS))

    def predict(self, record: dict) -> dict:
        if not isinstance(record, dict):
            raise ValueError("Record must be a dictionary of feature values.")
        missing = [c for c in self.feature_cols if c not in record]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")
        prepared = {col: float(record[col]) for col in self.feature_cols}
        X = pd.DataFrame([prepared])[self.feature_cols]
        if self.needs_scaling and self.scaler is not None:
            X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)[0]
        probs = {DEMAND_NAMES[int(c)]: round(float(p), 4)
                 for c, p in zip(self.model.classes_, proba)}
        return {"demand": max(probs, key=probs.get), "probabilities": probs}

    def forecast(self, record: dict) -> dict:
        return self.predict(record)

    def forecasting(self, record: dict) -> dict:
        return self.predict(record)

    def forcasting(self, record: dict) -> dict:
        return self.predict(record)


if __name__ == "__main__":
    m = train()
    print(f"\nBest: {m['best_model']}  params={m['best_params']}")
    print(f"Test F1 (weighted): {m['test_f1_weighted']:.4f}")
    print(m["classification_report"])