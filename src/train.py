"""
Power System Load Type Classification
Run from the project root: python src/train.py
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "load_data.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"
TARGET_COLUMN = "Load_Type"  # normalized header aliases are handled below
RANDOM_STATE = 42


def normalize_name(name: str) -> str:
    """Normalize headers to lowercase underscore-separated names."""
    import re
    name = str(name).strip().lower()
    name = name.replace("kvarh", "kvarh").replace("kw/h", "kwh")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")



def load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        candidates = (
            list((ROOT / "data").glob("*.csv"))
            + list((ROOT / "data").glob("*.xlsx"))
            + list((ROOT / "data").glob("*.xls"))
        )

        if len(candidates) == 1:
            path = candidates[0]
            print(f"Using detected data file: {path.name}")
        else:
            raise FileNotFoundError(
                f"Dataset not found at: {DATA_PATH}\n"
                "Place your CSV file in the data folder."
            )

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return pd.read_excel(path)



def resolve_target(columns):
    normalized = {normalize_name(c): c for c in columns}
    aliases = ["load_type", "loadtype", "load", "target"]
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(
        "Could not find the target column. Expected a column named 'Load Type' or 'Load_Type'.\n"
        f"Columns found: {list(columns)}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("POWER SYSTEM LOAD TYPE PREDICTION")
    print("=" * 68)

    df = load_excel(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Columns found:", list(df.columns))

    target_original = resolve_target(df.columns)
    df = df.drop_duplicates().copy()
    df[target_original] = df[target_original].astype("string").str.strip()
    df = df[df[target_original].notna() & (df[target_original] != "")].copy()

    # Parse and sort chronologically when Date is available.
    normalized_columns = {normalize_name(c): c for c in df.columns}
    date_col = next((normalized_columns[k] for k in ("date", "date_time", "datetime", "timestamp")if k in normalized_columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df = df[df[date_col].notna()].copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        if df.empty:
            raise ValueError("No rows remain after parsing the Date column. Check the date format.")
        latest_month = df[date_col].max().to_period("M")
        test_mask = df[date_col].dt.to_period("M") == latest_month
        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()
        print(f"\nChronological split: latest month ({latest_month}) is the test set.")
    else:
        raise ValueError(
            "A Date column is required because the problem statement specifies using the last month as the test set."
        )

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Cannot create a last-month test split. Make sure the dataset has at least two different months."
        )

    # Extract date features and exclude raw date to keep model inputs numeric.
    for part in (train_df, test_df):
        part["date_year"] = part[date_col].dt.year
        part["date_month"] = part[date_col].dt.month
        part["date_day"] = part[date_col].dt.day
        part["date_dayofweek"] = part[date_col].dt.dayofweek

    # Map source feature names into a normalized working frame.
    X_train = train_df.drop(columns=[target_original, date_col]).copy()
    X_test = test_df.drop(columns=[target_original, date_col]).copy()
    X_train.columns = [normalize_name(c) for c in X_train.columns]
    X_test.columns = [normalize_name(c) for c in X_test.columns]

    # Remove columns that are entirely empty in training.
    empty_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    if empty_cols:
        X_train = X_train.drop(columns=empty_cols)
        X_test = X_test.drop(columns=[c for c in empty_cols if c in X_test.columns])

    # Dataset is expected to contain numeric predictor columns. Coerce values safely.
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Drop columns with no numeric values after conversion.
    non_numeric_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    if non_numeric_cols:
        X_train = X_train.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=[c for c in non_numeric_cols if c in X_test.columns])

    if X_train.shape[1] == 0:
        raise ValueError("No usable numeric feature columns found. Inspect the Excel headers and cell values.")

    y_train = train_df[target_original].astype(str)
    y_test = test_df[target_original].astype(str)

    # A classifier cannot learn a class absent from the training period.
    unseen_test_labels = sorted(set(y_test) - set(y_train))
    if unseen_test_labels:
        print(
            "\nWARNING: These class(es) occur in the latest-month test set but not in earlier training data:",
            unseen_test_labels,
        )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
    }

    labels = sorted(set(y_train) | set(y_test))
    results = []
    best_name = None
    best_model = None
    best_f1 = -1.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
        results.append({
            "model": name,
            "accuracy": accuracy,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1,
        })
        print(f"{name}: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    results_df = pd.DataFrame(results).sort_values("weighted_f1", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_predictions = best_model.predict(X_test)
    report = classification_report(y_test, best_predictions, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    joblib.dump(
        {"model": best_model, "feature_columns": list(X_train.columns), "target_column": target_original,
         "date_column": date_col, "model_name": best_name},
        MODEL_DIR / "best_load_type_model.joblib",
    )

    summary = {
        "best_model": best_name,
        "test_month": str(latest_month),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_columns": list(X_train.columns),
        "metrics": results_df.iloc[0].to_dict(),
        "class_labels": labels,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nBEST MODEL:", best_name)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, best_predictions, zero_division=0))
    print("\nSaved outputs to:", OUTPUT_DIR)
    print("- model_comparison.csv")
    print("- classification_report.csv")
    print("- confusion_matrix.png")
    print("- summary.json")
    print("- model/best_load_type_model.joblib")


if __name__ == "__main__":
    main()
