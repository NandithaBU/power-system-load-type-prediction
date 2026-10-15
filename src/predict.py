"""Predict load type for rows in a new Excel file."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs" / "model" / "best_load_type_model.joblib"


def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to an Excel file containing new rows")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "predictions.xlsx"))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    df = pd.read_excel(args.input)
    normalized = {normalize_name(c): c for c in df.columns}
    date_col = next((normalized[k] for k in ("date", "datetime", "timestamp") if k in normalized), None)
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df["date_year"] = dates.dt.year
        df["date_month"] = dates.dt.month
        df["date_day"] = dates.dt.day
        df["date_dayofweek"] = dates.dt.dayofweek
        df = df.drop(columns=[date_col])

    X = df.drop(columns=[artifact["target_column"]], errors="ignore").copy()
    X.columns = [normalize_name(c) for c in X.columns]
    for col in artifact["feature_columns"]:
        if col not in X.columns:
            X[col] = pd.NA
    X = X[artifact["feature_columns"]]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    df["Predicted_Load_Type"] = artifact["model"].predict(X)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
