from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ec_biochar_adsorption_raw.csv"
TARGET_COL = "Capacity"


def load_raw_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the real emerging-contaminants adsorption-on-biochar dataset."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_adsorption_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove invalid targets and obvious nonphysical records."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    # Convert numeric columns that may be read as object
    for col in out.columns:
        if col not in ["Adsorbent", "Pollutant", "Wastewater type", "Adsorption type"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=[TARGET_COL])
    # Adsorption capacity can be negative in raw compiled literature data due formula/low removal artifacts.
    # For a practical selector, keep physically meaningful positive capacities.
    out = out[out[TARGET_COL] > 0].reset_index(drop=True)
    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in ["Adsorbent", "Pollutant", "Wastewater type", "Adsorption type"] if c in df.columns]
    numeric = [c for c in df.columns if c not in categorical + [TARGET_COL, "Final concentration"]]
    return numeric, categorical


if __name__ == "__main__":
    data = clean_adsorption_data(load_raw_data())
    print(data.shape)
    print(data.head())
