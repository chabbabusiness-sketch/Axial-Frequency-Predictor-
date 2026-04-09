from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

# =========================================================
# SAVE TRAINED MODELS AS PKL FILES
# Put this script in the same folder as Raw_Data_Set.xlsx
# Then run it once.
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Raw_Data_Set.xlsx"

# Output file names expected by your Streamlit app
XGB_MODEL_PATH = BASE_DIR / "xgb_raw_model.pkl"
POLY_MODEL_PATH = BASE_DIR / "poly_deg3_model.pkl"
POLY_TRANSFORMER_PATH = BASE_DIR / "poly_deg3_transformer.pkl"

RAW_FEATURE_COLS = ["E Fixed", "rho Fixed", "nu Fixed", "E Free", "rho Free", "nu Free"]
EQ_FEATURE_COLS = ["E Fixed (GPa)", "rho Fixed", "nu Fixed", "E Free (GPa)", "rho Free", "nu Free"]
TARGET_COL = "Axial Frequency (Hz)"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles Greek/English variants in the Excel headers.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    alias_map = {
        "E Fixed": ["E Fixed", "EFixed"],
        "rho Fixed": ["ρ Fixed", "rho Fixed", "? Fixed", "Density Fixed"],
        "nu Fixed": ["ν Fixed", "νFixed", "nu Fixed", "nuFixed", "?Fixed", "Poisson Fixed"],
        "E Free": ["E Free", "EFree"],
        "rho Free": ["ρ Free", "rho Free", "? Free", "Density Free"],
        "nu Free": ["ν Free", "νFree", "nu Free", "nuFree", "? Free.1", "Poisson Free"],
        "Axial Frequency (Hz)": ["Axial Frequency (Hz)", "Axial Frequency", "AxialFrequencyHz"],
    }

    rename_dict = {}
    for canonical, aliases in alias_map.items():
        found = None

        for c in df.columns:
            if c in aliases:
                found = c
                break

        if found is None:
            for c in df.columns:
                if canonical.lower().replace(" ", "") == c.lower().replace(" ", ""):
                    found = c
                    break

        if found is None:
            raise ValueError(
                f"Could not find column for: {canonical}\nAvailable columns: {list(df.columns)}"
            )

        rename_dict[found] = canonical

    return df.rename(columns=rename_dict)


def make_equation_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert E to GPa for the polynomial degree 3 model.
    """
    eq_df = raw_df.copy()
    eq_df["E Fixed (GPa)"] = eq_df["E Fixed"] / 1e9
    eq_df["E Free (GPa)"] = eq_df["E Free"] / 1e9
    return eq_df[EQ_FEATURE_COLS].copy()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Raw_Data_Set.xlsx not found in: {BASE_DIR}")

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    df = normalize_columns(df)
    df = df[RAW_FEATURE_COLS + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    X_raw = df[RAW_FEATURE_COLS]
    y = df[TARGET_COL]

    # =====================================================
    # Train XGBoost predictor
    # =====================================================
    print("Training XGBoost model...")
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_raw, y)

    # =====================================================
    # Train Polynomial Degree 3 equation model
    # =====================================================
    print("Training Polynomial Degree 3 model...")
    X_eq = make_equation_df(df)
    poly_transformer = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_transformer.fit_transform(X_eq)

    poly_model = Ridge(alpha=1.0, random_state=42)
    poly_model.fit(X_poly, y)

    # =====================================================
    # Save PKL files
    # =====================================================
    print("Saving PKL files...")
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    joblib.dump(poly_model, POLY_MODEL_PATH)
    joblib.dump(poly_transformer, POLY_TRANSFORMER_PATH)

    print("\nSaved:")
    print(f" - {XGB_MODEL_PATH.name}")
    print(f" - {POLY_MODEL_PATH.name}")
    print(f" - {POLY_TRANSFORMER_PATH.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
