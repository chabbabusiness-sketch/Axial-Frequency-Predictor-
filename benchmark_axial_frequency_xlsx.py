import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
)
from sklearn.inspection import permutation_importance

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR

# =========================================================
# USER SETTINGS
# =========================================================
DATA_PATH = Path(r"D:\Machine Learning\Raw_Data_Set.xlsx")
SHEET_NAME = 0   # first sheet
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Save outputs next to this script
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "axial_frequency_outputs_xlsx"
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# OPTIONAL PACKAGES
# =========================================================
HAS_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

HAS_PYGAM = True
try:
    from pygam import LinearGAM, s
except Exception:
    HAS_PYGAM = False

# =========================================================
# PLOT LABELS WITH GREEK SYMBOLS
# =========================================================
PLOT_LABELS = {
    "E Fixed": r"$E_{\mathrm{fixed}}$",
    "rho Fixed": r"$\rho_{\mathrm{fixed}}$",
    "nu Fixed": r"$\nu_{\mathrm{fixed}}$",
    "E Free": r"$E_{\mathrm{free}}$",
    "rho Free": r"$\rho_{\mathrm{free}}$",
    "nu Free": r"$\nu_{\mathrm{free}}$",
    "Axial Frequency (Hz)": r"$f_{\mathrm{axial}}$ (Hz)",
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


RMSE_SCORER = make_scorer(rmse_func, greater_is_better=False)


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse_func(y_true, y_pred),
    }


def simplify_name(name: str) -> str:
    """
    Normalizes Excel column names so both Greek and English variants work.
    Example:
    'ρ Fixed' -> 'rhofixed'
    'ν Fixed' -> 'nufixed'
    'Axial Frequency (Hz)' -> 'axialfrequencyhz'
    """
    s = str(name).strip().lower()

    # replace Greek symbols with words
    s = s.replace("ρ", "rho")
    s = s.replace("ν", "nu")

    # remove common separators
    for ch in [" ", "_", "-", "(", ")", "[", "]", "{", "}", ".", "/", "\\"]:
        s = s.replace(ch, "")

    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames actual Excel columns to canonical names used in the script.
    Works with:
    E Fixed / EFree / ρ Fixed / rho Fixed / ν Fixed / nu Fixed / νFixed / etc.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    canonical_targets = {
        "efixed": "E Fixed",
        "rhofixed": "rho Fixed",
        "nufixed": "nu Fixed",
        "efree": "E Free",
        "rhofree": "rho Free",
        "nufree": "nu Free",
        "axialfrequencyhz": "Axial Frequency (Hz)",
        "axialfrequency": "Axial Frequency (Hz)",
    }

    rename_dict = {}

    for original_col in df.columns:
        simplified = simplify_name(original_col)
        if simplified in canonical_targets:
            rename_dict[original_col] = canonical_targets[simplified]

    df = df.rename(columns=rename_dict)

    required_cols = [
        "E Fixed",
        "rho Fixed",
        "nu Fixed",
        "E Free",
        "rho Free",
        "nu Free",
        "Axial Frequency (Hz)",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def build_preprocessor(feature_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_models(preprocessor):
    models = {
        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "ExtraTrees": Pipeline([
            ("preprocessor", preprocessor),
            ("model", ExtraTreesRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "HistGradientBoosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                random_state=RANDOM_STATE
            ))
        ]),

        "SVR": Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVR(
                kernel="rbf",
                C=50,
                epsilon=0.1,
                gamma="scale"
            ))
        ]),

        "GradientBoosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                n_jobs=-1,
                verbosity=0
            ))
        ])
    else:
        print("\n[INFO] xgboost is not installed, so XGBoost will be skipped.")
        print("Install with: py -m pip install xgboost\n")

    return models


# =========================================================
# LOAD EXCEL DATA
# =========================================================
print(f"\nLoading Excel file from: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Excel file not found: {DATA_PATH}")

df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
df = normalize_columns(df)

feature_cols = ["E Fixed", "rho Fixed", "nu Fixed", "E Free", "rho Free", "nu Free"]
target_col = "Axial Frequency (Hz)"

data = df[feature_cols + [target_col]].copy()
data = data.dropna(subset=[target_col]).reset_index(drop=True)

print("\nColumns used:")
print(feature_cols + [target_col])

print("\nDataset shape:", data.shape)

# =========================================================
# CORRELATION HEATMAP
# =========================================================
corr = data.corr(numeric_only=True)
corr_plot = corr.copy()
corr_plot.index = [PLOT_LABELS.get(c, c) for c in corr_plot.index]
corr_plot.columns = [PLOT_LABELS.get(c, c) for c in corr_plot.columns]

plt.figure(figsize=(10, 8))
sns.heatmap(corr_plot, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv", index=True)

print("[SAVED] correlation_heatmap.png")
print("[SAVED] correlation_matrix.csv")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

if len(X_train) < N_SPLITS:
    raise ValueError(
        f"Training set has only {len(X_train)} rows, but {N_SPLITS}-fold CV was requested."
    )

preprocessor = build_preprocessor(feature_cols)
models = build_models(preprocessor)

# =========================================================
# 5-FOLD CROSS-VALIDATION
# =========================================================
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cv_summary_rows = []
fold_rows = []
fitted_models = {}

for model_name, model in models.items():
    print(f"Running CV for: {model_name}")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "R2": "r2",
            "MAE": "neg_mean_absolute_error",
            "RMSE": RMSE_SCORER,
        },
        return_train_score=False,
        n_jobs=1
    )

    fold_r2 = scores["test_R2"]
    fold_mae = -scores["test_MAE"]
    fold_rmse = -scores["test_RMSE"]

    for i in range(N_SPLITS):
        fold_rows.append({
            "Model": model_name,
            "Fold": i + 1,
            "R2": fold_r2[i],
            "MAE": fold_mae[i],
            "RMSE": fold_rmse[i],
        })

    cv_summary_rows.append({
        "Model": model_name,
        "CV_R2_mean": np.mean(fold_r2),
        "CV_R2_std": np.std(fold_r2),
        "CV_MAE_mean": np.mean(fold_mae),
        "CV_MAE_std": np.std(fold_mae),
        "CV_RMSE_mean": np.mean(fold_rmse),
        "CV_RMSE_std": np.std(fold_rmse),
    })

    model.fit(X_train, y_train)
    fitted_models[model_name] = model

cv_folds_df = pd.DataFrame(fold_rows)
cv_summary_df = pd.DataFrame(cv_summary_rows).sort_values(
    by=["CV_R2_mean", "CV_RMSE_mean"],
    ascending=[False, True]
).reset_index(drop=True)

cv_folds_df.to_csv(OUTPUT_DIR / "cv_fold_results.csv", index=False)
cv_summary_df.to_csv(OUTPUT_DIR / "cv_summary_results.csv", index=False)

print("\n===== CROSS-VALIDATION SUMMARY =====")
print(cv_summary_df.to_string(index=False))

best_model_name = cv_summary_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]

print(f"\nBest model under 5-fold CV: {best_model_name}")

# =========================================================
# CV ERROR CHANGE PLOTS
# =========================================================
plt.figure(figsize=(10, 6))
for model_name in cv_folds_df["Model"].unique():
    subset = cv_folds_df[cv_folds_df["Model"] == model_name]
    plt.plot(subset["Fold"], subset["RMSE"], marker="o", label=model_name)

plt.xlabel("Fold Number")
plt.ylabel("RMSE")
plt.title("5-Fold Cross-Validation: RMSE Across Folds")
plt.legend(title="Model")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cv_rmse_by_fold.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for model_name in cv_folds_df["Model"].unique():
    subset = cv_folds_df[cv_folds_df["Model"] == model_name]
    plt.plot(subset["Fold"], subset["R2"], marker="o", label=model_name)

plt.xlabel("Fold Number")
plt.ylabel(r"$R^2$")
plt.title("5-Fold Cross-Validation: $R^2$ Across Folds")
plt.legend(title="Model")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cv_r2_by_fold.png", dpi=300)
plt.close()

print("[SAVED] cv_fold_results.csv")
print("[SAVED] cv_summary_results.csv")
print("[SAVED] cv_rmse_by_fold.png")
print("[SAVED] cv_r2_by_fold.png")

# =========================================================
# FINAL TEST RESULTS
# =========================================================
test_rows = []
all_test_predictions = pd.DataFrame({"Actual": y_test.reset_index(drop=True)})

for model_name, model in fitted_models.items():
    y_pred = model.predict(X_test)
    m = regression_metrics(y_test, y_pred)

    test_rows.append({
        "Model": model_name,
        "Test_R2": m["R2"],
        "Test_MAE": m["MAE"],
        "Test_RMSE": m["RMSE"],
    })

    all_test_predictions[f"Pred_{model_name}"] = pd.Series(y_pred).reset_index(drop=True)

test_results_df = pd.DataFrame(test_rows).sort_values(
    by=["Test_R2", "Test_RMSE"],
    ascending=[False, True]
).reset_index(drop=True)

test_results_df.to_csv(OUTPUT_DIR / "test_results.csv", index=False)
all_test_predictions.to_csv(OUTPUT_DIR / "test_predictions_all_models.csv", index=False)

print("\n===== FINAL TEST SET RESULTS =====")
print(test_results_df.to_string(index=False))

# =========================================================
# PARITY PLOT WITH 0%, +/-5%, +/-10% ERROR LINES
# =========================================================
best_pred = best_model.predict(X_test)

y_true = np.array(y_test)
y_pred = np.array(best_pred)

min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

x_line = np.linspace(min_val, max_val, 300)
line_ideal = x_line
line_p5 = 1.05 * x_line
line_m5 = 0.95 * x_line
line_p10 = 1.10 * x_line
line_m10 = 0.90 * x_line

plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.75, label="Test Data")
plt.plot(x_line, line_ideal, linestyle="-", label="Ideal Fit (0% Error)")
plt.plot(x_line, line_p5, linestyle="--", label="+5% Error")
plt.plot(x_line, line_m5, linestyle="--", label="-5% Error")
plt.plot(x_line, line_p10, linestyle=":", label="+10% Error")
plt.plot(x_line, line_m10, linestyle=":", label="-10% Error")

plt.xlabel(f"Actual {PLOT_LABELS['Axial Frequency (Hz)']}")
plt.ylabel(f"Predicted {PLOT_LABELS['Axial Frequency (Hz)']}")
plt.title(f"Parity Plot - {best_model_name}")
plt.legend(title="Reference Lines")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"parity_plot_{best_model_name}.png", dpi=300)
plt.close()

print("[SAVED] test_results.csv")
print("[SAVED] test_predictions_all_models.csv")
print(f"[SAVED] parity_plot_{best_model_name}.png")

# =========================================================
# PERMUTATION IMPORTANCE
# =========================================================
perm = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=30,
    random_state=RANDOM_STATE,
    scoring="r2",
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance_Mean": perm.importances_mean,
    "Importance_STD": perm.importances_std
}).sort_values(by="Importance_Mean", ascending=False)

perm_df["Feature_Label"] = perm_df["Feature"].map(PLOT_LABELS)
perm_df.to_csv(OUTPUT_DIR / f"permutation_importance_{best_model_name}.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=perm_df, x="Importance_Mean", y="Feature_Label")
plt.xlabel(r"Permutation Importance (Mean Decrease in $R^2$)")
plt.ylabel("Input Variable")
plt.title(f"Permutation Importance - {best_model_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"permutation_importance_{best_model_name}.png", dpi=300)
plt.close()

print(f"[SAVED] permutation_importance_{best_model_name}.csv")
print(f"[SAVED] permutation_importance_{best_model_name}.png")

# =========================================================
# GAM PARTIAL EFFECT PLOTS
# =========================================================
# GAM is used for interpretation, not for choosing the best predictor
if HAS_PYGAM:
    print("\nFitting GAM for partial effect plots...")

    gam = LinearGAM(
        s(0) + s(1) + s(2) + s(3) + s(4) + s(5)
    ).fit(X_train.values, y_train.values)

    gam_rows = []
    n_features = len(feature_cols)
    ncols = 2
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(feature_cols):
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

        xvals = XX[:, i]
        feat_label = PLOT_LABELS.get(feat, feat)

        axes[i].plot(xvals, pdep, label="Partial Effect")
        axes[i].plot(xvals, confi[:, 0], linestyle="--", label="95% CI Lower")
        axes[i].plot(xvals, confi[:, 1], linestyle="--", label="95% CI Upper")

        axes[i].set_title(f"GAM Partial Effect Plot: {feat_label}", fontsize=12)
        axes[i].set_xlabel(feat_label)
        axes[i].set_ylabel(r"Effect on $f_{\mathrm{axial}}$ (Hz)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        gam_rows.append(pd.DataFrame({
            "Feature": feat,
            "Feature_Label": feat_label,
            "X_Value": xvals,
            "Partial_Effect": pdep,
            "CI_Lower": confi[:, 0],
            "CI_Upper": confi[:, 1],
        }))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gam_partial_effect_plots.png", dpi=300)
    plt.close()

    gam_effects_df = pd.concat(gam_rows, axis=0, ignore_index=True)
    gam_effects_df.to_csv(OUTPUT_DIR / "gam_partial_effect_values.csv", index=False)

    print("[SAVED] gam_partial_effect_plots.png")
    print("[SAVED] gam_partial_effect_values.csv")
else:
    print("\n[INFO] pygam is not installed, so GAM plots were skipped.")
    print("Install with: py -m pip install pygam")

print("\nDone.")
print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")