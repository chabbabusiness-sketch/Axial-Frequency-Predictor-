import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import PolynomialFeatures

try:
    from sklearn.preprocessing import SplineTransformer
    HAS_SPLINE = True
except Exception:
    HAS_SPLINE = False

try:
    from pygam import LinearGAM, s
    HAS_PYGAM = True
except Exception:
    HAS_PYGAM = False

try:
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except Exception:
    HAS_GPLEARN = False

DATA_PATH = Path(r"D:\Axial Frequency Predictor\Raw_Data_Set.xlsx")
SHEET_NAME = 0
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "equation_model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_LABELS = {
    "E_fixed_GPa": r"$E_{\mathrm{fixed}}$ (GPa)",
    "rho_fixed": r"$\rho_{\mathrm{fixed}}$",
    "nu_fixed": r"$\nu_{\mathrm{fixed}}$",
    "E_free_GPa": r"$E_{\mathrm{free}}$ (GPa)",
    "rho_free": r"$\rho_{\mathrm{free}}$",
    "nu_free": r"$\nu_{\mathrm{free}}$",
}

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

RMSE_SCORER = make_scorer(rmse, greater_is_better=False)

def simplify_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("ρ", "rho")
    s = s.replace("ν", "nu")
    for ch in [" ", "_", "-", "(", ")", "[", "]", "{", "}", ".", "/", "\\"]:
        s = s.replace(ch, "")
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    canonical = {
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
    for col in df.columns:
        simp = simplify_name(col)
        if simp in canonical:
            rename_dict[col] = canonical[simp]
    df = df.rename(columns=rename_dict)
    required = [
        "E Fixed", "rho Fixed", "nu Fixed",
        "E Free", "rho Free", "nu Free",
        "Axial Frequency (Hz)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")
    return df

def convert_to_engineering_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["E Fixed", "E Free"]:
        if out[c].abs().median() > 1e6:
            out[c] = out[c] / 1e9
    return out

def metrics_dict(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }

def format_number(x, digits=6):
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"

class GAMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, lam=0.6):
        self.lam = lam
        self.model_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        terms = s(0) + s(1) + s(2) + s(3) + s(4) + s(5)
        self.model_ = LinearGAM(terms, lam=self.lam).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))

    def equation_text(self, feature_cols):
        return (
            "f axial = β0 + s1(E fixed) + s2(ρ fixed) + s3(ν fixed) + "
            "s4(E free) + s5(ρ free) + s6(ν free)"
        )

class SymbolicWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        self.model_ = SymbolicRegressor(
            population_size=2000,
            generations=30,
            tournament_size=20,
            stopping_criteria=0.001,
            const_range=(-5.0, 5.0),
            init_depth=(2, 5),
            init_method='half and half',
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg'),
            metric='rmse',
            parsimony_coefficient=0.001,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=0,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model_.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))

    def equation_text(self, feature_cols):
        expr = str(self.model_._program)
        for i, feat in enumerate(feature_cols):
            expr = expr.replace(f"X{i}", feat.replace("_", " "))
        expr = expr.replace("rho", "ρ").replace("nu", "ν")
        return "f axial = " + expr

def build_models():
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.001, max_iter=50000, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=50000, random_state=RANDOM_STATE),
        "Polynomial_deg2": PipelinePoly(2),
        "Polynomial_deg3": PipelinePoly(3),
    }
    if HAS_SPLINE:
        models["Piecewise_ElasticNet"] = PiecewisePipeline()
    if HAS_PYGAM:
        models["GAM"] = GAMWrapper(lam=0.6)
    if HAS_GPLEARN:
        models["SymbolicRegression"] = SymbolicWrapper(random_state=RANDOM_STATE)
    return models

class PipelinePoly(BaseEstimator, RegressorMixin):
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = Ridge(alpha=1.0, random_state=RANDOM_STATE)

    def fit(self, X, y):
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self

    def predict(self, X):
        return self.model.predict(self.poly.transform(X))

class PiecewisePipeline(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.spline = SplineTransformer(degree=1, n_knots=4, include_bias=False)
        self.model = ElasticNet(alpha=0.0005, l1_ratio=0.3, max_iter=50000, random_state=RANDOM_STATE)

    def fit(self, X, y):
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
        Xs = self.spline.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X):
        return self.model.predict(self.spline.transform(X))

def equation_from_linear_model(name, model, feature_cols):
    coef = model.coef_.ravel()
    intercept = float(np.ravel(model.intercept_)[0]) if np.ndim(model.intercept_) else float(model.intercept_)
    pieces = [f"f axial = {format_number(intercept)}"]
    for c, feat in zip(coef, feature_cols):
        sign = "+" if c >= 0 else "-"
        feat2 = feat.replace("_", " ").replace("rho", "ρ").replace("nu", "ν")
        pieces.append(f" {sign} {format_number(abs(c))}*{feat2}")
    return "".join(pieces)

def equation_from_polynomial_pipeline(name, pipe, feature_cols):
    names = pipe.poly.get_feature_names_out(feature_cols)
    coef = pipe.model.coef_.ravel()
    intercept = float(np.ravel(pipe.model.intercept_)[0]) if np.ndim(pipe.model.intercept_) else float(pipe.model.intercept_)
    terms = []
    for c, n in zip(coef, names):
        if abs(c) < 1e-12:
            continue
        n2 = n.replace("_", " ").replace("rho", "ρ").replace("nu", "ν")
        sign = "+" if c >= 0 else "-"
        terms.append(f" {sign} {format_number(abs(c))}*{n2}")
    return f"f axial = {format_number(intercept)}" + "".join(terms)

def equation_from_piecewise_pipeline(pipe, feature_cols):
    basis_names = pipe.spline.get_feature_names_out(feature_cols)
    coef = pipe.model.coef_.ravel()
    intercept = float(np.ravel(pipe.model.intercept_)[0]) if np.ndim(pipe.model.intercept_) else float(pipe.model.intercept_)
    eq = [
        "f axial = " + format_number(intercept),
        "\nwhere B_j(x) are piecewise linear spline basis functions:",
    ]
    for c, n in zip(coef, basis_names):
        if abs(c) < 1e-10:
            continue
        sign = "+" if c >= 0 else "-"
        n2 = n.replace("_", " ").replace("rho", "ρ").replace("nu", "ν")
        eq.append(f"\n  {sign} {format_number(abs(c))}*{n2}")
    return "".join(eq)

def extract_equation(name, fitted_model, feature_cols):
    if name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
        return equation_from_linear_model(name, fitted_model, feature_cols)
    if name.startswith("Polynomial"):
        return equation_from_polynomial_pipeline(name, fitted_model, feature_cols)
    if name == "Piecewise_ElasticNet":
        return equation_from_piecewise_pipeline(fitted_model, feature_cols)
    if name == "GAM":
        return fitted_model.equation_text(feature_cols)
    if name == "SymbolicRegression":
        return fitted_model.equation_text(feature_cols)
    return f"Equation export not implemented for {name}"

print(f"Loading data from: {DATA_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
df = normalize_columns(df)
df = convert_to_engineering_units(df)

feature_cols = ["E Fixed", "rho Fixed", "nu Fixed", "E Free", "rho Free", "nu Free"]
target_col = "Axial Frequency (Hz)"

rename_for_equations = {
    "E Fixed": "E_fixed_GPa",
    "rho Fixed": "rho_fixed",
    "nu Fixed": "nu_fixed",
    "E Free": "E_free_GPa",
    "rho Free": "rho_free",
    "nu Free": "nu_free",
}

data = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
X = data[feature_cols].copy()
y = data[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
models = build_models()

cv_rows = []
fold_rows = []
fitted = {}

print("\nRunning equation focused benchmark...")
for name, model in models.items():
    print(f"  {name}")
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
        n_jobs=1,
        return_train_score=False,
    )
    r2s = scores["test_R2"]
    maes = -scores["test_MAE"]
    rmses = -scores["test_RMSE"]
    for i in range(N_SPLITS):
        fold_rows.append({
            "Model": name,
            "Fold": i + 1,
            "R2": r2s[i],
            "MAE": maes[i],
            "RMSE": rmses[i],
        })
    cv_rows.append({
        "Model": name,
        "CV_R2_mean": np.mean(r2s),
        "CV_R2_std": np.std(r2s),
        "CV_MAE_mean": np.mean(maes),
        "CV_MAE_std": np.std(maes),
        "CV_RMSE_mean": np.mean(rmses),
        "CV_RMSE_std": np.std(rmses),
    })
    fitted_model = clone(model).fit(X_train, y_train)
    fitted[name] = fitted_model

cv_summary = pd.DataFrame(cv_rows).sort_values(
    ["CV_R2_mean", "CV_RMSE_mean"], ascending=[False, True]
).reset_index(drop=True)
cv_folds = pd.DataFrame(fold_rows)
cv_summary.to_csv(OUTPUT_DIR / "cv_summary_equation_models.csv", index=False)
cv_folds.to_csv(OUTPUT_DIR / "cv_fold_equation_models.csv", index=False)

print("\nCV summary:")
print(cv_summary.to_string(index=False))

test_rows = []
preds_df = pd.DataFrame({"Actual": y_test.reset_index(drop=True)})
for name, model in fitted.items():
    pred = model.predict(X_test)
    m = metrics_dict(y_test, pred)
    test_rows.append({
        "Model": name,
        "Test_R2": m["R2"],
        "Test_MAE": m["MAE"],
        "Test_RMSE": m["RMSE"],
    })
    preds_df[f"Pred_{name}"] = pd.Series(pred).reset_index(drop=True)

test_results = pd.DataFrame(test_rows).sort_values(
    ["Test_R2", "Test_RMSE"], ascending=[False, True]
).reset_index(drop=True)
test_results.to_csv(OUTPUT_DIR / "test_results_equation_models.csv", index=False)
preds_df.to_csv(OUTPUT_DIR / "test_predictions_equation_models.csv", index=False)

best_name = cv_summary.iloc[0]["Model"]
best_model = fitted[best_name]

feature_cols_eq = [rename_for_equations[c] for c in feature_cols]
X_train_eq = X_train.copy()
X_train_eq.columns = feature_cols_eq
X_test_eq = X_test.copy()
X_test_eq.columns = feature_cols_eq

named_fitted = {}
for name, model in models.items():
    named_fitted[name] = clone(model).fit(X_train_eq, y_train.values)

equations = []
for name in cv_summary["Model"].tolist():
    try:
        eq = extract_equation(name, named_fitted[name], feature_cols_eq)
    except Exception as e:
        eq = f"Could not extract equation for {name}: {e}"
    equations.append((name, eq))

with open(OUTPUT_DIR / "equations.txt", "w", encoding="utf-8") as f:
    f.write("Equation focused models for Axial Frequency\n")
    f.write("Units used in equations: E in GPa, density in original dataset units, ν dimensionless.\n\n")
    for name, eq in equations:
        f.write(f"[{name}]\n{eq}\n\n")

best_pred = named_fitted[best_name].predict(X_test_eq)
y_true = np.asarray(y_test)
y_pred = np.asarray(best_pred)
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
x_line = np.linspace(min_val, max_val, 300)
plt.figure(figsize=(7.5, 7.5))
plt.scatter(y_true, y_pred, alpha=0.75, label="Test data")
plt.plot(x_line, x_line, label="Ideal fit")
plt.plot(x_line, 1.05*x_line, linestyle="--", label="+5%")
plt.plot(x_line, 0.95*x_line, linestyle="--", label="-5%")
plt.plot(x_line, 1.10*x_line, linestyle=":", label="+10%")
plt.plot(x_line, 0.90*x_line, linestyle=":", label="-10%")
plt.xlabel(r"Actual $f_{\mathrm{axial}}$ (Hz)")
plt.ylabel(r"Predicted $f_{\mathrm{axial}}$ (Hz)")
plt.title(f"Parity plot - best equation model ({best_name.replace('_', ' ')})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"parity_plot_{best_name}.png", dpi=300)
plt.close()

if "GAM" in named_fitted:
    gam = named_fitted["GAM"]
    gam_rows = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    for i, feat in enumerate(feature_cols_eq):
        XX = gam.model_.generate_X_grid(term=i)
        pdep, confi = gam.model_.partial_dependence(term=i, X=XX, width=0.95)
        xvals = XX[:, i]
        label = PLOT_LABELS.get(feat, feat.replace("_", " "))
        axes[i].plot(xvals, pdep, label="Partial effect")
        axes[i].plot(xvals, confi[:, 0], linestyle="--", label="95% CI lower")
        axes[i].plot(xvals, confi[:, 1], linestyle="--", label="95% CI upper")
        axes[i].set_title(f"GAM partial effect: {label}")
        axes[i].set_xlabel(label)
        axes[i].set_ylabel(r"Effect on $f_{\mathrm{axial}}$ (Hz)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        gam_rows.append(pd.DataFrame({
            "Feature": feat,
            "X_Value": xvals,
            "Partial_Effect": pdep,
            "CI_Lower": confi[:, 0],
            "CI_Upper": confi[:, 1],
        }))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gam_partial_effects_equation_models.png", dpi=300)
    plt.close()
    pd.concat(gam_rows, ignore_index=True).to_csv(OUTPUT_DIR / "gam_partial_effect_values_equation_models.csv", index=False)

print("\nSaved files:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(" -", p.name)
print("\nDone.")
