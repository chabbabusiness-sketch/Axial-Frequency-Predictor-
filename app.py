from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

# =========================================================
# APP PATHS
# Keep this app.py in the same folder as:
# - Raw_Data_Set.xlsx
# - Steel and Aluminium.png (optional)
# - Axial Deformation.mp4 (optional)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Raw_Data_Set.xlsx"

IMAGE_CANDIDATES = [
    BASE_DIR / "Steel and Aluminium.png",
]

VIDEO_CANDIDATES = [
    BASE_DIR / "Axial Deformation.mp4",
]

# Existing output-image folders from your earlier scripts
PRED_OUTPUT_DIR = BASE_DIR / "axial_frequency_outputs_xlsx"
EQ_OUTPUT_DIR = BASE_DIR / "equation_model_outputs"

PRED_IMAGE_CANDIDATES = [
    PRED_OUTPUT_DIR / "correlation_heatmap.png",
    PRED_OUTPUT_DIR / "cv_rmse_by_fold.png",
    PRED_OUTPUT_DIR / "cv_r2_by_fold.png",
    PRED_OUTPUT_DIR / "permutation_importance_XGBoost.png",
    PRED_OUTPUT_DIR / "parity_plot_XGBoost.png",
]

EQ_IMAGE_CANDIDATES = [
    EQ_OUTPUT_DIR / "parity_plot_Polynomial_deg3.png",
    EQ_OUTPUT_DIR / "gam_partial_effects_equation_models.png",
]

# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(page_title="Axial Frequency Predictor", layout="wide")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .pred-card {
        background: linear-gradient(135deg, #0f5132, #146c43);
        border-radius: 14px;
        padding: 22px 20px;
        min-height: 135px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        margin-bottom: 10px;
    }
    .pred-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #d7ffe4;
        margin-bottom: 16px;
    }
    .pred-value {
        font-size: 2.05rem;
        font-weight: 800;
        color: white;
        line-height: 1.1;
    }
    .small-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 95px;
        margin-bottom: 8px;
    }
    .small-title {
        font-size: 0.90rem;
        font-weight: 600;
        color: #cfd8dc;
        margin-bottom: 8px;
    }
    .small-value {
        font-size: 1.00rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.15;
    }
    .info-card {
        background: rgba(33, 150, 243, 0.16);
        border: 1px solid rgba(33, 150, 243, 0.35);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 88px;
        margin-bottom: 8px;
    }
    .info-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #9fd0ff;
        margin-bottom: 8px;
    }
    .info-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.15;
    }
    .limit-box {
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.35);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 12px;
        color: #ffe082;
        font-size: 0.95rem;
    }
    .section-note {
        font-size: 0.95rem;
        color: #c9d1d9;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DISPLAY LABELS
# =========================================================
DISPLAY_NAME_MAP = {
    "E Fixed": "E (Fixed) [N/m²]",
    "rho Fixed": "ρ (Fixed) [kg/m³]",
    "nu Fixed": "ν (Fixed) [-]",
    "E Free": "E (Free) [N/m²]",
    "rho Free": "ρ (Free) [kg/m³]",
    "nu Free": "ν (Free) [-]",
    "E Fixed (GPa)": "E (Fixed) [GPa]",
    "E Free (GPa)": "E (Free) [GPa]",
    "Axial Frequency (Hz)": "f axial (Hz)",
}

RAW_FEATURE_COLS = ["E Fixed", "rho Fixed", "nu Fixed", "E Free", "rho Free", "nu Free"]
EQ_FEATURE_COLS = ["E Fixed (GPa)", "rho Fixed", "nu Fixed", "E Free (GPa)", "rho Free", "nu Free"]
TARGET_COL = "Axial Frequency (Hz)"


def pretty_name(name: str) -> str:
    """Convert internal column names to nicer labels for display."""
    return DISPLAY_NAME_MAP.get(name, name)


def rename_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all columns for pretty display in Streamlit tables."""
    return df.rename(columns={col: pretty_name(col) for col in df.columns})


def optional_existing_path(candidates):
    """Return the first existing file path from a list, or None."""
    for path in candidates:
        if path.exists():
            return path
    return None


def fmt_sci(x: float) -> str:
    """Format numbers in scientific notation for the validity table."""
    return f"{x:.3e}"


# =========================================================
# DATA CLEANING
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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


@st.cache_data
def load_dataset():
    """Load the raw dataset and cache it."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Raw_Data_Set.xlsx not found in: {BASE_DIR}")

    df = pd.read_excel(DATA_PATH)
    df = normalize_columns(df)
    df = df[RAW_FEATURE_COLS + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df


def make_equation_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert E from N/m² to GPa for the polynomial equation model."""
    eq_df = raw_df.copy()
    eq_df["E Fixed (GPa)"] = eq_df["E Fixed"] / 1e9
    eq_df["E Free (GPa)"] = eq_df["E Free"] / 1e9
    return eq_df[EQ_FEATURE_COLS].copy()


# =========================================================
# EQUATION TEXT FORMATTING
# =========================================================
def greek_equation_name(name: str) -> str:
    name = name.replace("E Fixed (GPa)", "E Fixed")
    name = name.replace("E Free (GPa)", "E Free")
    name = name.replace("rho Fixed", "ρ Fixed")
    name = name.replace("rho Free", "ρ Free")
    name = name.replace("nu Fixed", "ν Fixed")
    name = name.replace("nu Free", "ν Free")
    name = name.replace("_", " ")
    return name


def greek_equation_text(text: str) -> str:
    text = text.replace("f_axial", "f axial")
    text = text.replace("_", " ")
    text = text.replace("rho", "ρ")
    text = text.replace("nu", "ν")
    text = text.replace("  ", " ")
    return text


def build_equation_text(intercept, coefs, names, target_name="f axial"):
    """Create the full polynomial equation text."""
    parts = [f"{target_name} = {intercept:.8g}"]
    for coef, name in zip(coefs, names):
        if abs(coef) < 1e-12:
            continue
        sign = "+" if coef >= 0 else "-"
        coef_abs = abs(coef)
        parts.append(f" {sign} {coef_abs:.8g}*({greek_equation_name(name)})")
    return "".join(parts)


def build_top_terms_text(intercept, coefs, names, target_name="f axial", top_n=15):
    """Show only the largest equation terms for a shorter view."""
    ranked = sorted(zip(coefs, names), key=lambda x: abs(x[0]), reverse=True)
    ranked = [(c, n) for c, n in ranked if abs(c) >= 1e-12][:top_n]

    parts = [f"{target_name} ≈ {intercept:.8g}"]
    for coef, name in ranked:
        sign = "+" if coef >= 0 else "-"
        coef_abs = abs(coef)
        parts.append(f" {sign} {coef_abs:.8g}*({greek_equation_name(name)})")
    return "".join(parts)


# =========================================================
# MODEL TRAINING
# =========================================================
@st.cache_resource
def train_models():
    df = load_dataset()

    # XGBoost predictor
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
    xgb_model.fit(df[RAW_FEATURE_COLS], df[TARGET_COL])

    # Polynomial degree 3 equation model
    eq_df = make_equation_df(df)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(eq_df)

    poly_model = Ridge(alpha=1.0, random_state=42)
    poly_model.fit(X_poly, df[TARGET_COL])

    # 5-fold CV R²
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    xgb_scores = cross_validate(
        XGBRegressor(
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
        ),
        df[RAW_FEATURE_COLS],
        df[TARGET_COL],
        cv=cv,
        scoring={"R2": "r2"},
        return_train_score=False,
        n_jobs=1,
    )

    poly_scores = cross_validate(
        Ridge(alpha=1.0, random_state=42),
        X_poly,
        df[TARGET_COL],
        cv=cv,
        scoring={"R2": "r2"},
        return_train_score=False,
        n_jobs=1,
    )

    feature_names = poly.get_feature_names_out(EQ_FEATURE_COLS)

    equation_full = build_equation_text(
        intercept=poly_model.intercept_,
        coefs=poly_model.coef_,
        names=feature_names,
        target_name="f axial",
    )

    equation_top = build_top_terms_text(
        intercept=poly_model.intercept_,
        coefs=poly_model.coef_,
        names=feature_names,
        target_name="f axial",
        top_n=15,
    )

    contour_df = build_contour_dataframe(df, xgb_model)

    return {
        "xgb_model": xgb_model,
        "poly_transformer": poly,
        "poly_model": poly_model,
        "equation_full": greek_equation_text(equation_full),
        "equation_top": greek_equation_text(equation_top),
        "xgb_r2_cv_mean": float(xgb_scores["test_R2"].mean()),
        "poly_r2_cv_mean": float(poly_scores["test_R2"].mean()),
        "contour_df": contour_df,
    }


# =========================================================
# CARD RENDERING
# =========================================================
def render_pred_card(title, value):
    st.markdown(
        f"""
        <div class="pred-card">
            <div class="pred-title">{title}</div>
            <div class="pred-value">{value:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_small_card(title, value):
    st.markdown(
        f"""
        <div class="small-card">
            <div class="small-title">{title}</div>
            <div class="small-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title, value):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">{title}</div>
            <div class="info-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# VALIDITY REGION TABLE
# =========================================================
def build_range_table(current_df: pd.DataFrame, reference_df: pd.DataFrame):
    rows = []
    current = current_df.iloc[0].to_dict()

    for feat, val in current.items():
        mn = float(reference_df[feat].min())
        mx = float(reference_df[feat].max())
        status = "Inside" if mn <= float(val) <= mx else "Outside"

        rows.append({
            "Feature": pretty_name(feat),
            "Current": fmt_sci(float(val)),
            "Lower boundary": fmt_sci(mn),
            "Upper boundary": fmt_sci(mx),
            "Status": status,
        })

    return pd.DataFrame(rows)


# =========================================================
# INPUT BUILDERS
# =========================================================
def create_input_dfs(E_fixed, rho_fixed, nu_fixed, E_free, rho_free, nu_free):
    raw_df = pd.DataFrame([{
        "E Fixed": E_fixed,
        "rho Fixed": rho_fixed,
        "nu Fixed": nu_fixed,
        "E Free": E_free,
        "rho Free": rho_free,
        "nu Free": nu_free,
    }])

    eq_df = pd.DataFrame([{
        "E Fixed (GPa)": E_fixed / 1e9,
        "rho Fixed": rho_fixed,
        "nu Fixed": nu_fixed,
        "E Free (GPa)": E_free / 1e9,
        "rho Free": rho_free,
        "nu Free": nu_free,
    }])

    return raw_df, eq_df


def create_prediction_comparison_plot(xgb_pred, poly_pred):
    """Create a simple bar chart for the two outputs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["XGBoost", "Polynomial Degree 3 - Equation"]
    values = [xgb_pred, poly_pred]

    ax.bar(names, values)
    ax.set_ylabel("Predicted f axial (Hz)")
    ax.set_title("Model Prediction Comparison")
    fig.tight_layout()
    return fig


# =========================================================
# CONTOUR MAP LOGIC
# x-axis:
#   Material (Free) - Material (Free)
# y-axis:
#   Material (Fixed) - Material (Fixed)
# intersection:
#   current mixed-material XGBoost prediction
# =========================================================
def build_single_material_cases(row_df: pd.DataFrame):
    """
    From one mixed-material input case, create:
    1. Material (Fixed) - Material (Fixed)
    2. Material (Free) - Material (Free)
    """
    fixed_row = pd.DataFrame([{
        "E Fixed": row_df.iloc[0]["E Fixed"],
        "rho Fixed": row_df.iloc[0]["rho Fixed"],
        "nu Fixed": row_df.iloc[0]["nu Fixed"],
        "E Free": row_df.iloc[0]["E Fixed"],
        "rho Free": row_df.iloc[0]["rho Fixed"],
        "nu Free": row_df.iloc[0]["nu Fixed"],
    }])

    free_row = pd.DataFrame([{
        "E Fixed": row_df.iloc[0]["E Free"],
        "rho Fixed": row_df.iloc[0]["rho Free"],
        "nu Fixed": row_df.iloc[0]["nu Free"],
        "E Free": row_df.iloc[0]["E Free"],
        "rho Free": row_df.iloc[0]["rho Free"],
        "nu Free": row_df.iloc[0]["nu Free"],
    }])

    return fixed_row, free_row


def build_contour_dataframe(df: pd.DataFrame, model):
    """
    Convert all training rows into contour points:
    - x = predicted Material (Free)-Material (Free) frequency
    - y = predicted Material (Fixed)-Material (Fixed) frequency
    - z = predicted mixed-material frequency
    """
    rows = []

    for _, row in df.iterrows():
        row_df = pd.DataFrame([row[RAW_FEATURE_COLS].to_dict()])
        fixed_row, free_row = build_single_material_cases(row_df)

        y_axis = float(model.predict(fixed_row[RAW_FEATURE_COLS])[0])
        x_axis = float(model.predict(free_row[RAW_FEATURE_COLS])[0])
        z_val = float(model.predict(row_df[RAW_FEATURE_COLS])[0])

        rows.append({
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_value": z_val,
        })

    contour_df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    contour_df = contour_df.drop_duplicates(subset=["x_axis", "y_axis"]).reset_index(drop=True)
    return contour_df


def create_user_contour_coordinates(model, raw_input_df):
    """Compute the current user-point for the contour plot."""
    fixed_row, free_row = build_single_material_cases(raw_input_df)

    y_axis = float(model.predict(fixed_row[RAW_FEATURE_COLS])[0])
    x_axis = float(model.predict(free_row[RAW_FEATURE_COLS])[0])
    z_actual = float(model.predict(raw_input_df[RAW_FEATURE_COLS])[0])

    return x_axis, y_axis, z_actual


def create_contour_plot(contour_df: pd.DataFrame, x_user: float, y_user: float, z_user: float):
    """
    Create contour map with:
    - background contour colors
    - Material (Free)-Material (Free) vertical line
    - Material (Fixed)-Material (Fixed) horizontal line
    - separate labels that do not overlap
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Build triangulated contour map
    triang = mtri.Triangulation(
        contour_df["x_axis"].to_numpy(),
        contour_df["y_axis"].to_numpy()
    )

    contour = ax.tricontourf(
        triang,
        contour_df["z_value"].to_numpy(),
        levels=20
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r"Predicted $f_{\mathrm{axial}}$ (Hz)")

    # Keep user point visible
    x_min = min(contour_df["x_axis"].min(), x_user)
    x_max = max(contour_df["x_axis"].max(), x_user)
    y_min = min(contour_df["y_axis"].min(), y_user)
    y_max = max(contour_df["y_axis"].max(), y_user)

    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

    x_lo = x_min - x_pad
    x_hi = x_max + x_pad
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Reference lines
    ax.axvline(
        x_user,
        linestyle="--",
        linewidth=1.8,
        label="Material (Free) - Material (Free)"
    )
    ax.axhline(
        y_user,
        linestyle="--",
        linewidth=1.8,
        label="Material (Fixed) - Material (Fixed)"
    )

    # Side marker for Material (Free) - Material (Free)
    ax.scatter(
        [x_user], [y_lo],
        s=85,
        marker="s",
        zorder=6,
        label="Material (Free) - Material (Free) value"
    )
    ax.annotate(
        f"Material (Free) - Material (Free)\n{x_user:.2f} Hz",
        (x_user, y_lo),
        textcoords="offset points",
        xytext=(15, 20),
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
    )

    # Side marker for Material (Fixed) - Material (Fixed)
    ax.scatter(
        [x_lo], [y_user],
        s=85,
        marker="^",
        zorder=6,
        label="Material (Fixed) - Material (Fixed) value"
    )
    ax.annotate(
        f"Material (Fixed) - Material (Fixed)\n{y_user:.2f} Hz",
        (x_lo, y_user),
        textcoords="offset points",
        xytext=(18, 18),
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
    )

    # Intersection marker = mixed-material case
    ax.scatter(
        [x_user], [y_user],
        s=140,
        facecolors="none",
        edgecolors="white",
        linewidths=2.5,
        zorder=7,
        label="Mixed material intersection"
    )
    ax.scatter(
        [x_user], [y_user],
        s=30,
        zorder=8,
    )
    ax.annotate(
        f"Mixed material case\n{z_user:.2f} Hz",
        (x_user, y_user),
        textcoords="offset points",
        xytext=(35, 45),
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
    )

    ax.set_xlabel(r"Material (Free) - Material (Free) predicted $f_{\mathrm{axial}}$ (Hz)")
    ax.set_ylabel(r"Material (Fixed) - Material (Fixed) predicted $f_{\mathrm{axial}}$ (Hz)")
    ax.set_title(r"Contour map of predicted $f_{\mathrm{axial}}$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


# =========================================================
# OPTIONAL MEDIA
# =========================================================
def show_output_image():
    img_path = optional_existing_path(IMAGE_CANDIDATES)
    if img_path is not None:
        st.image(str(img_path), use_container_width=True)


def show_output_video():
    video_path = optional_existing_path(VIDEO_CANDIDATES)
    if video_path is not None:
        st.subheader("Axial Deformation")
        st.video(video_path.read_bytes())


def show_existing_output_images():
    """Show already generated PNG outputs without captions."""
    pred_existing = [p for p in PRED_IMAGE_CANDIDATES if p.exists()]
    eq_existing = [p for p in EQ_IMAGE_CANDIDATES if p.exists()]

    if pred_existing:
        with st.expander("Prediction output images"):
            for p in pred_existing:
                st.image(str(p), use_container_width=True)

    if eq_existing:
        with st.expander("Equation output images"):
            for p in eq_existing:
                st.image(str(p), use_container_width=True)


# =========================================================
# MAIN GUI
# =========================================================
st.title("Axial Frequency Predictor")

# Show image at top
show_output_image()

# Sidebar inputs
with st.sidebar:
    st.header("Input Material Properties")
    st.caption("Use SI units: E in N/m², ρ in kg/m³, and ν is dimensionless [-].")

    st.subheader("Fixed Material")
    E_fixed = st.number_input("E (Fixed) [N/m²]", min_value=0.0, value=1.97e11, step=1e8, format="%.6e")
    rho_fixed = st.number_input("ρ (Fixed) [kg/m³]", min_value=0.0, value=7750.3, format="%.6f")
    nu_fixed = st.number_input("ν (Fixed) [-]", min_value=0.0, max_value=0.4999, value=0.29, format="%.6f")

    st.subheader("Free Material")
    E_free = st.number_input("E (Free) [N/m²]", min_value=0.0, value=4.24e8, step=1e8, format="%.6e")
    rho_free = st.number_input("ρ (Free) [kg/m³]", min_value=0.0, value=2200.5, format="%.6f")
    nu_free = st.number_input("ν (Free) [-]", min_value=0.0, max_value=0.4999, value=0.45, format="%.6f")

    predict_btn = st.button("Predict", use_container_width=True)

if predict_btn:
    try:
        if E_fixed <= 0 or rho_fixed <= 0 or E_free <= 0 or rho_free <= 0:
            raise ValueError("E and ρ values must be positive.")

        artifacts = train_models()
        raw_data = load_dataset()

        raw_input_df, eq_input_df = create_input_dfs(
            E_fixed, rho_fixed, nu_fixed, E_free, rho_free, nu_free
        )

        raw_bounds = build_range_table(raw_input_df, raw_data[RAW_FEATURE_COLS])

        xgb_pred = float(artifacts["xgb_model"].predict(raw_input_df[RAW_FEATURE_COLS])[0])

        X_eq = artifacts["poly_transformer"].transform(eq_input_df[EQ_FEATURE_COLS])
        poly_pred = float(artifacts["poly_model"].predict(X_eq)[0])

        x_user, y_user, z_user = create_user_contour_coordinates(
            artifacts["xgb_model"], raw_input_df
        )

        outside_count = int((raw_bounds["Status"] == "Outside").sum())
        if outside_count > 0:
            st.markdown(
                """
                <div class="limit-box">
                Warning: Data is out of the training validity region and needs deeper extrapolation to come across a more accurate value.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Predicted outputs
        st.subheader("Predicted Frequency (Hz)")
        c1, c2 = st.columns(2)
        with c1:
            render_pred_card("XGBoost Predictor", xgb_pred)
        with c2:
            render_pred_card("Polynomial Degree 3", poly_pred)

        # Extra info cards
        diff = abs(xgb_pred - poly_pred)
        c3, c4, c5 = st.columns(3)
        with c3:
            render_info_card("Absolute Difference", f"{diff:.2f} Hz")
        with c4:
            render_info_card("Predictor R²", f"{artifacts['xgb_r2_cv_mean']:.4f}")
        with c5:
            render_info_card("Equation R²", f"{artifacts['poly_r2_cv_mean']:.4f}")

        # Input summary
        st.subheader("Current Input Summary")
        s1, s2, s3 = st.columns(3)
        with s1:
            render_small_card("E (Fixed)", f"{E_fixed:.6e} N/m²")
        with s2:
            render_small_card("ρ (Fixed)", f"{rho_fixed:.6f} kg/m³")
        with s3:
            render_small_card("ν (Fixed)", f"{nu_fixed:.6f}")

        s4, s5, s6 = st.columns(3)
        with s4:
            render_small_card("E (Free)", f"{E_free:.6e} N/m²")
        with s5:
            render_small_card("ρ (Free)", f"{rho_free:.6f} kg/m³")
        with s6:
            render_small_card("ν (Free)", f"{nu_free:.6f}")

        show_output_video()

        # Prediction comparison bar chart
        st.subheader("Prediction Comparison")
        fig = create_prediction_comparison_plot(xgb_pred, poly_pred)
        st.pyplot(fig, use_container_width=False)

        # Contour map
        st.subheader("Contour Map")
        st.markdown(
            """
            <div class="section-note">
            x-axis = Material (Free) - Material (Free) predicted frequency from the current free-material inputs.<br>
            y-axis = Material (Fixed) - Material (Fixed) predicted frequency from the current fixed-material inputs.<br>
            The marked intersection is your current mixed-material XGBoost prediction, and the color map shows predicted f axial values.
            </div>
            """,
            unsafe_allow_html=True,
        )

        contour_fig = create_contour_plot(
            artifacts["contour_df"],
            x_user=x_user,
            y_user=y_user,
            z_user=z_user,
        )
        st.pyplot(contour_fig, use_container_width=True)

        c6, c7, c8 = st.columns(3)
        with c6:
            render_info_card("Material (Free) - Material (Free)", f"{x_user:.2f} Hz")
        with c7:
            render_info_card("Material (Fixed) - Material (Fixed)", f"{y_user:.2f} Hz")
        with c8:
            render_info_card("Intersection frequency", f"{z_user:.2f} Hz")

        # Model input tables
        with st.expander("Model inputs used"):
            st.write("XGBoost input:")
            st.dataframe(
                rename_display_columns(raw_input_df[RAW_FEATURE_COLS]),
                use_container_width=True,
                hide_index=True,
            )

            st.write("Polynomial Degree 3 input:")
            st.dataframe(
                rename_display_columns(eq_input_df[EQ_FEATURE_COLS]),
                use_container_width=True,
                hide_index=True,
            )

        # Equation section
        st.subheader("Polynomial Degree 3 Equation")
        st.markdown(
            """
            <div class="section-note">
            The equation model uses the raw variables only, with E converted to GPa internally
            to keep the polynomial coefficients more readable.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("**Compact view (largest terms):**")
        st.code(artifacts["equation_top"], language="text")

        with st.expander("Show full polynomial Degree 3 equation"):
            st.code(artifacts["equation_full"], language="text")

        # Validity region table
        st.subheader("Training / Validity Region")
        st.dataframe(raw_bounds, use_container_width=True, hide_index=True)

        # Existing saved images
        show_existing_output_images()

    except Exception as e:
        st.error(f"Error: {e}")
