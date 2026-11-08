"""Page 2: the data preparation steps and their effect (notebook Section 3)."""

import pandas as pd
import streamlit as st

from utils import CAT_FEATURES, MIN_HOURLY_WAGE, NUM_FEATURES, load_data, prepare, train_test_data

st.set_page_config(page_title="Preprocessing", page_icon="🧹", layout="wide")
st.title("🧹 Preprocessing")
st.markdown("Each step below is justified by a finding from the exploratory analysis.")

raw = load_data()

# --- Step 1: error removal (interactive) --------------------------------------
st.header("1. Correcting errors")
st.markdown("""
Wages are converted to 2024 dollars with the Census adjustment factor `ADJINC`. The **implied hourly wage**
= annual wage / (hours per week × weeks worked) is then checked: a full-time employee cannot legally earn less than the
**federal minimum wage of \\$7.25/hour**, so lower values are treated as reporting errors and removed.
Move the slider to see how the threshold changes the data.
""")
threshold = st.slider("Minimum hourly wage ($)", 0.0, 15.0, MIN_HOURLY_WAGE, 0.25)
cleaned = prepare(raw, min_hourly_wage=threshold)

col1, col2, col3 = st.columns(3)
col1.metric("Rows before", f"{len(raw):,}")
col2.metric("Rows removed", f"{len(raw) - len(cleaned):,}", f"{(len(raw) - len(cleaned)) / len(raw):.1%}",
            delta_color="off")
col3.metric("Skewness of log(wage)", f"{cleaned['log_wage'].skew():.2f}")
st.caption("With the chosen threshold of $7.25, 3,574 rows (3.9%) are removed and log(wage) becomes almost symmetric.")

# --- Step 2: missing values ----------------------------------------------------
st.header("2. Missing values")
st.markdown("All missing values are **structural** — the blank has a meaning — so they are filled with that meaning.")
missing = pd.DataFrame({
    "Column": ["ENG", "JWMNP", "JWTRNS"],
    "Missing before": [raw["ENG"].isna().mean(), raw["JWMNP"].isna().mean(), raw["JWTRNS"].isna().mean()],
    "Blank means": ["speaks only English", "works from home / not at work", "not at work last week"],
    "Filled with": ["0 (best English on the 0-4 scale)", "0 minutes", "used only for the work_from_home flag"],
})
st.dataframe(missing.style.format({"Missing before": "{:.1%}"}), hide_index=True)
st.markdown("`FOD1P` (field of degree, 57% blank for people without a degree) is dropped — it overlaps with education and occupation.")

# --- Step 3: feature engineering -------------------------------------------------
st.header("3. Feature engineering and selection")
features = pd.DataFrame({
    "Feature": ["occupation", "industry", "employer", "marital", "age_sq", "work_from_home"],
    "Created from": ["~520 OCCP codes → 25 groups", "~250 INDP codes → 17 sectors", "COW codes",
                     "MAR codes", "AGEP²", "JWTRNS = 11"],
    "Why": ["too many rare codes", "too many rare codes", "readable categories", "readable categories",
            "wages rise with age, then flatten", "remote workers earn more on average"],
})
st.dataframe(features, hide_index=True)
st.markdown(f"""
**Model features** — numerical: `{', '.join(NUM_FEATURES)}`; categorical: `{', '.join(CAT_FEATURES)}`.

**Deliberately excluded:** sex, race/ethnicity, nativity and disability (protected characteristics — a wage
benchmarking model should not set pay based on who someone is; they are used only for the fairness check),
and `hourly_wage` (it is computed from the wage, so it would leak the answer).
""")

# --- Step 4: split and encoding ----------------------------------------------------
st.header("4. Train/test split, encoding and scaling")
train_df, test_df = train_test_data()
col1, col2, col3 = st.columns(3)
col1.metric("Training set", f"{len(train_df):,}")
col2.metric("Test set", f"{len(test_df):,}")
col3.metric("Features after encoding", "58")
st.markdown("""
* **80/20 split**, stratified by wage decile, done **before** any scaling or encoding (no data leakage).
* **StandardScaler** for numerical features (essential for KNN, which uses distances).
* **One-hot encoding** for the 4 categorical features (no natural order).
* Education (`SCHL`) and English ability (`ENG`) stay numeric because they are ordinal.
* Both steps are inside a scikit-learn `Pipeline`, fitted on the training data only.
""")
