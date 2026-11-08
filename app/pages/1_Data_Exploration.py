"""Page 1: interactive dashboard for exploring the dataset (notebook Section 2)."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from utils import BLUE, add_labels, codes, load_data

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")
st.title("📊 Data Exploration")
st.markdown("Filter the 92,513 full-time employees in the sidebar — every number and chart on this page updates.")

df = add_labels(load_data())

# --- Sidebar filters --------------------------------------------------------
st.sidebar.header("Filters")
sexes = st.sidebar.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])
education = st.sidebar.multiselect("Education", codes.EDU_LEVELS, default=codes.EDU_LEVELS)
occupations = sorted(df["occupation"].dropna().unique())
chosen_occ = st.sidebar.multiselect("Occupation group", occupations, default=occupations)
age_range = st.sidebar.slider("Age", 18, 64, (18, 64))

view = df[
    df["sex"].isin(sexes)
    & df["education"].isin(education)
    & df["occupation"].isin(chosen_occ)
    & df["AGEP"].between(age_range[0], age_range[1])
]

if len(view) == 0:
    st.warning("No people match these filters — please widen the selection.")
    st.stop()

# --- Key figures ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("People", f"{len(view):,}")
col2.metric("Median wage", f"${view['WAGP'].median():,.0f}")
col3.metric("Mean wage", f"${view['WAGP'].mean():,.0f}")
col4.metric("Median age", f"{view['AGEP'].median():.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["Wage distribution", "Wages by group", "Correlations", "Data table"])

# --- Tab 1: distribution of the target ---------------------------------------
with tab1:
    log_scale = st.checkbox("Show log(wage) instead of wage", value=False)
    values = np.log(view["WAGP"]) if log_scale else view["WAGP"] / 1000
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(values, bins=60, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("log(wage)" if log_scale else "Annual wage, thousand USD")
    ax.set_ylabel("Number of people")
    ax.set_title(f"Skewness: {values.skew():.2f}")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Raw wages are strongly right-skewed; the log transformation makes the distribution almost symmetric. "
               "This is why the models predict log(wage).")

# --- Tab 2: median wage by a chosen group ------------------------------------
with tab2:
    group_options = {"Education": "education", "Occupation group": "occupation", "Industry": "industry",
                     "Employer type": "employer", "Sex": "sex", "Race / ethnicity": "race_eth",
                     "Marital status": "marital"}
    group_label = st.selectbox("Compare median wage by", list(group_options))
    group_col = group_options[group_label]

    summary = (view.groupby(group_col, observed=True)["WAGP"]
                   .agg(["count", "median"])
                   .rename(columns={"count": "people", "median": "median wage"}))
    summary = summary[summary["people"] >= 30].sort_values("median wage")   # hide groups too small to trust

    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(summary))))
    ax.barh(summary.index.astype(str), summary["median wage"] / 1000, color=BLUE, height=0.6)
    ax.set_xlabel("Median annual wage, thousand USD")
    fig.tight_layout()
    st.pyplot(fig)
    st.dataframe(summary.sort_values("median wage", ascending=False).style.format(
        {"people": "{:,}", "median wage": "${:,.0f}"}))

# --- Tab 3: correlation matrix ----------------------------------------------
with tab3:
    method = st.radio("Correlation method", ["spearman", "pearson"], horizontal=True)
    corr_view = view.assign(log_wage=np.log(view["WAGP"]))
    corr = corr_view[["log_wage", "SCHL", "AGEP", "WKHP", "WKWN", "JWMNP"]].corr(method=method)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("SCHL = education level, AGEP = age, WKHP = hours per week, WKWN = weeks worked, JWMNP = commute minutes. "
               "Spearman is recommended for the ordinal education variable.")

# --- Tab 4: data and summary statistics -------------------------------------
with tab4:
    st.subheader("Summary statistics")
    st.dataframe(view[["WAGP", "AGEP", "WKHP", "WKWN", "JWMNP"]].describe().T.style.format("{:,.1f}"))
    st.subheader("First 100 rows")
    st.dataframe(view[["AGEP", "sex", "education", "occupation", "industry", "employer", "WKHP", "WKWN", "WAGP"]].head(100))
