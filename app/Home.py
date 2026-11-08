"""Home page of the Streamlit app. Run with:  streamlit run app/Home.py"""

import streamlit as st

from utils import load_data, load_prepared, load_result

st.set_page_config(page_title="Texas Wage Predictor", page_icon="💼", layout="wide")

st.title("💼 Predicting Annual Wages of Full-Time Workers in Texas")
st.caption("Machine Learning and Data Analytics (6COSC017C) — Coursework 1 — Student ID 00017665")

st.markdown("""
This app presents an end-to-end machine learning project that predicts the **annual wage** of a full-time
employee in Texas from their education, occupation, industry and job characteristics.

**Why it matters:** wage estimates support **salary benchmarking** (what is a fair market salary for a profile),
**pay-equity audits** and **career and education decisions**.
""")

# Key numbers
data = load_data()
prepared = load_prepared()
test_results = load_result("test_results")
best = test_results.index[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("People in dataset", f"{len(data):,}")
col2.metric("After cleaning", f"{len(prepared):,}")
col3.metric("Best model", best)
col4.metric("Test R²", f"{test_results.loc[best, 'R²']:.3f}")

st.subheader("How to use this app")
st.markdown("""
Use the pages in the sidebar, in order:

1. **Data Exploration** — interactive dashboard of wages by education, occupation, sex and more.
2. **Preprocessing** — how the data was cleaned and prepared, with an interactive error threshold.
3. **Model Training** — train a model yourself with chosen hyperparameters and see its test score.
4. **Predict Wage** — enter a person's profile and get a wage estimate from the final model.
5. **Model Evaluation** — test-set results, overfitting check, feature importance and fairness.
""")

st.subheader("Data source")
st.markdown("""
American Community Survey (ACS) **2024 1-Year Public Use Microdata Sample (PUMS)**, Texas person file —
U.S. Census Bureau, public domain.
[Download page](https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/) (`csv_ptx.zip`).
The dataset is filtered to civilian employees aged 18–64 working 35+ hours per week with a positive wage.
""")
