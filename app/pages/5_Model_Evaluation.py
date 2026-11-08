"""Page 5: model comparison and evaluation results (notebook Sections 4.7 and 5)."""

import matplotlib.pyplot as plt
import streamlit as st

from utils import BLUE, ORANGE, load_result

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")
st.title("📈 Model Evaluation")
st.markdown("Results from the notebook: all models were tuned with 5-fold cross-validation and evaluated once on the test set.")

cv = load_result("cv_results")
test = load_result("test_results")
fit = load_result("fit_check")
importance = load_result("importance")
best = test.index[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final model", best)
col2.metric("Test R²", f"{test.loc[best, 'R²']:.3f}")
col3.metric("Mean absolute error", f"${test.loc[best, 'MAE ($)']:,.0f}")
col4.metric("Typical error", f"{test.loc[best, 'Median % error']:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Model comparison", "Overfitting check", "Feature importance", "Fairness"])

with tab1:
    st.subheader("Test set results")
    st.dataframe(test.style.format({"MAE (log)": "{:.4f}", "RMSE (log)": "{:.4f}", "R²": "{:.3f}",
                                    "MAE ($)": "${:,.0f}", "Median % error": "{:.1f}%"}))
    order = test.sort_values("R²")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(order.index, order["R²"], color=[ORANGE if name == best else BLUE for name in order.index], height=0.6)
    ax.set_xlabel("Test R² (higher is better)")
    ax.set_xlim(0, 0.7)
    fig.tight_layout()
    st.pyplot(fig)
    st.subheader("Cross-validation results (used to choose the final model)")
    st.dataframe(cv.style.format("{:.4f}"))
    st.caption("MAE, RMSE and R² are computed on log(wage); MAE ($) and the median % error convert predictions back to dollars.")

with tab2:
    st.dataframe(fit.loc[test.index].style.format("{:.3f}"))
    st.markdown("""
A large gap between train and test R² means the model memorised the training data (**overfitting**).
Linear Regression and XGBoost have small gaps; KNN and the fully grown Random Forest fit the training data almost
perfectly but still generalise reasonably well.
""")

with tab3:
    imp = importance["importance"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    ax.set_xlabel("Increase in MAE (log-wage) when the feature is shuffled")
    ax.set_title(f"Permutation importance ({best})")
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("Education, occupation, weeks worked and hours drive wages the most; commute time matters least.")

with tab4:
    st.markdown("""
Sex and race/ethnicity are **not** model features. The tables show whether the errors still differ between groups.
**Median error (%)** > 0 means the model typically over-predicts that group's wages.
""")
    fmt = {"people": "{:,}", "median actual ($)": "${:,.0f}", "median predicted ($)": "${:,.0f}",
           "median error (%)": "{:+.1f}%", "median % error": "{:.1f}%"}
    st.subheader("By sex")
    st.dataframe(load_result("fairness_sex").style.format(fmt))
    st.subheader("By race / ethnicity")
    st.dataframe(load_result("fairness_race").style.format(fmt))
    st.markdown("""
Women's wages are over-predicted and men's under-predicted: part of the real pay gap is **not explained** by job,
education and hours. For salary benchmarking this is the desired behaviour (pay based on the job, not on sex), and the
remaining gap is an estimate of the **unexplained pay gap** that a pay-equity audit would investigate.
""")
