"""Page 3: train a model interactively and evaluate it on the test set (notebook Section 4)."""

import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from utils import FEATURES, RANDOM_STATE, TARGET, load_result, make_preprocessor, train_test_data

st.set_page_config(page_title="Model Training", page_icon="🏋️", layout="wide")
st.title("🏋️ Model Training")
st.markdown("""
Choose a model and its hyperparameters, train it on a sample of the training data and see how it performs on the
**test set**. The final model in the notebook was tuned with GridSearchCV and 5-fold cross-validation;
its settings are pre-selected below.
""")

train_df, test_df = train_test_data()

# --- Choose the model and hyperparameters -------------------------------------
model_name = st.selectbox("Model", ["XGBoost", "Random Forest", "KNN", "Linear Regression"])

if model_name == "XGBoost":
    learning_rate = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    n_estimators = st.slider("n_estimators (number of trees)", 50, 500, 200, 50)
    max_depth = st.slider("max_depth", 2, 8, 5)
    model = XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                         random_state=RANDOM_STATE, n_jobs=-1)
elif model_name == "Random Forest":
    n_estimators = st.slider("n_estimators (number of trees)", 50, 300, 100, 50)
    max_depth = st.select_slider("max_depth", [5, 10, 20, None], value=20)
    min_samples_leaf = st.slider("min_samples_leaf", 1, 20, 1)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
elif model_name == "KNN":
    n_neighbors = st.slider("n_neighbors (k)", 1, 100, 30)
    weights = st.radio("weights", ["distance", "uniform"], horizontal=True)
    metric = st.radio("metric", ["manhattan", "euclidean"], horizontal=True)
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
else:
    st.info("Linear Regression has no hyperparameters to tune.")
    model = LinearRegression()

sample_size = st.slider("Training sample size (rows)", 5_000, 70_000, 20_000, 5_000,
                        help="Smaller samples train faster; the notebook's final model uses the full training set.")

# --- Train and evaluate ---------------------------------------------------------
if st.button("Train model", type="primary"):
    sample = train_df.sample(sample_size, random_state=RANDOM_STATE)
    pipeline = Pipeline([("prep", make_preprocessor()), ("model", model)])

    with st.spinner(f"Training {model_name} on {sample_size:,} rows..."):
        pipeline.fit(sample[FEATURES], sample[TARGET])
        train_pred = pipeline.predict(sample[FEATURES])
        test_pred = pipeline.predict(test_df[FEATURES])

    test_r2 = r2_score(test_df[TARGET], test_pred)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train R²", f"{r2_score(sample[TARGET], train_pred):.3f}")
    col2.metric("Test R²", f"{test_r2:.3f}")
    col3.metric("Test MAE (log)", f"{mean_absolute_error(test_df[TARGET], test_pred):.4f}")
    col4.metric("Test MAE ($)", f"${mean_absolute_error(np.exp(test_df[TARGET]), np.exp(test_pred)):,.0f}")

    gap = r2_score(sample[TARGET], train_pred) - test_r2
    if gap > 0.15:
        st.warning(f"Train R² is {gap:.2f} higher than test R² — the model is overfitting the training sample.")
    else:
        st.success(f"Train and test R² are close (gap {gap:.2f}) — the model generalises well.")

    final = load_result("test_results")
    best = final.index[0]
    st.caption(f"For comparison, the notebook's final model ({best}, tuned and trained on all "
               f"{len(train_df):,} rows) reaches a test R² of {final.loc[best, 'R²']:.3f}.")
