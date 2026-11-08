"""Shared helpers for the Streamlit app.

The data preparation here repeats Section 3 of the notebook (same steps, same
random seed), so the app works with exactly the same data and test set.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "acs_tx_2024_fulltime.csv"
MODEL_DIR = ROOT / "models"

sys.path.append(str(ROOT / "src"))
from mlda_cw1 import codes  # noqa: E402  (import after the path is added)

RANDOM_STATE = 42
MIN_HOURLY_WAGE = 7.25          # federal minimum wage, used to remove errors (notebook 3.1)

NUM_FEATURES = ["AGEP", "age_sq", "SCHL", "WKHP", "WKWN", "JWMNP", "ENG", "work_from_home"]
CAT_FEATURES = ["occupation", "industry", "employer", "marital"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "log_wage"

BLUE, ORANGE = "#2a78d6", "#eb6834"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Filtered dataset from notebook Section 1 (92,513 full-time employees)."""
    return pd.read_csv(DATA_FILE)


def add_labels(df):
    """Readable labels for charts (same decoding as notebook Section 2)."""
    out = df.copy()
    out["sex"] = out["SEX"].map(codes.SEX)
    out["education"] = codes.edu_level(out["SCHL"])
    out["occupation"] = codes.occupation_group(out["OCCP"])
    out["industry"] = codes.industry_sector(out["INDP"])
    out["employer"] = out["COW"].map(codes.COW)
    out["marital"] = out["MAR"].map(codes.MAR)
    out["race_eth"] = codes.race_ethnicity(out["RAC1P"], out["HISP"])
    return out


def prepare(df, min_hourly_wage=MIN_HOURLY_WAGE):
    """Notebook Sections 3.1-3.3: fix errors, fill missing values, create features."""
    data = df.copy()

    # 3.1 inflation adjustment and removal of impossible wages
    data["wage"] = data["WAGP"] * data["ADJINC"] / 1_000_000
    data["hourly_wage"] = data["wage"] / (data["WKHP"] * data["WKWN"])
    data = data[data["hourly_wage"] >= min_hourly_wage].copy()
    data["log_wage"] = np.log(data["wage"])

    # 3.2 structural missing values
    data["ENG"] = data["ENG"].fillna(0)        # blank = speaks only English
    data["JWMNP"] = data["JWMNP"].fillna(0)    # blank = works from home / not at work

    # 3.3 feature engineering
    data["occupation"] = codes.occupation_group(data["OCCP"])
    data["industry"] = codes.industry_sector(data["INDP"])
    data["employer"] = data["COW"].map(codes.COW)
    data["marital"] = data["MAR"].map(codes.MAR)
    data["age_sq"] = data["AGEP"] ** 2
    data["work_from_home"] = (data["JWTRNS"].fillna(0) == 11).astype(int)
    return data


@st.cache_data
def load_prepared():
    return prepare(load_data())


@st.cache_data
def train_test_data():
    """Notebook Section 3.5: the same 80/20 split, stratified by wage decile."""
    data = load_prepared()
    model_data = data[FEATURES + [TARGET, "wage", "SEX", "RAC1P", "HISP"]].reset_index(drop=True)
    wage_bins = pd.qcut(model_data[TARGET], q=10, labels=False)
    train_df, test_df = train_test_split(model_data, test_size=0.2, random_state=RANDOM_STATE,
                                         stratify=wage_bins)
    return train_df, test_df


def make_preprocessor():
    """Notebook Section 3.6: scaling for numbers, one-hot encoding for categories."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


# ---------------------------------------------------------------------------
# Saved model and results (created by notebook Section 5.7)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "wage_model.joblib")


def load_result(name):
    """Read one of the result tables saved by the notebook (e.g. 'test_results')."""
    return pd.read_csv(MODEL_DIR / f"{name}.csv", index_col=0)
