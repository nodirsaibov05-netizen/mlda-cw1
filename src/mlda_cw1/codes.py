"""Decoding tables for the ACS 2024 PUMS person file.

The Census Bureau stores every categorical variable as a numeric code.
These dictionaries translate the codes into readable labels, following the
official "2024 ACS PUMS Data Dictionary". They are used for EDA plots,
feature engineering and the Streamlit app.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Simple one-to-one code tables
# ---------------------------------------------------------------------------

SEX = {1: "Male", 2: "Female"}

# Educational attainment (ordinal: a higher code means more education)
SCHL = {
    1: "No schooling", 2: "Preschool", 3: "Kindergarten",
    4: "Grade 1", 5: "Grade 2", 6: "Grade 3", 7: "Grade 4", 8: "Grade 5",
    9: "Grade 6", 10: "Grade 7", 11: "Grade 8", 12: "Grade 9",
    13: "Grade 10", 14: "Grade 11", 15: "Grade 12, no diploma",
    16: "High school diploma", 17: "GED", 18: "Some college (<1 year)",
    19: "Some college (1+ years)", 20: "Associate's degree",
    21: "Bachelor's degree", 22: "Master's degree",
    23: "Professional degree", 24: "Doctorate",
}

# Education collapsed into 8 ordered levels (used in plots and as a feature)
EDU_LEVELS = [
    "Less than HS", "HS diploma / GED", "Some college", "Associate's",
    "Bachelor's", "Master's", "Professional", "Doctorate",
]


def edu_level(schl: pd.Series) -> pd.Series:
    """Map SCHL codes (1-24) to the 8 ordered education levels."""
    bins = [0, 15, 17, 19, 20, 21, 22, 23, 24]
    return pd.cut(schl, bins=bins, labels=EDU_LEVELS, ordered=True)


MAR = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
       5: "Never married"}

# Class of worker (the dataset keeps employees only, codes 1-5)
COW = {
    1: "Private for-profit", 2: "Private non-profit",
    3: "Local government", 4: "State government", 5: "Federal government",
    6: "Self-employed (not inc.)", 7: "Self-employed (inc.)",
    8: "Unpaid family worker", 9: "Unemployed",
}

NATIVITY = {1: "US-born", 2: "Foreign-born"}

# Ability to speak English. Missing (NaN) means "speaks only English".
ENG = {1: "Very well", 2: "Well", 3: "Not well", 4: "Not at all"}

DIS = {1: "With disability", 2: "No disability"}

MIL = {1: "Active duty", 2: "Veteran", 3: "Reserve/Guard training only",
       4: "Never served"}

# Means of transportation to work. Missing (NaN) = not at work last week.
JWTRNS = {
    1: "Car, truck or van", 2: "Bus", 3: "Subway", 4: "Long-distance train",
    5: "Light rail", 6: "Ferryboat", 7: "Taxicab", 8: "Motorcycle",
    9: "Bicycle", 10: "Walked", 11: "Worked from home", 12: "Other",
}

RAC1P = {
    1: "White", 2: "Black", 3: "American Indian", 4: "Alaska Native",
    5: "American Indian / Alaska Native", 6: "Asian",
    7: "Native Hawaiian / Pacific Islander", 8: "Some other race",
    9: "Two or more races",
}


def race_ethnicity(rac1p: pd.Series, hisp: pd.Series) -> pd.Series:
    """Combine race (RAC1P) and Hispanic origin (HISP, 1 = not Hispanic)."""
    result = rac1p.map({1: "White (non-Hisp.)", 2: "Black (non-Hisp.)",
                        6: "Asian (non-Hisp.)"}).fillna("Other (non-Hisp.)")
    result[hisp != 1] = "Hispanic"          # Hispanic origin overrides race
    return result


# ---------------------------------------------------------------------------
# Range-based groupings for occupation (OCCP) and industry (INDP)
# Each tuple is (first code, last code, label) from the PUMS data dictionary.
# ---------------------------------------------------------------------------

OCCP_GROUPS = [
    (10, 440, "Management"), (500, 750, "Business operations"),
    (800, 960, "Financial"), (1005, 1240, "Computer & math"),
    (1305, 1560, "Architecture & engineering"), (1600, 1980, "Science"),
    (2001, 2060, "Community & social service"), (2100, 2180, "Legal"),
    (2205, 2555, "Education"), (2600, 2920, "Arts, media & sports"),
    (3000, 3550, "Healthcare practitioners"), (3601, 3655, "Healthcare support"),
    (3700, 3960, "Protective service"), (4000, 4160, "Food preparation & serving"),
    (4200, 4255, "Cleaning & maintenance"), (4330, 4655, "Personal care"),
    (4700, 4965, "Sales"), (5000, 5940, "Office & admin support"),
    (6005, 6130, "Farming, fishing & forestry"), (6200, 6765, "Construction"),
    (6800, 6950, "Extraction"), (7000, 7640, "Installation & repair"),
    (7700, 8990, "Production"), (9005, 9760, "Transportation & moving"),
    (9800, 9830, "Military"),
]

INDP_SECTORS = [
    (170, 290, "Agriculture"), (370, 490, "Mining, oil & gas"),
    (570, 690, "Utilities"), (770, 770, "Construction"),
    (1070, 3990, "Manufacturing"), (4070, 4590, "Wholesale trade"),
    (4670, 5791, "Retail trade"), (6070, 6390, "Transportation & warehousing"),
    (6470, 6781, "Information"), (6870, 7190, "Finance, insurance & real estate"),
    (7270, 7790, "Professional & admin services"), (7860, 7890, "Education"),
    (7970, 8470, "Health care & social assistance"),
    (8561, 8690, "Arts, accommodation & food"), (8770, 9290, "Other services"),
    (9370, 9590, "Public administration"), (9670, 9870, "Military"),
]


def _group_by_range(codes: pd.Series, ranges: list) -> pd.Series:
    """Assign every numeric code to the label of the range it falls into."""
    out = pd.Series(pd.NA, index=codes.index, dtype="object")
    for low, high, label in ranges:
        out[codes.between(low, high)] = label
    return out


def occupation_group(occp: pd.Series) -> pd.Series:
    """Collapse ~520 detailed occupation codes into 25 major groups."""
    return _group_by_range(occp, OCCP_GROUPS)


def industry_sector(indp: pd.Series) -> pd.Series:
    """Collapse ~250 detailed industry codes into 18 sectors."""
    return _group_by_range(indp, INDP_SECTORS)
