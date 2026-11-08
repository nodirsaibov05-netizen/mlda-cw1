"""Page 4: predict the wage of a new person with the final model (inference)."""

import numpy as np
import pandas as pd
import streamlit as st

from utils import FEATURES, codes, load_model, load_result

st.set_page_config(page_title="Predict Wage", page_icon="🔮", layout="wide")
st.title("🔮 Predict a Wage")
st.markdown("Describe a full-time employee in Texas and the final model estimates their annual wage (2024 dollars).")

model = load_model()

# Options for the drop-down lists (taken from the decoding tables)
occupations = sorted({label for _, _, label in codes.OCCP_GROUPS if label != "Military"})
industries = sorted({label for _, _, label in codes.INDP_SECTORS if label != "Military"})
employers = [codes.COW[c] for c in [1, 2, 3, 4, 5]]
marital_options = list(codes.MAR.values())
english_options = {"Only English": 0, "Very well": 1, "Well": 2, "Not well": 3, "Not at all": 4}

with st.form("profile"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 64, 35)
        schl_label = st.selectbox("Highest education", list(codes.SCHL.values()),
                                  index=list(codes.SCHL.values()).index("Bachelor's degree"))
        occupation = st.selectbox("Occupation group", occupations, index=occupations.index("Computer & math"))
        industry = st.selectbox("Industry", industries,
                                index=industries.index("Professional & admin services"))
        employer = st.selectbox("Employer type", employers)
    with col2:
        hours = st.slider("Usual hours worked per week", 35, 80, 40)
        weeks = st.slider("Weeks worked in the last 12 months", 1, 52, 52)
        work_from_home = st.checkbox("Works from home", value=False)
        commute = 0 if work_from_home else st.slider("Commute time (minutes, one way)", 0, 120, 25)
        english = st.selectbox("English (if another language is spoken at home)", list(english_options))
        marital = st.selectbox("Marital status", marital_options)
    submitted = st.form_submit_button("Predict wage", type="primary")

if submitted:
    schl_code = [code for code, label in codes.SCHL.items() if label == schl_label][0]
    person = pd.DataFrame([{
        "AGEP": age, "age_sq": age ** 2, "SCHL": schl_code, "WKHP": hours, "WKWN": weeks,
        "JWMNP": commute, "ENG": english_options[english], "work_from_home": int(work_from_home),
        "occupation": occupation, "industry": industry, "employer": employer, "marital": marital,
    }])[FEATURES]

    predicted_wage = np.exp(model.predict(person)[0])        # the model predicts log(wage)
    typical_error = load_result("test_results").iloc[0]["Median % error"] / 100

    st.metric("Estimated annual wage", f"${predicted_wage:,.0f}")
    st.markdown(f"""
For a typical person the model's prediction is within about **±{typical_error:.0%}** of the true wage, so a realistic range
for this profile is **\\${predicted_wage * (1 - typical_error):,.0f} – \\${predicted_wage * (1 + typical_error):,.0f}**.
""")
    st.caption("The model does not use sex, race or other protected characteristics. It cannot see the employer, "
               "specific skills or experience in the role, which is why individual wages vary around the estimate.")
