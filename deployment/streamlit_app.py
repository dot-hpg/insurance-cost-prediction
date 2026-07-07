"""
Streamlit web calculator for the Insurance Cost Prediction model.

Run with:  streamlit run streamlit_app.py
(requires best_model.pkl, scaler.pkl, feature_list.pkl, uses_scaled_input.pkl
 in ../models/ — produced by notebooks/Insurance_Cost_Prediction.ipynb)
"""
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Insurance Premium Estimator", page_icon="🏥", layout="centered")

MODEL_DIR = "../models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    features = joblib.load(f"{MODEL_DIR}/feature_list.pkl")
    uses_scaled = joblib.load(f"{MODEL_DIR}/uses_scaled_input.pkl")
    return model, scaler, features, uses_scaled

model, scaler, FEATURES, USES_SCALED = load_artifacts()

st.title("🏥 Insurance Premium Estimator")
st.caption("Estimate annual health insurance premium (INR) from an individual's health profile.")

with st.form("premium_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 66, 35)
        height = st.slider("Height (cm)", 145, 188, 170)
        weight = st.slider("Weight (kg)", 51, 132, 75)
        surgeries = st.selectbox("Number of major surgeries", [0, 1, 2, 3])
    with col2:
        diabetes = st.checkbox("Diabetes")
        bp = st.checkbox("Blood pressure problems")
        transplant = st.checkbox("Any organ transplant")
        chronic = st.checkbox("Any chronic disease")
        allergies = st.checkbox("Known allergies")
        cancer_history = st.checkbox("Family history of cancer")

    submitted = st.form_submit_button("Estimate Premium")

if submitted:
    bmi = weight / ((height / 100) ** 2)
    total_conditions = sum([diabetes, bp, transplant, chronic, allergies, cancer_history])

    row = {
        "Age": age,
        "Diabetes": int(diabetes),
        "BloodPressureProblems": int(bp),
        "AnyTransplants": int(transplant),
        "AnyChronicDiseases": int(chronic),
        "Height": height,
        "Weight": weight,
        "KnownAllergies": int(allergies),
        "HistoryOfCancerInFamily": int(cancer_history),
        "NumberOfMajorSurgeries": surgeries,
        "BMI": bmi,
        "TotalConditions": total_conditions,
    }
    X = np.array([[row[f] for f in FEATURES]])
    X_in = scaler.transform(X) if USES_SCALED else X
    pred = model.predict(X_in)[0]
    pred = float(np.clip(pred, 15000, 40000))

    st.success(f"### Estimated Annual Premium: ₹{pred:,.0f}")
    st.write(f"**BMI:** {bmi:.1f}  |  **Risk factors present:** {total_conditions}/6")

    if pred > 30000:
        st.warning("This profile falls in the higher-risk pricing tier.")
    elif pred < 20000:
        st.info("This profile falls in the lower-risk pricing tier.")

st.divider()
st.caption(
    "Model: trained on the Insurance Cost Prediction dataset. "
    "This is a portfolio/demo tool, not a real underwriting system."
)
