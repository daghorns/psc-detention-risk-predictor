
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="PSC Detention Risk Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 PSC Detention Risk Predictor")

st.write(
    "This application estimates detention risk for a planned USCG Port State Control inspection "
    "using historical inspection data. The inspection type must already be known before the prediction is made."
)

st.info(
    "Decision-support tool only. The prediction is intended to support inspection preparation and resource prioritization, "
    "and should not replace inspector judgement."
)

model = joblib.load("psc_detention_model.pkl")

inspection_type = st.text_input(
    "Inspection type",
    "PSC B"
)

vessel_age = st.number_input(
    "Vessel age",
    min_value=0,
    max_value=100,
    value=15
)

gross_tonnage = st.number_input(
    "Gross tonnage",
    min_value=0.0,
    value=30000.0
)

deadweight = st.number_input(
    "Deadweight",
    min_value=0.0,
    value=50000.0
)

flag = st.text_input(
    "Flag",
    "PANAMA"
)

vessel_type = st.selectbox(
    "Vessel type",
    [
        "Freight Ship",
        "Tank Ship",
        "Passenger (Inspected)",
        "Public Freight",
        "Recreational",
        "Offshore Supply Vessel"
    ]
)

class_society = st.text_input(
    "Classification society",
    "AMERICAN BUREAU OF SHIPPING"
)

originating_unit = st.text_input(
    "USCG originating unit",
    "Sector New Orleans"
)

inspection_date = st.date_input(
    "Inspection date"
)

if st.button("Predict detention risk"):

    input_df = pd.DataFrame({
        "inspection_type_grouped": [inspection_type],
        "vessel_age_clean": [vessel_age],
        "gross_tonnage_clean": [gross_tonnage],
        "deadweight_clean": [deadweight],
        "flag_grouped": [flag],
        "Vessel Type": [vessel_type],
        "class_society_grouped": [class_society],
        "Originating Unit": [originating_unit],
        "year_feature": [inspection_date.year],
        "month_feature": [inspection_date.month]
    })

    probability = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction")

    st.metric(
        "Predicted detention risk",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "The score is a model-based estimate and should be interpreted together "
        "with operational and professional judgement."
    )
