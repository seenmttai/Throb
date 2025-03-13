import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load the saved preprocessor
try:
    preprocessor = joblib.load("preprocessor.pkl")
except FileNotFoundError:
    st.error("Error: Preprocessor file 'preprocessor.pkl' not found. Please ensure it's saved in the correct directory.")
    st.stop()

# Load the trained Keras model
try:
    model = tf.keras.models.load_model("heart_disease_detector.keras")
except FileNotFoundError:
    st.error("Error: Model file 'heart_disease_detector.keras' not found. Please ensure it's saved in the correct directory.")
    st.stop()

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .stTextInput, .stRadio, .stSelectbox, .stSlider, .stNumberInput, .stButton > button {
            font-size: 16px !important;
        }
        .stNumberInput input {
            padding: 10px;
            border-radius: 5px;
        }
        .stButton > button {
            background-color: #FF6B6B !important;
            color: white !important;
            border-radius: 5px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter your health details to predict the risk of heart disease.</p>", unsafe_allow_html=True)

# --- Input Fields ---
# Numeric features
bmi = st.slider('BMI', 12, 98, 25, step=1)
ment_hlth = st.slider('Mental Health in past 30 days (0-30)', 0, 30, 0)
phys_hlth = st.slider('Physical Health in past 30 days (0-30)', 0, 30, 0)
age = st.slider('Age (1-13, where 1 is 18-24 and 13 is 80+)', 1, 13, 1)

# Categorical features
high_bp = st.radio('High Blood Pressure (0 for no, 1 for yes)', [0, 1])
high_chol = st.radio('High Cholesterol (0 for no, 1 for yes)', [0, 1])
chol_check = st.radio('Cholesterol Check (0 for no, 1 for yes)', [0, 1])
smoker = st.radio('Smoker (0 for no, 1 for yes)', [0, 1])
stroke = st.radio('Stroke (0 for no, 1 for yes)', [0, 1])
diabetes = st.radio('Diabetes (0 for no, 1 for yes, 2 for borderline)', [0, 1, 2])
phys_activity = st.radio('Physical Activity (0 for no, 1 for yes)', [0, 1])
hvy_alcohol_consump = st.radio('Heavy Alcohol Consumption (0 for no, 1 for yes)', [0, 1])
any_healthcare = st.radio('Any Healthcare Coverage (0 for no, 1 for yes)', [0, 1])
no_doc_bc_cost = st.radio('Could not see doctor because of cost (0 for no, 1 for yes)', [0, 1])
gen_hlth = st.radio('General Health (1 for excellent, up to 5 for poor)', [1, 2, 3, 4, 5])
sex = st.radio('Sex (0 for female, 1 for male)', [0, 1])

if st.button("Predict Heart Disease Risk ❤️"):
    # Create a DataFrame with the same column names as used in the preprocessor
    raw_input = pd.DataFrame({
        "BMI": [bmi],
        "MentHlth": [ment_hlth],
        "PhysHlth": [phys_hlth],
        "Age": [age],
        "HighBP": [high_bp],
        "HighChol": [high_chol],
        "CholCheck": [chol_check],
        "Smoker": [smoker],
        "Stroke": [stroke],
        "Diabetes": [diabetes],
        "PhysActivity": [phys_activity],
        "HvyAlcoholConsump": [hvy_alcohol_consump],
        "AnyHealthcare": [any_healthcare],
        "NoDocbcCost": [no_doc_bc_cost],
        "GenHlth": [gen_hlth],
        "Sex": [sex]
    })
    
    # Transform the raw input using the preprocessor
    input_data = preprocessor.transform(raw_input)
    
    # Make the prediction (assuming a binary classifier with threshold 0.5)
    prediction = model.predict(input_data)
    risk = "High Risk" if prediction[0] >= 0.5 else "Low Risk"
    
    # Display the prediction result
    if risk == "High Risk":
        st.markdown(
            """<div style='background-color: #ffe6e6; padding: 15px; border-radius: 5px; text-align: center;'>
            <h3 style='color: #a30000'>High Risk: The model predicts a <span style='color: #a30000;'>high risk</span> of heart disease.</h3>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style='background-color: #e6ffe6; padding: 15px; border-radius: 5px; text-align: center;'>
            <h3 style='color: #006600'>Low Risk: The model predicts a <span style='color: #006600;'>low risk</span> of heart disease.</h3>
            </div>""",
            unsafe_allow_html=True
        )
