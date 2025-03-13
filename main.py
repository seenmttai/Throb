import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Try to load the trained model (consider using the .h5 format if issues persist)
try:
    model = load_model("heart_disease_detector.keras")
    # If issues persist, consider re-saving the model to .h5 and updating this path.
except FileNotFoundError:
    st.error("Error: Model file 'heart_disease_detector.keras' not found. Please make sure the model is trained and saved in the correct directory.")
    st.stop()

# Customizing the page layout
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

# Styling
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# App Header
st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Enter your health details to predict the risk of heart disease.</p>",
    unsafe_allow_html=True
)

# Input fields
age = st.slider('Age (1-13, where 1 is 18-24 and 13 is 80+)', 1, 13, 1)
sex = st.radio('Sex (0 for female, 1 for male)', [0, 1])
high_bp = st.radio('High Blood Pressure (0 for no, 1 for yes)', [0, 1])
high_chol = st.radio('High Cholesterol (0 for no, 1 for yes)', [0, 1])
chol_check = st.radio('Cholesterol Check in 5 years (0 for no, 1 for yes)', [0, 1])
# Use integer slider for BMI
bmi = st.slider('BMI', 12, 98, 25, step=1)
smoker = st.radio('Smoker (0 for no, 1 for yes)', [0, 1])
stroke = st.radio('Stroke (0 for no, 1 for yes)', [0, 1])
diabetes = st.radio('Diabetes (0 for no, 1 for yes, 2 if borderline)', [0, 1, 2])
phys_activity = st.radio('Physical Activity in past 30 days (0 for no, 1 for yes)', [0, 1])
hvy_alcohol_consump = st.radio('Heavy Alcohol Consumption (0 for no, 1 for yes)', [0, 1])
any_healthcare = st.radio('Any Healthcare Coverage (0 for no, 1 for yes)', [0, 1])
no_doc_bc_cost = st.radio('Could not see doctor because of cost (0 for no, 1 for yes)', [0, 1])
gen_hlth = st.slider('General Health (1-5, where 1 is excellent, 5 is poor)', 1, 5, 3)
ment_hlth = st.slider('Mental Health in past 30 days (0-30)', 0, 30, 0)
phys_hlth = st.slider('Physical Health in past 30 days (0-30)', 0, 30, 0)

# Predict button
if st.button("Predict Heart Disease Risk ❤️"):
    # Prepare the input data as a NumPy array
    input_data = np.array([
        high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes,
        phys_activity, hvy_alcohol_consump, any_healthcare, no_doc_bc_cost,
        gen_hlth, ment_hlth, phys_hlth, sex, age
    ]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
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
