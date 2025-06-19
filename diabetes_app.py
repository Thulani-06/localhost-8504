import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("diabetesmodel.pkl", "rb") as file:
    model = pickle.load(file)

st.title("AI Diabetes Risk Assessment")

# Input fields with min and max values
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("You are likely to have diabetes.")
    else:
        st.success("You are not likely to have diabetes.")