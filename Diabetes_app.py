import streamlit as st
import numpy as np
import pandas as pd
import pickle


# Load the pickle files
model = pickle.load(open('model.pkl', 'rb'))
scaler =pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("Diabetes Prediction System")

#crete 2 columns for better layout
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glu = st.number_input("Glucose", min_value=0)
    bp = st.number_input("BloodPressure", min_value=0)
    skin = st.number_input("SkinThickness", min_value=0)

with col2:
    ins = st.number_input("Insuline", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("DibetesPedigreeFunction", min_value=0.0)
    age = st.number_input("Age",  min_value=1)

if st.button("Predict"):
    features = np.array([[ preg, glu, bp, skin, ins, bmi, dpf, age ]])
    features_scaled = scaler.transform(features)    
    Prediction = model.predict(features_scaled)

    if Prediction[0] == 1:
        st.error("The model predicts: Diabetes Positive ")
    else:
        st.success("The model predicts: Diabetes Negative")
