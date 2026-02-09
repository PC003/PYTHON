import streamlit as st
import joblib as jb
import numpy as np
model=jb.load("regression_model.pkl")
scaler=jb.load("scaler.pkl")
st.title("Linear Regression Predictor")

x1 = st.number_input("Enter Your Weight in Kg")

input_data=np.array([[x1]])

if st.button("Predict"):
    prediction = model.predict(scaler.transform(input_data))
    st.success(f"Predicted value: {prediction[0]} cm")
