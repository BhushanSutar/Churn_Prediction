import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------
# Load trained objects
# -----------------------
model = load_model("Churn_Modelling.h5", compile=False, safe_mode=False)  # your trained ANN model

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

# -----------------------
# Streamlit UI
# -----------------------
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# Example input fields (modify based on your dataset)
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, step=100.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0)

# -----------------------
# Preprocessing
# -----------------------
if st.button("Predict"):
    # Encode categorical features
    gender_encoded = label_encoder.transform([gender])[0]  # label encoding
    geography_encoded = one_hot_encoder.transform([[geography]]).toarray()[0]  # one-hot encoding

    # Combine all inputs
    features = np.array([
        credit_score,
        gender_encoded,
        age,
        tenure,
        balance,
        num_of_products,
        1 if has_cr_card == "Yes" else 0,
        1 if is_active_member == "Yes" else 0,
        estimated_salary
    ])

    # Append one-hot encoded geography
    features = np.concatenate([features, geography_encoded])

    # Scale features
    features_scaled = scaler.transform([features])

    # Predict
    prediction = model.predict(features_scaled)[0][0]
    result = "Churn ❌" if prediction > 0.5 else "No Churn ✅"

    st.subheader(f"Prediction: {result}")
    st.write(f"Churn Probability: {prediction:.2f}")
