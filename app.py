
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
def load_model():
    return joblib.load("churn_prediction_model_v1_0.joblib")

model = load_model()

# Streamlit UI for Customer Churn Prediction
st.title("Customer Churn Prediction App")
st.write("This tool predicts customer churn risk based on their details. Enter the required information below.")

# User Inputs
SeniorCitizen = st.selectbox("Is the customer a senior citizen?", [0, 1])

Partner = st.selectbox("Does the customer have a partner?", ["Yes", "No"])
Dependents = st.selectbox("Does the customer have dependents?", ["Yes", "No"])
PhoneService = st.selectbox("Does the customer have phone service?", ["Yes", "No"])
InternetService = st.selectbox(
    "Type of Internet Service",
    ["DSL", "Fiber optic", "No"]
)
Contract = st.selectbox(
    "Type of Contract",
    ["Month-to-month", "One year", "Two year"]
)
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

tenure = st.number_input(
    "Tenure (Months with the company)",
    min_value=0,
    value=12
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=600.0
)

# Create input DataFrame
input_data = pd.DataFrame([{
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "Partner": Partner,
    "Dependents": Dependents,
    "PhoneService": PhoneService,
    "InternetService": InternetService,
    "Contract": Contract,
    "PaymentMethod": PaymentMethod
}])

# Set classification threshold
classification_threshold = 0.5

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Prediction: The customer is likely to **{result}**.")
    st.write(f"Churn Probability: {prediction_proba:.2f}")
