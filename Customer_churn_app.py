import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

@st.cache_resource
def load_resources():
    try:
        # Load the MinMaxScaler
        with open('min_max_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Load the Logistic Regression model
        with open('logistic_regression_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)

        # Load the Random Forest model
        with open('random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)

        # Load the Keras Neural Network model
        nn_model = keras.models.load_model('neural_network_model.h5')
        return scaler, lr_model, rf_model, nn_model
    except FileNotFoundError as e:
        st.error(f"Error loading model files. Make sure 'min_max_scaler.pkl', 'logistic_regression_model.pkl', 'random_forest_model.pkl', and 'neural_network_model.h5' are in the same directory as this script. Missing file: {e.filename}")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()

scaler, lr_model, rf_model, nn_model = load_resources()

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction App")
st.write("""
This application predicts whether a customer will churn based on their demographic and banking information.
""")

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 350, 850, 600)
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
    num_of_products = st.slider("Number of Products", 1, 4, 1)

with col2:
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 75000.0, step=1000.0)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Female", "Male"])

# Convert Yes/No to 1/0
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_member_val = 1 if is_active_member == "Yes" else 0

def preprocess_input(credit_score, age, tenure, balance, num_of_products,
                     has_cr_card_val, is_active_member_val, estimated_salary,
                     geography, gender):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_val],
        'IsActiveMember': [is_active_member_val],
        'EstimatedSalary': [estimated_salary],
        'Gender_M': [1 if gender == 'Male' else 0], # Male is 1, Female is 0 (due to drop_first=True)
        'Geography_Germany': [1 if geography == 'Germany' else 0],
        'Geography_Spain': [1 if geography == 'Spain' else 0]
    })

    scaled_input = scaler.transform(input_data)
    return scaled_input

if st.button("Predict Churn"):
    processed_input = preprocess_input(
        credit_score, age, tenure, balance, num_of_products,
        has_cr_card_val, is_active_member_val, estimated_salary,
        geography, gender
    )

    st.subheader("Prediction Results")

    # Neural Network Prediction
    nn_prediction_proba = nn_model.predict(processed_input)[0][0]
    nn_prediction = "Churn" if nn_prediction_proba > 0.5 else "No Churn"
    st.write(f"**Neural Network Prediction:** {nn_prediction} (Probability: {nn_prediction_proba:.2f})")

    # Logistic Regression Prediction
    lr_prediction_proba = lr_model.predict_proba(processed_input)[0][1]
    lr_prediction = "Churn" if lr_prediction_proba > 0.5 else "No Churn"
    st.write(f"**Logistic Regression Prediction:** {lr_prediction} (Probability: {lr_prediction_proba:.2f})")

    # Random Forest Prediction
    rf_prediction_proba = rf_model.predict_proba(processed_input)[0][1]
    rf_prediction = "Churn" if rf_prediction_proba > 0.5 else "No Churn"
    st.write(f"**Random Forest Prediction:** {rf_prediction} (Probability: {rf_prediction_proba:.2f})")

    st.markdown("---")
    st.write("A 'Churn' prediction means the customer is likely to leave the bank.")
    st.write("A 'No Churn' prediction means the customer is likely to stay.")

st.markdown("---")
st.write("Developed by Prince Mishra")
