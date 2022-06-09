import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np

st.set_page_config(
    page_title="Concrete CS Estimator",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://google.com',
        'Report a bug': "https://github.com/4dhil",
        'About': "Telco Churn Prediction"
    }
)

st.header('Telco Churn Predictor')
st.write("""
Created by Fadhil Muhammad Irfan FTDS-10
""")

def user_input():
    SeniorCitizen = st.radio('Senior Citizen', ['yes', 'no'])
    Partner = st.radio("Partner", ['Yes', 'No'])
    Dependents = st.radio("Dependents", ['Yes', 'No'])
    tenure = st.number_input("Tenure Length")
    OnlineSecurity = st.radio("Online Security", ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.radio("Online Backup", ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.radio("Device Protection", ['No', 'Yes', 'No internet service'])
    TechSupport = st.radio("Tech Support", ['No', 'Yes', 'No internet service'])
    Contract = st.radio("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.radio("Paperless Billing", ['Yes', 'No'])
    PaymentMethod = st.radio("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])
    MonthlyCharges = st.number_input("Monthly Charges")
    TotalCharges = st.number_input("Total Charges")

    data = {
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'Contract' : Contract,
        'PaperlessBilling' : PaperlessBilling,
        'PaymentMethod' : PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges' : TotalCharges
    }
    features = pd.DataFrame(data, index=[0])
    return features


input = user_input()

st.subheader('User Input')
st.write(input)

transformer = joblib.load("transformer.pkl")
input_trans = transformer.transform(input)

model_inf = keras.models.load_model("modelque.h5")
prediction = model_inf.predict(input_trans)
prediction = np.where(prediction >= 0.5, 1, 0)

st.write('Based on user input, the customer will:')
if prediction == 1:
    st.write('Churn')
else:
    st.write('Not Churn')