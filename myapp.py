import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Prosper Loan Eligibility Predictor",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("Prosper Loan Eligibility Predictor")
st.markdown("""
This application predicts loan eligibility based on applicant information using a pre-trained machine learning model.
Fill in the form below and click 'Predict' to see if you're eligible for a loan.
""")

# Function to load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open('preprocessor.pkl', 'rb') as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)
            
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model or preprocessor files not found. Please ensure 'model.pkl' and 'preprocessor.pkl' exist in the current directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {str(e)}")
        return None, None

# Load model and preprocessor
model, preprocessor = load_model_and_preprocessor()

# Create form for user inputs
with st.form("loan_eligibility_form"):
    st.subheader("Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Applicant's age in years")
        
        stated_monthly_income = st.number_input(
            "Stated Monthly Income (USD)", 
            min_value=0.0, 
            max_value=100000.0, 
            value=5000.0,
            step=100.0,
            help="Applicant's stated monthly income in USD"
        )
        
        loan_amount = st.number_input(
            "Loan Amount (USD)", 
            min_value=1000.0, 
            max_value=50000.0, 
            value=10000.0,
            step=500.0,
            help="Requested loan amount in USD"
        )
        
        employment_status = st.selectbox(
            "Employment Status", 
            options=["Employed", "Self-employed", "Unemployed"],
            help="Current employment status of the applicant"
        )
    
    with col2:
        credit_score = st.slider(
            "Credit Score", 
            min_value=300, 
            max_value=850, 
            value=700,
            help="Applicant's credit score (FICO)"
        )
        
        loan_term = st.selectbox(
            "Loan Term (months)", 
            options=[12, 36, 60],
            help="Term of the loan in months"
        )
        
        loan_purpose = st.selectbox(
            "Loan Purpose", 
            options=["Debt Consolidation", "Home Improvement", "Business", "Education", "Other"],
            help="Purpose of the loan"
        )
        
        delinquencies = st.number_input(
            "Number of Delinquencies in the Last 2 Years", 
            min_value=0, 
            max_value=20, 
            value=0,
            help="Number of times the applicant was delinquent on payments in the last 2 years"
        )
    
    # Submit button
    submit_button = st.form_submit_button("Predict")

# Prediction logic
if submit_button and model and preprocessor:
    try:
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Age': [age],
            'StatedMonthlyIncome': [stated_monthly_income],
            'LoanAmount': [loan_amount],
            'EmploymentStatus': [employment_status],
            'CreditScore': [credit_score],
            'LoanTerm': [loan_term],
            'LoanPurpose': [loan_purpose],
            'Delinquencies': [delinquencies]
        })
        
        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get prediction probability
        prediction_proba = model.predict_proba(processed_data)[0]
        confidence_score = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        # Display prediction results
        st.subheader("Prediction Result")
        
        if prediction == 0:
            st.success("**Eligible for Loan**")
            st.write(f"Confidence Score: {confidence_score:.2%}")
            st.write("Congratulations! Based on the provided information, you are eligible for a loan.")
        else:
            st.error("**High Credit Risk**")
            st.write(f"Confidence Score: {confidence_score:.2%}")
            st.write("Based on the provided information, you are considered a high credit risk for this loan.")
        
        # Display a table of the input information for reference
        st.subheader("Applicant Information Summary")
        summary_data = {
            "Parameter": ["Age", "Monthly Income (USD)", "Loan Amount (USD)", "Employment Status", 
                         "Credit Score", "Loan Term (months)", "Loan Purpose", "Delinquencies"],
            "Value": [age, f"${stated_monthly_income:,.2f}", f"${loan_amount:,.2f}", employment_status, 
                     credit_score, loan_term, loan_purpose, delinquencies]
        }
        st.table(pd.DataFrame(summary_data))
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Please check your inputs and try again.")