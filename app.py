import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Prosper Loan Eligibility Predictor",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("Prosper Loan Eligibility Predictor")
st.markdown("""
This application predicts loan eligibility based on applicant information using a pre-trained machine learning pipeline.
Fill in the form below and click 'Predict' to see if you're eligible for a loan.
""")

# Load the pipeline
@st.cache_resource
def load_model_pipeline():
    try:
        with open('model_pipeline_Nb.pkl', 'rb') as file:
            pipeline = pickle.load(file)
        return pipeline
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model_pipeline_Nb.pkl' exists in the directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model pipeline: {str(e)}")
        return None

pipeline = load_model_pipeline()

# Create form for user input
with st.form("loan_eligibility_form"):
    st.subheader("Applicant Information")
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        stated_monthly_income = st.number_input("Stated Monthly Income (USD)", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
        loan_amount = st.number_input("Loan Amount (USD)", min_value=1000.0, max_value=50000.0, value=10000.0, step=500.0)
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed"])

    with col2:
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
        loan_term = st.selectbox("Loan Term (months)", [12, 36, 60])
        loan_purpose = st.selectbox("Loan Purpose", ["Debt Consolidation", "Home Improvement", "Business", "Education", "Other"])
        delinquencies = st.number_input("Number of Delinquencies in Last 2 Years", min_value=0, max_value=20, value=0)

    submit_button = st.form_submit_button("Predict")

# Prediction logic
if submit_button and pipeline:
    try:
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

        # Predict using the full pipeline
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)[0]
        confidence_score = prediction_proba[1] if prediction == 1 else prediction_proba[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Eligible for Loan")
            st.write(f"Confidence Score: {confidence_score:.2%}")
            st.markdown("Based on your inputs, you are likely to be **approved** for a loan.")
        else:
            st.error("❌ High Credit Risk")
            st.write(f"Confidence Score: {confidence_score:.2%}")
            st.markdown("Based on your inputs, you are considered a **high credit risk** for this loan.")

        # Summary Table
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
