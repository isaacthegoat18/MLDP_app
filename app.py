import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Job Salary Predictor",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Title Section ---
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>AI Job Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Predict your AI job salary based on real global data</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("Set Prediction Parameters")

job_titles = [
    'Data Scientist', 'Machine Learning Engineer', 'Data Analyst', 'AI Researcher',
    'Business Intelligence Analyst', 'Data Engineer', 'AI Developer'
]

experience_levels = ['EN', 'MI', 'SE', 'EX']
employment_types = ['FT', 'PT', 'CT', 'FL']
company_locations = ['US', 'GB', 'IN', 'DE', 'CA']
employee_residences = ['US', 'GB', 'IN', 'DE', 'CA']
industries = ['Tech', 'Finance', 'Health', 'Education']

job_title = st.sidebar.selectbox("Job Title", job_titles)
experience_level = st.sidebar.selectbox("Experience Level", experience_levels)
employment_type = st.sidebar.selectbox("Employment Type", employment_types)
company_location = st.sidebar.selectbox("Company Location", company_locations)
employee_residence = st.sidebar.selectbox("Employee Residence", employee_residences)
company_size = st.sidebar.number_input("Company Size (Number of Employees)", min_value=1, max_value=100000, value=50)
remote_ratio = st.sidebar.slider("Remote Work Ratio (%)", 0, 100, 50)
industry = st.sidebar.selectbox("Industry", industries)

# --- Load Model and Columns ---
model = joblib.load("ai_salary_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# --- Make Prediction ---
if st.sidebar.button("Predict Salary"):
    # Create DataFrame
    input_data = pd.DataFrame([{
        'job_title': job_title,
        'experience_level': experience_level,
        'employment_type': employment_type,
        'company_location': company_location,
        'employee_residence': employee_residence,
        'company_size': company_size,
        'remote_ratio': remote_ratio,
        'industry': industry
    }])

    # One-hot encode to match training data
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    with st.spinner("Predicting salary..."):
        time.sleep(1)
        prediction = model.predict(input_encoded)[0]

    st.success(f"💰 Estimated Salary: **${int(prediction):,} USD**")

# --- How It Works Section ---
st.markdown("---")
st.markdown("## 🔍 How It Works")

# Step 1
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://i.ibb.co/svzTktJ2/Screenshot-2025-08-01-084431.png", width=300)
with col2:
    st.subheader("Step 1: Set Parameters")
    st.write("Choose the job details from the sidebar to reflect your desired role and company context.")

# Step 2
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://i.ibb.co/C4J2nWs/Screenshot-2025-08-01-085150.png", width=300)
with col2:
    st.subheader("Step 2: Click Predict")
    st.write("Press the **Predict Salary** button to submit your job profile to our AI model.")

# Step 3
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://i.postimg.cc/zfHkzw6g/Screenshot-2025-08-01-085917.png", width=300)
with col2:
    st.subheader("Step 3: View Results")
    st.write("View your estimated salary immediately on the screen.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9rem;'>Built with ❤️ using Streamlit and Machine Learning</p>", unsafe_allow_html=True)
