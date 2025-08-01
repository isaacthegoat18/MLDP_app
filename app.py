import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

st.set_page_config(
    page_title="AI Job Salary Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
}

/* Background and blur effect on main only, not whole container */
.main {
    background: url("https://www.aihr.com/wp-content/uploads/salary-benchmarking-cover-image.png") no-repeat center center fixed;
    background-size: cover;
    position: relative;
    border-radius: 1rem;
    padding: 2rem;
}

/* Optional: Add dark blur only to content area, not full container */
.block-container {
    backdrop-filter: blur(6px);
    background-color: rgba(0, 0, 0, 0.7);
    border-radius: 1rem;
    padding: 2rem;
}

/* Fix for sidebar visibility */
section[data-testid="stSidebar"] {
    background-color: #1e1e1e;
    color: white;
    padding: 1rem;
    z-index: 999;
}

section[data-testid="stSidebar"] * {
    color: white;
}

h1, h2, h3, h4, h5, h6 {
    color: #0080ff;
}

/* Hide default Streamlit header/footer */
#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Load model and columns
model = joblib.load("ai_salary_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("💰AI Job Salary Predictor")
st.markdown("Use the sidebar to select the parameters for salary prediction. The predicted salary will appear below.")

result_placeholder = st.empty()
loading_placeholder = st.empty()

# Sidebar options
employment_options = {
    "Full-time": "FT",
    "Part-time": "PT",
    "Contract": "CT",
    "Freelance": "FL"
}

experience_level_options = {
    "Entry-level": "EN",
    "Mid-level": "MI",
    "Senior-level": "SE",
    "Executive-level": "EX"
}

with st.sidebar:
    st.markdown("## Set Prediction Parameters")
    job_title = st.selectbox("Job Title", ['AI Architect', 'AI Consultant', 'AI Product Manager', 'AI Research Scientist', 'AI Software Engineer', 'AI Specialist', 'Autonomous Systems Engineer', 'Computer Vision Engineer', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Deep Learning Engineer','Head of AI', 'Machine Learning Engineer', 'Machine Learning Researcher' ,'ML Ops Engineer', 'NLP Engineer', 'Principal Data Scientist', 'Research Scientist', 'Robotics Engineer'])
    employment_display = st.selectbox("Employment Type", list(employment_options.keys()))
    employment_type = employment_options[employment_display]
    company_location = st.selectbox("Company Location", ['Australia', 'Austria', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'India', 'Ireland', 'Israel', 'Japan' ,'Netherlands' ,'Norway' ,'Singapore' ,'South Korea', 'Sweden', 'Switzerland','United Kingdom','United States'])
    employee_residence = st.selectbox("Employee Residence", ['Australia', 'Austria', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'India', 'Ireland', 'Israel', 'Japan' ,'Netherlands' ,'Norway' ,'Singapore' ,'South Korea', 'Sweden', 'Switzerland','United Kingdom','United States'])
    remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, value=0, step=50)
    number_of_req_skills = st.number_input("Number of Required Skills", 0, 20, value=4, step=1)
    years_experience = st.number_input("Years of Experience", 0.0, 50.0, value=1.0, step=0.5)
    industry = st.selectbox("Industry", ['Automotive', 'Consulting', 'Education', 'Energy', 'Finance', 'Gaming', 'Government', 'Healthcare', 'Manufacturing', 'Media', 'Real Estate', 'Retail', 'Technology', 'Telecommunications', 'Transportation'])
    benefits_score = st.slider("Benefits Score (0-10)", 0, 10, value=2, step=1)
    num_employees = st.number_input("Number of Employees", min_value=1, max_value=100000, value=1, step=1)
    experience_label = st.selectbox("Experience Level", list(experience_level_options.keys()))
    experience_level = experience_level_options[experience_label]
    education_required = st.selectbox("Education Level", ['Associate', 'Bachelor', 'Master', 'PhD'])
    predict_button = st.button("Predict Salary")

# Company size encoding
if num_employees < 50:
    company_size = 'S'
elif num_employees < 250:
    company_size = 'M'
else:
    company_size = 'L'

# Prediction
if predict_button:
    result_placeholder.empty()
    with loading_placeholder.container():
        st.spinner("Predicting salary...")
        time.sleep(1.5)

    input_df = pd.DataFrame([{
        'job_title': job_title,
        'employment_type': employment_type,
        'company_location': company_location,
        'employee_residence': employee_residence,
        'remote_ratio': remote_ratio,
        'number_of_req_skills': number_of_req_skills,
        'years_experience': years_experience,
        'industry': industry,
        'benefits_score': benefits_score,
        'company_size': company_size,
        'experience_level': experience_level,
        'education_required': education_required
    }])

    input_df['company_size_encoded'] = input_df['company_size'].map({'S': 0, 'M': 1, 'L': 2})
    input_df['experience_level_encoded'] = input_df['experience_level'].map({'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3})
    input_df['education_required_encoded'] = input_df['education_required'].map({'Associate': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3})
    input_df.drop(['company_size', 'experience_level', 'education_required'], axis=1, inplace=True)
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    predicted_salary = model.predict(input_df)[0]
    loading_placeholder.empty()
    result_placeholder.success(f"Predicted Yearly AI job salary: ${predicted_salary:,.2f} USD")


st.markdown("""
    <div class="info-box">
        <h3>How the Prediction Works (Model-Based)</h3>
        <p>This application uses a pre-trained machine learning model (<code>ai_salary_model.pkl</code>) to predict the salary based on the input features. The model processes the categorical inputs through ordinal and one-hot encoding to align them with its training data.</p>
        <p class="mt-2">This model has a deviation of $15,000 to $20,000 US Dollars</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="mission-background">
    <div class="mission-card-container mission-card-shape">
        <div class="mission-content">
            <h2>My Mission</h2>
            <p>
                I recognise that individuals may often feel confused or overwhelmed by the
                ever changing job market of AI. This might make them feel unsure of their worth
                and shortchange them of their talents.<br><br>
                My Mission is to empower individuals using data to bring clarity,
                fairness, and confidence to salary expectations in the global AI job market.
                I harness the power of machine learning to uncover real-time, personalized salary
                predictions so that you can stop guessing your worth and start making smarter, more informed
                career decisions.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


st.markdown("""
<div class="how-it-works-section">
    <h2 class="section-title">How It Works</h2>
    <div class="steps">
        <div class="step-card ">
            <div class="step-image">
            <img src="https://i.ibb.co/svzTktJ2/Screenshot-2025-08-01-084431.png" alt="Adjust Parameters">
            </div>
            <div class="step-content">
                <div class="icon-wrapper">✏ </div>
                <h3>Step 1: Set parameters</h3>
                <p>Simply set the job's parameters on the left hand side of the screen to the job you are currently looking at</p>
            </div>
        </div>
""", unsafe_allow_html=True)

st.markdown("""
        <div class="step-card ">
            <div class="step-image">
                <img src="https://i.ibb.co/C4J2nWs/Screenshot-2025-08-01-085150.png" alt="ML Step">
            </div>
            <div class="step-content">
                <div class="icon-wrapper">🧠</div>
                <h3>Step 2: Click the button</h3>
                <p>Just click the "Predict Salary" button. This will trigger our highly skilled and trained model to give you the best prediction results for salary as possible.</p>
            </div>
        </div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

st.markdown("""
        <div class="step-card">
            <div class="step-image">
                <img src="https://i.postimg.cc/zfHkzw6g/Screenshot-2025-08-01-085917.png" alt="Detailed Guidance Step">
            </div>
            <div class="step-content">
                <div class="icon-wrapper">💡</div>
                <h3>Step 3: Prediction Value</h3>
                <p>Receive an accurate prediction value based on the parameters YOU set!</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
