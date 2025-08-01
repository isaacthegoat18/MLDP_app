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


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');



/* Right side container */
[data-testid="stAppViewContainer"] {
    background: url("https://www.aihr.com/wp-content/uploads/salary-benchmarking-cover-image.png") no-repeat center center fixed;
    background-size: cover;         
    border-radius: 1rem;
    margin: 0.4rem auto;
    padding: 0;
    max-width: 1400px;
    display: flex;
    flex-direction: column;
    min-height: calc(100vh - 4rem);
}
            
    [data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    backdrop-filter: blur(6px);
    background-color: rgba(0, 0, 0, 0.7); /* Adjust darkness */
    z-index: 0;
}
            
@media (min-width: 768px) {
    [data-testid="stAppViewContainer"] {
        flex-direction: row;
    }
}



/* Main area */
/*[data-testid="stAppViewContainer"] > .main {
    padding: 0rem;
    background-color: #2d3748;
    border-radius: 0 0 1rem 1rem;
}
@media (min-width: 768px) {
    [data-testid="stAppViewContainer"] > .main {
        width: 55%;
        border-radius: 0 1rem 1rem 0;
    }
}*/
            

/* Text and headers */
h1, h2, h3, h4, h5, h6 {
    color: #0080ff;
}


/* Info box */
.info-box {
    background-color: rgba(45, 55, 72, 0.9);
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin-top: 2rem;
    color: #cbd5e0;
}
.info-box p{
            color:#fff
}

.info-box h3 {
    margin-bottom: 1rem;
    color: #e2e8f0;
}

/* Hide Streamlit header/footer */
#MainMenu, footer, header {
    visibility: hidden;
}
            
/* Mission Background */
.mission-background {
    padding: 2px 4px;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}

.mission-card-container {
    position: relative;
    max-width: 896px;
    margin: auto;
    padding: 32px;
    background-color: rgba(45, 55, 72, 0.9);
    border-radius: 0.75rem;
    box-shadow: 0 20px 25px -5px rgba(0,0,0,0.2),
                0 10px 10px -5px rgba(0,0,0,0.04);
    z-index: 10;
    overflow: hidden;
}

.mission-card-container::before {
    content: '';
    position: absolute;
    top: -20px;
    right: -20px;
    width: 80px;
    height: 80px;
    background-color: blueviolet;
    border-radius: 20px;
    transform: rotate(15deg);
    opacity: 0.3;
    z-index: 0;
}

.mission-card-container::after {
    content: '';
    position: absolute;
    bottom: -20px;
    left: -20px;
    width: 100px;
    height: 100px;
    background-color: #EF4444;
    border-radius: 50%;
    opacity: 0.2;
    z-index: 0;
}

.mission-content {
    position: relative;
    text-align: center;
    z-index: 1;
    color: white;
}

.mission-content h2 {
    font-size: 2.25rem;
    font-weight: 800;
    margin-bottom: 32px;
}

.mission-content p {
    font-size: 1.125rem;
    color: #fff;
    line-height: 1.625;
}

/* How It Works Section */
.how-it-works-section {
   
    padding: 2rem 1rem;
    color: white;
}

.section-title {
    font-size: 2.25rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 3rem;
}

.steps {
    display: flex;
    flex-direction: column;
    gap: 3rem;
}

.step-card {
    max-width: 850px;
    margin: auto;
    background-color: #2d3748;
    color: #1f2937;
    border-radius: 0.5rem;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1),
                0 4px 6px -2px rgba(0,0,0,0.05);
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 2rem;
    border: 1px solid #e5e7eb;
}
            


@media (min-width: 768px) {
    .step-card {
        flex-direction: row;
    }
    .step-card.reverse {
        flex-direction: row-reverse;
    }
}

.step-image img {
    width: 100%;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.step-content {
    flex: 1;
    text-align: left;
}

.icon-wrapper {
    width: 4rem;
    height: 4rem;
    border-radius: 9999px;
    background-color: #4f46e5;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.875rem;
    margin-bottom: 1rem;
}

.step-content h3 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.step-content p {
    color: #fff;
    line-height: 1.625;
}

.feature-list {
    list-style: none;
    padding-left: 0;
    font-size: 1rem;
    color: #374151;
}
.feature-list li {
    margin-bottom: 0.5rem;
}

 .section-gap {
    height: 1.5rem; 
}   
            

@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        overflow-y: auto;
        max-height: 100vh;
        padding-right: 1rem;
    }
}      
            
</style>
""", unsafe_allow_html=True)



model = joblib.load("ai_salary_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("💰AI Job Salary Predictor")
st.markdown("Use the sidebar to select the parameters for salary prediction. The predicted salary will appear below.")

result_placeholder = st.empty()

loading_placeholder = st.empty()

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

    job_title = st.selectbox("Job Title", ['AI Architect', 'AI Consultant', 'AI Product Manager', 'AI Research Scientist', 'AI Software Engineer', 'AI Specialist', 'Autonomous Systems Engineer', 'Computer Vision Engineer', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Deep Learning Engineer','Head of AI', 'Machine Learning Engineer', 'Machine Learning Researcher' ,'ML Ops Engineer', 'NLP Engineer', 'Principal Data Scientist', 'Research Scientist', 'Robotics Engineer'], index=0,help="Select the name of the job") # Default to NLP Engineer
    
    employment_display = st.selectbox("Employment Type", list(employment_options.keys()),index=0,help='Select the employment type of the job')
    employment_type = employment_options[employment_display] 
    
    company_location = st.selectbox("Company Location", ['Australia', 'Austria', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'India', 'Ireland', 'Israel', 'Japan' ,'Netherlands' ,'Norway' ,'Singapore' ,'South Korea', 'Sweden', 'Switzerland',' United Kingdom' ,'United States'], index=0,help='Select the country location of the company') # Default to Switzerland
    
    employee_residence = st.selectbox("Employee Residence", ['Australia', 'Austria', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'India', 'Ireland', 'Israel', 'Japan' ,'Netherlands' ,'Norway' ,'Singapore' ,'South Korea', 'Sweden', 'Switzerland',' United Kingdom' ,'United States'], index=0,help='Select the location that you are residing in') # Default to India
    
    remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, value=0, step=50,help='Select how much does the company work remotely: 0 (No remote), 50 (Hybrid), 100 (Fully remote)') 
    
    number_of_req_skills = st.number_input("Number of Required Skills", 0, 20, value=4, step=1,help='Select the number of required skills for the job')
    
    years_experience = st.number_input("Years of Experience", 0.0, 50.0, value=1.0, step=0.5,help='Enter the amount of years of experience for the job')
    
    industry = st.selectbox("Industry", ['Automotive', 'Consulting', 'Education', 'Energy', 'Finance', 'Gaming', 'Government', 'Healthcare', 'Manufacturing', 'Media', 'Real Estate', 'Retail', 'Technology', 'Telecommunications', 'Transportation'], index=0,help='Select the industry of the company') 
    
    benefits_score = st.slider("Benefits Score (0-10)", 0, 10, value=2, step=1,help='Enter the benefit score of the job')
    
    num_employees = st.number_input("Number of Employees", min_value=1, max_value=100000, value=1, step=1,help='Enter the amount of employees the company has')

    experience_label = st.selectbox("Experience Level", list(experience_level_options.keys()), index=0,help='Select the experience level of the job required')
    experience_level = experience_level_options[experience_label]
    
    education_required = st.selectbox("Education Level", ['Associate', 'Bachelor', 'Master', 'PhD'], index=0,help='Select the education level required for the job') 

    predict_button = st.button("Predict Salary")

# --- Prediction Logic ---
# Number of employees
if num_employees < 50:
    company_size = 'S'
elif num_employees < 250:
    company_size = 'M'
else:
    company_size = 'L'

if predict_button:
    # Clear previous results and show loading spinner
    result_placeholder.empty()
    with loading_placeholder.container():
        st.spinner("Predicting salary...")
        time.sleep(1.5) # Simulate prediction time

    # Input as DataFrame
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
                <div class="icon-wrapper">✏    </div>
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
