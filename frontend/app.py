"""
frontend/app.py
Streamlit app for loan default risk prediction.
Connects to the FastAPI backend (local or deployed).
"""

import streamlit as st
import requests
import json
import pandas as pd

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💰",
    layout="wide"
)

# Default API URL (update when deploying)
DEFAULT_API_URL = "http://localhost:8001"

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
api_url = st.sidebar.text_input(
    "API URL",
    value=DEFAULT_API_URL,
    help="URL of the FastAPI prediction endpoint (e.g., http://localhost:8000 or your Render URL)"
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This app sends applicant data to the FastAPI backend and displays the predicted "
    "risk of loan default along with top 3 driving factors (SHAP)."
)

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def predict(applicant_data: dict, url: str) -> dict:
    """
    Send applicant data to the /predict endpoint and return the JSON response.
    """
    payload = {"data": applicant_data}
    response = requests.post(f"{url}/predict", json=payload, timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error ({response.status_code}): {response.text}")

def display_gauge(score: float):
    """Display a custom radial gauge using Plotly."""
    import plotly.graph_objects as go

    color = "green" if score < 0.3 else "orange" if score < 0.6 else "red"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 60], "color": "lightyellow"},
                {"range": [60, 100], "color": "lightcoral"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score * 100
            }
        },
        title={"text": "Risk Score"}
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------------
st.title("💰 Loan Default Risk Predictor")
st.markdown("Fill in the applicant details below and click **Predict** to assess the risk of default.")

# Split into two columns: input and result
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("👤 Applicant Details")
    
    with st.form("applicant_form"):
        # Personal & financial
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        gender = st.selectbox("Gender", ["M", "F"])
        car_owner = st.selectbox("Owns a Car", ["Y", "N"])
        realty_owner = st.selectbox("Owns Real Estate", ["Y", "N"])
        children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
        income = st.number_input("Total Annual Income ($)", min_value=0.0, value=200000.0, step=1000.0)
        credit = st.number_input("Credit Amount ($)", min_value=0.0, value=500000.0, step=1000.0)
        annuity = st.number_input("Loan Annuity ($)", min_value=0.0, value=25000.0, step=100.0)
        goods_price = st.number_input("Goods Price ($)", min_value=0.0, value=450000.0, step=100.0)
        
        # Age (we'll convert to DAYS_BIRTH for the API)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        # Convert to DAYS_BIRTH (negative number of days)
        days_birth = -age * 365
        
        # Employment
        days_employed_placeholder = st.checkbox(
            "DAYS_EMPLOYED anomaly (unemployed/retired)?",
            value=False,
            help="Check if the applicant has the special DAYS_EMPLOYED value 365243"
        )
        days_employed = 365243 if days_employed_placeholder else st.number_input(
            "Days Employed (negative, e.g., -2000 for ~5.5 years)", value=-2000
        )
        
        income_type = st.selectbox(
            "Income Type",
            ["Working", "State servant", "Commercial associate", "Pensioner", "Unemployed", "Student", "Businessman", "Maternity leave"]
        )
        education = st.selectbox(
            "Education Level",
            ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
        )
        family_status = st.selectbox(
            "Family Status",
            ["Single / not married", "Married", "Civil marriage", "Widow", "Separated", "Unknown"]
        )
        housing = st.selectbox(
            "Housing Type",
            ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"]
        )
        
        # External scores (very predictive)
        ext_source_2 = st.slider("External Source 2 Score", 0.0, 1.0, 0.5, step=0.01)
        ext_source_3 = st.slider("External Source 3 Score", 0.0, 1.0, 0.5, step=0.01)
        
        submitted = st.form_submit_button("🔮 Predict Risk")

# ------------------------------------------------------------------
# ON PREDICT
# ------------------------------------------------------------------
with col_result:
    if submitted:
        with st.spinner("Calling prediction API..."):
            # Construct full applicant data (only known features; rest will be NaN)
            applicant = {
                "NAME_CONTRACT_TYPE": contract_type,
                "CODE_GENDER": gender,
                "FLAG_OWN_CAR": car_owner,
                "FLAG_OWN_REALTY": realty_owner,
                "CNT_CHILDREN": children,
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods_price,
                "DAYS_BIRTH": days_birth,
                "DAYS_EMPLOYED": days_employed,
                "NAME_INCOME_TYPE": income_type,
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family_status,
                "NAME_HOUSING_TYPE": housing,
                "EXT_SOURCE_2": ext_source_2,
                "EXT_SOURCE_3": ext_source_3
                # SK_ID_CURR will be autogenerated by the API as None if missing
            }
            
            try:
                result = predict(applicant, api_url)
                
                st.subheader("📈 Prediction Result")
                # Risk score and gauge
                score = result["risk_score"]
                risk_label = result["risk_label"]
                
                # Gauge
                display_gauge(score)
                
                # Label with color
                label_color = (
                    "green" if risk_label == "Low" else
                    "orange" if risk_label == "Medium" else "red"
                )
                st.markdown(
                    f"<h3 style='text-align: center; color: {label_color};'>Risk Level: {risk_label}</h3>",
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                st.subheader("🔍 Top 3 Reasons (SHAP)")
                # Display each reason with arrow and value
                for i, reason in enumerate(result["top_reasons"]):
                    feat = reason["feature"]
                    # Make feature name more readable (remove prefixes like num__ or cat__)
                    clean_feat = feat.split("__")[-1] if "__" in feat else feat
                    direction = reason["direction"]
                    shap_val = reason["shap_value"]
                    arrow = "⬆️ increases" if direction == "increases" else "⬇️ decreases"
                    emoji = "🔴" if direction == "increases" else "🟢"
                    st.write(f"**{i+1}. {clean_feat}** {emoji} {arrow} risk (|SHAP|={abs(shap_val):.3f})")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Make sure the FastAPI server is running and the API URL is correct.")