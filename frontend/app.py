"""
frontend/app.py
Streamlit app for loan default risk prediction.
"""

import streamlit as st
import requests
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------------------------------------------------
# API CONFIG
# ---------------------------------------------------
DEFAULT_API_URL = "https://loan-default-risk-predictor-ifcb.onrender.com"

st.sidebar.title("⚙️ Configuration")

api_url = st.sidebar.text_input(
    "API URL",
    value=DEFAULT_API_URL
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This app sends applicant data to the FastAPI backend "
    "and predicts loan default risk."
)

# ---------------------------------------------------
# API CALL FUNCTION
# ---------------------------------------------------
def predict(applicant_data: dict, url: str):

    payload = {
        "data": applicant_data
    }

    response = requests.post(
        f"{url}/predict",
        json=payload,
        timeout=30
    )

    if response.status_code == 200:
        return response.json()

    raise Exception(
        f"API error ({response.status_code}): {response.text}"
    )

# ---------------------------------------------------
# GAUGE CHART
# ---------------------------------------------------
def display_gauge(score: float):

    color = (
        "green"
        if score < 0.3
        else "orange"
        if score < 0.6
        else "red"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,

            number={
                "suffix": "%",
                "font": {"size": 40}
            },

            gauge={
                "axis": {
                    "range": [0, 100]
                },

                "bar": {
                    "color": color
                },

                "steps": [
                    {
                        "range": [0, 30],
                        "color": "lightgreen"
                    },
                    {
                        "range": [30, 60],
                        "color": "lightyellow"
                    },
                    {
                        "range": [60, 100],
                        "color": "lightcoral"
                    }
                ],

                "threshold": {
                    "line": {
                        "color": "black",
                        "width": 4
                    },
                    "value": score * 100
                }
            },

            title={"text": "Risk Score"}
        )
    )

    fig.update_layout(
        height=350
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("💰 Loan Default Risk Predictor")

st.markdown(
    "Fill applicant details and predict the probability of loan default."
)

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------
col_input, col_result = st.columns([1, 1])

# ---------------------------------------------------
# INPUT FORM
# ---------------------------------------------------
with col_input:

    st.subheader("👤 Applicant Details")

    with st.form("prediction_form"):

        contract_type = st.selectbox(
            "Contract Type",
            ["Cash loans", "Revolving loans"]
        )

        gender = st.selectbox(
            "Gender",
            ["M", "F"]
        )

        car_owner = st.selectbox(
            "Owns a Car",
            ["Y", "N"]
        )

        realty_owner = st.selectbox(
            "Owns Real Estate",
            ["Y", "N"]
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=20,
            value=0
        )

        income = st.number_input(
            "Total Annual Income ($)",
            min_value=0.0,
            value=200000.0
        )

        credit = st.number_input(
            "Credit Amount ($)",
            min_value=0.0,
            value=500000.0
        )

        annuity = st.number_input(
            "Loan Annuity ($)",
            min_value=0.0,
            value=25000.0
        )

        goods_price = st.number_input(
            "Goods Price ($)",
            min_value=0.0,
            value=450000.0
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        days_birth = -age * 365

        unemployed_flag = st.checkbox(
            "Special unemployed value (365243)",
            value=False
        )

        days_employed = (
            365243
            if unemployed_flag
            else st.number_input(
                "Days Employed",
                value=-2000
            )
        )

        income_type = st.selectbox(
            "Income Type",
            [
                "Working",
                "State servant",
                "Commercial associate",
                "Pensioner"
            ]
        )

        education = st.selectbox(
            "Education Level",
            [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher"
            ]
        )

        family_status = st.selectbox(
            "Family Status",
            [
                "Single / not married",
                "Married",
                "Civil marriage"
            ]
        )

        housing = st.selectbox(
            "Housing Type",
            [
                "House / apartment",
                "With parents",
                "Rented apartment"
            ]
        )

        ext_source_2 = st.slider(
            "External Source 2",
            0.0,
            1.0,
            0.5
        )

        ext_source_3 = st.slider(
            "External Source 3",
            0.0,
            1.0,
            0.5
        )

        submitted = st.form_submit_button(
            "🔮 Predict Risk"
        )

# ---------------------------------------------------
# RESULT SECTION
# ---------------------------------------------------
with col_result:

    if submitted:

        with st.spinner("Predicting..."):

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
            }

            try:

                prediction = predict(
                    applicant,
                    api_url
                )

                score = prediction["risk_score"]

                risk_label = prediction["risk_label"]

                st.subheader("📈 Prediction Result")

                # Gauge
                display_gauge(score)

                # Risk label
                label_color = (
                    "green"
                    if risk_label == "Low"
                    else "orange"
                    if risk_label == "Medium"
                    else "red"
                )

                st.markdown(
                    f"""
                    <h2 style='text-align:center;
                    color:{label_color};'>
                    Risk Level: {risk_label}
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("---")

                # SHAP explanations
                if (
                    prediction.get("top_reasons")
                    and len(prediction["top_reasons"]) > 0
                ):

                    st.subheader("🔍 Top 3 Reasons (SHAP)")

                    for reason in prediction["top_reasons"]:

                        direction_emoji = (
                            "🔺"
                            if reason["direction"] == "increases"
                            else "🔻"
                        )

                        st.write(
                            f"{direction_emoji} "
                            f"**{reason['feature']}** "
                            f"({reason['direction']} risk)"
                        )

                        st.progress(
                            min(
                                abs(reason["shap_value"]),
                                1.0
                            )
                        )

                else:

                    st.info(
                        "SHAP explanations are currently unavailable."
                    )

            except Exception as e:

                st.error(
                    f"❌ Prediction failed: {str(e)}"
                )

                st.info(
                    "Check API URL and backend deployment."
                )