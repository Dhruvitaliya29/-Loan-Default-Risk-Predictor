"""
api/main.py
FastAPI application for loan default risk prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import joblib
import pandas as pd
import os
import sys
import traceback

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global variables for loaded assets
model_pipeline = None
explainer = None
feature_names = None
preprocessor = None

# Thresholds for risk labels
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.6

app = FastAPI(
    title="Loan Default Risk Predictor",
    description="API for predicting loan default probability with SHAP explanations",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Loan Default Risk Predictor API is running"
    }

class ApplicantData(BaseModel):
    data: Dict[str, Any]

class RiskReason(BaseModel):
    feature: str
    shap_value: float
    direction: str

class PredictionResponse(BaseModel):
    applicant_id: Optional[int] = None
    risk_score: float
    risk_label: str
    top_reasons: List[RiskReason]

def load_assets():
    """Load model, SHAP explainer, and feature names."""

    global model_pipeline, explainer, feature_names, preprocessor

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, "models", "loan_default_pipeline.pkl")
    explainer_path = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
    feature_path = os.path.join(BASE_DIR, "models", "feature_names.txt")

    # Load model
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")

    model_pipeline = joblib.load(model_path)
    print(f"✅ Model loaded")

    # Extract preprocessor
    preprocessor = model_pipeline.named_steps['preprocessor']

    # Load SHAP explainer safely
    try:
        if os.path.exists(explainer_path):
            explainer = joblib.load(explainer_path)
            print("✅ SHAP explainer loaded")
        else:
            explainer = None
            print("⚠️ SHAP explainer file not found")

    except Exception as e:
        explainer = None
        print(f"⚠️ SHAP explainer failed to load: {e}")

    # Load feature names
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ Feature names loaded")
    else:
        feature_names = []

@app.on_event("startup")
def startup_event():
    try:
        load_assets()
        print("✅ Assets loaded successfully")
    except Exception as e:
        print(f"❌ Startup error: {e}")

def preprocess_input(data_dict: dict):

    from src.preprocess import clean_and_create_features

    # Convert input to DataFrame
    df = pd.DataFrame([data_dict])

    # Add dummy target
    df['TARGET'] = 0

    # Clean features
    X_cleaned, _ = clean_and_create_features(df)

    # Match training columns
    fitted_feature_names = preprocessor.feature_names_in_

    X_cleaned = X_cleaned.reindex(
        columns=fitted_feature_names,
        fill_value=np.nan
    )

    # Transform
    processed = preprocessor.transform(X_cleaned)

    return processed

def get_risk_label(score: float):

    if score < LOW_THRESHOLD:
        return "Low"

    elif score < HIGH_THRESHOLD:
        return "Medium"

    return "High"

@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantData):

    try:
        data = applicant.data

        applicant_id = data.get('SK_ID_CURR', None)

        # Preprocess
        X_processed = preprocess_input(data)

        # Predict
        classifier = model_pipeline.named_steps['classifier']

        risk_score = float(
            classifier.predict_proba(X_processed)[0, 1]
        )

        risk_label = get_risk_label(risk_score)

        # Default empty explanations
        top_reasons = []

        # SHAP explanation
        if explainer is not None:

            try:
                shap_values = explainer.shap_values(X_processed)

                # Multi-class handling
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                shap_values = shap_values[0]

                # Top features
                top3_idx = np.argsort(
                    np.abs(shap_values)
                )[-3:][::-1]

                for idx in top3_idx:

                    feat_name = feature_names[idx]

                    feat_name = (
                        feat_name
                        .replace("num__", "")
                        .replace("cat__", "")
                    )

                    shap_val = shap_values[idx]

                    direction = (
                        "increases"
                        if shap_val > 0
                        else "decreases"
                    )

                    top_reasons.append(
                        RiskReason(
                            feature=feat_name,
                            shap_value=round(float(shap_val), 4),
                            direction=direction
                        )
                    )

            except Exception as shap_error:
                print(f"⚠️ SHAP Error: {shap_error}")

        return PredictionResponse(
            applicant_id=applicant_id,
            risk_score=round(risk_score, 4),
            risk_label=risk_label,
            top_reasons=top_reasons
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None
    }

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )