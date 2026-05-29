"""
api/main.py
FastAPI application for loan default risk prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    """
    Full applicant data matching the original dataset fields.
    All columns present in the original CSV should be included.
    We'll only process the ones our pipeline needs (others ignored).
    """
    data: Dict[str, Any]  # Key-value pairs of all applicant attributes

    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "SK_ID_CURR": 100001,
                    "AMT_INCOME_TOTAL": 202500.0,
                    "AMT_CREDIT": 406597.5,
                    "AMT_ANNUITY": 24700.5,
                    "AMT_GOODS_PRICE": 351000.0,
                    "NAME_CONTRACT_TYPE": "Cash loans",
                    "CODE_GENDER": "M",
                    "FLAG_OWN_CAR": "N",
                    "FLAG_OWN_REALTY": "Y",
                    "CNT_CHILDREN": 0,
                    "DAYS_BIRTH": -9461,
                    "DAYS_EMPLOYED": 365243,
                    "NAME_INCOME_TYPE": "Working",
                    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                    "NAME_FAMILY_STATUS": "Single / not married",
                    "NAME_HOUSING_TYPE": "House / apartment",
                    "EXT_SOURCE_2": 0.262949,
                    "EXT_SOURCE_3": 0.1393768,
                    # ... add all other fields as needed
                }
            }
        }

class RiskReason(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases" or "decreases" risk

class PredictionResponse(BaseModel):
    applicant_id: Optional[int] = None
    risk_score: float
    risk_label: str
    top_reasons: List[RiskReason]

def load_assets():
    """Load model, preprocessor, SHAP explainer, and feature names at startup."""
    global model_pipeline, explainer, feature_names, preprocessor
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "loan_default_pipeline.pkl")
    explainer_path = os.path.join(BASE_DIR, "models", "shap_explainer.pkl")
    feature_path = os.path.join(BASE_DIR, "models", "feature_names.txt")
    
    # Load model pipeline
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}. Run train.py first.")
    model_pipeline = joblib.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    # Extract preprocessor (used without SMOTE for prediction)
    preprocessor = model_pipeline.named_steps['preprocessor']
    
    # Load SHAP explainer
    if not os.path.exists(explainer_path):
        raise RuntimeError(f"SHAP explainer not found at {explainer_path}. Run explain.py first.")
    explainer = joblib.load(explainer_path)
    print(f"✅ SHAP explainer loaded from {explainer_path}")
    
    # Load feature names
    if not os.path.exists(feature_path):
        raise RuntimeError(f"Feature names file not found at {feature_path}. Run explain.py first.")
    with open(feature_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✅ Feature names loaded ({len(feature_names)} features)")

@app.on_event("startup")
def startup_event():
    """Load assets when the server starts."""
    load_assets()

def preprocess_input(data_dict: dict) -> np.ndarray:
    """
    Convert a single applicant dictionary into a processed numpy array.
    Steps:
      1. Convert to DataFrame (single row)
      2. Apply cleaning (DAYS_EMPLOYED anomaly, drop high-missing columns)
         using the same clean_and_create_features function (without target)
      3. Ensure all original columns are present (missing ones filled with NaN)
      4. Transform using the saved preprocessor
    """
    from src.preprocess import clean_and_create_features
    
    # Create DataFrame from input dict
    df = pd.DataFrame([data_dict])
    
    # The cleaning function expects the full dataframe with target optionally
    # We'll supply a dummy target column to pass through, then remove it
    df_dummy_target = df.copy()
    df_dummy_target['TARGET'] = 0  # dummy value, won't be used
    X_cleaned, _ = clean_and_create_features(df_dummy_target)
    
    # Now X_cleaned has only the features the pipeline expects (without TARGET and ID)
    # But some columns might be missing if the input dict didn't include them.
    # The preprocessor will handle missing columns by imputing or dropping.
    # We'll align columns to match what the preprocessor was fitted on.
    # Get the columns that the preprocessor was fitted on.
    fitted_feature_names = preprocessor.feature_names_in_
    # Ensure X_cleaned has exactly those columns (fill missing with NaN)
    X_cleaned = X_cleaned.reindex(columns=fitted_feature_names, fill_value=np.nan)
    
    # Transform using the preprocessor (impute, scale, one-hot, etc.)
    processed = preprocessor.transform(X_cleaned)
    
    # Get feature names for SHAP
    # (the preprocessor's get_feature_names_out() might differ; we'll use the global feature_names)
    return processed

def get_risk_label(score: float) -> str:
    if score < LOW_THRESHOLD:
        return "Low"
    elif score < HIGH_THRESHOLD:
        return "Medium"
    else:
        return "High"

@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantData):
    """
    Predict loan default risk for a single applicant.
    """
    try:
        data = applicant.data
        
        # Optional: extract applicant ID if present
        applicant_id = data.get('SK_ID_CURR', None)
        
        # Preprocess the raw input
        X_processed = preprocess_input(data)
        
        # Use only the classifier (last step of pipeline) for prediction
        classifier = model_pipeline.named_steps['classifier']
        risk_score = float(classifier.predict_proba(X_processed)[0, 1])
        risk_label = get_risk_label(risk_score)
        
        # SHAP explanation
        shap_values = explainer.shap_values(X_processed)
        # If it returns a list (for multi-class), take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_values = shap_values[0]  # single sample
        
        # Get top 3 absolute SHAP values
        top3_idx = np.argsort(np.abs(shap_values))[-3:][::-1]
        
        top_reasons = []
        for idx in top3_idx:
            feat_name = feature_names[idx]

            feat_name = (
                feat_name
                .replace("num__", "")
                .replace("cat__", "")
            )
            shap_val = shap_values[idx]
            direction = "increases" if shap_val > 0 else "decreases"
            top_reasons.append(RiskReason(
                feature=feat_name,
                shap_value=round(float(shap_val), 4),
                direction=direction
            ))
        
        return PredictionResponse(
            applicant_id=applicant_id,
            risk_score=round(risk_score, 4),
            risk_label=risk_label,
            top_reasons=top_reasons
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)