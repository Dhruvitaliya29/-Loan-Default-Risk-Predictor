import os
import sys
import joblib
import shap
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import prepare_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "loan_default_pipeline.pkl")

print("📦 Loading trained pipeline...")
pipeline = joblib.load(MODEL_PATH)

print("📦 Preparing sample training data...")
X_train, X_test, y_train, y_test = prepare_data()

# Get preprocessor and classifier
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']

# Transform sample data
X_processed = preprocessor.transform(X_train.iloc[:1000])

print("🧠 Creating SHAP explainer...")

# TreeExplainer for XGBoost
explainer = shap.TreeExplainer(classifier)

# Save explainer
explainer_path = os.path.join(MODEL_DIR, "shap_explainer.pkl")
joblib.dump(explainer, explainer_path)

print(f"✅ SHAP explainer saved at:")
print(explainer_path)

# Save feature names
feature_names = preprocessor.get_feature_names_out()

feature_path = os.path.join(MODEL_DIR, "feature_names.txt")

with open(feature_path, "w") as f:
    for feat in feature_names:
        f.write(str(feat) + "\n")

print(f"✅ Feature names saved at:")
print(feature_path)

print("🎉 Explainability assets generated successfully.")