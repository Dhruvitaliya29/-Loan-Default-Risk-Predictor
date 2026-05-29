
"""
train.py
Train and compare ML models for loan default prediction
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_curve
)

import joblib
import os
import sys
import time

# local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import (
    prepare_data,
    build_preprocessing_pipeline,
    build_training_pipeline
)

# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def train_and_compare():

    print("=" * 60)
    print("📦 Loading and preparing data...")

    X_train, X_test, y_train, y_test = prepare_data()

    preprocessor = build_preprocessing_pipeline()

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------
    models = {

        "Logistic Regression": LogisticRegression(
            solver='liblinear',
            random_state=42,
            max_iter=1000,
            C=0.1
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
    }

    results = {}

    roc_curves = {}
    pr_curves = {}

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for name, model in models.items():

        print(f"\n{'='*40}")
        print(f"🔧 Training {name}...")

        start = time.time()

        pipeline = build_training_pipeline(
            model,
            preprocessor
        )

        pipeline.fit(X_train, y_train)

        train_time = time.time() - start

        print(f"✅ Training completed in {train_time:.1f} sec")

        # Predict probabilities
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Custom threshold tuning
        threshold = 0.20

        # Convert probabilities into predictions
        y_pred = (y_proba >= threshold).astype(int)

        # metrics
        f1 = f1_score(y_test, y_pred)

        roc_auc = roc_auc_score(y_test, y_proba)

        precision, recall, _ = precision_recall_curve(
            y_test,
            y_proba
        )

        pr_auc = auc(recall, precision)

        results[name] = {
            "F1-Score": f1,
            "AUC-ROC": roc_auc,
            "AUC-PR": pr_auc,
            "Pipeline": pipeline
        }

        # curves
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        roc_curves[name] = (fpr, tpr, roc_auc)

        pr_curves[name] = (recall, precision, pr_auc)

        # output
        print("\n📊 Classification Report:")
        print(classification_report(
            y_test,
            y_pred,
            digits=4
        ))

        print(f"F1-Score : {f1:.4f}")
        print(f"AUC-ROC  : {roc_auc:.4f}")
        print(f"AUC-PR   : {pr_auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)

        print("\nConfusion Matrix:")
        print(cm)

    # ---------------------------------------------------------
    # Results summary
    # ---------------------------------------------------------
    summary_df = pd.DataFrame(results).T.drop(
        columns=['Pipeline']
    )

    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON")
    print(summary_df)

    # ---------------------------------------------------------
    # ROC CURVES
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for name, (fpr, tpr, auc_val) in roc_curves.items():

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc_val:.3f})"
        )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_DIR, "07_roc_curves.png"),
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # PR CURVES
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for name, (recall, precision, pr_auc) in pr_curves.items():

        plt.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC={pr_auc:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_DIR, "08_pr_curves.png"),
        dpi=150
    )

    plt.close()

    print("\n✅ ROC and PR curves saved.")

    # ---------------------------------------------------------
    # Save best model
    # ---------------------------------------------------------
    best_model_name = summary_df["AUC-ROC"].idxmax()

    best_pipeline = results[best_model_name]["Pipeline"]

    save_path = os.path.join(
        MODEL_DIR,
        "loan_default_pipeline.pkl"
    )

    joblib.dump(best_pipeline, save_path)

    print(f"\n✅ Best model saved: {best_model_name}")

    print(f"📁 Path: {save_path}")

    return best_pipeline


if __name__ == "__main__":

    train_and_compare()

