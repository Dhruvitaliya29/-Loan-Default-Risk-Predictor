# 💰 Loan Default Risk Predictor

An end-to-end Machine Learning system that predicts the probability of loan default using applicant financial and personal information from the Home Credit Default Risk dataset.

The project includes:
- Complete ML preprocessing & training pipeline
- XGBoost-based risk prediction
- Explainable AI using SHAP
- FastAPI backend API
- Interactive Streamlit frontend
- Cloud deployment using Render & Streamlit Cloud

---

# 🚀 Project Overview

Financial institutions lose billions annually due to loan defaults. This project builds an explainable ML-powered risk prediction system that can:

✅ Predict loan default probability  
✅ Categorize applicants into Low / Medium / High risk  
✅ Explain predictions using SHAP values  
✅ Serve predictions through a REST API  
✅ Provide a user-friendly interactive dashboard  

This project demonstrates practical ML engineering skills including:
- data preprocessing
- feature engineering
- model deployment
- API development
- frontend/backend integration
- explainable AI

---

# 📊 Dataset

### Home Credit Default Risk Dataset
Source:  
https://www.kaggle.com/competitions/home-credit-default-risk

### Dataset Details

| Attribute | Value |
|---|---|
| Rows | 307,511 |
| Columns | 122 |
| Features After Cleaning | 75 |
| Non-Default Clients | 282,686 |
| Default Clients | 24,825 |
| Imbalance Ratio | 11.39 : 1 |

### Important EDA Findings

- `FLOORSMIN_MEDI` contains ~67.85% missing values
- `YEARS_BUILD_AVG` contains ~66.50% missing values
- `YEARS_BUILD_MODE` contains ~66.50% missing values
- External credit scores (`EXT_SOURCE_1/2/3`) are highly predictive
- Older applicants tend to default less frequently
- `DAYS_EMPLOYED = 365243` represents unemployed/retired applicants

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalanced Data Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Deployment | Render, Streamlit Cloud |
| Version Control | Git, GitHub |

---

# 📁 Project Structure

```bash
Loan-Default-Risk-Predictor/
│
├── api/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── loan_default_pipeline.pkl
│   ├── feature_names.txt
│
├── notebooks/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── explain.py
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Dhruvitaliya29/-Loan-Default-Risk-Predictor.git
cd -Loan-Default-Risk-Predictor
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Download Dataset

Download dataset from Kaggle:

https://www.kaggle.com/competitions/home-credit-default-risk

Place:
```bash
application_train.csv
```

inside your dataset folder.

---

# 🧠 Machine Learning Pipeline

The ML workflow includes:

### ✅ Data Cleaning
- Missing value handling
- Outlier/anomaly correction
- Feature alignment
- Dropped 49 columns with >40% missing values

### ✅ Feature Engineering
- Ratio-based features
- Categorical encoding
- Scaling & normalization

### ✅ Imbalanced Data Handling
- SMOTE oversampling

### ✅ Model Training
Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost

Final selected model:
✅ XGBoost

---

# 📈 Model Performance

## Initial XGBoost Model

| Metric | Score |
|---|---|
| Accuracy | 91.94% |
| Precision | 0.5135 |
| Recall | 0.0230 |
| F1-Score | 0.0440 |
| ROC-AUC | 0.7548 |
| AUC-PR | 0.2380 |

---

# 🚀 Improved Model Performance

| Metric | Before | After |
|---|---|---|
| F1-Score | 0.044 | 0.292 |
| Recall | 0.023 | 0.327 |
| Accuracy | 0.919 | 0.872 |
| ROC-AUC | 0.755 | 0.755 |

### 🔥 Key Improvement

The model was optimized for better minority-class detection (loan defaults).

Although overall accuracy decreased slightly, recall and F1-score improved significantly, making the system much more practical for financial risk prediction.

This improvement demonstrates the importance of handling class imbalance correctly instead of relying only on raw accuracy.

---

# 🔍 Explainable AI with SHAP

This project uses SHAP (SHapley Additive exPlanations) to generate interpretable predictions.

For every applicant:
- Top 3 important features are shown
- Risk increasing/decreasing factors are highlighted
- Predictions become transparent instead of black-box outputs

Example:
- Low external credit score → increases risk
- High income → decreases risk

---

# ⚡ FastAPI Backend

The backend is built using FastAPI.

### Main Endpoint

```http
POST /predict
```

### Example Response

```json
{
  "risk_score": 0.72,
  "risk_label": "High",
  "top_reasons": [
    {
      "feature": "EXT_SOURCE_2",
      "direction": "increases",
      "shap_value": 0.42
    }
  ]
}
```

---

# 🎨 Streamlit Frontend

The frontend dashboard allows users to:
- Enter applicant details
- Predict loan default risk
- View risk gauge visualization
- View SHAP explanations interactively

---

# ☁️ Deployment

## Backend Deployment
Platform: Render

Features:
- FastAPI REST API
- Public prediction endpoint
- Real-time model inference

---

## Frontend Deployment
Platform: Streamlit Community Cloud

Features:
- Interactive UI
- Live API integration
- Responsive dashboard

---

# 📸 Demo Features

✅ Low risk prediction  
✅ High risk prediction  
✅ Interactive risk gauge  
✅ SHAP explanations  
✅ Real-time API predictions  
✅ End-to-end deployed ML workflow  

---

# 💡 What I Learned

Through this project I learned:
- End-to-end ML system development
- Explainable AI workflows
- API deployment using FastAPI
- Frontend/backend integration
- Handling real-world deployment/debugging issues
- Cloud deployment workflows
- Importance of recall-focused optimization for imbalanced datasets

---

# 🔗 GitHub Repository

:contentReference[oaicite:0]{index=0}

---

# 🙌 Future Improvements

- Improve recall performance further
- Add authentication system
- Add database integration
- Add monitoring & logging
- Dockerize the project
- CI/CD integration

---

# 🤝 Contributing

Contributions, suggestions, and feedback are welcome.

Feel free to fork the repository and create pull requests.

---

# 📄 License

This project is for educational and portfolio purposes.

---

# 👨‍💻 Author

Dhruv Italiya

If you found this project interesting, feel free to connect or provide feedback.