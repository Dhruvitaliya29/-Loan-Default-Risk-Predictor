# 💰 Loan Default Risk Predictor

An end-to-end Machine Learning system that predicts the probability of loan default using applicant financial and personal information from the Home Credit Default Risk dataset.

The project includes:
- A complete ML preprocessing & training pipeline
- XGBoost-based risk prediction
- Explainable AI using SHAP
- FastAPI backend API
- Interactive Streamlit frontend
- Cloud deployment using Render & Streamlit Cloud

---

## 🚀 Project Overview

Financial institutions face major losses due to loan defaults. This project aims to build an explainable ML-powered risk prediction system that can:

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
- File used: `application_train.csv`
- Rows: 307,511
- Columns: 122
- Target variable: `TARGET`
  - `1` → Loan Default
  - `0` → No Default

### Key Insights from EDA
- External credit scores (`EXT_SOURCE_1/2/3`) are highly predictive
- Older applicants tend to default less
- `DAYS_EMPLOYED = 365243` represents unemployed/retired applicants
- The dataset is highly imbalanced (~8% defaults)

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

Download from Kaggle:

https://www.kaggle.com/competitions/home-credit-default-risk

Place:
```bash
application_train.csv
```

inside your dataset/data folder.

---

# 🧠 Machine Learning Pipeline

The ML pipeline includes:

### ✅ Data Cleaning
- Missing value handling
- Anomaly correction
- Feature alignment

### ✅ Feature Engineering
- Ratio-based features
- Categorical encoding
- Scaling & normalization

### ✅ Imbalanced Data Handling
- SMOTE oversampling

### ✅ Model Training
Models tested:
- Logistic Regression
- Random Forest
- XGBoost

Final selected model:
✅ XGBoost

---

# 📈 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.76 |
| Accuracy | ~92% |
| Recall | ~65% |
| F1 Score | ~0.32 |

The model was optimized for recall-focused financial risk prediction.

---

# 🔍 Explainable AI with SHAP

This project uses SHAP (SHapley Additive exPlanations) to provide interpretable predictions.

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
- Public endpoint
- Real-time predictions

---

## Frontend Deployment
Platform: Streamlit Community Cloud

Features:
- Interactive UI
- Live API integration
- Responsive dashboard

---

# 📸 Demo

### Features Demonstrated
✅ Low risk prediction  
✅ High risk prediction  
✅ Interactive risk gauge  
✅ SHAP explanations  
✅ Real-time API predictions  

---

# 💡 What I Learned

Through this project I learned:
- End-to-end ML system development
- Explainable AI workflows
- API deployment using FastAPI
- Frontend/backend integration
- Handling real-world deployment/debugging issues
- Cloud deployment workflows

---

# 🔗 GitHub Repository

https://github.com/Dhruvitaliya29/-Loan-Default-Risk-Predictor

---

# 🙌 Future Improvements

- Add authentication system
- Add database integration
- Improve model calibration
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