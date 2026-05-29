# 💰 Loan Default Risk Predictor

A production‑ready machine learning system that predicts the probability of loan default using application‑level data from the **Home Credit Default Risk** competition. The project includes a full preprocessing pipeline, model comparison, SHAP explainability, a FastAPI REST backend, and an interactive Streamlit frontend.

![Streamlit UI](https://via.placeholder.com/800x400?text=Streamlit+Loan+Default+Risk+Predictor)

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create a virtual environment](#2-create-a-virtual-environment)
  - [3. Download the dataset](#3-download-the-dataset)
  - [4. Run the pipeline](#4-run-the-pipeline)
- [Usage](#usage)
  - [Running the API locally](#running-the-api-locally)
  - [Running the Streamlit frontend](#running-the-streamlit-frontend)
- [Deployment](#deployment)
  - [FastAPI on Render](#fastapi-on-render)
  - [Streamlit on Streamlit Cloud](#streamlit-on-streamlit-cloud)
- [Model & Methodology](#model--methodology)
- [Explainability (SHAP)](#explainability-shap)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Project Overview

Financial institutions lose billions annually due to loan defaults. This project builds an **explainable ML system** that:
- Ingests applicant data (income, credit history, external scores, etc.)
- Outputs a **risk score** (0–1) and a **risk label** (Low / Medium / High)
- Provides **top 3 reasons** behind the decision using SHAP values

The system is designed as a **showcase for ML Engineering / Data Science internships**, demonstrating end‑to‑end skills: data cleaning, feature engineering, model selection, deployment, and explainability.

---

## 📊 Dataset

- **Home Credit Default Risk** – [Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk)
- **File used:** `application_train.csv` (307,511 rows × 122 columns)
- **Target:** `TARGET` = 1 if the client defaulted, 0 otherwise.
- **Imbalance:** ~8% default rate.

Key insights from EDA:
- External credit scores (`EXT_SOURCE_1/2/3`) are the strongest predictors.
- Older clients default less.
- `DAYS_EMPLOYED` contains a placeholder value (365243) for unemployed/retired applicants.

---

## 🛠 Tech Stack

| Category         | Tools / Libraries |
|------------------|-------------------|
| Data Processing  | Pandas, NumPy |
| Visualization    | Matplotlib, Seaborn |
| Modeling         | Scikit‑learn, XGBoost |
| Imbalance Handling | imbalanced‑learn (SMOTE) |
| Explainability   | SHAP |
| Backend API      | FastAPI, Uvicorn |
| Frontend         | Streamlit, Plotly |
| Deployment       | Render, Streamlit Cloud |
| Version Control  | Git, Git LFS |

---

## 📁 Project Structure
