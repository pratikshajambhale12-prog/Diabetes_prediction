# Diabetes Risk Prediction Web Application

An interactive Streamlit web application that uses a Machine Learning classification model to predict the likelihood of diabetes based on diagnostic clinical measurements.

## 📊 Project Overview
The goal of this project is to build an end-to-end predictive framework for early diabetes risk assessment. The application accepts clinical features as input (such as Glucose levels, Insulin, and BMI) and instantly outputs a diagnostic risk classification, helping users or healthcare providers catch early warning signs.

## 🛠️ Tech Stack & Advanced Workflow
* **Frontend Interface & Deployment:** Streamlit
* **Language:** Python
* **Machine Learning Framework:** Scikit-Learn (Random Forest Classifier)
* **Data Preprocessing:** * Features normalized using `StandardScaler` to handle magnitude differences.
  * Class imbalance resolved using `SMOTE` (Synthetic Minority Over-sampling Technique) for more reliable predictive power.
* **Model Pipeline:** Saved and loaded using serialized Pickle files (`.pkl`) for real-time inference.

## 💡 Key Features
* **Interactive Form:** Clean, double-column layout for intuitive data entry (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age).
* **Instant Model Inference:** High-speed risk evaluation processed instantly when clicking the "Predict" button.
* **Visual Status Feedback:** Clear, color-coded warning boxes (Red for Positive Risk, Green for Negative Risk) based on model outcomes.

## 💻 How to Run This Project Locally

1. **Clone the repository:**
   '''bash
   git clone https://github.com/pratikshajambhale12-prog/Diabetes_prediction.git
   cd Diabetes_prediction
   '''
