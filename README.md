# 🏦 Loan Default Prediction System

This project is a **Machine Learning** application designed to predict the likelihood of a borrower defaulting on a loan. It uses an **XGBoost Classifier** model and is deployed as an interactive web dashboard using **Streamlit**.

---

## 📊 Project Overview
The system analyzes applicant data (like credit score, income, and loan amount) to assess risk. To handle the data imbalance in the dataset, **SMOTE** (Synthetic Minority Over-sampling Technique) was used during training to ensure the model can accurately identify potential defaulters.

## 🚀 Key Features
* **Real-time Prediction:** Enter applicant details and get an instant risk assessment.
* **Interactive UI:** Clean and simple dashboard built with Streamlit.
* **High Performance:** Achieved ~88.02% accuracy on test data.
* **Risk Insights:** Provides recommendations based on the prediction (Low Risk vs. High Risk).

## 📁 Project Structure
* `app.py`: The main Streamlit application script.
* `xgb_model.json`: The trained XGBoost model.
* `loan_default_prediction.ipynb`: Jupyter notebook containing data analysis and model training.
* `feature_names.pkl` & `label_encoders.pkl`: Preprocessing files for data transformation.
* `requirements.txt`: List of necessary Python libraries.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Library:** XGBoost, Scikit-learn
* **Data Handling:** Pandas, Numpy
* **Imbalance Handling:** Imbalanced-learn (SMOTE)
* **Deployment:** Streamlit

## 📈 Model Performance
* **Train Accuracy:** 90.67%
* **Test Accuracy:** 88.02%
* **Algorithm:** XGBoost Classifier
* **Features Used:** 16 variables (Age, Income, Credit Score, Loan Purpose, etc.)

---
