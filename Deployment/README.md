# 🏦 Loan Default Prediction System

A machine learning application for predicting loan defaults using XGBoost.

## 📊 Model Performance

- **Train Accuracy:** 90.67%
- **Test Accuracy:** 88.02%
- **Algorithm:** XGBoost Classifier
- **Features:** 16

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

## 📁 Files

- `xgb_model.json` - Trained XGBoost model
- `feature_names.pkl` - List of feature names
- `label_encoders.pkl` - Encoders for categorical variables
- `model_info.pkl` - Model metadata and statistics
- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies

## 🎯 Features

Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner

## 📝 Usage

1. Enter loan application data in the web interface
2. Click "Predict Loan Default Risk"
3. View prediction results and recommendations

## 🛠️ Technical Details

- **Model:** XGBoost Classifier (n_estimators=100, max_depth=10)
- **Imbalance Handling:** SMOTE (sampling_strategy=0.5)
- **Train/Test Split:** 80/20 with stratification
- **Random State:** 42 (for reproducibility)

## 📄 License

This project is for educational purposes.
