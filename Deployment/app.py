import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    """Load all required models and data"""
    try:
        # Load XGBoost model
        model = XGBClassifier()
        model.load_model('xgb_model.json')
        
        # Load other files
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('model_info.pkl', 'rb') as f:
            info = pickle.load(f)
        
        return model, features, encoders, info
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

# Load everything
model, feature_names, label_encoders, model_info = load_models()
st.success("✅ Model loaded successfully!")

# Main title
st.title('🏦 Loan Default Prediction System')
st.markdown('### Predict the likelihood of loan default based on applicant data')

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Train Accuracy", f"{model_info['train_accuracy']:.2%}")
    st.metric("Test Accuracy", f"{model_info['test_accuracy']:.2%}")
    st.markdown("---")
    st.info(f"**Features:** {len(feature_names)}")
    st.info(f"**Model:** {model_info.get('model_type', 'XGBoost')}")
    st.info(f"**Sampling:** {model_info.get('sampling_strategy', 'SMOTE')}")

st.markdown("---")
st.subheader("📝 Enter Loan Application Data")

# Initialize inputs
inputs = {}

# Separate features
categorical_features = list(label_encoders.keys())
if 'Default' in categorical_features:
    categorical_features.remove('Default')

numeric_features = [f for f in feature_names if f not in categorical_features]

# Numeric inputs
if numeric_features:
    st.markdown("#### 🔢 Numeric Information")
    cols = st.columns(3)
    
    for idx, feature in enumerate(numeric_features):
        with cols[idx % 3]:
            inputs[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                help=f"Enter value for {feature}"
            )

# Categorical inputs
if categorical_features:
    st.markdown("---")
    st.markdown("#### 📋 Categorical Information")
    
    cols = st.columns(3)
    for idx, feature in enumerate(categorical_features):
        with cols[idx % 3]:
            possible_values = label_encoders[feature].classes_.tolist()
            possible_values = [v for v in possible_values if str(v) != 'nan']
            
            inputs[feature] = st.selectbox(
                f"{feature}",
                options=possible_values,
                help=f"Select {feature}"
            )

# Prediction button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        '🔍 Predict Loan Default Risk',
        type='primary',
        use_container_width=True
    )

if predict_button:
    try:
        # Prepare input data
        input_data = pd.DataFrame([inputs])
        
        # Encode categorical features
        for col in categorical_features:
            if col in input_data.columns:
                input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Ensure correct order
        input_data = input_data[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Status")
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK")
                st.markdown("**Default Likely**")
            else:
                st.success("### ✅ LOW RISK")
                st.markdown("**Repayment Likely**")
        
        with col2:
            st.markdown("### No Default Probability")
            st.info(f"## {prediction_proba[0]:.1%}")
            st.progress(float(prediction_proba[0]))
        
        with col3:
            st.markdown("### Default Probability")
            st.warning(f"## {prediction_proba[1]:.1%}")
            st.progress(float(prediction_proba[1]))
        
        # Chart
        st.markdown("---")
        st.markdown("### 📈 Probability Distribution")
        
        chart_data = pd.DataFrame({
            'Status': ['No Default', 'Default'],
            'Probability': [prediction_proba[0] * 100, prediction_proba[1] * 100]
        })
        
        st.bar_chart(chart_data.set_index('Status'))
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        if prediction == 1:
            st.warning("""
            **⚠️ Warning - High Risk:**
            - High probability of loan default detected
            - Additional credit assessment recommended
            - May require additional guarantees or collateral
            - Consider rejecting application or offering modified terms
            """)
        else:
            st.success("""
            **✅ Positive Assessment - Low Risk:**
            - Good probability of successful repayment
            - Applicant appears qualified for the loan
            - Can proceed with standard approval procedures
            - Regular monitoring recommended
            """)
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info("Please ensure all fields are filled correctly")

# Footer
st.markdown("---")
with st.expander("ℹ️ About This System"):
    st.markdown(f"""
    ### Loan Default Prediction System
    
    **Model Details:**
    - Algorithm: {model_info.get('model_type', 'XGBoost Classifier')}
    - Imbalance Handling: {model_info.get('sampling_strategy', 'SMOTE')}
    
    **Performance:**
    - Train Accuracy: {model_info['train_accuracy']:.2%}
    - Test Accuracy: {model_info['test_accuracy']:.2%}
    - Features: {len(feature_names)}
    
    **Features Used:**
    {', '.join(feature_names)}
    """)

st.markdown(
    """
    <div style='text-align: center; margin-top: 30px;'>
        <p style='color: #666;'>
            Developed with Streamlit | Powered by XGBoost<br>
            Machine Learning for Financial Risk Assessment
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
