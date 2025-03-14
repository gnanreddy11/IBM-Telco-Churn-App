import streamlit as st
import joblib
import numpy as np

# Load trained XGBoost model
model = joblib.load("final_xgboost_churn_model.pkl")

# Streamlit UI Customization
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Sidebar Navigation
with st.sidebar:
    st.image("https://www.chargebee.com/blog/wp-content/uploads/2023/05/Customer-Retention-Benefits-Your-Business.png", width=250)
    st.title("🔍 Churn Predictor")
    st.markdown("### Use AI to predict customer churn and reduce revenue loss.")

st.title("📊 Customer Churn Prediction App")
st.markdown("#### Enter customer details below to predict whether they will churn or not.")

st.divider()  # Adds a visual separator

# Split UI into Two Columns
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("📅 Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("💵 Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0)
    total_charges = st.number_input("💰 Total Charges ($)", min_value=0.0, max_value=8000.0, value=600.0)

    contract = st.selectbox("📜 Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("🖥️ Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    churn_score = st.number_input("📈 Churn Score", min_value=0, max_value=100, value=50)
    cltv = st.number_input("💎 Customer Lifetime Value (CLTV)", min_value=0, max_value=8000, value=3000)

with col2:
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("🧓 Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("❤️ Partner", ["Yes", "No"])
    dependents = st.selectbox("👶 Dependents", ["Yes", "No"])
    phone_service = st.selectbox("📞 Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("📶 Multiple Lines", ["Yes", "No"])

    internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("🔐 Online Security", ["Yes", "No"])
    online_backup = st.selectbox("💾 Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("🛡️ Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("🛠️ Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("📺 Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("🎬 Streaming Movies", ["Yes", "No"])

st.divider()

# Encoding categorical inputs
binary_mapping = {"No": 0, "Yes": 1}
gender_mapping = {"Male": 1, "Female": 0}
contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_mapping = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}
internet_mapping = {"DSL": 0, "Fiber optic": 1, "No": 2}

# Prepare input data for prediction (Ensure 21 Features)
input_data = np.array([
    tenure, monthly_charges, total_charges, 
    gender_mapping[gender], binary_mapping[senior_citizen], binary_mapping[partner], binary_mapping[dependents], 
    binary_mapping[phone_service], binary_mapping[multiple_lines], internet_mapping[internet_service], 
    binary_mapping[online_security], binary_mapping[online_backup], binary_mapping[device_protection], 
    binary_mapping[tech_support], binary_mapping[streaming_tv], binary_mapping[streaming_movies], 
    contract_mapping[contract], binary_mapping[paperless_billing], payment_mapping[payment_method],
    churn_score, cltv  # 🚀 Added Missing Features!
]).reshape(1, -1)  # Reshape to match model input shape

# Predict Churn
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_data)
    churn_probability = model.predict_proba(input_data)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ This customer is **likely to churn** (Risk: {churn_probability:.2f}%)")
    else:
        st.success(f"✅ This customer is **not likely to churn** (Confidence: {100 - churn_probability:.2f}%)")

st.markdown("---")
st.markdown("🌟 **Built with Streamlit & XGBoost**")
