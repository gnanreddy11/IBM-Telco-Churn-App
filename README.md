# 📊 Customer Churn Prediction – IBM Telco Data  

## 📖 Overview  
This project predicts **customer churn** for a telecom company using **machine learning**. It analyzes key customer attributes to determine if they are likely to **stay or leave**, helping businesses improve **customer retention strategies**.  

An **interactive Streamlit app** allows real-time predictions based on user inputs, providing insights into churn risks.  

## 🚀 Features  
✔ **Exploratory Data Analysis (EDA)** to identify churn patterns  
✔ **Trained multiple ML models** (Logistic Regression, Random Forest, XGBoost)  
✔ **Optimized XGBoost to 93.3% accuracy** with **hyperparameter tuning**  
✔ **Built & deployed a Streamlit web app** for real-time predictions  
✔ **Modern UI with a sidebar & interactive inputs**  

## 🔬 Dataset Details  
- **Source**: IBM Telco Customer Churn Dataset  
- **Size**: **7,043 customers**  
- **Target Variable**: `Churn Label` (Yes/No)  
- **Features**: **21+ customer attributes**, including contract type, payment method, tenure, and monthly charges  

## 🏗 Tech Stack  
- **Programming**: Python  
- **Libraries**: Pandas, NumPy, Scikit-Learn, XGBoost, Streamlit  
- **Deployment**: Streamlit Cloud  

## 🎯 How to Run Locally  
1️⃣ Clone the repository:  
```bash
git clone https://github.com/gnanreddy11/IBM-Telco-Churn-App.git
cd IBM-Telco-Churn-App

2️⃣ Install dependencies:
pip install -r requirements.txt

3️⃣ Run the Streamlit app:
streamlit run streamlit_app.py

4️⃣ Open http://localhost:8501 in your browser

🌍 Live Deployment
🔗 https://ibm-telco-churn-app-smktsm8agsd7crhndl6xkn.streamlit.app/
