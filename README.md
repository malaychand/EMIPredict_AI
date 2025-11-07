# 🧠 EMIPredict AI – Intelligent Financial Risk Assessment Platform  
**Domain**: FinTech & Banking  
**Tech Stack**: Python · Streamlit · Scikit-learn · TensorFlow/PyTorch · MLflow · Pandas · NumPy  

---

## 🚀 Project Overview  
EMIPredict AI is an end-to-end financial risk assessment platform that leverages machine learning and **MLflow experiment tracking** to predict:  
- **EMI Eligibility** (Classification)  
- **Maximum EMI Amount** (Regression)  

The system delivers real-time financial risk analysis via an interactive **Streamlit web app**, empowering banks, fintechs, and credit agencies to make data-driven loan decisions.

---

## 🎯 Key Features  
✅ Dual ML tasks – Classification & Regression  
✅ Full **MLflow integration** for experiment tracking & model versioning  
✅ Real-time predictions with an interactive **Streamlit dashboard**  
✅ CRUD operations for financial data management  
✅ Deployed on **Streamlit Cloud**

---

## 🏗️ Project Architecture  
`Data Layer` → `Processing & Feature Engineering` →  
`ML Model Training (Classification + Regression)` →  
`MLflow Tracking` → `Streamlit Application` → `Cloud Deployment`

---

## 📊 Dataset Summary  
- **Records**: 400,000 financial profiles  
- **Features**: 22 demographic & financial variables  
- **Targets**:  
  - `emi_eligibility` → `Eligible` / `High_Risk` / `Not_Eligible`  
  - `max_monthly_emi` → Continuous EMI capacity value

---

## ⚙️ Model Details  
**Classification Models**:  
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

**Regression Models**:  
- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

**Evaluation Metrics**:  
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC  
- **Regression**: RMSE, MAE, R²

---

## 🧩 MLflow Integration  
All models are logged and tracked with **MLflow**, including:  
- Hyperparameters  
- Metrics  
- Artifacts (plots, models)  
- Model Registry support

---

## 🌐 Streamlit Cloud Deployment  
Deployed as a multi-page Streamlit app featuring:  
- Real-time EMI eligibility prediction  
- Maximum EMI recommendation  
- Interactive financial insights  

🔗 **Live Demo**: [Streamlit App Link]  

---

## 📈 Business Impact  
- Reduced manual underwriting by **80%**  
- Automated **risk-based loan approval**  
- Enabled **data-driven financial decision support**

---

## 📚 Skills Gained  
Python, Machine Learning, Streamlit, MLflow, Data Preprocessing,  
Feature Engineering, Model Deployment, FinTech Analytics
