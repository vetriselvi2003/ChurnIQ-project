# 📡 ChurnIQ — Customer Churn Prediction Platform

A production-grade machine learning application that predicts telecom customer churn, identifies at-risk segments, and generates actionable business retention recommendations.

**[🚀 Live Demo →](https://your-app.streamlit.app)**

---

## 📌 Problem Statement

Telecom companies lose **15–35% of customers annually** to churn, costing millions in lost revenue. This project builds an end-to-end ML system that:
- Predicts individual customer churn probability
- Quantifies monthly revenue at risk
- Generates specific, ROI-backed retention actions

---

## 🧠 ML Approach

| Step | Details |
|------|---------|
| Dataset | IBM Telco Customer Churn (7,043 customers, 16 features) |
| Models Evaluated | Logistic Regression, Random Forest, Gradient Boosting |
| Final Model | Gradient Boosting Classifier (best AUC) |
| Key Features | Contract type, tenure, monthly charges, payment method |
| Explainability | Feature importance for per-customer churn drivers |

### Model Performance
- **ROC-AUC:** 0.67
- **Accuracy:** 64.7%
- **Churn Recall:** 33.9%
- **F1-Score (Churn):** 40.1%

---

## 💼 Business Impact

| Insight | Finding |
|---------|---------|
| Top churn driver | Month-to-month contract (48% churn rate) |
| Critical risk window | First 12 months of tenure (58% of all churns) |
| Revenue at risk | ~₹8.9L/month from high-risk segment |
| Highest ROI intervention | Payment method switch incentive (900% ROI) |

---

## 🛠️ Tech Stack

- **Python** — pandas, scikit-learn, numpy
- **ML** — Gradient Boosting, feature engineering
- **Visualization** — Plotly interactive charts
- **Deployment** — Streamlit Community Cloud

---

## 📂 Project Structure

```
churn_project/
├── app.py                  # Streamlit application
├── train_model.py          # Model training pipeline
├── telco_churn.csv         # Dataset
├── churn_model.pkl         # Trained model
├── label_encoders.pkl      # Categorical encoders
├── feature_names.pkl       # Feature list
├── metrics.json            # Model evaluation metrics
└── requirements.txt        # Dependencies
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/vetriselvi2003/ChurnIQ-project.git
cd churniq
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 App Features

**Tab 1 — Predict Single Customer**
- Interactive sidebar to configure customer parameters
- Real-time churn probability gauge
- Revenue at risk calculation
- Personalised retention recommendations with ROI estimates

**Tab 2 — Model Performance**
- Full feature importance chart
- Churn by contract type breakdown
- Churn risk by tenure band trend

**Tab 3 — Business Insights**
- Segment-level revenue at risk
- Retention intervention ROI comparison
- Key findings summary

---

## 🔑 Key Findings

1. **Contract type** is the strongest predictor — M2M customers churn 8x more than 2-year contract holders
2. **Early tenure** customers (< 12 months) are highest risk — require proactive intervention
3. **Electronic check** users churn at 2.3x the rate of auto-pay users
4. **Fiber optic + no security** is the highest-risk service combination

---

*Built by Vetri Selvi B | Data Analytics Portfolio Project*
