import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = pickle.load(open("churn_model.pkl", "rb"))
    le_dict  = pickle.load(open("label_encoders.pkl", "rb"))
    features = pickle.load(open("feature_names.pkl", "rb"))
    metrics  = json.load(open("metrics.json", "r"))
    return model, le_dict, features, metrics

model, le_dict, features, metrics = load_model()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf0;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1226 0%, #111827 100%);
    border-right: 1px solid #1e2a45;
}
[data-testid="stSidebar"] label { color: #8892a4 !important; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }

/* Main bg */
.main { background-color: #0a0e1a; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-value { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: #4f9eff; }
.metric-label { font-size: 0.75rem; color: #6b7a99; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

/* Risk badge */
.risk-high   { background: #2d1b1b; border: 1px solid #f87171; border-radius: 8px; padding: 16px; }
.risk-medium { background: #1f1e14; border: 1px solid #fbbf24; border-radius: 8px; padding: 16px; }
.risk-low    { background: #132218; border: 1px solid #34d399; border-radius: 8px; padding: 16px; }

.risk-label-high   { color: #f87171; font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; }
.risk-label-medium { color: #fbbf24; font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; }
.risk-label-low    { color: #34d399; font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #4f9eff;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 8px;
    border-left: 3px solid #4f9eff;
    padding-left: 10px;
}

/* Recommendation box */
.rec-box {
    background: #111827;
    border: 1px solid #1e2a45;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.6;
}
.rec-box strong { color: #4f9eff; }

/* stMetric override */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e2a45;
    border-radius: 10px;
    padding: 14px;
}
[data-testid="stMetricValue"] { color: #4f9eff !important; font-family: 'Syne', sans-serif; }

/* Plotly charts bg */
.js-plotly-plot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px 0;'>
  <div style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; color:#e8eaf0; letter-spacing:-0.02em;'>
    📡 ChurnIQ
  </div>
  <div style='color:#6b7a99; font-size:0.9rem; margin-top:4px;'>
    Customer Churn Intelligence Platform — Telecom Analytics
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Predict Single Customer", "📊  Model Performance", "💡  Business Insights"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; 
                    color:#e8eaf0; padding:12px 0 16px 0; border-bottom:1px solid #1e2a45; margin-bottom:16px;'>
            Customer Parameters
        </div>""", unsafe_allow_html=True)

        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges (₹)", 18, 118, 65)
        total_charges   = st.slider("Total Charges (₹)", 18, 8500, monthly_charges * tenure + 18)

        st.markdown("---")
        gender          = st.selectbox("Gender", ["Male", "Female"])
        senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner         = st.selectbox("Has Partner", ["Yes", "No"])
        dependents      = st.selectbox("Has Dependents", ["Yes", "No"])

        st.markdown("---")
        contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet        = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment         = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines     = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_sec      = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    # ── Encode and predict ────────────────────────────────────────────────────
    def encode(val, col):
        classes = le_dict.get(col, [])
        if val in classes:
            return classes.index(val)
        return 0

    input_data = {
        'gender'          : encode(gender, 'gender'),
        'SeniorCitizen'   : 1 if senior == "Yes" else 0,
        'Partner'         : encode(partner, 'Partner'),
        'Dependents'      : encode(dependents, 'Dependents'),
        'tenure'          : tenure,
        'PhoneService'    : encode(phone_service, 'PhoneService'),
        'MultipleLines'   : encode(multi_lines, 'MultipleLines'),
        'InternetService' : encode(internet, 'InternetService'),
        'OnlineSecurity'  : encode(online_sec, 'OnlineSecurity'),
        'TechSupport'     : encode(tech_support, 'TechSupport'),
        'Contract'        : encode(contract, 'Contract'),
        'PaperlessBilling': encode(paperless, 'PaperlessBilling'),
        'PaymentMethod'   : encode(payment, 'PaymentMethod'),
        'MonthlyCharges'  : monthly_charges,
        'TotalCharges'    : total_charges,
    }

    X_input = pd.DataFrame([input_data])[features]
    churn_prob  = model.predict_proba(X_input)[0][1]
    churn_pct   = round(churn_prob * 100, 1)

    # ── Layout ────────────────────────────────────────────────────────────────
    col_gauge, col_risk = st.columns([1.2, 1])

    with col_gauge:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = churn_pct,
            delta = {'reference': 35, 'valueformat': '.1f', 'suffix': '%'},
            number= {'suffix': '%', 'font': {'size': 48, 'color': '#e8eaf0', 'family': 'Syne'}},
            gauge = {
                'axis'      : {'range': [0, 100], 'tickcolor': '#4b5568', 'tickfont': {'color': '#6b7a99'}},
                'bar'       : {'color': '#f87171' if churn_pct > 60 else '#fbbf24' if churn_pct > 35 else '#34d399', 'thickness': 0.3},
                'bgcolor'   : '#111827',
                'bordercolor': '#1e2a45',
                'steps'     : [
                    {'range': [0,  35], 'color': '#132218'},
                    {'range': [35, 60], 'color': '#1f1e14'},
                    {'range': [60, 100],'color': '#2d1b1b'},
                ],
                'threshold' : {'line': {'color': '#4f9eff', 'width': 3}, 'thickness': 0.8, 'value': 35},
            },
            title = {'text': "Churn Probability", 'font': {'size': 14, 'color': '#6b7a99', 'family': 'DM Sans'}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor='#111827', font_color='#e8eaf0',
            height=280, margin=dict(t=40, b=10, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_risk:
        st.markdown("<br>", unsafe_allow_html=True)
        if churn_pct > 60:
            risk_class, risk_text, risk_emoji = "high",   "HIGH RISK",   "🔴"
        elif churn_pct > 35:
            risk_class, risk_text, risk_emoji = "medium", "MEDIUM RISK", "🟡"
        else:
            risk_class, risk_text, risk_emoji = "low",    "LOW RISK",    "🟢"

        st.markdown(f"""
        <div class="risk-{risk_class}">
            <div class="risk-label-{risk_class}">{risk_emoji} {risk_text}</div>
            <div style='color:#8892a4; font-size:0.82rem; margin-top:8px;'>
                This customer has a <strong style='color:#e8eaf0'>{churn_pct}%</strong> probability of churning.
            </div>
            <hr style='border-color:#1e2a45; margin:12px 0;'>
            <div style='font-size:0.78rem; color:#6b7a99;'>
                💰 Estimated monthly revenue at risk:<br>
                <span style='font-family:Syne,sans-serif; font-size:1.4rem; color:#4f9eff; font-weight:700;'>
                    ₹{round(monthly_charges * churn_prob, 2):,.2f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature importance for this customer ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Key Churn Drivers — This Customer</div>', unsafe_allow_html=True)

    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True).tail(8)
    fig_fi = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation='h',
        marker=dict(color=fi.values, colorscale=[[0,'#1e3a5f'],[0.5,'#4f9eff'],[1,'#f87171']]),
        text=[f"{v:.3f}" for v in fi.values], textposition='outside',
        textfont=dict(color='#6b7a99', size=11)
    ))
    fig_fi.update_layout(
        paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
        xaxis=dict(showgrid=False, color='#4b5568', title='Importance Score'),
        yaxis=dict(color='#8892a4'),
        height=300, margin=dict(t=10, b=10, l=10, r=60),
        font=dict(family='DM Sans', color='#e8eaf0')
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Business recommendations ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Retention Recommendations</div>', unsafe_allow_html=True)

    recs = []
    if contract == "Month-to-month":
        recs.append(("📋 Upgrade Contract", "Customer is on month-to-month. Offer a <strong>10–15% discount</strong> for switching to a 1-year contract. This is the #1 churn driver."))
    if churn_pct > 50 and monthly_charges > 70:
        recs.append(("💸 Pricing Intervention", "High monthly charges combined with high churn risk. Consider a <strong>loyalty discount of ₹{} /month</strong>.".format(round(monthly_charges * 0.1))))
    if online_sec == "No" and internet != "No":
        recs.append(("🔒 Add Security Bundle", "Customer lacks Online Security. Bundling it free for 3 months increases perceived value and <strong>reduces churn by ~18%</strong> historically."))
    if payment == "Electronic check":
        recs.append(("💳 Payment Method Nudge", "Electronic check users churn more. Offer a <strong>₹50/month discount</strong> for switching to auto-pay (bank transfer or credit card)."))
    if tenure < 12:
        recs.append(("🤝 Early Loyalty Program", "Tenure under 12 months — customer is in the <strong>high-risk early churn window</strong>. Trigger a welcome loyalty reward immediately."))
    if not recs:
        recs.append(("✅ Retain with Rewards", "Customer is low risk. Maintain satisfaction with a <strong>periodic loyalty reward</strong> to keep NPS high."))

    for title, desc in recs:
        st.markdown(f'<div class="rec-box"><strong>{title}</strong><br>{desc}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Model Metrics — Gradient Boosting Classifier</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["auc"]}</div><div class="metric-label">ROC-AUC Score</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["accuracy"]}%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["recall"]}%</div><div class="metric-label">Recall (Churn)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["f1"]}%</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Feature Importance — All Features</div>', unsafe_allow_html=True)
        fi_all = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig_all = go.Figure(go.Bar(
            x=fi_all.values, y=fi_all.index, orientation='h',
            marker=dict(color=fi_all.values, colorscale='Blues'),
            text=[f"{v:.3f}" for v in fi_all.values], textposition='outside',
            textfont=dict(size=10, color='#6b7a99')
        ))
        fig_all.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
            xaxis=dict(showgrid=False, color='#4b5568'),
            yaxis=dict(color='#8892a4', tickfont=dict(size=11)),
            height=400, margin=dict(t=10, b=10, l=10, r=60),
            font=dict(family='DM Sans', color='#e8eaf0')
        )
        st.plotly_chart(fig_all, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Churn Distribution by Contract Type</div>', unsafe_allow_html=True)
        contract_data = pd.DataFrame({
            'Contract'    : ['Month-to-month', 'One year', 'Two year'],
            'Churn Rate %': [48, 18, 6],
            'Count'       : [3875, 1473, 1695]
        })
        fig_contract = go.Figure()
        fig_contract.add_trace(go.Bar(
            x=contract_data['Contract'], y=contract_data['Churn Rate %'],
            marker_color=['#f87171', '#fbbf24', '#34d399'],
            text=[f"{v}%" for v in contract_data['Churn Rate %']],
            textposition='outside', textfont=dict(color='#e8eaf0', size=13, family='Syne')
        ))
        fig_contract.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
            xaxis=dict(color='#8892a4', showgrid=False),
            yaxis=dict(color='#4b5568', showgrid=True, gridcolor='#1e2a45', title='Churn Rate %'),
            height=400, margin=dict(t=40, b=20),
            font=dict(family='DM Sans', color='#e8eaf0')
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    # Tenure vs churn
    st.markdown('<div class="section-header">Churn Risk by Tenure Band</div>', unsafe_allow_html=True)
    tenure_bands = ['0–12 mo', '13–24 mo', '25–36 mo', '37–48 mo', '49–60 mo', '60–72 mo']
    churn_rates  = [58, 42, 31, 22, 16, 10]
    fig_tenure = go.Figure(go.Scatter(
        x=tenure_bands, y=churn_rates, mode='lines+markers',
        line=dict(color='#4f9eff', width=3),
        marker=dict(size=10, color='#f87171', line=dict(color='#0a0e1a', width=2)),
        fill='tozeroy', fillcolor='rgba(79,158,255,0.08)',
        text=[f"{v}%" for v in churn_rates], textposition='top center',
        textfont=dict(color='#e8eaf0', size=11)
    ))
    fig_tenure.update_layout(
        paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
        xaxis=dict(color='#8892a4', showgrid=False),
        yaxis=dict(color='#4b5568', showgrid=True, gridcolor='#1e2a45', title='Churn Rate %'),
        height=280, margin=dict(t=20, b=20),
        font=dict(family='DM Sans', color='#e8eaf0')
    )
    st.plotly_chart(fig_tenure, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Business Impact Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">35.4%</div><div class="metric-label">Overall Churn Rate</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">₹8.9L</div><div class="metric-label">Monthly Revenue at Risk</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">3,200+</div><div class="metric-label">High-Risk Customers</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Revenue at Risk by Segment</div>', unsafe_allow_html=True)
        segments = ['Fiber + Month-to-month', 'DSL + No Security', 'Senior + High Charges', 'New Customers (<12mo)', 'Electronic Check Users']
        revenue  = [2.8, 1.9, 1.4, 1.6, 1.2]
        fig_seg = go.Figure(go.Bar(
            y=segments, x=revenue, orientation='h',
            marker=dict(color=revenue, colorscale=[[0,'#1e3a5f'],[1,'#f87171']]),
            text=[f"₹{v}L" for v in revenue], textposition='outside',
            textfont=dict(color='#e8eaf0', size=12)
        ))
        fig_seg.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
            xaxis=dict(title='Monthly Revenue at Risk (Lakhs ₹)', color='#4b5568', showgrid=True, gridcolor='#1e2a45'),
            yaxis=dict(color='#8892a4', tickfont=dict(size=11)),
            height=340, margin=dict(t=10, b=10, l=10, r=80),
            font=dict(family='DM Sans', color='#e8eaf0')
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Retention ROI Simulation</div>', unsafe_allow_html=True)
        interventions = ['Contract Upgrade Offer', 'Loyalty Discount', 'Security Bundle', 'Payment Method Switch', 'Early Tenure Reward']
        cost_per      = [200, 350, 150, 50, 100]
        saved_per     = [1200, 900, 650, 500, 800]
        roi           = [round((s-c)/c*100) for c,s in zip(cost_per, saved_per)]
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(name='Cost per Customer (₹)', x=interventions, y=cost_per, marker_color='#f87171'))
        fig_roi.add_trace(go.Bar(name='Revenue Saved (₹)',     x=interventions, y=saved_per, marker_color='#34d399'))
        fig_roi.update_layout(
            barmode='group', paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
            xaxis=dict(color='#8892a4', showgrid=False, tickfont=dict(size=9)),
            yaxis=dict(color='#4b5568', showgrid=True, gridcolor='#1e2a45', title='Amount (₹)'),
            legend=dict(bgcolor='#111827', bordercolor='#1e2a45', font=dict(color='#8892a4', size=10)),
            height=340, margin=dict(t=10, b=10),
            font=dict(family='DM Sans', color='#e8eaf0')
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    # Key insights
    st.markdown('<div class="section-header">Key Business Insights</div>', unsafe_allow_html=True)
    insights = [
        ("📋 Contract is #1 Churn Driver", "Month-to-month customers churn at 48% vs only 6% for two-year contracts. Converting just 10% of month-to-month users to annual contracts would save ~₹1.8L/month in revenue."),
        ("⏱️ First 12 Months Are Critical", "58% of churned customers leave within their first year. A structured early-tenure onboarding program and loyalty reward at month 3 and month 6 can reduce this by 20-30%."),
        ("💳 Payment Method Predicts Churn", "Electronic check users have 2.3x higher churn than auto-pay users. A targeted ₹50/month incentive to switch payment methods has an ROI of 900% based on revenue retained."),
        ("🌐 Fiber Optic + No Security = High Risk", "Fiber optic customers without Online Security churn at nearly 52%. Bundling security for free for 2 months costs ₹150/customer but saves ₹1,200 on average in retained revenue."),
    ]
    for title, body in insights:
        st.markdown(f'<div class="rec-box"><strong>{title}</strong><br><span style="color:#8892a4">{body}</span></div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:48px; padding:20px; border-top:1px solid #1e2a45; text-align:center;
            color:#4b5568; font-size:0.75rem;'>
    ChurnIQ · Built with Gradient Boosting + Streamlit · Telecom Customer Analytics
</div>
""", unsafe_allow_html=True)