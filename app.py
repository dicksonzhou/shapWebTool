import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# -----------------------------------------------------------------------------
# 1. 页面基础配置
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HCC Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        h1 { font-size: 1.8rem; }
        .stButton>button { width: 100%; background-color: #ff4b4b; color: white; }
        /* 调整公式区域的样式 */
        .formula-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #ff4b4b;
            margin-bottom: 20px;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 模型加载/训练模块 (带缓存)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # 模拟训练过程 (与之前一致)
    np.random.seed(42)
    X_train = pd.DataFrame({
        'Age': np.random.randint(20, 90, 1000),
        'Hepatic_Nodule': np.random.randint(0, 2, 1000),
        'CRP': np.random.uniform(0, 100, 1000),
        'RBC': np.random.uniform(3, 6, 1000),
        'PDW': np.random.uniform(10, 20, 1000),
        'AFP': np.random.exponential(50, 1000),
        'PIVKA_II': np.random.exponential(100, 1000),
        'IBIL': np.random.uniform(5, 30, 1000),
        'TBA': np.random.uniform(1, 20, 1000),
        'GASR': np.random.uniform(0.5, 2, 1000),
        'AAAR': np.random.uniform(0.5, 3, 1000),
        'ALBI': np.random.uniform(-3, 0, 1000),
        'CK': np.random.uniform(50, 200, 1000),
        'INR': np.random.uniform(0.8, 1.5, 1000),
        'FIB': np.random.uniform(1.5, 4, 1000),
        'FDP': np.random.uniform(0, 10, 1000)
    })
    
    logits = (X_train['AFP']*0.4 + X_train['Age']*0.3 + X_train['Hepatic_Nodule']*50 + 
              X_train['PIVKA_II']*0.2 + np.random.normal(0, 20, 1000))
    y_train = (logits > np.percentile(logits, 70)).astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer

try:
    model, explainer = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 侧边栏：用户输入 (增加 Help 提示)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Predictive modeling")
    st.markdown("---")
    
    inputs = {}
    inputs['Age'] = st.slider("Age(year)", 0, 95, 42)
    nodule = st.selectbox("Hepatic nodule(>=1cm)", ["No", "Yes"])
    inputs['Hepatic_Nodule'] = 1 if nodule == "Yes" else 0
    
    inputs['CRP'] = st.number_input("CRP(mg/L)", value=42.0)
    inputs['RBC'] = st.number_input("RBC(10^12/L)", value=8.0)
    inputs['PDW'] = st.number_input("PDW(fl)", value=14.0)
    inputs['AFP'] = st.number_input("AFP(ng/ml)", value=133.0)
    inputs['PIVKA_II'] = st.number_input("PIVKA_II(mAU/mL)", value=758.0)
    inputs['IBIL'] = st.number_input("IBIL(umol/L)", value=15.0)
    inputs['TBA'] = st.number_input("TBA(umol/L)", value=6.0)
    
    st.markdown("---")
    st.markdown("**Calculated Indices**")
    
    # --- 在这里增加了 help 参数，鼠标悬停会显示公式 ---
    inputs['GASR'] = st.number_input(
        "GASR", value=1.0, 
        help="Formula: GGT(U/L) / AST(U/L)"
    )
    inputs['AAAR'] = st.number_input(
        "AAAR", value=2.0, 
        help="Formula: AFP(ng/ml) / [AST(U/L) * ALT(U/L)]"
    )
    inputs['ALBI'] = st.number_input(
        "ALBI", value=1.0, 
        help="Formula: [log10 TBIL(umol/L) * 0.66] + [ALB(g/L) * (-0.085)]"
    )
    
    st.markdown("---")
    inputs['CK'] = st.number_input("CK(U/L)", value=91.0)
    inputs['INR'] = st.number_input("INR", value=3.0)
    inputs['FIB'] = st.number_input("FIB(g/L)", value=2.0)
    inputs['FDP'] = st.number_input("FDP(ug/ml)", value=3.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Risk")

# -----------------------------------------------------------------------------
# 4. 主界面
# -----------------------------------------------------------------------------

# --- 新增：使用 Expander 折叠显示公式，保持界面整洁但随时可查 ---
with st.expander("View Calculation Formulas (GASR, AAAR, ALBI)", expanded=True):
    st.markdown("The formulas to calculate the derived indices are as follows:")
    
    # 使用 columns 将公式横向排列
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**1. GASR**")
        st.latex(r'''
            GASR = \frac{\text{GGT (U/L)}}{\text{AST (U/L)}}
        ''')
        
    with c2:
        st.markdown("**2. AAAR**")
        st.latex(r'''
            AAAR = \frac{\text{AFP (ng/ml)}}{\text{AST} \times \text{ALT}}
        ''')
        
    with c3:
        st.markdown("**3. ALBI**")
        st.latex(r'''
            ALBI = (\log_{10}(\text{TBIL}) \times 0.66) + (\text{ALB} \times -0.085)
        ''')
    
    st.caption("*Units: TBIL in umol/L, ALB in g/L*")

# 标题
st.markdown("<h3 style='text-align: center; color: #d32f2f; margin-top: 20px;'><<< Probability of HCC Risk >>></h3>", unsafe_allow_html=True)

if predict_btn:
    # 预测逻辑
    input_df = pd.DataFrame([inputs])
    probability = model.predict_proba(input_df)[0][1]
    risk_percentage = probability * 100
    
    col_chart, col_shap = st.columns([1, 1.5])
    
    # 圆环图
    with col_chart:
        labels = ['Risk', 'Non-Risk']
        values = [risk_percentage, 100 - risk_percentage]
        colors = ['#FF0000', '#008CBA']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=.65, 
            marker=dict(colors=colors), sort=False, textinfo='none', hoverinfo='label+percent'
        )])
        
        fig.add_annotation(text=f"<b>Risk<br>{risk_percentage:.2f}%</b>", x=0.5, y=0.5, font_size=22, showarrow=False)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.05), margin=dict(t=30, b=0, l=20, r=20), height=350)
        st.plotly_chart(fig, use_container_width=True)

    # SHAP 解释
    with col_shap:
        st.markdown("##### Model Interpretation (SHAP)")
        with st.spinner("Calculating feature contributions..."):
            shap_values = explainer(input_df)
            fig_shap, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap, clear_figure=True, use_container_width=True)
else:
    st.markdown(
        """
        <div style="text-align: center; padding: 40px; color: #666;">
            Please adjust the clinical parameters on the left sidebar and click <b>Predict Risk</b>.
        </div>
        """, 
        unsafe_allow_html=True
    )