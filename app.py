import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. 页面基础配置
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HCC Risk Prediction (Custom Data)",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        h1 { font-size: 1.8rem; }
        .stButton>button { width: 100%; background-color: #ff4b4b; color: white; }
        .formula-box { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #ff4b4b; margin-bottom: 20px; font-size: 0.9em; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 定义必要的特征列 (必须与界面输入框对应)
# -----------------------------------------------------------------------------
REQUIRED_FEATURES = [
    'Age', 'Hepatic_Nodule', 'CRP', 'RBC', 'PDW', 'AFP', 
    'PIVKA_II', 'IBIL', 'TBA', 'GASR', 'AAAR', 'ALBI', 
    'CK', 'INR', 'FIB', 'FDP'
]

# -----------------------------------------------------------------------------
# 3. 侧边栏：数据导入与模型训练
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("1. Model Training")
    st.info("Upload your historical dataset (.csv) to train the model.")
    
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    
    model = None
    explainer = None
    is_model_ready = False

    if uploaded_file is not None:
        # 读取数据
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows.")
            
            # 检查列名是否匹配
            missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing columns in CSV: {', '.join(missing_cols)}")
                st.warning("Please ensure your CSV headers match the input fields strictly.")
            else:
                # 选择目标变量 (Y)
                target_col = st.selectbox("Select Target Column (Outcome)", [c for c in df.columns if c not in REQUIRED_FEATURES])
                
                if st.button("Train Model Now"):
                    with st.spinner("Training XGBoost Model..."):
                        # 数据预处理
                        X = df[REQUIRED_FEATURES].copy()
                        y = df[target_col]
                        
                        # 处理 Hepatic_Nodule 如果是字符串 'Yes'/'No'
                        if X['Hepatic_Nodule'].dtype == object:
                             X['Hepatic_Nodule'] = X['Hepatic_Nodule'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)
                        
                        # 训练模型
                        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
                        model.fit(X, y)
                        
                        # 创建解释器
                        explainer = shap.TreeExplainer(model)
                        
                        # 保存到 Session State 防止刷新丢失
                        st.session_state['model'] = model
                        st.session_state['explainer'] = explainer
                        st.session_state['model_trained'] = True
                        st.success("Model Trained Successfully!")
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # 检查 Session State 中是否有模型
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        model = st.session_state['model']
        explainer = st.session_state['explainer']
        is_model_ready = True

# -----------------------------------------------------------------------------
# 4. 侧边栏：预测输入 (只有模型就绪才显示)
# -----------------------------------------------------------------------------
inputs = {}

if is_model_ready:
    with st.sidebar:
        st.markdown("---")
        st.title("2. Patient Input")
        
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
        
        st.markdown("**Calculated Indices**")
        inputs['GASR'] = st.number_input("GASR", value=1.0, help="GGT/AST")
        inputs['AAAR'] = st.number_input("AAAR", value=2.0, help="AFP/[AST*ALT]")
        inputs['ALBI'] = st.number_input("ALBI", value=1.0, help="log10 TBIL*0.66 + ALB*-0.085")
        
        inputs['CK'] = st.number_input("CK(U/L)", value=91.0)
        inputs['INR'] = st.number_input("INR", value=3.0)
        inputs['FIB'] = st.number_input("FIB(g/L)", value=2.0)
        inputs['FDP'] = st.number_input("FDP(ug/ml)", value=3.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Predict Risk")
else:
    # 如果没有模型，侧边栏显示锁定状态或提示
    with st.sidebar:
        st.markdown("---")
        st.warning(⚠️ Please upload data and train model first.")
        predict_btn = False

# -----------------------------------------------------------------------------
# 5. 主界面内容
# -----------------------------------------------------------------------------

# 公式说明
with st.expander("ℹ️ View Calculation Formulas (GASR, AAAR, ALBI)", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1: st.latex(r'GASR = \frac{\text{GGT}}{\text{AST}}')
    with c2: st.latex(r'AAAR = \frac{\text{AFP}}{\text{AST} \times \text{ALT}}')
    with c3: st.latex(r'ALBI = 0.66 \log_{10}(\text{TBIL}) - 0.085 \text{ALB}')

st.markdown("<h3 style='text-align: center; color: #d32f2f;'><<< Probability of HCC Risk >>></h3>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. 预测逻辑与展示
# -----------------------------------------------------------------------------

if is_model_ready:
    if predict_btn:
        # 构造输入 DataFrame
        input_df = pd.DataFrame([inputs])
        
        # 确保列顺序与训练时完全一致
        input_df = input_df[REQUIRED_FEATURES]

        # 预测
        probability = model.predict_proba(input_df)[0][1]
        risk_percentage = probability * 100
        
        # 布局
        col_chart, col_shap = st.columns([1, 1.5])
        
        # 1. 圆环图
        with col_chart:
            labels = ['Risk', 'Non-Risk']
            values = [risk_percentage, 100 - risk_percentage]
            colors = ['#FF0000', '#008CBA']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=.65, 
                marker=dict(colors=colors), sort=False, textinfo='none'
            )])
            fig.add_annotation(text=f"<b>Risk<br>{risk_percentage:.2f}%</b>", x=0.5, y=0.5, font_size=22, showarrow=False)
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.05), margin=dict(t=30, b=0, l=20, r=20), height=350)
            st.plotly_chart(fig, use_container_width=True)

        # 2. SHAP 解释
        with col_shap:
            st.markdown("##### Model Interpretation (SHAP)")
            shap_values = explainer(input_df)
            fig_shap, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap, clear_figure=True, use_container_width=True)
    else:
        st.info("👈 Adjust patient parameters on the left and click 'Predict Risk'.")
else:
    # 引导用户上传数据的空状态页
    st.markdown(
        """
        <div style="text-align: center; padding: 50px; background-color: #f0f2f6; border-radius: 10px; border: 2px dashed #ccc;">
            <h2>📂 Waiting for Data</h2>
            <p>Please upload a CSV file in the sidebar to initialize the model.</p>
            <p style="font-size: 0.9em; color: #666;">
                <b>Required CSV Columns:</b><br>
                Age, Hepatic_Nodule, CRP, RBC, PDW, AFP, PIVKA_II, IBIL, TBA, GASR, AAAR, ALBI, CK, INR, FIB, FDP<br>
                + <i>One target column (e.g., 'Outcome' or 'Risk')</i>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )