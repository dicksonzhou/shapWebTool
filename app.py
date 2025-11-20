import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# -----------------------------------------------------------------------------
# 1. 页面基础配置 (必须是第一行代码)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HCC Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS: 调整顶部留白，使其看起来更像原生软件
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 { font-size: 1.8rem; }
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 模型加载/训练模块 (带缓存)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    在真实部署中，这里应该加载已经训练好的模型文件，例如:
    model = joblib.load('hcc_model.pkl')
    此处为了演示，我们现场快速训练一个模拟模型。
    """
    np.random.seed(42)
    # 模拟训练数据 (1000条)
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
    
    # 模拟标签逻辑：AFP高、Age大、有结节 -> 风险高
    logits = (X_train['AFP']*0.4 + X_train['Age']*0.3 + X_train['Hepatic_Nodule']*50 + 
              X_train['PIVKA_II']*0.2 + np.random.normal(0, 20, 1000))
    y_train = (logits > np.percentile(logits, 70)).astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # 创建解释器
    explainer = shap.TreeExplainer(model)
    
    return model, explainer

# 初始化模型 (利用缓存，只会运行一次)
try:
    model, explainer = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 侧边栏：用户输入
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Predictive modeling")
    st.markdown("---")
    
    # 收集输入数据的字典
    inputs = {}
    
    # 按照图片顺序排列
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
    st.caption("Calculated Scores")
    inputs['GASR'] = st.number_input("GASR", value=1.0)
    inputs['AAAR'] = st.number_input("AAAR", value=2.0)
    inputs['ALBI'] = st.number_input("ALBI", value=1.0)
    
    st.markdown("---")
    inputs['CK'] = st.number_input("CK(U/L)", value=91.0)
    inputs['INR'] = st.number_input("INR", value=3.0)
    inputs['FIB'] = st.number_input("FIB(g/L)", value=2.0)
    inputs['FDP'] = st.number_input("FDP(ug/ml)", value=3.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Risk")

# -----------------------------------------------------------------------------
# 4. 主界面：展示结果
# -----------------------------------------------------------------------------

# 顶部公式说明
st.caption("The formulas to calculate GASR, AAAR, and ALBI are as follows: GASR=GGT/AST; AAAR=AFP/[AST*ALT]; ALBI= [log10 TBIL* 0.66] + [ALB * (-0.085)]")

st.markdown("<h3 style='text-align: center; color: #d32f2f;'><<< Probability of HCC Risk >>></h3>", unsafe_allow_html=True)

if predict_btn:
    # 4.1 准备数据
    input_df = pd.DataFrame([inputs])
    
    # 4.2 预测
    # predict_proba 返回 [[prob_class_0, prob_class_1]]
    probability = model.predict_proba(input_df)[0][1]
    risk_percentage = probability * 100
    
    # -------------------------------------------------------
    # 4.3 布局：左侧圆环图，右侧SHAP图
    # -------------------------------------------------------
    col_chart, col_shap = st.columns([1, 1.5])
    
    with col_chart:
        # 圆环图
        labels = ['Risk', 'Non-Risk']
        values = [risk_percentage, 100 - risk_percentage]
        colors = ['#FF0000', '#008CBA'] # 红色和蓝色
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.65, 
            marker=dict(colors=colors),
            sort=False,
            textinfo='none',
            hoverinfo='label+percent'
        )])
        
        fig.add_annotation(
            text=f"<b>Risk<br>{risk_percentage:.2f}%</b>", 
            x=0.5, y=0.5, font_size=22, showarrow=False
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=30, b=0, l=20, r=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_shap:
        st.markdown("##### Model Interpretation (Why?)")
        
        # 计算 SHAP 值
        shap_values = explainer(input_df)
        
        # 绘制瀑布图
        fig_shap, ax = plt.subplots(figsize=(6, 4))
        # max_display 控制显示前几个最重要的特征
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)
        
        # 调整 Matplotlib 样式以适应网页
        plt.tight_layout()
        st.pyplot(fig_shap, clear_figure=True, use_container_width=True)
        
        st.info("SHAP Waterfall Plot explains which features pushed the risk up (Red) or down (Blue) from the baseline.")

else:
    # 初始状态提示
    st.markdown(
        """
        <div style="text-align: center; padding: 50px; background-color: #f0f2f6; border-radius: 10px;">
            <h4> Please input parameters on the left sidebar and click "Predict Risk"</h4>
        </div>
        """, 
        unsafe_allow_html=True
    )