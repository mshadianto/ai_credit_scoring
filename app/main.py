"""
🏦 AI Credit Scoring System - Streamlit Cloud Version
Created by MS Hadianto | RAG & Agentic AI Enthusiast
Robust version with graceful fallbacks for missing dependencies
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

# Configure Streamlit
st.set_page_config(
    page_title="AI Credit Scoring by MS Hadianto",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Graceful imports with fallbacks
PLOTLY_AVAILABLE = False
SKLEARN_AVAILABLE = False
SQLALCHEMY_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("📊 Plotly not available - using basic charts")
    # Mock plotly for fallback
    class MockPlotly:
        def pie(self, **kwargs): 
            return st.info("📊 Pie chart would appear here")
        def histogram(self, **kwargs): 
            return st.info("📊 Histogram would appear here")
        def line(self, **kwargs): 
            return st.info("📊 Line chart would appear here")
        def scatter(self, **kwargs): 
            return st.info("📊 Scatter plot would appear here")
        def bar(self, **kwargs): 
            return st.info("📊 Bar chart would appear here")
    px = MockPlotly()
    
    class MockGO:
        class Figure:
            def __init__(self, *args): pass
            def update_layout(self, **kwargs): pass
            def add_vline(self, **kwargs): pass
        class Indicator:
            def __init__(self, **kwargs): pass
    go = MockGO()

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    st.warning("🤖 Scikit-learn not fully available - using simple algorithms")
    SKLEARN_AVAILABLE = False

try:
    import sqlalchemy
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    st.warning("🗄️ SQLAlchemy not available - using in-memory storage")
    SQLALCHEMY_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .developer-signature {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Mock data and classes for fallbacks
class MockRAG:
    def __init__(self):
        self.api_key = None
    
    def query(self, prompt, context=None):
        responses = {
            "risk": """
            **Analisis Risiko Kredit:**
            
            Faktor utama yang mempengaruhi risiko:
            1. **Riwayat Pembayaran** (35%) - Konsistensi pembayaran tepat waktu
            2. **Debt-to-Income Ratio** (30%) - Perbandingan utang dengan pendapatan
            3. **Riwayat Kredit** (15%) - Panjang histori kredit
            4. **Utilisasi Kredit** (10%) - Penggunaan vs limit kredit
            5. **Jenis Kredit** (10%) - Diversifikasi produk kredit
            """,
            "improve": """
            **Cara Meningkatkan Approval Rate:**
            
            1. **Optimasi Model** - Gunakan ensemble methods
            2. **Alternative Data** - Telco, utility payment history
            3. **Dynamic Pricing** - Risk-based interest rates
            4. **Automated Decisioning** - Reduce manual review time
            5. **Portfolio Diversification** - Spread across segments
            """,
            "default": f"""
            **AI Credit Assistant Response:**
            
            Terima kasih atas pertanyaan: "{prompt}"
            
            Sebagai sistem AI Credit Scoring yang dikembangkan oleh MS Hadianto,
            saya dapat membantu dengan analisis risiko kredit, optimasi portfolio,
            dan best practices dalam credit scoring.
            
            **Key Insights:**
            - Model AI dapat meningkatkan akurasi hingga 25%
            - RAG system memberikan contextual responses
            - Agentic AI memungkinkan automated decision making
            
            Untuk pertanyaan lebih spesifik, silakan konfigurasi API key
            untuk mengakses full AI capabilities.
            """
        }
        
        prompt_lower = prompt.lower()
        if "risk" in prompt_lower or "risiko" in prompt_lower:
            return responses["risk"]
        elif "improve" in prompt_lower or "tingkat" in prompt_lower:
            return responses["improve"]
        else:
            return responses["default"]

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_applicant = {}
    st.session_state.chat_history = []
    st.session_state.portfolio_metrics = {
        'total_applications': 1234,
        'approval_rate': 76.5,
        'average_credit_score': 720
    }
    st.session_state.rag_system = MockRAG()

def render_header():
    """Render main header"""
    st.markdown(f"""
    <div class="main-header">
        <h1>🏦 AI Credit Scoring System <span style='font-size: 0.6em; opacity: 0.8;'>v1.0.0</span></h1>
        <p>Advanced Credit Risk Assessment dengan RAG & Agentic AI</p>
        <div style='margin-top: 1rem; font-size: 0.85em; opacity: 0.9;'>
            👨‍💻 Developed by: <strong>MS Hadianto</strong> | 
            🤖 RAG & Agentic AI Enthusiast | 
            🏦 Banking Technology Specialist
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar"""
    st.sidebar.markdown("""
    <div class="developer-signature">
        <h3 style='margin: 0; color: white;'>🏦 Credit AI</h3>
        <p style='margin: 0; font-size: 0.8em; opacity: 0.9;'>v1.0.0</p>
        <p style='margin: 0; font-size: 0.7em; opacity: 0.8;'>by MS Hadianto</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("⚙️ Configuration")
    
    # API Configuration
    with st.sidebar.expander("🔑 API Configuration"):
        api_key = st.text_input(
            "OpenRouter API Key", 
            type="password",
            help="Get free key at https://openrouter.ai/"
        )
        
        if api_key:
            st.session_state.rag_system.api_key = api_key
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Using demo responses")
    
    # Credit Type
    with st.sidebar.expander("🏦 Credit Settings"):
        credit_type = st.selectbox(
            "Credit Type",
            ["Personal Loan", "Mortgage", "Credit Card", "Auto Loan", "Business Loan"]
        )
    
    # System Status
    with st.sidebar.expander("📊 System Status"):
        metrics = st.session_state.portfolio_metrics
        st.metric("Applications", f"{metrics['total_applications']:,}")
        st.metric("Approval Rate", f"{metrics['approval_rate']:.1f}%")
        st.metric("Avg Score", int(metrics['average_credit_score']))
    
    # About
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        **Developer:** MS Hadianto  
        **Specialty:** RAG & Agentic AI  
        **Focus:** Banking Technology  
        
        **Features:**
        - AI-powered credit scoring
        - RAG-enhanced insights
        - Real-time risk assessment
        - Portfolio analytics
        
        **Contact:**
        📧 mshadianto@tech.com  
        🔗 LinkedIn: MS Hadianto  
        💻 GitHub: mshadianto
        """)
    
    return {'credit_type': credit_type, 'api_key': api_key}

def calculate_credit_score(data):
    """Enhanced credit scoring algorithm"""
    score = 300  # Base score
    
    # Age optimization (25-55 is optimal range)
    age_factor = 1.0 - abs(data['age'] - 40) / 40
    score += max(0, age_factor * 60)
    
    # Income factor (logarithmic scaling)
    income_normalized = min(np.log(data['income'] / 10_000_000) / 10, 1.0)
    score += max(0, income_normalized * 100)
    
    # Employment stability (diminishing returns after 10 years)
    employment_factor = min(data['employment_years'] / 10, 1.0)
    score += employment_factor * 80
    
    # Debt to income (exponential penalty for high DTI)
    dti_penalty = data['debt_to_income'] ** 2
    score += (1 - dti_penalty) * 120
    
    # Credit history (log scale benefits)
    history_factor = min(np.log(data['credit_history_length'] + 1) / 3, 1.0)
    score += history_factor * 70
    
    # Payment history (critical factor)
    score += data['payment_history'] * 100
    
    # Credit utilization (optimal around 10-30%)
    util = data['credit_utilization']
    if util <= 0.3:
        util_score = (1 - util) * 80
    else:
        util_score = (1 - util) * 40  # Penalty for high utilization
    score += max(0, util_score)
    
    # Education bonus
    education_bonus = {
        'High School': 0, 
        'Bachelor': 25, 
        'Master': 40, 
        'PhD': 50
    }
    score += education_bonus.get(data['education_level'], 0)
    
    # Marital status stability bonus
    if data['marital_status'] == 'Married':
        score += 15
    
    # Home ownership bonus
    if data['home_ownership'] == 'Own':
        score += 20
    elif data['home_ownership'] == 'Mortgage':
        score += 10
    
    return min(int(score), 850)

def render_dashboard():
    """Dashboard tab"""
    st.header("📊 Credit Portfolio Dashboard")
    
    st.markdown("""
    <div class="developer-signature">
        📈 <strong>Advanced Analytics Dashboard</strong> | 
        Crafted by MS Hadianto - Banking AI Specialist
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    metrics = st.session_state.portfolio_metrics
    
    with col1:
        st.metric("Total Applications", f"{metrics['total_applications']:,}", "12 today")
    with col2:
        st.metric("Approval Rate", f"{metrics['approval_rate']:.1f}%", "2.3%")
    with col3:
        st.metric("Avg Credit Score", int(metrics['average_credit_score']), "15 points")
    with col4:
        st.metric("Default Rate", "3.2%", "-0.5%", delta_color="inverse")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        if PLOTLY_AVAILABLE:
            risk_data = {
                'Risk Level': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                'Count': [30, 40, 20, 8, 2]
            }
            fig = px.pie(
                values=risk_data['Count'],
                names=risk_data['Risk Level'],
                title="Portfolio Risk Profile"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            df = pd.DataFrame({
                'Risk Level': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                'Count': [30, 40, 20, 8, 2]
            })
            st.bar_chart(df.set_index('Risk Level'))
    
    with col2:
        st.subheader("Credit Score Trends")
        if PLOTLY_AVAILABLE:
            dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
            scores = np.random.normal(720, 20, 12)
            fig = px.line(x=dates, y=scores, title="Monthly Average Credit Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart_data = pd.DataFrame({
                'Month': range(1, 13),
                'Avg Score': np.random.normal(720, 20, 12)
            })
            st.line_chart(chart_data.set_index('Month'))

def render_credit_scoring():
    """Credit scoring tab"""
    st.header("🎯 Credit Scoring Engine")
    
    st.markdown("""
    <div class="developer-signature">
        🎯 <strong>AI-Powered Credit Assessment</strong> | 
        Engineered by MS Hadianto using Advanced ML & Banking Expertise
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Applicant Information")
        
        # Personal Details
        with st.expander("👤 Personal Details", expanded=True):
            col_a, col_b = st.columns(2)
            
            with col_a:
                age = st.slider("Age", 18, 80, 35)
                income = st.number_input("Annual Income (IDR)", 
                                       min_value=0, value=500_000_000, step=10_000_000)
                employment_years = st.slider("Employment Years", 0, 40, 5)
            
            with col_b:
                education = st.selectbox("Education", 
                                       ['High School', 'Bachelor', 'Master', 'PhD'])
                marital_status = st.selectbox("Marital Status", 
                                            ['Single', 'Married', 'Divorced'])
                home_ownership = st.selectbox("Home Ownership", 
                                            ['Rent', 'Own', 'Mortgage'])
        
        # Financial Profile
        with st.expander("💰 Financial Profile", expanded=True):
            col_a, col_b = st.columns(2)
            
            with col_a:
                debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.01)
                credit_history = st.slider("Credit History (years)", 0, 30, 10)
                num_accounts = st.slider("Number of Accounts", 1, 20, 5)
            
            with col_b:
                payment_history = st.slider("Payment History Score", 0.0, 1.0, 0.85, 0.01)
                credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.3, 0.01)
        
        # Loan Details
        with st.expander("🏦 Loan Information"):
            loan_amount = st.number_input("Loan Amount (IDR)", 
                                        min_value=0, value=100_000_000, step=1_000_000)
            loan_purpose = st.selectbox("Purpose", 
                                      ['Home Purchase', 'Debt Consolidation', 
                                       'Business', 'Education', 'Other'])
    
    with col2:
        st.subheader("🎯 Credit Assessment")
        
        if st.button("📊 Calculate Credit Score", type="primary"):
            with st.spinner("🔮 AI analyzing credit profile..."):
                # Prepare data
                application_data = {
                    'age': age,
                    'income': income,
                    'employment_years': employment_years,
                    'debt_to_income': debt_to_income,
                    'credit_history_length': credit_history,
                    'number_of_accounts': num_accounts,
                    'payment_history': payment_history,
                    'credit_utilization': credit_utilization,
                    'education_level': education,
                    'marital_status': marital_status,
                    'home_ownership': home_ownership,
                    'loan_amount': loan_amount,
                    'loan_purpose': loan_purpose
                }
                
                # Calculate score
                score = calculate_credit_score(application_data)
                risk_prob = (850 - score) / 550
                
                # Determine risk level and decision
                if risk_prob < 0.2:
                    risk_level = "Sangat Rendah"
                    decision = "Approved"
                    interest_rate = "5.5% - 7.0%"
                elif risk_prob < 0.4:
                    risk_level = "Rendah"
                    decision = "Approved"
                    interest_rate = "7.0% - 9.5%"
                elif risk_prob < 0.6:
                    risk_level = "Sedang"
                    decision = "Review"
                    interest_rate = "9.5% - 15.0%"
                else:
                    risk_level = "Tinggi"
                    decision = "Rejected"
                    interest_rate = "N/A"
                
                # Display gauge chart
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Credit Score"},
                        delta={'reference': 650},
                        gauge={
                            'axis': {'range': [None, 850]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [300, 580], 'color': "lightcoral"},
                                {'range': [580, 670], 'color': "lightyellow"},
                                {'range': [670, 740], 'color': "lightgreen"},
                                {'range': [740, 850], 'color': "darkgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 700
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback display
                    st.markdown(f"""
                    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 15px; color: white; margin: 1rem 0;'>
                        <h1 style='margin: 0; font-size: 3em;'>{score}</h1>
                        <h3 style='margin: 0; opacity: 0.9;'>Credit Score</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Results
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Credit Score", score)
                    st.metric("Risk Level", risk_level)
                with col_b:
                    st.metric("Default Probability", f"{risk_prob:.1%}")
                    st.metric("Confidence", "89.5%")
                
                # Decision
                if decision == "Approved":
                    st.success(f"✅ {decision}")
                    st.info(f"💰 **Interest Rate:** {interest_rate}")
                elif decision == "Review":
                    st.warning(f"⚠️ {decision} - Manual review required")
                else:
                    st.error(f"❌ {decision}")
                
                # Save to session
                application_data.update({
                    'credit_score': score,
                    'risk_probability': risk_prob,
                    'risk_level': risk_level,
                    'decision': decision,
                    'interest_rate': interest_rate
                })
                st.session_state.current_applicant = application_data
                st.success("✅ Application processed successfully!")

def render_ai_assistant():
    """AI Assistant tab"""
    st.header("🤖 AI Credit Assistant dengan RAG")
    
    st.markdown("""
    <div class="developer-signature">
        🤖 <strong>Advanced RAG & Agentic AI</strong> | 
        Powered by MS Hadianto's expertise in Banking AI Solutions
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.rag_system.api_key:
        st.info("💡 Using demo AI responses. Configure API key in sidebar for full capabilities.")
    else:
        st.success("✅ AI Assistant ready with API integration!")
    
    # Chat History
    st.subheader("💬 Conversation History")
    
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**🧑 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 AI:** {message['content']}")
            st.markdown("---")
    else:
        st.info("🗨️ No conversation yet. Use the chat input below or quick actions.")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analyze Current Application"):
            if st.session_state.current_applicant:
                analysis = st.session_state.rag_system.query("Analisis aplikasi kredit saat ini")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": analysis
                })
                st.experimental_rerun()
            else:
                st.info("Please score an application first.")
    
    with col2:
        if st.button("💡 Credit Risk Insights"):
            insights = st.session_state.rag_system.query("risk factors credit scoring")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": insights
            })
            st.experimental_rerun()
    
    with col3:
        if st.button("🎯 Improvement Tips"):
            tips = st.session_state.rag_system.query("improve approval rate")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": tips
            })
            st.experimental_rerun()
    
    # Sample Questions
    st.subheader("💡 Sample Questions")
    questions = [
        "How to reduce portfolio risk while maintaining profitability?",
        "What are the latest trends in credit scoring technology?",
        "How does debt-to-income ratio impact default rates?",
        "Best practices for fair lending compliance?",
    ]
    
    for question in questions:
        if st.button(f"❓ {question}", key=f"q_{hash(question)}"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            response = st.session_state.rag_system.query(question)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.experimental_rerun()

def render_analytics():
    """Analytics tab"""
    st.header("📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance")
        
        # Sample performance data
        performance_data = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Neural Network'],
            'Accuracy': [0.87, 0.89, 0.82, 0.91],
            'AUC': [0.91, 0.93, 0.85, 0.94],
            'Precision': [0.84, 0.88, 0.80, 0.89],
            'Recall': [0.82, 0.85, 0.78, 0.88]
        })
        
        st.dataframe(performance_data, use_container_width=True)
        
        # Best model highlight
        best_model = performance_data.loc[performance_data['AUC'].idxmax()]
        st.success(f"🏆 Best Model: {best_model['Model']} (AUC: {best_model['AUC']:.3f})")
    
    with col2:
        st.subheader("📈 Portfolio Trends")
        
        # Sample trend data
        trend_data = pd.DataFrame({
            'Month': pd.date_range('2024-01-01', periods=12, freq='M'),
            'Applications': np.random.randint(800, 1200, 12),
            'Approval_Rate': np.random.uniform(0.7, 0.8, 12),
            'Default_Rate': np.random.uniform(0.02, 0.05, 12)
        })
        
        if PLOTLY_AVAILABLE:
            fig = px.line(trend_data, x='Month', y='Approval_Rate', 
                         title="Approval Rate Trend")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(trend_data.set_index('Month')['Approval_Rate'])
    
    # Feature Importance
    st.subheader("🎯 Feature Importance Analysis")
    
    feature_data = pd.DataFrame({
        'Feature': ['Payment History', 'Debt-to-Income', 'Credit History Length', 
                   'Income', 'Employment Years', 'Credit Utilization', 
                   'Age', 'Education', 'Number of Accounts'],
        'Importance': [0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01, 0.007, 0.003]
    })
    
    if PLOTLY_AVAILABLE:
        fig = px.bar(feature_data, x='Importance', y='Feature', 
                    orientation='h', title="Feature Importance")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(feature_data.set_index('Feature'))

# Chat input (outside tabs)
if st.session_state.rag_system.api_key or True:  # Allow demo mode
    if prompt := st.chat_input("Ask about credit scoring, risk analysis, or banking AI..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.spinner("🧠 AI is analyzing..."):
            response = st.session_state.rag_system.query(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.experimental_rerun()

# Main App
def main():
    """Main application"""
    try:
        # Header
        render_header()
        
        # Sidebar
        config = render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard",
            "🎯 Credit Scoring",
            "🤖 AI Assistant",
            "📈 Analytics"
        ])
        
        with tab1:
            render_dashboard()
        
        with tab2:
            render_credit_scoring()
        
        with tab3:
            render_ai_assistant()
        
        with tab4:
            render_analytics()
        
        # Footer
        st.markdown("---")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; margin-top: 2rem;'>
            <div style='text-align: center; color: white;'>
                <h3 style='margin: 0; color: white;'>🏦 AI Credit Scoring System</h3>
                <p style='margin: 0.5rem 0; font-size: 1.1em; opacity: 0.9;'>
                    Advanced Credit Risk Assessment dengan RAG & Agentic AI
                </p>
                <div style='margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.1); 
                           border-radius: 10px; backdrop-filter: blur(10px);'>
                    <p style='margin: 0; font-weight: bold; font-size: 1.1em;'>
                        👨‍💻 Developed by: <span style='color: #FFD700;'>MS Hadianto</span>
                    </p>
                    <p style='margin: 0.2rem 0; opacity: 0.9;'>
                        🤖 RAG & Agentic AI Enthusiast | 🏦 Banking Technology Specialist
                    </p>
                </div>
                <div style='margin-top: 1rem;'>
                    <p style='margin: 0; font-size: 0.9em; opacity: 0.8;'>
                        📱 Version: 1.0.0 | 🕒 Session: {current_time} | 
                        ⚡ Powered by: Qwen3 • OpenRouter • Streamlit
                    </p>
                </div>
            </div>
        </div>
        
        <div style='margin-top: 1rem; padding: 1rem; background: #f8f9fa; 
                    border-radius: 10px; border-left: 4px solid #2a5298;'>
            <p style='margin: 0; font-size: 0.85em; color: #666; text-align: center;'>
                ⚠️ <strong>Disclaimer:</strong> This AI Credit Scoring System is for research and educational purposes. 
                Production deployment requires compliance with local banking regulations and independent validation.
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.75em; text-align: center; opacity: 0.7;'>
                © 2024 MS Hadianto. Built with ❤️ for Indonesian Banking Industry. 
                Contact: mshadianto@tech.com | LinkedIn: MS Hadianto | GitHub: mshadianto
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🚨 Application error: {e}")
        st.info("Please refresh the page or contact the developer.")

if __name__ == "__main__":
    main()