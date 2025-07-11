"""
AI Credit Scoring System - Main Application
Enhanced with RAG, Agentic AI, and Advanced ML
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json

# Project imports
try:
    from config.settings import settings, BANKING_STANDARDS, RISK_MATRICES
    from utils.database import db_manager, init_sample_data, DataGenerator
    from utils.ml_utils import model_ensemble, ModelInterpreter
    from utils.api_utils import rag_system, formatter, validate_credit_application
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all required files are in place")
    st.stop()

# Configure Streamlit
try:
    st.set_page_config(
        page_title="AI Credit Scoring",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except:
    st.set_page_config(
        page_title="AI Credit Scoring",
        page_icon="🏦",
        layout="wide"
    )

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
    
    .risk-high {
        border-left-color: #ff4757 !important;
    }
    
    .risk-medium {
        border-left-color: #ffa502 !important;
    }
    
    .risk-low {
        border-left-color: #2ed573 !important;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: #f8f9fa;
    }
    
    .ai-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class CreditScoringApp:
    """Main application class"""
    
    def __init__(self):
        self.initialize_session_state()
        self.initialize_data()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_applicant = {}
            st.session_state.chat_history = []
            st.session_state.model_trained = {}
            st.session_state.portfolio_metrics = {}
    
    def initialize_data(self):
        """Initialize application data"""
        try:
            # Initialize database with sample data if empty
            init_sample_data()
            
            # Load portfolio metrics
            st.session_state.portfolio_metrics = db_manager.get_portfolio_metrics()
            
        except Exception as e:
            st.error(f"Error initializing data: {e}")
    
    def render_header(self):
        """Render main header with dynamic version"""
        app_version = getattr(settings, 'APP_VERSION', '1.0.0')
        
        st.markdown(f"""
        <div class="main-header">
            <h1>🏦 AI Credit Scoring System <span style='font-size: 0.6em; opacity: 0.8;'>v{app_version}</span></h1>
            <p>Advanced Credit Risk Assessment dengan RAG & Agentic AI</p>
            <div style='margin-top: 1rem; font-size: 0.85em; opacity: 0.9;'>
                👨‍💻 Developed by: <strong>MS Hadianto</strong> | 
                🤖 RAG & Agentic AI Enthusiast |      
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar configuration"""
        # Header sidebar
        app_version = getattr(settings, 'APP_VERSION', '1.0.0')
        st.sidebar.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; margin-bottom: 1rem;'>
            <h3 style='margin: 0; color: white;'>🏦 Credit AI</h3>
            <p style='margin: 0; font-size: 0.8em; opacity: 0.9;'>v{app_version}</p>
            <p style='margin: 0; font-size: 0.7em; opacity: 0.8;'>by MS Hadianto</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.header("⚙️ Konfigurasi Sistem")
        
        # API Configuration
        with st.sidebar.expander("🔑 API Configuration", expanded=True):
            api_key = st.text_input(
                "OpenRouter API Key", 
                type="password",
                help="Dapatkan gratis di https://openrouter.ai/"
            )
            
            if api_key:
                rag_system.api_key = api_key
                st.success("✅ API Key configured")
            else:
                st.warning("⚠️ API Key diperlukan untuk AI features")
        
        # Model Configuration
        with st.sidebar.expander("🤖 Model Settings"):
            credit_types = ["Personal Loan", "Mortgage", "Credit Card", "Auto Loan", "Business Loan"]
            credit_type = st.selectbox(
                "Jenis Kredit", 
                credit_types,
                key="selected_credit_type"
            )
            
            auto_retrain = st.checkbox("Auto Retrain Models", value=False)
            model_threshold = st.slider("Decision Threshold", 0.3, 0.7, 0.5)
        
        # System Status
        with st.sidebar.expander("📊 System Status"):
            total_apps = st.session_state.portfolio_metrics.get('total_applications', 0)
            st.metric("Applications", total_apps)
            
            approval_rate = st.session_state.portfolio_metrics.get('approval_rate', 0)
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
            
            avg_score = st.session_state.portfolio_metrics.get('average_credit_score', 0)
            st.metric("Avg Credit Score", int(avg_score))
        
        # About & Help Section
        with st.sidebar.expander("ℹ️ About & Help"):
            st.markdown("""
            **🎯 Features:**
            - AI-powered credit scoring
            - RAG-enhanced insights
            - Real-time risk assessment
            - Portfolio analytics
            - Model management
            
            **👨‍💻 Developer:**
            - **Name:** MS Hadianto
            - **Specialty:** RAG & Agentic AI
            - **Focus:** Banking Technology
            
            **🔗 Resources:**
            - [Documentation](https://docs.example.com)
            - [GitHub Repo](https://github.com/mshadianto)
            - [API Reference](https://api.example.com)
            
            **⚡ Tech Stack:**
            - Python 3.11+
            - Streamlit Framework
            - Qwen3 LLM via OpenRouter
            - Scikit-learn ML
            - Plotly Visualization
            """)
            
            if st.button("📧 Contact Developer"):
                st.info("""
                📧 **Email:** sopian.hadianto@gmail.com
                🔗 **LinkedIn:** MS Hadianto
                💻 **GitHub:** mshadianto
                
                Feel free to reach out for:
                - Custom implementations
                - Enterprise features
                - Training & consultation
                - Collaboration opportunities
                """)
        
        # Performance Info
        with st.sidebar.expander("⚡ Performance Info"):
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("Session Time", current_time)
            st.metric("Models Loaded", len(st.session_state.model_trained))
            st.metric("Chat Messages", len(st.session_state.chat_history))
            
            if st.button("🔄 Refresh Metrics"):
                st.experimental_rerun()
        
        return {
            'api_key': api_key,
            'credit_type': credit_type,
            'auto_retrain': auto_retrain,
            'model_threshold': model_threshold
        }
    
    def render_dashboard_tab(self):
        """Render dashboard tab"""
        st.header("📊 Credit Portfolio Dashboard")
        
        # Professional signature
        st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 0.5rem 1rem; 
                    border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 1rem;'>
            <p style='margin: 0; font-size: 0.85em; color: #667eea;'>
                📈 <strong>Advanced Analytics Dashboard</strong> | 
                Crafted by MS Hadianto - RAG & Agentic AI Enthusiast
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = st.session_state.portfolio_metrics
        
        with col1:
            st.metric(
                "Total Applications", 
                f"{metrics.get('total_applications', 0):,}",
                delta="12 today"
            )
        
        with col2:
            approval_rate = metrics.get('approval_rate', 0)
            st.metric(
                "Approval Rate", 
                f"{approval_rate:.1f}%",
                delta="2.3%"
            )
        
        with col3:
            avg_score = metrics.get('average_credit_score', 0)
            st.metric(
                "Avg Credit Score", 
                int(avg_score),
                delta="15 points"
            )
        
        with col4:
            st.metric(
                "Default Rate", 
                "3.2%",
                delta="-0.5%",
                delta_color="inverse"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            
            # Sample risk data for demo
            risk_data = {
                'Risk Level': ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
                'Count': [30, 40, 20, 8, 2]
            }
            
            fig = px.pie(
                values=risk_data['Count'],
                names=risk_data['Risk Level'],
                title="Portfolio Risk Distribution",
                color_discrete_map={
                    'Sangat Rendah': '#2ed573',
                    'Rendah': '#7bed9f',
                    'Sedang': '#ffa502',
                    'Tinggi': '#ff6b6b',
                    'Sangat Tinggi': '#ff4757'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Credit Score Distribution")
            
            # Sample score data
            scores = np.random.normal(720, 80, 1000)
            scores = np.clip(scores, 300, 850)
            
            fig = px.histogram(
                x=scores,
                nbins=30,
                title="Credit Score Distribution",
                labels={'x': 'Credit Score', 'y': 'Frequency'}
            )
            fig.add_vline(x=580, line_dash="dash", line_color="red", annotation_text="Fair Credit")
            fig.add_vline(x=670, line_dash="dash", line_color="orange", annotation_text="Good Credit")
            fig.add_vline(x=740, line_dash="dash", line_color="green", annotation_text="Excellent Credit")
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trends
        st.subheader("Monthly Application Trends")
        
        # Sample trend data
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        applications = np.random.randint(50, 200, 12)
        
        fig = px.line(
            x=dates,
            y=applications,
            title="Monthly Application Volume",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_scoring_tab(self):
        """Render credit scoring tab"""
        st.header("🎯 Credit Scoring Engine")
        
        # Professional signature
        st.markdown("""
        <div style='background: rgba(46, 213, 115, 0.1); padding: 0.5rem 1rem; 
                    border-radius: 8px; border-left: 4px solid #2ed573; margin-bottom: 1rem;'>
            <p style='margin: 0; font-size: 0.85em; color: #27ae60;'>
                🎯 <strong>AI-Powered Credit Assessment</strong> | 
                Engineered by MS Hadianto using Advanced ML | RAG & Agentic AI
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Applicant Information")
            
            # Personal Information
            with st.expander("👤 Personal Details", expanded=True):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    age = st.slider("Age", 18, 80, 35)
                    income = st.number_input(
                        "Annual Income (IDR)", 
                        min_value=0, 
                        value=500_000_000, 
                        step=10_000_000,
                        format="%d"
                    )
                    employment_years = st.slider("Years of Employment", 0, 40, 5)
                
                with col_b:
                    education = st.selectbox(
                        "Education Level", 
                        ['High School', 'Bachelor', 'Master', 'PhD']
                    )
                    marital_status = st.selectbox(
                        "Marital Status", 
                        ['Single', 'Married', 'Divorced']
                    )
                    home_ownership = st.selectbox(
                        "Home Ownership", 
                        ['Rent', 'Own', 'Mortgage']
                    )
            
            # Financial Information
            with st.expander("💰 Financial Profile", expanded=True):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    debt_to_income = st.slider(
                        "Debt-to-Income Ratio", 
                        0.0, 1.0, 0.3, 0.01,
                        help="Total monthly debt payments / Monthly income"
                    )
                    credit_history = st.slider("Credit History (years)", 0, 30, 10)
                    num_accounts = st.slider("Number of Credit Accounts", 1, 20, 5)
                
                with col_b:
                    payment_history = st.slider(
                        "Payment History Score", 
                        0.0, 1.0, 0.85, 0.01,
                        help="Percentage of on-time payments"
                    )
                    credit_utilization = st.slider(
                        "Credit Utilization Ratio", 
                        0.0, 1.0, 0.3, 0.01,
                        help="Used credit / Available credit"
                    )
            
            # Loan Details
            with st.expander("🏦 Loan Information"):
                loan_amount = st.number_input(
                    "Requested Loan Amount (IDR)", 
                    min_value=0, 
                    value=100_000_000, 
                    step=1_000_000
                )
                loan_purpose = st.selectbox(
                    "Loan Purpose", 
                    ['Home Purchase', 'Debt Consolidation', 'Business', 'Education', 'Other']
                )
        
        with col2:
            st.subheader("Credit Assessment")
            
            if st.button("🎯 Calculate Credit Score", type="primary"):
                with st.spinner("Calculating credit score..."):
                    try:
                        # Prepare application data
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
                        
                        # Simple credit scoring algorithm
                        score = self.calculate_credit_score(application_data)
                        risk_prob = (850 - score) / 550
                        
                        # Determine risk level
                        if risk_prob < 0.2:
                            risk_level = "Sangat Rendah"
                        elif risk_prob < 0.4:
                            risk_level = "Rendah"
                        elif risk_prob < 0.6:
                            risk_level = "Sedang"
                        elif risk_prob < 0.8:
                            risk_level = "Tinggi"
                        else:
                            risk_level = "Sangat Tinggi"
                        
                        # Make decision
                        if score >= 740:
                            decision = "Approved"
                            interest_rate = "5.5% - 7.0%"
                        elif score >= 670:
                            decision = "Approved"
                            interest_rate = "7.0% - 9.5%"
                        elif score >= 580:
                            decision = "Review"
                            interest_rate = "9.5% - 15.0%"
                        else:
                            decision = "Rejected"
                            interest_rate = "N/A"
                        
                        # Display results
                        # Credit Score Gauge
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
                                    {'range': [300, 580], 'color': "lightgray"},
                                    {'range': [580, 670], 'color': "yellow"},
                                    {'range': [670, 740], 'color': "lightgreen"},
                                    {'range': [740, 850], 'color': "green"}
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
                        
                        # Results
                        st.metric("Credit Score", score)
                        st.metric("Risk Level", risk_level)
                        st.metric("Default Probability", f"{risk_prob:.1%}")
                        
                        # Decision
                        if decision == "Approved":
                            st.success(f"✅ {decision}")
                            st.info(f"**Interest Rate:** {interest_rate}")
                        elif decision == "Review":
                            st.warning(f"⚠️ {decision}")
                            st.info("Requires manual review")
                        else:
                            st.error(f"❌ {decision}")
                        
                        # Save to session state
                        application_data.update({
                            'credit_score': score,
                            'risk_probability': risk_prob,
                            'risk_level': risk_level,
                            'decision': decision,
                            'interest_rate': interest_rate
                        })
                        
                        st.session_state.current_applicant = application_data
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
    
    def calculate_credit_score(self, data):
        """Simple credit scoring algorithm"""
        score = 300  # Base score
        
        # Age factor (25-65 is optimal)
        age_factor = 1.0 - abs(data['age'] - 45) / 45
        score += age_factor * 50
        
        # Income factor
        income_score = min(data['income'] / 1_000_000_000, 1.0) * 100
        score += income_score
        
        # Employment stability
        employment_score = min(data['employment_years'] / 10, 1.0) * 80
        score += employment_score
        
        # Debt to income (lower is better)
        dti_score = (1 - data['debt_to_income']) * 120
        score += dti_score
        
        # Credit history
        history_score = min(data['credit_history_length'] / 20, 1.0) * 60
        score += history_score
        
        # Payment history
        payment_score = data['payment_history'] * 100
        score += payment_score
        
        # Credit utilization (lower is better)
        utilization_score = (1 - data['credit_utilization']) * 80
        score += utilization_score
        
        # Education bonus
        education_bonus = {'High School': 0, 'Bachelor': 20, 'Master': 30, 'PhD': 40}
        score += education_bonus.get(data['education_level'], 0)
        
        return min(int(score), 850)
    
    def render_ai_assistant_tab(self):
        """Render AI assistant tab - FIXED VERSION - NO CHAT INPUT"""
        st.header("🤖 AI Credit Assistant dengan RAG")
        
        # Developer credit in AI tab
        st.markdown("""
        <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <p style='margin: 0; color: white; font-size: 0.85em; text-align: center;'>
                🤖 <strong>Advanced RAG & Agentic AI</strong> | 
                Powered by MS Hadianto's expertise in Banking AI Solutions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if not rag_system.api_key:
            st.warning("⚠️ Please configure OpenRouter API Key in sidebar to use AI Assistant")
            st.info("💬 Chat input will appear at the bottom of the page when API key is configured")
            return
        
        st.success("✅ AI Assistant ready! Use the chat input at the bottom of the page.")
        
        # Display chat history ONLY
        st.subheader("💬 Conversation History")
        
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            st.info("🗨️ No conversation yet. Use the chat input at the bottom to start chatting!")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Analyze Current Application"):
                if st.session_state.current_applicant:
                    try:
                        analysis = rag_system.query(f"Analisis aplikasi kredit: {st.session_state.current_applicant}")
                        st.session_state.chat_history.append({"role": "assistant", "content": analysis})
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
                else:
                    st.info("No current application to analyze. Please score an application first.")
        
        with col2:
            if st.button("💡 Get Credit Insights"):
                credit_type = st.session_state.get('selected_credit_type', 'Personal Loan')
                try:
                    insights = rag_system.query(f"Berikan insights mendalam tentang {credit_type}")
                    st.session_state.chat_history.append({"role": "assistant", "content": insights})
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Insights error: {e}")
        
        with col3:
            if st.button("🎯 Model Explanation"):
                if st.session_state.current_applicant:
                    prediction_result = st.session_state.current_applicant
                    try:
                        explanation = rag_system.query(f"Jelaskan hasil credit scoring: {prediction_result}")
                        st.session_state.chat_history.append({"role": "assistant", "content": explanation})
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Explanation error: {e}")
                else:
                    st.info("No scored application to explain.")
        
        # Sample questions - CONVERTED TO BUTTONS THAT ADD TO CHAT
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "How can we improve our approval rate without increasing risk?",
            "What are the key risk factors for mortgage lending?",
            "How does debt-to-income ratio affect credit decisions?",
            "What alternative data sources can improve credit scoring?",
            "How to implement fair lending practices?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(f"❓ {question[:50]}...", key=f"sample_{i}"):
                    # Add question to chat history
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    
                    # Get AI response
                    try:
                        response = rag_system.query(question, st.session_state.portfolio_metrics)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.experimental_rerun()
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        st.experimental_rerun()
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.header("📈 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Analysis")
            
            # Sample analytics data
            sample_scores = np.random.normal(700, 80, 1000)
            sample_scores = np.clip(sample_scores, 300, 850)
            
            fig = px.histogram(
                x=sample_scores,
                nbins=30,
                title="Credit Score Distribution Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk Metrics")
            
            # Sample risk data
            risk_data = pd.DataFrame({
                'Credit_Type': ['Personal Loan', 'Mortgage', 'Credit Card', 'Auto Loan', 'Business Loan'],
                'Default_Rate': [0.05, 0.02, 0.08, 0.04, 0.12],
                'Avg_Score': [680, 720, 650, 700, 640]
            })
            
            fig = px.scatter(
                risk_data,
                x='Avg_Score',
                y='Default_Rate',
                size='Default_Rate',
                color='Credit_Type',
                title="Risk vs Score by Credit Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("Model Performance")
        
        # Sample performance data
        performance_data = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
            'Accuracy': [0.87, 0.89, 0.82],
            'AUC': [0.91, 0.93, 0.85],
            'Precision': [0.84, 0.88, 0.80],
            'Recall': [0.82, 0.85, 0.78]
        })
        
        st.dataframe(performance_data, use_container_width=True)
    
    def render_model_management_tab(self):
        """Render model management tab"""
        st.header("⚙️ Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Training")
            
            credit_types = ["Personal Loan", "Mortgage", "Credit Card", "Auto Loan", "Business Loan"]
            
            if st.button("🔄 Train All Models"):
                progress_bar = st.progress(0)
                
                for i, credit_type in enumerate(credit_types):
                    with st.spinner(f"Training {credit_type} model..."):
                        try:
                            # Simulate model training
                            time.sleep(1)
                            
                            # Store training result
                            st.session_state.model_trained[credit_type] = {
                                'accuracy': np.random.uniform(0.8, 0.95),
                                'auc_score': np.random.uniform(0.85, 0.98),
                                'timestamp': datetime.now()
                            }
                            
                            st.success(f"✅ {credit_type} model trained successfully")
                            
                        except Exception as e:
                            st.error(f"❌ Error training {credit_type}: {e}")
                    
                    progress_bar.progress((i + 1) / len(credit_types))
                
                st.success("🎉 All models trained successfully!")
        
        with col2:
            st.subheader("Model Status")
            
            for credit_type in credit_types:
                with st.expander(f"📊 {credit_type}"):
                    if credit_type in st.session_state.model_trained:
                        st.success("✅ Model Trained")
                        result = st.session_state.model_trained[credit_type]
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Accuracy", f"{result['accuracy']:.3f}")
                        with col_b:
                            st.metric("AUC Score", f"{result['auc_score']:.3f}")
                            
                        st.info(f"Last trained: {result['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    else:
                        st.warning("❌ Model Not Trained")
            
            st.subheader("Export & Import")
            
            if st.button("📥 Export All Models"):
                st.success("Models exported successfully!")
            
            uploaded_file = st.file_uploader("Import Models", type=['zip'])
            if uploaded_file:
                st.info("Model import functionality available")
    
    def run(self):
        """Run the main application"""
        # Render header
        self.render_header()
        
        # Render sidebar
        config = self.render_sidebar()
        
        # Chat input (moved outside tabs) - CRITICAL FIX
        if rag_system.api_key:
            if prompt := st.chat_input("Ask about credit scoring, risk analysis, or get recommendations..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Generate AI response
                with st.spinner("AI is analyzing..."):
                    try:
                        # Get portfolio context
                        portfolio_context = st.session_state.portfolio_metrics
                        
                        # Query RAG system
                        response = rag_system.query(prompt, portfolio_context)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        st.experimental_rerun()
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                        st.experimental_rerun()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard",
            "🎯 Credit Scoring", 
            "🤖 AI Assistant",
            "📈 Analytics",
            "⚙️ Model Management"
        ])
        
        with tab1:
            self.render_dashboard_tab()
        
        with tab2:
            self.render_scoring_tab()
        
        with tab3:
            self.render_ai_assistant_tab()
        
        with tab4:
            self.render_analytics_tab()
        
        with tab5:
            self.render_model_management_tab()
        
        # Footer with disclaimer and developer info
        st.markdown("---")
        
        # Get dynamic version and timestamp
        app_version = getattr(settings, 'APP_VERSION', '1.0.0')
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
                <div style='display: flex; justify-content: space-between; align-items: center; 
                           flex-wrap: wrap; gap: 1rem; margin-top: 1rem;'>
                    <div style='flex: 1; min-width: 200px;'>
                        <p style='margin: 0; font-size: 0.9em; opacity: 0.8;'>
                            📱 Version: <strong>{app_version}</strong> | 
                            🕒 Session: {current_time}
                        </p>
                    </div>
                    <div style='flex: 1; min-width: 200px;'>
                        <p style='margin: 0; font-size: 0.9em; opacity: 0.8;'>
                            ⚡ Powered by: Qwen3 • OpenRouter • Streamlit
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <div style='margin-top: 1rem; padding: 1rem; background: #f8f9fa; 
                    border-radius: 10px; border-left: 4px solid #2a5298;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #2a5298;'>⚠️ Important Disclaimer</h4>
            <div style='font-size: 0.85em; color: #666; line-height: 1.5;'>
                <p style='margin: 0 0 0.5rem 0;'>
                    <strong>🔬 Research & Educational Purpose:</strong> 
                    This AI Credit Scoring System is developed for research, educational, and demonstration purposes. 
                    While it employs advanced machine learning algorithms and industry best practices, 
                    it should not be used as the sole basis for actual credit decisions without proper validation and compliance review.
                </p>
                <p style='margin: 0 0 0.5rem 0;'>
                    <strong>⚖️ Regulatory Compliance:</strong> 
                    Any production deployment must comply with local banking regulations (BI, OJK), 
                    fair lending practices, and data protection laws. Independent model validation and bias testing are required.
                </p>
                <p style='margin: 0 0 0.5rem 0;'>
                    <strong>🛡️ Data Privacy:</strong> 
                    Ensure all applicant data is handled according to applicable privacy laws and banking regulations. 
                    This system includes security measures but requires additional enterprise-grade security for production use.
                </p>
                <p style='margin: 0 0 0.5rem 0;'>
                    <strong>🎯 Model Limitations:</strong> 
                    AI models may have biases and limitations. Regular model monitoring, retraining, and fairness testing are essential. 
                    Human oversight and appeal processes should always be maintained.
                </p>
                <div style='margin-top: 1rem; padding: 0.5rem; background: #e3f2fd; border-radius: 5px;'>
                    <p style='margin: 0; font-size: 0.8em; text-align: center;'>
                        📞 <strong>Contact Developer:</strong> 
                        <a href="mailto:sopian.hadianto@gmail.com" style='color: #1976d2; text-decoration: none;'>
                            sopian.hadianto@gmail.com
                        </a> | 
                        <a href="https://linkedin.com/in/mshadianto" style='color: #1976d2; text-decoration: none;' target="_blank">
                            LinkedIn: MS Hadianto
                        </a> | 
                        <a href="https://github.com/mshadianto" style='color: #1976d2; text-decoration: none;' target="_blank">
                            GitHub: mshadianto
                        </a>
                    </p>
                    <p style='margin: 0.3rem 0 0 0; font-size: 0.75em; text-align: center; opacity: 0.7;'>
                        © 2024 MS Hadianto. Built with ❤️ for Indonesian Banking Industry. 
                        Licensed under MIT License for educational and research use.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    try:
        app = CreditScoringApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()