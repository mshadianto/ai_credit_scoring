import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Credit Scoring",
    page_icon="Ìø¶",
    layout="wide"
)

st.title("Ìø¶ AI Credit Scoring System")
st.markdown("**Created by MS Hadianto | RAG & Agentic AI Enthusiast**")

st.success("‚úÖ Basic app deployed successfully!")
st.info("Full features will be available once all dependencies are resolved.")

# Simple credit scoring demo
st.header("ÌæØ Quick Credit Score Demo")

age = st.slider("Age", 18, 80, 35)
income = st.number_input("Annual Income (IDR)", value=500_000_000)
debt_ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)

if st.button("Calculate Score"):
    # Simple scoring algorithm
    score = 300 + (1 - debt_ratio) * 200 + min(age/80, 1) * 150 + min(income/1_000_000_000, 1) * 200
    score = min(int(score), 850)
    
    st.metric("Credit Score", score)
    
    if score >= 700:
        st.success("‚úÖ Excellent Credit")
    elif score >= 600:
        st.info("‚ÑπÔ∏è Good Credit") 
    else:
        st.warning("‚ö†Ô∏è Fair Credit")

st.markdown("---")
st.markdown("¬© 2024 MS Hadianto - AI Credit Scoring System")
