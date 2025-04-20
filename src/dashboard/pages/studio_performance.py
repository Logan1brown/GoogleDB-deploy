"""Studio Performance page for the dashboard."""

import streamlit as st
from src.dashboard.auth.auth_required import auth_required
from src.dashboard.utils.timing import time_page

@auth_required
def show():
    """Display the studio performance page."""
    st.title("Studio Performance")
    
    # Studio metrics
    st.header("Studio Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Studios", "342", "+5%")
        st.markdown("### Top Studios")
        st.bar_chart({"Warner": 50, "Universal": 40, "Sony": 30})
    
    with col2:
        st.metric("Average Shows per Studio", "7.4", "+0.3")
        st.markdown("### Genre Focus")
        st.bar_chart({"Drama": 45, "Comedy": 35, "Action": 20})
    
    with col3:
        st.metric("New Productions", "89", "+12")
        st.markdown("### Platform Distribution")
        st.bar_chart({"Streaming": 60, "Cable": 25, "Network": 15})
    
    # Performance trends
    st.header("Performance Trends")
    st.line_chart({"Q1 2023": 100, "Q2 2023": 110, "Q3 2023": 125, "Q4 2023": 140})
