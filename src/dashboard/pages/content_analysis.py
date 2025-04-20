"""Content Analysis page for the dashboard."""

import streamlit as st
from src.dashboard.auth.auth_required import auth_required
from src.dashboard.utils.timing import time_page

@auth_required
def show():
    """Display the content analysis page."""
    st.title("Content Analysis")
    
    # Content overview
    st.header("Content Overview")
    
    # Content metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Shows", "1,234", "+15%")
        st.markdown("### Top Genres")
        st.bar_chart({"Drama": 40, "Comedy": 30, "Action": 20})
    
    with col2:
        st.metric("Average Episodes", "12", "-2")
        st.markdown("### Runtime Distribution")
        st.bar_chart({"30min": 45, "60min": 40, "Other": 15})
    
    with col3:
        st.metric("New Shows", "127", "+5%")
        st.markdown("### Platform Distribution")
        st.bar_chart({"Netflix": 35, "Amazon": 25, "HBO": 20})
    
    # Content trends
    st.header("Content Trends")
    st.line_chart({"2020": 100, "2021": 120, "2022": 150, "2023": 180})
