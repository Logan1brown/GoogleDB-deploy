"""Overview page for the TV Series Database Dashboard."""

import streamlit as st
from src.dashboard.auth.auth_required import auth_required
from src.dashboard.utils.timing import time_page
from src.dashboard.utils.style_config import COLORS, FONTS

@auth_required
def show():
    """Display the overview page."""
    # Page title using style from style_config
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Overview</p>', unsafe_allow_html=True)

    # Overview content
    st.markdown("### TV Series Market Analysis Dashboard")
    
    # Welcome message
    st.markdown("""
    Welcome to the TV Series Market Analysis Dashboard. This tool provides insights into:
    
    - Market trends and patterns
    - Content analysis across platforms
    - Studio performance metrics
    - Audience engagement data
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Series", "2,547", "+12%")
    
    with col2:
        st.metric("Platforms", "15", "+2")
    
    with col3:
        st.metric("Studios", "342", "+5%")
