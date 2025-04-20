"""Overview page for the TV Series Database Dashboard."""

import streamlit as st
from src.dashboard.auth.auth_required import auth_required


@auth_required
def show():
    """Display the overview page."""
    # Page title
    st.title("Overview")

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
