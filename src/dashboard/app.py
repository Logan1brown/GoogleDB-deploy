"""
TV Series Database Dashboard
Main application file for the Streamlit dashboard.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Set page config must be first Streamlit command
st.set_page_config(
    page_title="TV Series Dashboard",
    page_icon="📺",
    layout="wide"
)

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required, init_auth_state, login, logout, refresh_session, get_user_role
from src.dashboard.utils.style_config import COLORS, FONTS

def show_login():
    """Show login form in sidebar."""
    with st.sidebar:
        st.markdown("### Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            try:
                if login(email, password):
                    st.rerun()
                else:
                    st.error("Invalid email or password")
            except Exception as e:
                st.error(str(e))

def main_page():
    """Main dashboard page."""
    # Page title
    st.markdown(f'<h1 style="font-family: {FONTS["primary"]["family"]}; color: {COLORS["accent"]}; margin-bottom: 1em;">STS Sales Analysis</h1>', unsafe_allow_html=True)

    # Welcome message
    st.markdown("""
        Explore straight-to-series TV shows through multiple lenses: market dynamics, content strategies, and studio relationships. 
        This dashboard provides data-driven insights from comprehensive tracking of straight-to-series orders across networks and studios.
    """)

    # Dashboard Sections
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Dashboard Sections</p>', unsafe_allow_html=True)

    st.markdown("""
    **Show Detail** *(Coming Soon)*  
    Search and explore detailed information about specific shows:
    - Comprehensive show information including title, network, studios, and creative team
    - View similar shows in the market
    - Analyze network patterns across similar show groups
    - Identify trends and relationships in show packaging and development

    **Market Snapshot**  
    Track the pulse of the straight-to-series market with real-time metrics and trends. Analyze order volumes, 
    network activities, and success rates through interactive visualizations. Filter by shows, creators, genres, 
    and networks to uncover market patterns and opportunities.

    **Content Analysis**  
    Deep dive into content strategies across three key phases:
    - Acquisition: Analyze network preferences
    - Packaging: Explore successful creative team combinations
    - Development: Understand format decisions and their impact

    Filter by source type and genre to reveal patterns in successful content development.

    **Studio Performance**  
    Visualize studio relationships and performance metrics through an interactive network graph. Track studio 
    collaborations, success rates, and partnership patterns. Identify key players and emerging trends in the 
    production landscape.
    """)

    # Dataset section
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Dataset</p>', unsafe_allow_html=True)

    st.markdown("""
    This analysis is based on a curated dataset of straight-to-series orders tracked from Deadline Hollywood.
    While comprehensive within its scope, please note:
    - Focus is on straight-to-series orders rather than traditional pilots
    - Data collection is more complete for recent years
    - Some historical data may be incomplete
    """)

def main():
    """Main application entry point."""
    # Initialize auth state
    init_auth_state()
    
    # Show login form if not authenticated
    if not st.session_state.get('authenticated'):
        show_login()
        st.warning("Please log in to access the dashboard")
        return
    
    # Show logout button
    if st.sidebar.button('Logout'):
        logout()
        st.rerun()
    
    # Show role
    role = get_user_role()
    if role:
        st.sidebar.info(f"Role: {role}")
    else:
        st.sidebar.warning("Your account has not been assigned a role yet. Please contact an administrator.")
        return
    
    # Add navigation menu
    
    try:
        # Set up navigation
        pages = {
            "Home": [st.Page(main_page)],
            "Dashboard": [
                st.Page("src/dashboard/pages/1_show_detail.py"),
                st.Page("src/dashboard/pages/2_market_snapshot.py"),
                st.Page("src/dashboard/pages/3_content_analysis.py"),
                st.Page("src/dashboard/pages/3_1_comp_builder.py"),
                st.Page("src/dashboard/pages/4_studio_performance.py")
            ],
            "Data Management": [st.Page("src/dashboard/pages/5_data_entry.py")],
            "Admin": [st.Page("src/dashboard/pages/6_admin.py")]
        }
        
        # Run navigation
        pg = st.navigation(pages)
        pg.run()
    except Exception as e:
        st.error("An error occurred loading the page. Please try again.")
        st.error(str(e))

if __name__ == "__main__":
    main()
