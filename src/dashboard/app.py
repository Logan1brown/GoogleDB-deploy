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

from src.dashboard.auth.auth_required import auth_required
from src.dashboard.auth.auth_state import init_auth_state, login, logout, refresh_session
from src.dashboard.utils.style_config import COLORS, FONTS
from src.shared.auth import get_user_role

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
    st.markdown("""
    ### Welcome to the TV Series Database Dashboard
    
    This dashboard provides tools for:
    - 📊 Market analysis and insights
    - 📈 Content performance tracking
    - 🏢 Studio performance metrics
    - ✨ Data entry and management
    
    Select a page from the menu to get started.
    """)

def main():
    """Main application entry point."""
    # Initialize auth state
    init_auth_state()
    
    # If not authenticated, show login in sidebar
    if not st.session_state.get('authenticated'):
        show_login()
    
    # Session refresh is now handled by auth_required decorator
    
    # Show user info and logout in sidebar if authenticated
    if st.session_state.get('authenticated') and st.session_state.get('user'):
        st.sidebar.info(f"Logged in as: {st.session_state.user.email}")
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()
    
    # Get user role if authenticated
    role = None
    if st.session_state.get('authenticated') and st.session_state.get('user'):
        role = get_user_role()
        if not role:
            st.sidebar.warning("Your account has not been assigned a role yet. Please contact an administrator.")
    
    # Initialize pages for navigation
    # All pages are always visible, access control is handled by auth_required decorator
    st.sidebar.title("Navigation")
    
    # Main dashboard pages
    page = st.sidebar.selectbox("Dashboard", ["Overview", "Market Snapshot", "Content Analysis", "Studio Performance"])
    
    if page == "Overview":
        from src.dashboard.pages import overview
        overview.show()
    elif page == "Market Snapshot":
        from src.dashboard.pages import market_snapshot
        market_snapshot.show()
    elif page == "Content Analysis":
        from src.dashboard.pages import content_analysis
        content_analysis.show()
    elif page == "Studio Performance":
        from src.dashboard.pages import studio_performance
        studio_performance.show()
    
    # Data Management and Admin sections
    section = st.sidebar.selectbox("Other", ["Data Management", "Admin"])
    if section == "Data Management":
        from src.dashboard.pages import data_entry
        data_entry.show()
    elif section == "Admin":
        from src.dashboard.pages import admin
        admin.show()
    
    # Set up navigation
    pages = {
        "Home": [st.Page(main_page)],
        "Dashboard": [
            st.Page("pages/1_overview.py"),
            st.Page("pages/2_market_snapshot.py"),
            st.Page("pages/3_content_analysis.py"),
            st.Page("pages/4_studio_performance.py")
        ],
        "Data Management": [st.Page("pages/5_data_entry.py")],
        "Admin": [st.Page("pages/6_admin.py")]
    }
    
    # Run navigation
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()
