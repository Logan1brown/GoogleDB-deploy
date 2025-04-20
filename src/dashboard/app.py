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
    
    # Navigation
    st.sidebar.title("Navigation")
    
    # Main dashboard pages
    page = st.sidebar.selectbox("Dashboard", ["Overview", "Market Snapshot", "Content Analysis", "Studio Performance"])
    
    try:
        if page == "Overview":
            from src.dashboard.pages.overview import show
            show()
        elif page == "Market Snapshot":
            from src.dashboard.pages.market_snapshot import show
            show()
        elif page == "Content Analysis":
            from src.dashboard.pages.content_analysis import show
            show()
        elif page == "Studio Performance":
            from src.dashboard.pages.studio_performance import show
            show()
        
        # Data Management and Admin sections (only show if appropriate role)
        if role in ['admin', 'editor']:
            section = st.sidebar.selectbox("Other", ["Data Management", "Admin"])
            if section == "Data Management":
                from src.dashboard.pages.data_entry import show
                show()
            elif section == "Admin" and role == 'admin':
                from src.dashboard.pages.admin import show
                show()
    except Exception as e:
        st.error("An error occurred loading the page. Please try again.")
        st.error(str(e))

if __name__ == "__main__":
    main()
