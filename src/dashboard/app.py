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
    
    # Add navigation menu
    
    try:
        # Set up navigation
        pages = {
            "Home": [st.Page(main_page)],
            "Dashboard": [
                st.Page("src/dashboard/pages/1_overview.py"),
                st.Page("src/dashboard/pages/2_market_snapshot.py"),
                st.Page("src/dashboard/pages/3_content_analysis.py"),
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
