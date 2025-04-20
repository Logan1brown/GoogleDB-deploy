"""Login page for the dashboard."""

import streamlit as st
import sys
import os

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from dashboard.auth.auth_state import init_auth_state, login

# Page config
st.set_page_config(
    page_title="Login - TV Series Database",
    page_icon="🔐",
    layout="centered"
)

def main():
    """Render login page."""
    # Initialize auth state
    init_auth_state()
    
    # If already logged in, redirect to home
    if st.session_state.authenticated:
        st.switch_page("app.py")
        return
        
    # Header
    st.title("🔐 Login")
    st.write("Please log in to access the TV Series Database Dashboard")
    
    # Login form
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if login(email, password):
                st.success("Logged in successfully!")
                st.rerun()  # Rerun to trigger redirect

if __name__ == "__main__":
    main()
