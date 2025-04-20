"""Custom navigation menu for the data entry app."""
import streamlit as st

def show_auth_status():
    """Show authentication status and controls in sidebar."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        st.sidebar.success("Logged in")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    else:
        st.sidebar.warning("Please log in to access this page")
        from data_entry_app_v3 import login
        login()

def menu():
    """Show auth status and let Streamlit handle navigation."""
    # Always show auth status at top of sidebar
    show_auth_status()
    
    # Let Streamlit handle the page navigation
    if not st.session_state.authenticated:
        st.error("Please log in to access the data entry system")
        st.stop()

def menu_with_redirect():
    """Show menu and handle auth state."""
    menu()
