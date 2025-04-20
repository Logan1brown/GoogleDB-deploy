"""Admin page for the dashboard."""
import streamlit as st
from src.shared.auth import auth_required

@auth_required
def show():
    """Display the admin page."""
    st.title("Admin Dashboard")
    # Import content from 6_admin.py
    with open("/mount/src/googledb-deploy/src/dashboard/pages/6_admin.py") as f:
        exec(f.read())
