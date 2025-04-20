"""Studio Performance page for the dashboard."""
import streamlit as st
from src.dashboard.auth.auth_required import auth_required

@auth_required
def show():
    """Display the studio performance page."""
    st.title("Studio Performance")
    # Import content from 4_studio_performance.py
    with open("/mount/src/googledb-deploy/src/dashboard/pages/4_studio_performance.py") as f:
        exec(f.read())
