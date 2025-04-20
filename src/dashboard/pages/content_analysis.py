"""Content Analysis page for the dashboard."""
import streamlit as st
from src.dashboard.auth.auth_required import auth_required

@auth_required
def show():
    """Display the content analysis page."""
    st.title("Content Analysis")
    # Import content from 3_content_analysis.py
    with open("/mount/src/googledb-deploy/src/dashboard/pages/3_content_analysis.py") as f:
        exec(f.read())
