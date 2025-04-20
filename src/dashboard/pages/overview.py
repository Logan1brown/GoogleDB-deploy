"""Overview page for the dashboard."""
import streamlit as st
from src.dashboard.auth.auth_required import auth_required

@auth_required
def show():
    """Display the overview page."""
    st.title("TV Series Market Analysis")
    # Import content from 1_overview.py
    with open("/mount/src/googledb-deploy/src/dashboard/pages/1_overview.py") as f:
        exec(f.read())
