"""Data Entry page for the dashboard."""
import streamlit as st
from src.dashboard.auth.auth_required import auth_required

@auth_required
def show():
    """Display the data entry page."""
    st.title("Data Entry")
    # Import content from 5_data_entry.py
    with open("/mount/src/googledb-deploy/src/dashboard/pages/5_data_entry.py") as f:
        exec(f.read())
