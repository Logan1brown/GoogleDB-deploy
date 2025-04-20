"""Market Snapshot page for the dashboard."""
import streamlit as st
from src.dashboard.auth.auth_required import auth_required

@auth_required
def show():
    """Display the market snapshot page."""
    st.title("Market Snapshot")
    # Import content from 2_market_snapshot.py
    with open("/mount/src/googledb-deploy/src/dashboard/pages/2_market_snapshot.py") as f:
        exec(f.read())
