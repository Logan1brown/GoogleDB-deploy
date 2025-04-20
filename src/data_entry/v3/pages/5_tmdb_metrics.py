"""View TMDB integration metrics and status."""
import streamlit as st
import sys
import os

# Add parent directory to path so we can import from the parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import menu
from data_entry_menu import menu_with_redirect

# Show menu and redirect if not authenticated
menu_with_redirect()

st.title("TMDB Metrics")
st.info("Coming soon! This page will show TMDB integration metrics and status.")
