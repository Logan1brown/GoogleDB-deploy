"""Review and approve TMDB matches for shows."""
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

st.title("TMDB Matches")
st.info("Coming soon! This page will allow you to review and approve TMDB matches for shows.")
