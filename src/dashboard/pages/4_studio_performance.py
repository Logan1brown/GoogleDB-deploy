"""
Studio Performance Page

Analyzes and visualizes studio performance metrics.
"""

import streamlit as st
import sys
import os

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required
from src.dashboard.utils.timing import time_page
from src.dashboard.utils.style_config import COLORS, FONTS
from src.data_processing.studio_performance.studio_analyzer import StudioAnalyzer
from src.dashboard.components.studio_view import render_studio_performance_dashboard
from src.dashboard.state.session import get_page_state

@time_page
@auth_required()
def show():
    # Page title using style from style_config
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Studio Performance</p>', unsafe_allow_html=True)

    try:
        # Get page state
        state = get_page_state("studio_performance")
        
        # Initialize StudioAnalyzer and fetch data
        studio_analyzer = StudioAnalyzer()
        shows_df, studio_categories_df = studio_analyzer.fetch_studio_data()
        
        # Render the studio performance dashboard
        render_studio_performance_dashboard(shows_df, studio_categories_df)
    except Exception as e:
        st.error(f"Error loading studio performance data: {str(e)}")

if __name__ == "__main__":
    show()
