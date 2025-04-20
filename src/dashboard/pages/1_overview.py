"""Overview page for the TV Series Database Dashboard."""

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

@auth_required()
def show():
    """Main page content."""
    # Page title using style from style_config
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Overview</p>', unsafe_allow_html=True)

    # Welcome message
    st.markdown("""
        Explore straight-to-series TV shows through multiple lenses: market dynamics, content strategies, and studio relationships. 
        This dashboard provides data-driven insights from comprehensive tracking of straight-to-series orders across networks and studios.
    """)

    # Dashboard Sections
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Dashboard Sections</p>', unsafe_allow_html=True)

    st.markdown("""
    **Market Snapshot**  
    Track the pulse of the straight-to-series market with real-time metrics and trends. Analyze order volumes, 
    network activities, and success rates through interactive visualizations. Filter by shows, creators, genres, 
    and networks to uncover market patterns and opportunities.

    **Content Analysis**  
    Deep dive into content strategies across three key phases:
    - Acquisition: Analyze network preferences and creator track records
    - Packaging: Explore successful creative team combinations
    - Development: Understand format decisions and their impact

    Filter by source type and genre to reveal patterns in successful content development.

    **Studio Performance**  
    Visualize studio relationships and performance metrics through an interactive network graph. Track studio 
    collaborations, success rates, and partnership patterns. Identify key players and emerging trends in the 
    production landscape.
    """)

    # Dataset section
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Dataset</p>', unsafe_allow_html=True)

    st.markdown("""
    This analysis is based on a curated dataset of straight-to-series orders tracked from Deadline Hollywood.
    While comprehensive within its scope, please note:
    - Focus is on straight-to-series orders rather than traditional pilots
    - Data collection is more complete for recent years
    - Some historical data may be incomplete
    """)

if __name__ == "__main__":
    show()
