"""
Content Analysis Page

Provides detailed analysis of TV series content across acquisition, packaging, and development.
"""

import streamlit as st
import sys
import os

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.dashboard.auth.auth_required import auth_required
from src.dashboard.utils.timing import time_page
from dataclasses import asdict, dataclass, field
from src.dashboard.utils.style_config import COLORS, FONTS
from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.success_analysis.success_analyzer import SuccessAnalyzer
from src.dashboard.components.unified_view import render_unified_dashboard
from src.dashboard.state.session import get_page_state, FilterState

@dataclass
class UnifiedState:
    """State for unified dashboard page."""
    analysis_type: str = "Acquisition"
    source_type: str = "All"
    genre: str = "All"
    network: str = "All"
    year_range: tuple[int, int] = field(default_factory=lambda: (2020, 2025))
    selected_shows: list[str] = field(default_factory=list)
    selected_networks: list[str] = field(default_factory=list)
    success_filter: str = "All"

@time_page
@auth_required()
def show():
    try:
        # Page title using style from style_config
        st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Content Analysis</p>', unsafe_allow_html=True)

        # Get page state
        state = get_page_state("content_analysis")
        if "unified" not in state:
            state["unified"] = asdict(UnifiedState())
        
        # Initialize data and analyzers
        success_analyzer = SuccessAnalyzer()
        
        # Update state with filter values
        unified_state = state["unified"]
        if "analysis_type" in st.session_state:
            unified_state["analysis_type"] = st.session_state["analysis_type"]
        if "source_type" in st.session_state:
            unified_state["source_type"] = st.session_state["source_type"]
        if "genre" in st.session_state:
            unified_state["genre"] = st.session_state["genre"]
        if "network" in st.session_state:
            unified_state["network"] = st.session_state["network"]
        if "year_range" in st.session_state:
            unified_state["year_range"] = st.session_state["year_range"]
        
        # Render view with state
        render_unified_dashboard(success_analyzer)
        
    except Exception as e:
        st.error(f"Error displaying unified dashboard: {str(e)}")
        st.info("An error occurred while loading the content analysis view.")

if __name__ == "__main__":
    show()
