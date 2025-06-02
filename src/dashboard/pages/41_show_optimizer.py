"""Show Optimizer Page.

This page provides an interface for optimizing show concepts based on
historical success patterns and network preferences.
"""

import streamlit as st
import logging
import pandas as pd
import sys
import os
import traceback
from typing import Dict, Any

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required
from src.dashboard.utils.style_config import COLORS, FONTS, CHART_DEFAULTS
from src.dashboard.components.optimizer_view import OptimizerView
from src.data_processing.show_optimizer.show_optimizer import ShowOptimizer
from src.dashboard.state.session import get_page_state, update_page_state

# Configure logging
logger = logging.getLogger(__name__)

@auth_required()
def show():
    """Main page content."""
    # Page title using style from style_config
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Show Optimizer</p>', unsafe_allow_html=True)
    
    # Add description
    st.write("Optimize your show concept based on historical success patterns and network preferences. This tool analyzes your concept against our database to calculate success probability, identify best-matching networks, and recommend optimizations.")
    
    try:
        # Get page state
        state = get_page_state("show_optimizer")
        if not state:
            state = {}
        
        # Initialize state if needed
        if "criteria" not in state:
            state["criteria"] = {}
        if "field_options" not in state:
            state["field_options"] = {}
        if "display_options" not in state:
            state["display_options"] = {}
        if "results" not in state:
            state["results"] = False
        if "summary" not in state:
            state["summary"] = None
        
        # Render the optimizer view
        optimizer_view = OptimizerView()
        optimizer_view.render(state)
        
        # Save state back to session
        update_page_state("show_optimizer", state)
        
    except Exception as e:
        st.error(f"Error in show optimizer: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    show()
