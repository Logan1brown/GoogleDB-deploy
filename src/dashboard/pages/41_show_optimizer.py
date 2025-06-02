"""Show Optimizer Page.

This page provides an interface for optimizing show concepts based on
historical success patterns and network preferences.
"""

import streamlit as st
import logging
import pandas as pd
import sys
import os
from typing import Dict, Any

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required
from src.dashboard.utils.style_config import COLORS, FONTS, CHART_DEFAULTS
from src.dashboard.components.optimizer_view import OptimizerView
from src.data_processing.show_optimizer.show_optimizer import ShowOptimizer

# Configure logging
logger = logging.getLogger(__name__)

# Initialize session state
if "optimizer_initialized" not in st.session_state:
    st.session_state.optimizer_initialized = False

@auth_required()
def show():
    """Main page content."""
    # Page header
    st.write("# 🎯 Show Optimizer")
    
    st.markdown("""
    Optimize your show concept based on historical success patterns and network preferences.
    
    This tool analyzes your show concept against our database of successful shows to:
    - Calculate your concept's success probability
    - Identify the best-matching networks
    - Recommend optimizations to improve your chances
    """)
    
    # Render the optimizer view
    optimizer_view = OptimizerView()
    optimizer_view.render()

if __name__ == "__main__":
    show()
