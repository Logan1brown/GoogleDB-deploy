"""Show Optimizer Page.

This page provides an interface for optimizing show concepts based on
historical success patterns and network preferences.
"""

import streamlit as st
import logging
import pandas as pd
from typing import Dict, Any

from ..components.optimizer_view import OptimizerView
from ...data_processing.show_optimizer.show_optimizer import ShowOptimizer

# Configure page
st.set_page_config(
    page_title="Show Optimizer",
    page_icon="🎯",
    layout="wide"
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize session state
if "optimizer_initialized" not in st.session_state:
    st.session_state.optimizer_initialized = False

def main():
    """Main function to render the Show Optimizer page."""
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
    main()
