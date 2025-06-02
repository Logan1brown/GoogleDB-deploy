"""Comp Builder Page

Interactive tool for exploring show comps by criteria.
"""

import streamlit as st
from typing import Dict

import sys
import os

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required
import traceback
import streamlit as st

from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.components.comp_view import render_comp_builder
from src.dashboard.state.session import get_page_state

@auth_required()
def show():
    """Main page content."""
    # Page title using style from style_config
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Comp Builder</p>', unsafe_allow_html=True)

    try:
        # Get page state
        state = get_page_state("comp_builder")
        if not state:
            state = {}
        if "criteria" not in state:
            state["criteria"] = {}

        # Render comp builder view
        render_comp_builder(state)

    except Exception as e:
        st.error(f"Error in comp builder: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    show()
