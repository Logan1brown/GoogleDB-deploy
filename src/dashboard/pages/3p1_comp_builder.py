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
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.components.comp_view import render_comp_builder
from src.dashboard.state.session import get_page_state


@auth_required()
def main():
    """Main page content."""
    # Page title using style from style_config
    st.markdown(
        f'<p style="font-family: {FONTS["primary"]["family"]}; '
        f'font-size: {FONTS["primary"]["sizes"]["header"]}px; '
        f'text-transform: uppercase; font-weight: 600; '
        f'letter-spacing: 0.1em; color: {COLORS["accent"]}; '
        f'margin-bottom: 1em;">Comp Builder</p>',
        unsafe_allow_html=True
    )

    try:
        # Get page state
        state = get_page_state("comp_builder")
        if "criteria" not in state:
            state["criteria"] = {}

        # Render comp builder view
        render_comp_builder(state)

    except Exception as e:
        st.error(f"Error in comp builder: {str(e)}")
