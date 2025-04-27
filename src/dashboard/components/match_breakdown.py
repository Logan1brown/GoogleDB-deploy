"""Component for displaying show match breakdowns.

This component handles the display of match score breakdowns in an expandable container.
"""

import streamlit as st
from typing import Any, Dict, Optional

from src.data_processing.show_detail.show_detail_analyzer import SimilarShow
from ..utils.style_config import COLORS, FONTS

def render_match_breakdown(
    show: SimilarShow,
    expanded: bool = False,
) -> None:
    """Render a match breakdown in an expandable container.
    
    Args:
        show: SimilarShow object containing match data
        expanded: Whether the container should be expanded by default
    """
    # Format title with scores
    title = f"{show.title} "
    title += f"(Match: {show.match_score['total']}, "
    title += f"Success: {show.success_score if show.success_score is not None else 'N/A'})"
    
    with st.expander(title, expanded=expanded):
        # Header
        st.markdown(
            f'<p style="font-family: {FONTS["primary"]["family"]}; '
            f'font-size: {FONTS["primary"]["sizes"]["header"]}px; '
            f'font-weight: 600; color: {COLORS["text"]["primary"]}; '
            f'margin-bottom: 10px;">Match Score Breakdown</p>',
            unsafe_allow_html=True
        )
        
        # Use columns for better layout
        col1, col2 = st.columns(2)
        
        # Style for score items
        score_style = (
            f'font-family: {FONTS["primary"]["family"]}; '
            f'font-size: {FONTS["primary"]["sizes"]["body"]}px; '
            f'color: {COLORS["text"]["secondary"]}; '
            f'margin: 5px 0;'
        )
        
        with col1:
            st.markdown(f'<p style="{score_style}">Genre Score: {show.match_score["genre_score"]} points</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="{score_style}">Team Score: {show.match_score["team_score"]} points</p>', unsafe_allow_html=True)
            
        with col2:
            st.markdown(f'<p style="{score_style}">Source Score: {show.match_score["source_score"]} points</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="{score_style}">Release Window Score: {show.match_score["date_score"]} points</p>', unsafe_allow_html=True)
        
        # Total score with primary color
        st.markdown(
            f'<p style="font-family: {FONTS["primary"]["family"]}; '
            f'font-size: {FONTS["primary"]["sizes"]["body"]}px; '
            f'font-weight: 600; color: {COLORS["text"]["primary"]}; '
            f'margin-top: 10px;">Total Match Score: {show.match_score["total"]} points</p>',
            unsafe_allow_html=True
        )
