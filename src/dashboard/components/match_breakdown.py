"""Component for displaying show match breakdowns.

This component handles the display of match score breakdowns in an expandable container.
"""

import streamlit as st
from typing import Optional

from src.data_processing.show_detail.show_detail_analyzer import SimilarShow

def render_match_breakdown(
    show: SimilarShow,
    expanded: bool = False,
) -> None:
    """Render a match breakdown in an expandable container.
    
    Args:
        show: SimilarShow object containing match data
        expanded: Whether the container should be expanded by default
    """
    show_match_breakdown(show, expanded)

def show_match_breakdown(show, expanded=False):
    """Show match breakdown for a show in an expander.
    
    Args:
        show: SimilarShow object
        expanded: Whether expander should be expanded by default
    """
    scores = show.match_score
    success = show.success_score if show.success_score is not None else 'N/A'
    
    # Build title
    title = f"{show.title} (Match: {scores['total']}, Success: {success})"
    
    with st.expander(title, expanded=expanded):
        st.write(f"Network: {show.network_name}\n")
        
        # Content Match section
        st.markdown(f"**Content Match** ({scores['content_total']}/85 points)")
        st.write(f"    Genre: {scores['genre_score']}/45")
        st.write(f"    Source Type: {scores['source_score']}/15")
        st.write(f"    Team: {scores['team_score']}/25")
        
        # Format Match section
        st.markdown(f"\n**Format Match** ({scores['format_total']}/15 points)")
        st.write(f"    Episodes: {scores['episode_score']}/8")
        st.write(f"    Order Type: {scores['order_score']}/4")
        st.write(f"    Timing: {scores['date_score']}/3")
        

