"""Component for displaying unmatched show information."""

import streamlit as st
from typing import Dict, Any, Callable, List

def render_unmatched_show_header():
    """Render the header row for unmatched shows table."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.markdown("**Title**")
    with col2:
        st.markdown("**Network**")
    with col3:
        st.markdown("**Date**")
    with col4:
        st.markdown("**Actions**")

def render_unmatched_show_row(
    show: Dict[str, Any], 
    on_find_matches: Callable[[Dict[str, Any]], None]
):
    """Render a row for an unmatched show with a Find Matches button.
    
    Args:
        show: Dictionary containing show data (show_id, title, network_name, date, team_members)
        on_find_matches: Callback when Find Matches is clicked
    """
    with st.container():
        # Display show info in columns
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(show['title'])
        with col2:
            st.write(show.get('network_name', ''))
        with col3:
            st.write(show.get('year', ''))
        with col4:
            # Only show Find Matches button
            if st.button("Find Matches", 
                        key=f"find_{show['id']}", 
                        use_container_width=True):
                on_find_matches(show)
