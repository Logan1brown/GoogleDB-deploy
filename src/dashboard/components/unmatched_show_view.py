"""Component for displaying unmatched show information."""

import streamlit as st
from typing import Dict, Any, Callable, List

def render_unmatched_shows_table(
    shows: List[Dict[str, Any]], 
    on_find_matches: Callable[[Dict[str, Any]], None]
):
    """Render all unmatched shows in a scrollable table.
    
    Args:
        shows: List of show dictionaries
        on_find_matches: Callback when Find Matches is clicked
    """
    # Add CSS from style config
    from ..utils.style_config import CSS_CLASSES
    st.markdown(CSS_CLASSES['unmatched_shows'], unsafe_allow_html=True)
    
    # Header stays outside scroll area
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.markdown("**Title**")
    with col2:
        st.markdown("**Network**")
    with col3:
        st.markdown("**Date**")
    with col4:
        st.markdown("**Actions**")

    # Start scrollable container
    st.markdown('<div class="unmatched-shows">', unsafe_allow_html=True)
    
    # Render each show row
    for show in shows:
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
    
    # End scrollable container
    st.markdown('</div>', unsafe_allow_html=True)
