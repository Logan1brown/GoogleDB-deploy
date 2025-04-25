"""Component for displaying unmatched show information."""

import streamlit as st
from typing import Dict, Any, Callable, List
from ..state.admin_state import TMDBMatchState
from .tmdb_match_view import render_match_card
from ..services.tmdb.match_service import TMDBMatchService

def render_unmatched_shows_table(
    shows: List[Dict[str, Any]], 
    on_find_matches: Callable[[Dict[str, Any]], None],
    matching: TMDBMatchState,
    validate_match: Callable[[TMDBMatchState], None]
):
    """Render all unmatched shows in a table.
    
    Args:
        shows: List of show dictionaries
        on_find_matches: Callback when Find Matches is clicked
        matching: Current TMDB matching state
        validate_match: Callback for validating matches
    """
    # Create table header
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.markdown("**Title**")
    with col2:
        st.markdown("**Network**")
    with col3:
        st.markdown("**Date**")
    with col4:
        st.markdown("**Actions**")
    
    # Render each show row
    for show in shows:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(show['title'])
        with col2:
            st.write(show.get('network_name', ''))
        with col3:
            st.write(show.get('year', ''))
        with col4:
            # Stack buttons vertically
            if st.button("Find Matches",
                       key=f"find_{show['id']}",
                       type="primary",
                       use_container_width=True):
                on_find_matches(show)
            
            match_service = TMDBMatchService()
            if st.button("No Matches", 
                       key=f"no_match_{show['id']}",
                       type="secondary",
                       use_container_width=True):
                if match_service.mark_as_no_match(show['id']):
                    st.rerun()
        
        # Show matches right after this row if this is the show we just searched
        if matching.matches and matching.search_query == show['title']:
            st.subheader(f"Matches for '{matching.search_query}'")
            for match in matching.matches:
                match.our_eps = matching.our_eps
                render_match_card(match, validate_match)
