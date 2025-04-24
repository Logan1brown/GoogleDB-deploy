"""Component for displaying TMDB match information."""

import streamlit as st
from ..state.admin_state import TMDBMatch

def render_match_card(match: TMDBMatch, on_propose=None):
    """Render a TMDB match card with side-by-side comparison.
    
    Args:
        match: TMDBMatch object containing all match data
        on_propose: Optional callback when match is proposed
    """
    with st.expander(f"{match.name}", expanded=True):
        col1, col2 = st.columns(2)
        
        # Our Show Details
        with col1:
            st.markdown("**Our Show Data**")
            st.markdown(f"**Title:** {match.show_title}")
            st.markdown(f"**Network:** {match.show_network or 'Unknown'}")
            st.markdown(f"**Year:** {match.show_year or 'Unknown'}")
        
        # TMDB Details
        with col2:
            st.markdown("**TMDB Data**")
            st.markdown(f"**Title:** {match.name}")
            st.markdown(f"**Network:** {', '.join(match.networks) if match.networks else 'Unknown'}")
            st.markdown(f"**First Air:** {match.first_air_date or 'Unknown'}")
            st.markdown(f"**Status:** {match.status or 'Unknown'}")
        
        # Overview
        st.markdown("**Overview**")
        st.write(match.overview or "No overview available")
        
        # Episode Data
        st.markdown("**Episodes & Seasons**")
        if match.episodes_per_season:
            season_data = [f"S{i+1}: {count} eps" for i, count in enumerate(match.episodes_per_season)]
            st.markdown(" | ".join(season_data))
        else:
            st.markdown("No episode data available")
        
        # Match Score Details
        st.markdown("**Match Details**")
        score_col1, score_col2, score_col3 = st.columns(3)
        with score_col1:
            st.metric("Title Match", f"{match.title_score}%")
        with score_col2:
            st.metric("Network Match", f"{match.network_score}%")
        with score_col3:
            st.metric("Year Match", f"{match.year_score}%")
        
        # Action buttons - side by side at bottom
        st.markdown("---")
        
        # Two equal columns for the buttons
        button_col1, button_col2 = st.columns(2)
        
        # Propose Match button
        with button_col1:
            if st.button(f"Propose Match ({match.confidence}%)", 
                       key=f"propose_{match.show_id}_{match.tmdb_id}",
                       use_container_width=True):
                if on_propose:
                    on_propose(match)
        
        # No Match button
        with button_col2:
            if st.button("No Match", 
                       key=f"no_match_{match.show_id}_{match.tmdb_id}",
                       type="secondary",
                       use_container_width=True):
                if on_propose:
                    # Create a dummy match object with tmdb_id = -1
                    no_match = TMDBMatch(
                        our_show_id=match.show_id,
                        our_show_title=match.show_title,
                        our_network=match.show_network,
                        tmdb_id=-1,
                        name="N/A",
                        confidence=100,  # We're 100% sure there's no match
                        title_score=0,
                        network_score=0,
                        ep_score=0
                    )
                    on_propose(no_match)
