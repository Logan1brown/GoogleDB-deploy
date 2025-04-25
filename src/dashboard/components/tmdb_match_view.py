"""Component for displaying TMDB match information.

This component handles the display of individual TMDB matches in the admin dashboard.
It uses the TMDBMatchState class for state management and provides validation controls.
"""

import streamlit as st
from ..state.admin_state import TMDBMatchState
from ..state.session import clear_match_session_state

def render_match_card(match: TMDBMatchState, on_validate=None):
    """Render a TMDB match card with side-by-side comparison.
    
    Args:
        match: TMDBMatchState object containing match data and UI state
        on_validate: Optional callback when match is validated
    
    The card shows:
    1. Our show data vs TMDB data
    2. Match confidence scores
    3. Validation controls
    
    UI state is persisted in the match state object and session state.
    """
    # Generate unique key for this match card
    card_key = f"tmdb_match_{match.our_show_id}_{match.tmdb_id}"
    
    # Initialize or get expanded state
    if f"{card_key}_expanded" not in st.session_state:
        st.session_state[f"{card_key}_expanded"] = match.expanded
    
    # Update match state from session state
    match.expanded = st.session_state[f"{card_key}_expanded"]
    
    with st.expander(f"{match.name}", expanded=match.expanded):
        col1, col2 = st.columns(2)
        
        # Our Show Details
        with col1:
            st.markdown("**Our Show Data**")
            st.markdown(f"**Title:** {match.our_show_title}")
            st.markdown(f"**Network:** {match.our_network or 'Unknown'}")
            st.markdown(f"**Year:** {match.our_year or 'Unknown'}")
            st.markdown("**Executive Producers:**")
            if match.our_eps:
                for ep in match.our_eps:
                    matched = ep in match.executive_producers
                    st.write("🟢" if matched else "⚫", ep)
            else:
                st.markdown("*No executive producers listed*")
        
        # TMDB Details
        with col2:
            st.markdown("**TMDB Data**")
            st.markdown(f"**Title:** {match.name}")
            st.markdown(f"**Network:** {', '.join(match.networks) if match.networks else 'Unknown'}")
            st.markdown(f"**First Air:** {match.first_air_date or 'Unknown'}")
            st.markdown("**Executive Producers:**")
            if match.executive_producers:
                for ep in match.executive_producers:
                    matched = ep in match.our_eps
                    st.write("🟢" if matched else "⚫", ep)
            else:
                st.markdown("*No executive producers found*")
        
        # Match Score Details
        st.markdown("**Match Details**")
        score_col1, score_col2, score_col3 = st.columns(3)
        with score_col1:
            st.metric("Title Match", f"{match.title_score}%")
        with score_col2:
            st.metric("Network Match", f"{match.network_score}%")
        with score_col3:
            st.metric("EP Match", f"{match.ep_score}%")
        
        # Action buttons - side by side at bottom
        st.markdown("---")
        
        # Two equal columns for the buttons
        button_col1, button_col2 = st.columns(2)
        
        # Validate Match button
        with button_col1:
            if st.button(f"Validate Match ({match.confidence}%)", 
                       key=f"{card_key}_validate",
                       use_container_width=True):
                if on_validate:
                    # Clear expanded state and all session state before validation
                    match.expanded = False
                    clear_match_session_state(match.our_show_id)
                    on_validate(match)
        
        # No Match button
        with button_col2:
            if st.button("No Match", 
                       key=f"{card_key}_no_match",
                       type="secondary",
                       use_container_width=True):
                if on_validate:
                    # Create a dummy match state object with tmdb_id = -1
                    no_match = TMDBMatchState(
                        our_show_id=match.our_show_id,
                        our_show_title=match.our_show_title,
                        our_network=match.our_network,
                        tmdb_id=-1,
                        name="N/A",
                        confidence=100,  # We're 100% sure there's no match
                        title_score=0,
                        network_score=0,
                        ep_score=0
                    )
                    on_validate(no_match)
