"""Component for displaying TMDB match information.

This component handles the display of individual TMDB matches in the admin dashboard.
It uses the TMDBMatchState class for state management and provides validation controls.
"""

import streamlit as st

from ..services.tmdb.match_service import TMDBMatchService
from ..state.admin_state import TMDBMatchState
from ..state.session import clear_match_session_state, update_admin_state
from ..utils.style_config import COLORS, FONTS

def find_matches_for_show(show_data: dict, state) -> None:
    """Find TMDB matches for a show and update state."""
    matching = state.tmdb_matching
    match_service = TMDBMatchService()
    
    try:
        with st.spinner(f"Searching TMDB for {show_data['title']}..."):
            # Get our EPs first
            team_members = show_data.get('team_members', [])
            our_eps = [member['name'] for member in team_members 
                     if member['role'].lower() == 'executive producer']
            
            # Get TMDB matches
            matches = match_service.search_and_match(show_data)
            
            if not matches:
                st.warning("No matches found")
                return
            
            # Store matches in state
            matching.matches = matches
            matching.search_query = show_data['title']
            matching.our_eps = our_eps
            matching.last_validation = None  # Clear any previous validation
            update_admin_state(state)
            
    except Exception as e:
        st.error(f"Error searching TMDB: {str(e)}")

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
    
    # Show title and first air date
    title_col, score_col = st.columns([3, 1])
    with title_col:
        st.markdown(
            f"<h3 style='margin: 0; font-size: {FONTS['primary']['sizes']['title']}px;'>"
            f"{match.name}"
            f"<span style='color: {COLORS['text']['secondary']}; font-size: {FONTS['primary']['sizes']['body']}px;'>"
            f" ({match.first_air_date or 'Unknown'})"
            f"</span>"
            f"</h3>",
            unsafe_allow_html=True
        )
    
    # Show match scores
    with score_col:
        # Title: 60%, Network: 25%, EP: 15%
        total_score = int(match.title_score + match.network_score + match.ep_score)
        st.metric("Match Score", f"{total_score}%")
    
    # Show details in columns
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
        st.markdown(f"**Year:** {match.first_air_date[:4] if match.first_air_date else 'Unknown'}")
        st.markdown("**Executive Producers:**")
        if match.executive_producers:
            for ep in match.executive_producers:
                matched = ep in match.our_eps
                st.write("🟢" if matched else "⚫", ep)
        else:
            st.markdown("*No executive producers found*")
    
    # Score details
    st.markdown(
        f"<div style='font-size: {FONTS['primary']['sizes']['header']}px; margin-top: 1em;'>"
        "Title Match: {title}% · Network Match: {network}% · EP Match: {ep}%"
        f"</div>".format(
            title=int(match.title_score),
            network=int(match.network_score),
            ep=int(match.ep_score)
        ),
        unsafe_allow_html=True
    )
    
    # Add validation controls
    if not match.validated:
        if st.button("Validate Match", key=f"validate_{card_key}", type="primary", use_container_width=True):
            if on_validate:
                on_validate(match)
    else:
        st.success("Match validated!")
