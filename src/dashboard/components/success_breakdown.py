"""Component for displaying success score breakdown."""

import streamlit as st
import pandas as pd
from typing import Dict


def render_success_breakdown(match: Dict) -> None:
    """Render success score and breakdown for a show.
    
    Args:
        match: Show data dictionary containing success metrics
    """
    st.markdown("### Success Score")
    success_score = match.get('success_score', 0)
    st.metric("Success Score", f"{int(success_score)}/100", label_visibility="collapsed")
    
    st.markdown("### Score Breakdown")
    
    # Calculate scores first
    season_score = 0
    if pd.notna(match.get('tmdb_seasons')):
        seasons = int(match['tmdb_seasons'])
        if seasons >= 2:
            season_score += 40
            extra_seasons = seasons - 2
            if extra_seasons > 0:
                season_score += min(extra_seasons * 20, 40)
    
    episode_score = 0
    avg_eps = match.get('tmdb_avg_eps', 0)
    if pd.notna(avg_eps):
        avg_eps = float(avg_eps)
        if avg_eps >= 10:
            episode_score = 40
        elif avg_eps >= 8:
            episode_score = 20
    
    # Display breakdown
    if pd.notna(match.get('tmdb_seasons')):
        seasons = int(match['tmdb_seasons'])
        if seasons >= 2:
            st.write("**Renewed for Season 2** _(+40 points)_")
            extra_seasons = seasons - 2
            if extra_seasons > 0:
                bonus = min(extra_seasons * 20, 40)
                st.write(f"**Additional seasons bonus** _(+{bonus} points)_")
    
    if pd.notna(match.get('tmdb_avg_eps')):
        avg_eps = float(match['tmdb_avg_eps'])
        if avg_eps >= 10:
            st.write("**High episode volume** _(+40 points)_")
        elif avg_eps >= 8:
            st.write("**Standard episode volume** _(+20 points)_")
    
    # Status modifier (only show if it affects score)
    status = match.get('status_name')
    if status == 'Returning Series':
        st.write("**Active show bonus:** _Score multiplied by 1.2_")
    elif status == 'Canceled':
        st.write("**Canceled show penalty:** _Score multiplied by 0.8_")
