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
        
        # Genre details
        details = scores['details']
        genre = details['genre']
        st.write(f"    Genre: {scores['genre_score']}/45")
        if genre['primary_match']:
            st.write(f"        ✓ Primary genre match: {genre['primary']} (+30)")
        if genre['shared_subgenres']:
            points = genre['subgenre_points']
            genres = ', '.join(genre['shared_subgenres'])
            st.write(f"        ✓ Shared subgenres: {genres} (+{points})")
        
        # Source details
        source = details['source']
        st.write(f"    Source Type: {scores['source_score']}/15")
        if source['match']:
            st.write(f"        ✓ Both {source['type1']}")
        else:
            st.write(f"        × {source['type1']} vs {source['type2']}")
        
        # Team details
        team = details['team']
        st.write(f"    Team: {scores['team_score']}/25")
        if team['shared_members']:
            for name, role in team['shared_members']:
                st.write(f"        ✓ {name} ({role})")
        
        # Format Match section
        st.markdown(f"\n**Format Match** ({scores['format_total']}/15 points)")
        
        # Episode format
        format = details['format']
        st.write(f"    Episodes: {scores['episode_score']}/8")
        eps1, eps2 = format['eps_per_season1'], format['eps_per_season2']
        if eps1 is not None and eps2 is not None:
            st.write(f"        {eps1:.1f} vs {eps2:.1f} eps/season")
        
        # Order type
        st.write(f"    Order Type: {scores['order_score']}/4")
        if format['order_type1'] == format['order_type2']:
            st.write(f"        ✓ Both {format['order_type1']}")
        else:
            st.write(f"        × {format['order_type1']} vs {format['order_type2']}")
        
        # Timing
        st.write(f"    Timing: {scores['date_score']}/3")
        
        # Show description
        if show.description:
            st.write("\n---")
            st.write(show.description)
