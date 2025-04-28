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
    title = f"{show.title} (Match: {int(scores['total'])}, Success: {int(success) if success is not None else 'N/A'})"
    
    with st.expander(title, expanded=expanded):
        # Title as a header
        st.markdown(f"### {show.title}")
        st.markdown(f"Match: {int(scores['total'])}, Success: {int(success) if success is not None else 'N/A'}")
        
        # Content Match section
        st.markdown(f"**Content Match** ({scores['content_total']}/85 points)")
        
        # Genre and Source (core content)
        col1, col2 = st.columns(2)
        details = scores['details']
        
        with col1:
            # Genre details
            genre = details['genre']
            st.write(f"Genre: {scores['genre_score']}/40")
            if genre['primary_match']:
                st.write(f"✓ Primary: {genre['primary']} (+27)")
            if genre['shared_subgenres']:
                points = genre['subgenre_points']
                genres = ', '.join(genre['shared_subgenres'])
                st.write(f"✓ Subgenres: {genres} (+{points})")
        
        with col2:
            # Source details
            source = details['source']
            st.write(f"Source: {scores['source_score']}/15")
            if source['match']:
                st.write(f"✓ Both {source['type1']}")
            else:
                st.write(f"× {source['type1']} vs {source['type2']}")
        
        # Production section
        st.markdown(f"\n**Production** ({scores['team_score'] + scores['network_score'] + scores['studio_score']}/30 points)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Team details
            team = details['team']
            st.write(f"Team: {scores['team_score']}/20")
            if team['shared_members']:
                for name, role in team['shared_members']:
                    st.write(f"✓ {name} ({role})")
        
        with col2:
            # Network details
            network = details['network']
            st.write(f"Network: {scores['network_score']}/7")
            if network['match']:
                st.write(f"✓ Both {network['name1']}")
            else:
                st.write(f"× {network['name1']} vs {network['name2']}")
            
            # Studio details
            studio = details['studio']
            st.write(f"\nStudio: {scores['studio_score']}/3")
            if studio['match']:
                st.write(f"✓ Both {studio['name1']}")
            elif studio['name1'] and studio['name2']:
                st.write(f"× {studio['name1']} vs {studio['name2']}")
            else:
                st.write(f"× Missing studio data")
        
        # Format Match section
        st.markdown(f"\n**Format Match** ({scores['format_total']}/15 points)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Episode format
            format = details['format']
            st.write(f"Episodes: {scores['episode_score']}/8")
            eps1, eps2 = format['eps_per_season1'], format['eps_per_season2']
            if eps1 is not None and eps2 is not None:
                st.write(f"{int(eps1)} vs {int(eps2)} eps/season")
            
            # Order type
            st.write(f"\nOrder Type: {scores['order_score']}/4")
            if format['order_type1'] == format['order_type2']:
                st.write(f"✓ Both {format['order_type1']}")
            else:
                st.write(f"× {format['order_type1']} vs {format['order_type2']}")
        
        with col2:
            # Timing
            st.write(f"Timing: {scores['date_score']}/3")
        
        # Show description
        if show.description:
            st.write("\n---")
            st.write(show.description)
