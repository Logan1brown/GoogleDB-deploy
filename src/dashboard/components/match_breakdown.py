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
    
    # Build expander title - this appears in the collapsed view
    # Format success score - only convert to int if it's a number
    success_display = str(int(success)) if isinstance(success, (int, float)) else 'N/A'
    title = f"{show.title} (Match: {int(scores['total'])}, Success: {success_display})"
    
    with st.expander(title, expanded=expanded):
        
        # Content Match section
        st.markdown(f"**Content Match** ({scores['content_total']}/82 points)")
        
        # Genre and Source (core content)
        col1, col2 = st.columns(2)
        details = scores['details']
        
        # Plot Elements
        plot = details.get('plot_elements', {})
        st.write(f"\nPlot Elements ({scores.get('plot_score', 0)}/12):")
        shared = plot.get('shared_elements', [])
        if shared:
            # First match gets 9 points (75%)
            st.write(f"• {shared[0]} (+9)")
            # Second match gets 3 points (25%)
            if len(shared) > 1:
                st.write(f"• {shared[1]} (+3)")
            # Show remaining matches without points
            for element in shared[2:]:
                st.write(f"• {element}")
        elif plot.get('elements1') and plot.get('elements2'):
            st.write("No matching plot elements")
            st.write(f"Show 1: {', '.join(plot['elements1'])}")
            st.write(f"Show 2: {', '.join(plot['elements2'])}")
        else:
            st.write("• Missing plot element data")
        
        # Theme Elements
        themes = details.get('theme_elements', {})
        st.write(f"\nTheme Elements ({scores.get('theme_score', 0)}/13):")
        shared = themes.get('shared_themes', [])
        if shared:
            # First match gets 9.75 points (75%)
            st.write(f"• {shared[0]} (+9.75)")
            # Second match gets 3.25 points (25%)
            if len(shared) > 1:
                st.write(f"• {shared[1]} (+3.25)")
            # Show remaining matches without points
            for theme in shared[2:]:
                st.write(f"• {theme}")
        elif themes.get('themes1') and themes.get('themes2'):
            st.write("No matching theme elements")
            st.write(f"Show 1: {', '.join(themes['themes1'])}")
            st.write(f"Show 2: {', '.join(themes['themes2'])}")
        else:
            st.write("• Missing theme element data")
        
        # Character Types
        chars = details.get('character_types', {})
        st.write(f"\nCharacter Types ({scores.get('character_score', 0)}/14):")
        shared = chars.get('shared_types', [])
        if shared:
            # First match gets 10.5 points (75%)
            st.write(f"• {shared[0]} (+10.5)")
            # Second match gets 3.5 points (25%)
            if len(shared) > 1:
                st.write(f"• {shared[1]} (+3.5)")
            # Show remaining matches without points
            for char_type in shared[2:]:
                st.write(f"• {char_type}")
        elif chars.get('types1') and chars.get('types2'):
            st.write("No matching character types")
            st.write(f"Show 1: {', '.join(chars['types1'])}")
            st.write(f"Show 2: {', '.join(chars['types2'])}")
        else:
            st.write("• Missing character type data")
        
        # Tone
        tone = details.get('tone', {})
        st.write(f"\nTone: {scores.get('tone_score', 0)}/9")
        if tone.get('match'):
            st.write(f"• Both {tone.get('tone1', 'Unknown')}")
        elif tone.get('tone1') and tone.get('tone2'):
            st.write(f"• {tone.get('tone1')} vs {tone.get('tone2')}")
        else:
            st.write("• Missing tone data")
        
        with col1:
            # Genre details
            genre = details['genre']
            st.write(f"Genre: {scores['genre_score']}/17")
            if genre['primary_match']:
                st.write(f"• Primary: {genre['primary']} (+9)")
            if genre['shared_subgenres']:
                genres = ', '.join(genre['shared_subgenres'])
                st.write(f"• Subgenres: {genres} (+8)")
        
        with col2:
            # Source details
            source = details['source']
            st.write(f"Source: {scores['source_score']}/15")
            if source['match']:
                st.write(f"• Both {source['type1']}")
            else:
                st.write(f"× {source['type1']} vs {source['type2']}")
        
        # Production section
        st.markdown(f"\n**Production** ({scores['team_score'] + scores['network_score'] + scores['studio_score']}/30 points)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Network details
            network = details['network']
            st.write(f"Network: {scores['network_score']}/7")
            if network['match']:
                st.write(f"• Both {network['name1']}")
            else:
                st.write(f"× {network['name1']} vs {network['name2']}")
            
            # Studio details
            studio = details['studio']
            st.write(f"\nStudio: {scores['studio_score']}/3")
            if studio['match']:
                st.write(f"• Both {studio['name1']}")
            elif studio['name1'] and studio['name2']:
                st.write(f"• {studio['name1']} vs {studio['name2']}")
            else:
                st.write(f"• Missing studio data")
        
        with col2:
            # Team details
            team = details['team']
            st.write(f"Team: {scores['team_score']}/20")
            if team['shared_members']:
                for name, role in team['shared_members']:
                    st.write(f"• {name} ({role})")
        
        # Format Match section
        st.markdown(f"\n**Format Match** ({scores['format_total']}/5 points)")
        
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
                st.write(f"• Both {format['order_type1']}")
            else:
                st.write(f"• {format['order_type1']} vs {format['order_type2']}")
        
        with col2:
            # Timing
            st.write(f"Timing: {scores['date_score']}/3")
        
        # Show description
        if show.description:
            st.write("\n---")
            st.write(show.description)
