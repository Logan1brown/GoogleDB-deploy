"""Base component for displaying show match breakdowns.

This component provides the core functionality for displaying match breakdowns,
which can be used by both the comp builder and show detail pages.
"""

import streamlit as st
from typing import Dict, Optional

def render_base_match_breakdown(
    title: str,
    scores: Dict,
    details: Dict,
    success_score: Optional[float] = None,
    expanded: bool = False,
    use_expander: bool = True,
    description: Optional[str] = None
) -> None:
    """Render a match breakdown with consistent styling.
    
    Args:
        title: Show title
        scores: Dictionary of score components
        details: Dictionary of match details
        success_score: Optional success score
        expanded: Whether the section should be expanded by default
        use_expander: Whether to wrap in an expander
        description: Optional show description
    """
    # Build title with scores
    header = f"{title} (Match: {int(scores['total'])}"
    if success_score is not None:
        header += f", Success: {int(success_score) if success_score is not None else 'N/A'}"
    header += ")"
    
    def render_content():
        # Content Match section
        st.markdown(f"**Content Match** ({scores['content_total']}/85 points)")
        
        # Genre and Source (core content)
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre details
            genre = details['genre']
            if genre['selected']:
                st.write(f"Genre: {scores['genre_score']}/40")
                if genre['primary_match']:
                    st.write(f"✓ Primary: {genre['primary']} (+27)")
                if genre.get('shared_subgenres'):
                    points = genre['subgenre_points']
                    genres = ', '.join(genre['shared_subgenres'])
                    st.write(f"✓ Subgenres: {genres} (+{points})")
            else:
                st.write("Genre: (not selected)")
                st.write(f"• {genre['primary']}")
                if genre.get('shared_subgenres'):
                    genres = ', '.join(genre['shared_subgenres'])
                    st.write(f"• Subgenres: {genres}")
                
            # Character types
            if 'characters' in details:
                chars = details['characters']
                if chars['selected']:
                    st.write(f"\nCharacter Types: {scores['character_score']}/10")
                    if chars['matches']:
                        for char_type in chars['matches']:
                            st.write(f"✓ {char_type}")
                    if chars['mismatches']:
                        for char_type in chars['mismatches']:
                            st.write(f"× {char_type}")
                else:
                    st.write("\nCharacter Types: (not selected)")
                    for char_type in chars['matches'] + chars['mismatches']:
                        st.write(f"• {char_type}")
                        
            # Plot elements
            if 'plot' in details:
                plot = details['plot']
                if plot['selected'] or plot['matches']:
                    st.write(f"\nPlot Elements: {scores['plot_score']}/10")
                    if plot['matches']:
                        for element in plot['matches']:
                            st.write(f"✓ {element} (+{scores['plot_score']})")
                    if plot['mismatches']:
                        for element in plot['mismatches']:
                            st.write(f"× {element}")
                else:
                    st.write("\nPlot Elements: (not selected)")
                    for element in plot['matches'] + plot['mismatches']:
                        st.write(f"• {element}")
        
        with col2:
            # Source details
            source = details['source']
            if source['selected']:
                st.write(f"Source: {scores['source_score']}/15")
                if source['match']:
                    st.write(f"✓ Both {source['type1']}")
                else:
                    st.write(f"× {source['type1']} vs {source['type2']}")
            else:
                st.write("Source: (not selected)")
                st.write(f"• {source['type1']}")
                
            # Theme elements
            if 'themes' in details:
                themes = details['themes']
                if themes['selected']:
                    st.write(f"\nTheme Elements: {scores['theme_score']}/10")
                    if themes['matches']:
                        for theme in themes['matches']:
                            st.write(f"✓ {theme}")
                    if themes['mismatches']:
                        for theme in themes['mismatches']:
                            st.write(f"× {theme}")
                else:
                    st.write("\nTheme Elements: (not selected)")
                    for theme in themes['matches'] + themes['mismatches']:
                        st.write(f"• {theme}")
                        
            # Tone
            if 'tone' in details:
                tone = details['tone']
                if tone['selected']:
                    st.write(f"\nTone: {scores['tone_score']}/5")
                    if tone['tone1'] == tone['tone2']:
                        st.write(f"✓ Both {tone['tone1']}")
                    else:
                        st.write(f"× {tone['tone1']} vs {tone['tone2']}")
                else:
                    st.write("\nTone: (not selected)")
                    st.write(f"• {tone['tone1']}")
        
        # Production section
        prod_total = scores['team_score'] + scores.get('network_score', 0) + scores.get('studio_score', 0)
        st.markdown(f"\n**Production** ({prod_total}/30 points)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Network details
            if 'network' in details:
                network = details['network']
                st.write(f"Network: {scores['network_score']}/7")
                if network['match']:
                    st.write(f"✓ Both {network['name1']}")
                else:
                    st.write(f"× {network['name1']} vs {network['name2']}")
            
            # Studio details
            if 'studio' in details:
                studio = details['studio']
                st.write(f"\nStudio: {scores['studio_score']}/3")
                if studio['match']:
                    st.write(f"✓ Both {studio['name1']}")
                elif studio['name1'] and studio['name2']:
                    st.write(f"× {studio['name1']} vs {studio['name2']}")
                else:
                    st.write(f"× Missing studio data")
        
        with col2:
            # Team details
            if 'team' in details:
                team = details['team']
                st.write(f"Team: {scores['team_score']}/20")
                if team.get('shared_members'):
                    for name, role in team['shared_members']:
                        st.write(f"✓ {name} ({role})")
        
        # Format Match section
        st.markdown(f"\n**Format Match** ({scores['format_total']}/15 points)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Format details
            if 'format' in details:
                format = details['format']
                if format['selected']:
                    st.write(f"\nFormat Match: {scores['format_score']}/15")
                    
                    # Episodes
                    st.write(f"Episodes: {scores['episode_score']}/8")
                    if format['episode_match']:
                        st.write(f"✓ Both {format['episode_count1']}")
                    else:
                        st.write(f"× {format['episode_count1']} vs {format['episode_count2']}")
                        
                    # Order type
                    st.write(f"\nOrder Type: {scores['order_score']}/4")
                    if format['order_type1'] == format['order_type2']:
                        st.write(f"✓ Both {format['order_type1']}")
                    else:
                        st.write(f"× {format['order_type1']} vs {format['order_type2']}")
                else:
                    st.write("\nFormat: (not selected)")
                    st.write(f"• Episodes: {format['episode_count1']}")
                    st.write(f"• Order Type: {format['order_type1']}")
        
        with col2:
            # Timing
            if 'date_score' in scores:
                st.write(f"Timing: {scores['date_score']}/3")
        
        # Setting Match section (if present)
        if 'setting_total' in scores:
            st.markdown(f"\n**Setting Match** ({scores['setting_total']}/7 points)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Time setting
                if 'time_setting' in details:
                    time = details['time_setting']
                    st.write(f"Time Setting: {scores['time_score']}/4")
                    if time['match']:
                        st.write(f"✓ Both {time['time1']}")
                    else:
                        st.write(f"× {time['time1']} vs {time['time2']}")
            
            with col2:
                # Location setting
                if 'location' in details:
                    loc = details['location']
                    st.write(f"Location: {scores['location_score']}/3")
                    if loc['match']:
                        st.write(f"✓ Both {loc['location1']}")
                    else:
                        st.write(f"× {loc['location1']} vs {loc['location2']}")
        
        # Show description
        if description:
            st.write("\n---")
            st.write(description)
    
    # Render with or without expander
    if use_expander:
        with st.expander(header, expanded=expanded):
            render_content()
    else:
        st.markdown(f"### {header}")
        render_content()
