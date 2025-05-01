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
                    st.markdown(f":green[✓] Subgenres: {genres} (+{points})")
            else:
                st.write(f"Genre (0/40)")
                st.write(f"• {genre['primary']} (not selected)")
                if genre.get('shared_subgenres'):
                    genres = ', '.join(genre['shared_subgenres'])
                    st.markdown(f":green[✓] Subgenres: {genres} (not selected)")
                
            # Character types
            if 'characters' in details:
                chars = details['characters']
                if chars['selected']:
                    st.write(f"\nCharacter Types: {scores['character_score']}/10")
                    if chars['matches']:
                        for char_type in chars['matches']:
                            st.markdown(f":green[✓] {char_type}")
                    if chars['mismatches']:
                        for char_type in chars['mismatches']:
                            st.write(f"× {char_type}")
                else:
                    st.write("\nCharacter Types (0/10)")
                    for char_type in chars['matches'] + chars['mismatches']:
                        st.markdown(f":green[✓] {char_type} (not selected)")
                        
            # Plot elements
            if 'plot' in details:
                plot = details['plot']
                if plot['selected'] or plot['matches']:
                    st.write(f"\nPlot Elements: {scores['plot_score']}/10")
                    if plot['matches']:
                        for element in plot['matches']:
                            st.markdown(f":green[✓] {element} (+{scores['plot_score']})")
                    if plot['mismatches']:
                        for element in plot['mismatches']:
                            st.write(f"× {element}")
                else:
                    st.write("\nPlot Elements (0/10)")
                    for element in plot['matches'] + plot['mismatches']:
                        st.markdown(f":green[✓] {element} (not selected)")
        
        with col2:
            # Source details
            source = details['source']
            if source['selected']:
                st.write(f"Source: {scores['source_score']}/15")
                if source['match']:
                    st.markdown(f":green[✓] Both {source['type1']}")
                else:
                    st.write(f"× {source['type1']} vs {source['type2']}")
            else:
                st.write(f"Source ({scores['source_score']}/15)")
                st.write(f"• {source['type1']} (not selected)")
                
            # Theme elements
            if 'themes' in details:
                themes = details['themes']
                if themes['selected'] or themes['matches'] or themes['mismatches']:
                    st.write(f"\nTheme Elements: {scores['theme_score']}/10")
                    if themes['matches']:
                        for theme in themes['matches']:
                            st.markdown(f":green[✓] {theme} (+{scores['theme_score']})")
                    if themes['mismatches']:
                        for theme in themes['mismatches']:
                            st.write(f"× {theme}")
                else:
                    st.write(f"\nTheme Elements ({scores['theme_score']}/10)")
                    if themes['matches'] or themes['mismatches']:
                        for theme in themes['matches'] + themes['mismatches']:
                            st.markdown(f":green[✓] {theme} (not selected)")
                    else:
                        st.write("• No theme elements")
                        
            # Tone
            if 'tone' in details:
                tone = details['tone']
                if tone['selected']:
                    st.write(f"\nTone: {scores['tone_score']}/5")
                    if tone['tone1'] == tone['tone2']:
                        st.markdown(f":green[✓] Both {tone['tone1']}")
                    else:
                        st.write(f"× {tone['tone1']} vs {tone['tone2']}")
                else:
                    st.write(f"\nTone ({scores['tone_score']}/5)")
                    st.write(f"• {tone['tone1']} (not selected)")
        
        # Production section
        prod_total = scores['team_score'] + scores.get('network_score', 0) + scores.get('studio_score', 0)
        st.markdown(f"\n**Production** ({prod_total}/30 points)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Network details
            if 'network' in details:
                network = details['network']
                if network['selected'] or network['match']:
                    st.write(f"Network: {scores['network_score']}/7")
                    if network['match']:
                        st.markdown(f":green[✓] Both {network['name1']} (+{scores['network_score']})")
                    else:
                        st.write(f"× {network['name1']} vs {network['name2']}")
                else:
                    st.write(f"Network ({scores['network_score']}/7)")
                    st.write(f"• {network['name1']} (not selected)")
            
            # Studio details
            if 'studio' in details:
                studio = details['studio']
                if studio['selected'] or studio['match']:
                    st.write(f"\nStudio: {scores['studio_score']}/3")
                    if studio['match']:
                        st.markdown(f":green[✓] Both {studio['name1']} (+{scores['studio_score']})")
                    elif studio['name1'] and studio['name2']:
                        st.write(f"× {studio['name1']} vs {studio['name2']}")
                    else:
                        st.write(f"× Missing studio data")
                else:
                    st.write(f"\nStudio ({scores['studio_score']}/3)")
                    st.write(f"• {studio['name1']} (not selected)")
        
        with col2:
            # Team details
            if 'team' in details:
                team = details['team']
                if team['selected'] or team.get('shared_members'):
                    st.write(f"Team: {scores['team_score']}/20")
                    if team.get('shared_members'):
                        for name, role in team['shared_members']:
                            st.markdown(f":green[✓] {name} ({role}) (+{scores['team_score']})")
                else:
                    st.write(f"Team ({scores['team_score']}/20)")
                    st.write("• No shared team members (not selected)")
        
        # Format Match section
        if 'format' in details:
            format = details['format']
            if format['selected'] or format['episode_match'] or format['order_match']:
                st.markdown(f"\n**Format Match** ({scores['format_total']}/5 points)")
                
                # Episodes
                st.write(f"Episodes: {scores['episode_score']}/4")
                if format['episode_count1'] == format['episode_count2']:
                    st.markdown(f":green[✓] Both {format['episode_count1']} (+4)")
                else:
                    diff = abs(format['episode_count1'] - format['episode_count2'])
                    if diff <= 2:
                        st.markdown(f":green[✓] {format['episode_count1']} vs {format['episode_count2']} (+4)")
                    elif diff <= 4:
                        st.markdown(f":green[✓] {format['episode_count1']} vs {format['episode_count2']} (+3)")
                    elif diff <= 6:
                        st.markdown(f":green[✓] {format['episode_count1']} vs {format['episode_count2']} (+2)")
                    else:
                        st.write(f"× {format['episode_count1']} vs {format['episode_count2']}")
                    
                # Order type
                st.write(f"\nOrder Type: {scores['order_score']}/1")
                if format['order_type1'] == format['order_type2']:
                    st.markdown(f":green[✓] Both {format['order_type1']} (+1)")
                else:
                    st.write(f"× {format['order_type1']} vs {format['order_type2']}")
            else:
                st.markdown(f"\n**Format ({scores['format_total']}/5)**")
                st.write(f"• Episodes: {format['episode_count1']} (not selected)")
                st.write(f"• Order Type: {format['order_type1']} (not selected)")
        
        # Setting Match section
        if 'setting' in details:
            setting = details['setting']
            if setting['selected'] or setting['time_match'] or setting['location_match']:
                st.markdown(f"\n**Setting Match** ({scores['setting_total']}/7 points)")
                
                # Time period
                st.write(f"Time Period: {scores['time_score']}/4")
                if setting['time_match']:
                    st.markdown(f":green[✓] Both {setting['time']} (+4)")
                else:
                    st.write(f"× {setting['time']}")
                
                # Location
                st.write(f"\nLocation: {scores['location_score']}/3")
                if setting['location_match']:
                    st.markdown(f":green[✓] Both {setting['location']} (+3)")
                else:
                    st.write(f"× {setting['location']}")
            else:
                st.markdown(f"\n**Setting ({scores['setting_total']}/7)**")
                st.write(f"• Time Period: {setting['time']} (not selected)")
                st.write(f"• Location: {setting['location']} (not selected)")
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
