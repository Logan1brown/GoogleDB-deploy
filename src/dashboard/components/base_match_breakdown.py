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
    matches: Dict,
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
        matches: Dictionary of raw match data
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
        st.markdown(f"\n**Content Match** ({scores['content_total']}/82 points)")
        
        # Genre and Source (core content)
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre details
            genre = details['genre']
            if genre['selected']:
                st.write(f"Genre ({scores['genre_score']}/17)")
                # Primary genre
                if genre['primary_match']:
                    st.write(f"🟢 Primary: {genre['primary']} (+9)")
                else:
                    # Just show the unmatched genre without comparison
                    st.write(f"⚫ {genre['primary']}")
                    
                # Subgenres
                if genre.get('subgenre_points', 0) > 0 and genre.get('subgenre_matches'):
                    st.write(f"🟢 Subgenres: {', '.join(genre['subgenre_matches'])} (+{genre['subgenre_points']})")
                if genre.get('subgenre_mismatches'):
                    st.write(f"⚫ Subgenres: {', '.join(genre['subgenre_mismatches'])}")
            else:
                st.write(f"Genre ({scores['genre_score']}/17)")
                st.write(f"⚫ {genre['primary']} (not selected)")
                if genre.get('shared_subgenres'):
                    genres = ', '.join(genre['shared_subgenres'])
                    st.write(f"⚫ Subgenres: {genres} (not selected)")
                
            # Character types
            if 'characters' in details:
                chars = details['characters']
                st.write(f"Character Types ({scores['character_score']}/14):")
                if chars['selected']:
                    if chars['matches']:
                        first = True
                        for char_type in chars['matches']:
                            if first:
                                st.write(f"🟢 {char_type} (+5)")
                                first = False
                            else:
                                st.write(f"🟢 {char_type} (+1.8)")
                    if chars['mismatches']:
                        for char_type in chars['mismatches']:
                            st.write(f"⚫ {char_type}")
                else:
                    if chars.get('matches') or chars.get('mismatches'):
                        for char_type in (chars.get('matches', []) + chars.get('mismatches', [])):
                            st.write(f"⚫ {char_type} (not selected)")
                    else:
                        st.write("⚫ No character types")
                        
            # Plot elements
            if 'plot' in details:
                plots = details['plot']
                st.write(f"Plot Elements ({scores['plot_score']}/12):")
                if plots['selected']:
                    # Show matches first
                    for plot in plots.get('matches', []):
                        st.write(f"🟢 {plot} (+2.4)")
                    # Then show mismatches
                    for plot in plots.get('mismatches', []):
                        st.write(f"⚫ {plot}")
                else:
                    if plots.get('plot_element_names'):
                        for plot in plots['plot_element_names']:
                            st.write(f"⚫ {plot} (not selected)")
                    elif plots.get('matches') or plots.get('mismatches'):
                        for plot in (plots.get('matches', []) + plots.get('mismatches', [])):
                            st.write(f"⚫ {plot} (not selected)")
                    else:
                        st.write("⚫ No plot elements")
        
        with col2:
            # Source details
            source = details['source']
            if source['selected']:
                st.write(f"Source: {scores['source_score']}/15")
                if source['match']:
                    st.write(f"🟢 Both {source['type1']} (+{scores['source_score']})")
                else:
                    st.write(f"⚫ {source['type1']} vs {source['type2']}")
            else:
                st.write(f"Source ({scores['source_score']}/15):")
                st.write(f"⚫ {source['type1']} (not selected)")
                
            # Theme elements
            if 'themes' in details:
                themes = details['themes']
                st.write(f"\nTheme Elements ({scores['theme_score']}/13):")
                if themes['selected']:
                    if themes.get('matches'):
                        for theme in themes['matches']:
                            st.write(f"🟢 {theme}")
                    if themes.get('mismatches'):
                        for theme in themes['mismatches']:
                            st.write(f"⚫ {theme}")
                else:
                    if 'thematic_element_names' in matches and matches['thematic_element_names']:
                        for theme in matches['thematic_element_names']:
                            st.write(f"⚫ {theme} (not selected)")
                    elif themes.get('matches') or themes.get('mismatches'):
                        for theme in (themes.get('matches', []) + themes.get('mismatches', [])):
                            st.write(f"⚫ {theme} (not selected)")
                    else:
                        st.write("⚫ No theme elements")
                        
            # Tone
            if 'tone' in details:
                tone = details['tone']
                if tone['selected']:
                    st.write(f"\nTone ({scores['tone_score']}/9):")
                    if tone['match']:
                        st.write(f"🟢 Both {tone['tone1']} (+9)")
                    else:
                        st.write(f"⚫ {tone['tone1']} vs {tone['tone2']}")
                else:
                    st.write(f"\nTone ({scores['tone_score']}/9):")
                    st.write(f"⚫ {tone['tone1']} (not selected)")
        
        # Production section
        prod_total = scores['team_score'] + scores.get('network_score', 0) + scores.get('studio_score', 0)
        st.markdown(f"\n**Production Match** ({prod_total}/30 points)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Network details
            st.write(f"Network ({scores['network_score']}/5):")
            if 'network' in details and details['network']['selected']:
                network = details['network']
                if network['match']:
                    st.write(f"🟢 Both {network['name1']} (+5)")
                elif network.get('name1') and network.get('name2'):
                    st.write(f"⚫ {network['name1']} vs {network['name2']}")
                else:
                    st.write(f"⚫ {network['name1']}")
            else:
                if 'network_name' in matches:
                    st.write(f"⚫ {matches['network_name']} (not selected)")
                else:
                    st.write("⚫ No network")
            
            # Studio details
            st.write(f"\nStudio ({scores['studio_score']}/3):")
            if 'studio' in details and details['studio']['selected']:
                studio = details['studio']
                if studio['match']:
                    st.write(f"🟢 Both {studio['name1']} (+3)")
                elif studio.get('name1') and studio.get('name2'):
                    st.write(f"⚫ {studio['name1']} vs {studio['name2']}")
                else:
                    st.write(f"⚫ {studio['name1']}")
            else:
                if 'studio_names' in matches and matches['studio_names']:
                    st.write(f"⚫ {matches['studio_names'][0]} (not selected)")
                else:
                    st.write("⚫ No studio")
                    
        with col2:
            # Team details
            st.write(f"Team ({scores['team_score']}/5):")
            if 'team' in details and details['team']['selected']:
                team = details['team']
                if team.get('shared_members'):
                    # Always show matched members first
                    for name, role in team['shared_members']:
                        st.write(f"🟢 {name} ({role}) (+1)")
                else:
                    st.write("⚫ No team members")
            else:
                if 'team_member_names' in matches and matches['team_member_names']:
                    # Show first 2 members
                    shown_members = matches['team_member_names'][:2]
                    remaining = len(matches['team_member_names']) - 2
                    
                    for name in shown_members:
                        st.write(f"⚫ {name} (not selected)")
                        
                    if remaining > 0:
                        st.write(f"⚫ +{remaining} more team members")
                else:
                    st.write("⚫ No team members")
        
        # Format Match section
        if 'format' in details:
            format = details['format']
            if format['selected'] or format['episode_match'] or format['order_match']:
                st.markdown(f"\n**Format Match** ({scores['format_total']}/5 points)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Episodes
                    st.write(f"Episodes ({scores['episode_score']}/4):")
                    if format['episode_count1'] == format['episode_count2']:
                        st.write(f"🟢 Both {int(format['episode_count1'])} (+4)")
                    else:
                        diff = abs(format['episode_count1'] - format['episode_count2'])
                        if diff <= 2:
                            st.write(f"🟢 {int(format['episode_count1'])} vs {int(format['episode_count2'])} (+4)")
                        elif diff <= 4:
                            st.write(f"🟢 {int(format['episode_count1'])} vs {int(format['episode_count2'])} (+3)")
                        elif diff <= 6:
                            st.write(f"🟢 {int(format['episode_count1'])} vs {int(format['episode_count2'])} (+2)")
                        else:
                            st.write(f"⚫ {int(format['episode_count1'])} vs {int(format['episode_count2'])}")
                
                with col2:
                    # Order type
                    st.write(f"Order Type ({scores['order_score']}/1):")
                    if format['order_type1'] == format['order_type2']:
                        st.write(f"🟢 Both {format['order_type1']} (+1)")
                    else:
                        # Show mismatch with actual score
                        st.write(f"⚫ {format['order_type1']} vs {format['order_type2']} (+0)")
            else:
                st.markdown(f"\n**Format ({scores['format_total']}/5)**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Episodes ({scores['episode_score']}/4):")
                    st.write(f"⚫ {int(format['episode_count1'])} (not selected)")
                
                with col2:
                    st.write(f"Order Type ({scores['order_score']}/1):")
                    st.write(f"⚫ {format['order_type1']} (not selected)")
        
        # Setting Match section
        if 'setting' in details:
            setting = details['setting']
            if setting['selected'] or setting['time_match'] or setting['location_match']:
                st.markdown(f"\n**Setting Match** ({scores['setting_total']}/7 points)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Time period
                    st.write(f"Time Period ({scores['time_score']}/4):")
                    if setting['time_match']:
                        st.write(f"🟢 Both {setting['time']} (+4)")
                    else:
                        st.write(f"⚫ {setting['time']}")
                
                with col2:
                    # Location
                    st.write(f"Location ({scores['location_score']}/3):")
                    if setting['location_match']:
                        st.write(f"🟢 Both {setting['location']} (+3)")
                    else:
                        st.write(f"⚫ {setting['location']}")
            else:
                st.markdown(f"\n**Setting ({scores['setting_total']}/7)**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"Time Period ({scores['time_score']}/4):")
                    st.write(f"⚫ {setting['time']} (not selected)")
                
                with col2:
                    st.write(f"Location ({scores['location_score']}/3):")
                    st.write(f"⚫ {setting['location']} (not selected)")
                # Time setting
                if 'time_setting' in details:
                    time = details['time_setting']
                    st.write(f"Time Setting: {scores['time_score']}/4")
                    if time['match']:
                        st.write(f"⚫ Both {time['time1']}")
                    else:
                        st.write(f"⚫ {time['time1']} vs {time['time2']}")
            
            with col2:
                # Location setting
                if 'location' in details:
                    loc = details['location']
                    st.write(f"Location: {scores['location_score']}/3")
                    if loc['match']:
                        st.write(f"• Both {loc['location1']}")
                    else:
                        st.write(f"• {loc['location1']} vs {loc['location2']}")
        
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
