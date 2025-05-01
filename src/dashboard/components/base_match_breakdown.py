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
        st.markdown(f"\n**Content Match** ({scores['content_total']}/82 points)")
        
        # Genre and Source (core content)
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre details
            genre = details['genre']
            if genre['selected']:
                st.write("Genre:")
                st.write(f"Score: {scores['genre_score']}/17")
                # Primary genre
                if genre['primary_match']:
                    st.write(f"🟢 Primary: {genre['primary']} (+9)")
                else:
                    st.write(f"⚫ {genre['primary']} vs {genre['primary2']}")
                    
                # Subgenres
                if genre.get('subgenre_points', 0) > 0 and genre.get('subgenre_matches'):
                    st.write(f"🟢 Subgenres: {', '.join(genre['subgenre_matches'])} (+{genre['subgenre_points']})")
                if genre.get('subgenre_mismatches'):
                    st.write(f"⚫ Subgenres: {', '.join(genre['subgenre_mismatches'])}")
            else:
                st.write("Genre:")
                st.write(f"⚫ {genre['primary']} (not selected)")
                if genre.get('shared_subgenres'):
                    genres = ', '.join(genre['shared_subgenres'])
                    st.write(f"⚫ Subgenres: {genres} (not selected)")
                
            # Character types
            if 'characters' in details:
                chars = details['characters']
                if chars['selected']:
                    st.write("\nCharacter Types:")
                    st.write(f"Score: {scores['character_score']}/14")
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
                    st.write("\nCharacter Types:")
                    if chars.get('matches') or chars.get('mismatches'):
                        for char_type in (chars.get('matches', []) + chars.get('mismatches', [])):
                            st.write(f"⚫ {char_type} (not selected)")
                    else:
                        st.write("⚫ No character types")
                        
            # Plot elements
            if 'plot' in details:
                plot = details['plot']
                if plot['selected']:
                    st.write("\nPlot Elements:")
                    st.write(f"Score: {scores['plot_score']}/10")
                    if plot['matches']:
                        for element in plot['matches']:
                            st.write(f"🟢 {element} (+{scores['plot_score']})")
                    if plot['mismatches']:
                        for element in plot['mismatches']:
                            st.write(f"⚫ {element}")
                else:
                    st.write("\nPlot Elements:")
                    for element in plot['matches'] + plot['mismatches']:
                        st.write(f"⚫ {element} (not selected)")
        
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
                st.write(f"Source ({scores['source_score']}/15)")
                st.write(f"⚫ {source['type1']} (not selected)")
                
            # Theme elements
            if 'themes' in details:
                themes = details['themes']
                if themes['selected']:
                    st.write("\nTheme Elements:")
                    st.write(f"Score: {scores['theme_score']}/13")
                    if themes['matches']:
                        for theme in themes['matches']:
                            st.write(f"🟢 {theme} (+2.6)")
                    if themes['mismatches']:
                        for theme in themes['mismatches']:
                            st.write(f"⚫ {theme}")
                else:
                    st.write("\nTheme Elements:")
                    if themes.get('matches') or themes.get('mismatches'):
                        for theme in (themes.get('matches', []) + themes.get('mismatches', [])):
                            st.write(f"⚫ {theme} (not selected)")
                    else:
                        st.write("⚫ No theme elements")
                        
            # Tone
            if 'tone' in details:
                tone = details['tone']
                if tone['selected']:
                    st.write("\nTone:")
                    st.write(f"Score: {scores['tone_score']}/5")
                    if tone['tone1'] == tone['tone2']:
                        st.write(f"🟢 Both {tone['tone1']} (+5)")
                    else:
                        st.write(f"⚫ {tone['tone1']} vs {tone['tone2']}")
                else:
                    st.write("\nTone:")
                    st.write(f"⚫ {tone['tone1']} (not selected)")
        
        # Production section
        prod_total = scores['team_score'] + scores.get('network_score', 0) + scores.get('studio_score', 0)
        st.markdown(f"\n**Production Match** ({prod_total}/30 points)")
        
        # Network details
        if 'network' in details:
            network = details['network']
            st.write("Network:")
            st.write(f"Score: {scores['network_score']}/7")
            if network['match'] and network['selected']:
                st.write(f"🟢 Both {network['name1']} (+{scores['network_score']})")
            elif network['name1'] and network['name2']:
                st.write(f"⚫ {network['name1']} vs {network['name2']}")
            else:
                st.write(f"⚫ {network['name1']} (not selected)")
        
        # Studio details
        if 'studio' in details:
            studio = details['studio']
            st.write("\nStudio:")
            st.write(f"Score: {scores['studio_score']}/3")
            if studio['match'] and studio['selected']:
                st.write(f"🟢 Both {studio['name1']} (+{scores['studio_score']})")
            elif studio['name1'] and studio['name2']:
                st.write(f"⚫ {studio['name1']} vs {studio['name2']}")
            else:
                st.write(f"⚫ {studio['name1']} (not selected)")
        
        # Team details
        if 'team' in details:
            team = details['team']
            st.write("\nTeam:")
            st.write(f"Score: {scores['team_score']}/20")
            if team['selected'] and team.get('shared_members'):
                for name, role in team['shared_members']:
                    st.write(f"🟢 {name} ({role}) (+{scores['team_score']})")
            elif team.get('team_members'):
                for name in team['team_members']:
                    st.write(f"⚫ {name} (not selected)")
            else:
                st.write("⚫ No team members")
        
        # Format Match section
        if 'format' in details:
            format = details['format']
            if format['selected'] or format['episode_match'] or format['order_match']:
                st.markdown(f"\n**Format Match** ({scores['format_total']}/5 points)")
                
                # Episodes
                st.write("Episodes:")
                st.write(f"Score: {scores['episode_score']}/4")
                if format['episode_count1'] == format['episode_count2']:
                    st.write(f"🟢 Both {format['episode_count1']} (+4)")
                else:
                    diff = abs(format['episode_count1'] - format['episode_count2'])
                    if diff <= 2:
                        st.write(f"🟢 {format['episode_count1']} vs {format['episode_count2']} (+4)")
                    elif diff <= 4:
                        st.write(f"🟢 {format['episode_count1']} vs {format['episode_count2']} (+3)")
                    elif diff <= 6:
                        st.write(f"🟢 {format['episode_count1']} vs {format['episode_count2']} (+2)")
                    else:
                        st.write(f"⚫ {format['episode_count1']} vs {format['episode_count2']}")
                    
                # Order type
                st.write("\nOrder Type:")
                st.write(f"Score: {scores['order_score']}/1")
                if format['order_type1'] == format['order_type2']:
                    st.write(f"🟢 Both {format['order_type1']} (+1)")
                else:
                    st.write(f"⚫ {format['order_type1']} vs {format['order_type2']}")
            else:
                st.markdown(f"\n**Format ({scores['format_total']}/5)**")
                st.write("Episodes:")
                st.write(f"⚫ {format['episode_count1']} (not selected)")
                st.write("\nOrder Type:")
                st.write(f"⚫ {format['order_type1']} (not selected)")
        
        # Setting Match section
        if 'setting' in details:
            setting = details['setting']
            if setting['selected'] or setting['time_match'] or setting['location_match']:
                st.markdown(f"\n**Setting Match** ({scores['setting_total']}/7 points)")
                
                # Time period
                st.write("Time Period:")
                st.write(f"Score: {scores['time_score']}/4")
                if setting['time_match']:
                    st.write(f"🟢 Both {setting['time']} (+4)")
                else:
                    st.write(f"⚫ {setting['time']}")
                
                # Location
                st.write("\nLocation:")
                st.write(f"Score: {scores['location_score']}/3")
                if setting['location_match']:
                    st.write(f"🟢 Both {setting['location']} (+3)")
                else:
                    st.write(f"⚫ {setting['location']}")
            else:
                st.markdown(f"\n**Setting ({scores['setting_total']}/7)**")
                st.write("Time Period:")
                st.write(f"⚫ {setting['time']} (not selected)")
                st.write("\nLocation:")
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
