"""Base component for displaying show match breakdowns.

This component provides the core functionality for displaying match breakdowns,
which can be used by both the comp builder and show detail pages.
"""

import streamlit as st
from typing import Dict, Optional

def render_base_match_breakdown(
    title: str,
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
        details: Dictionary of match details and scores
        matches: Dictionary of raw match data
        success_score: Optional success score
        expanded: Whether the section should be expanded by default
        use_expander: Whether to wrap in an expander
        description: Optional show description
    """
    # Build title with scores
    header = f"{title} (Match: {details['total']:.1f}"
    if success_score is not None:
        success_display = f"{success_score:.1f}" if success_score is not None else "N/A"
        header += f", Success: {success_display}"
    header += ")"
    
    def render_content():
        # Content Match section
        content_details = details['content']
        st.markdown(f"\n**Content Match** ({content_details['score']}/{content_details['max']} points)")
        
        # Genre and Source (core content)
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre details
            genre = details['genre']
            if genre['selected']:
                st.write(f"Genre ({content_details['components']['genre']}/17)")
                # Primary genre
                if genre['primary_match']:
                    st.write(f"🟢 Primary: {genre['primary']} (+9)")
                else:
                    # Just show the unmatched genre without comparison
                    st.write(f"⚫ {genre['primary']}")
                    
                # Subgenres
                # Show matching subgenres with full 8 points
                if genre.get('subgenre_points', 0) > 0 and genre.get('subgenre_matches'):
                    st.write(f"🟢 Subgenres: {', '.join(genre['subgenre_matches'])} (+8)")
                if genre.get('subgenre_mismatches'):
                    st.write(f"⚫ Subgenres: {', '.join(genre['subgenre_mismatches'])}")
            else:
                st.write(f"Genre ({content_details['components']['genre']}/17)")
                st.write(f"⚫ {genre['primary']} (not selected)")
                if genre.get('shared_subgenres'):
                    genres = ', '.join(genre['shared_subgenres'])
                    st.write(f"⚫ Subgenres: {genres} (not selected)")
                
            # Character types
            if 'characters' in details:
                chars = details['characters']
                st.write(f"Character Types ({content_details['components']['character_types']}/14):")
                if chars['selected']:
                    if chars['matches']:
                        first = True
                        for char_type in chars['matches']:
                            if first:
                                st.write(f"🟢 {char_type} (+10.5)")
                                first = False
                            else:
                                st.write(f"🟢 {char_type} (+3.5)")
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
                st.write(f"Plot Elements ({content_details['components']['plot_elements']}/12):")
                if plots['selected']:
                    # Show matches (elements we selected that the show has)
                    if plots.get('matches'):
                        first = True
                        for plot in plots['matches']:
                            if first:
                                st.write(f"🟢 {plot} (+9)")
                                first = False
                            else:
                                st.write(f"🟢 {plot} (+3)")
                    
                    # Show elements in show that we didn't select
                    if plots.get('mismatches'):
                        for plot in plots['mismatches']:
                            st.write(f"⚫ {plot}")
                else:
                    # Show all plot elements as unselected
                    if 'plot' in matches and matches['plot']:
                        for plot in matches['plot']:
                            st.write(f"⚫ {plot} (not selected)")
                    else:
                        st.write("⚫ No plot elements")
        
        with col2:
            # Source details
            source = details['source']
            if source['selected']:
                st.write(f"Source: {content_details['components']['source_type']}/15")
                if source['match']:
                    st.write(f"🟢 Both {source['name1']} (+{content_details['components']['source_type']})")
                else:
                    st.write(f"⚫ {source['name1']} vs {source['name2']}")
            else:
                st.write(f"Source ({content_details['components']['source_type']}/15):")
                st.write(f"⚫ {source['name1']} (not selected)")
                
            # Theme elements
            if 'themes' in details:
                themes = details['themes']
                st.write(f"\nTheme Elements ({content_details['components']['theme_elements']}/13):")
                if themes['selected']:
                    if themes['matches']:
                        first = True
                        for theme in themes['matches']:
                            if first:
                                st.write(f"🟢 {theme} (+9.75)")
                                first = False
                            else:
                                st.write(f"🟢 {theme} (+3.25)")
                    if themes['mismatches']:
                        for theme in themes['mismatches']:
                            st.write(f"⚫ {theme}")
                else:
                    if themes.get('matches') or themes.get('mismatches'):
                        for theme in (themes.get('matches', []) + themes.get('mismatches', [])):
                            st.write(f"⚫ {theme} (not selected)")
                    else:
                        st.write("⚫ No theme elements")
                        
            # Tone
            if 'tone' in details:
                tone = details['tone']
                if tone['selected']:
                    st.write(f"\nTone ({content_details['components']['tone']}/9):")
                    if tone['match']:
                        st.write(f"🟢 Both {tone['name1']} (+9)")
                    else:
                        st.write(f"⚫ {tone['name1']} vs {tone['name2']}")
                else:
                    st.write(f"\nTone ({content_details['components']['tone']}/9):")
                    st.write(f"⚫ {tone['name1']} (not selected)")
        
        # Production match section
        prod_details = details['production']
        st.markdown(f"\n**Production Match** ({prod_details['score']}/{prod_details['max']} points)")
        
        # Network match
        if 'network' in details:
            network = details['network']
            st.write(f"Network ({prod_details['components']['network']}/5):")
            if network['selected']:
                if network['match']:
                    st.write(f"🟢 Both {network['name1']} (+5)")
                else:
                    st.write(f"⚫ {network['name1']} vs {network['name2']}")
            else:
                st.write(f"⚫ {network['name1']} (not selected)")
                
        # Studio match
        if 'studios' in details:
            studios = details['studios']
            st.write(f"\nStudio ({prod_details['components']['studio']}/3):")
            if studios['selected']:
                if studios['matches']:
                    first = True
                    for studio in studios['matches']:
                        if first:
                            st.write(f"🟢 {studio} (+2)")
                            first = False
                        else:
                            st.write(f"🟢 {studio} (+0.5)")
                if studios['mismatches']:
                    for studio in studios['mismatches']:
                        st.write(f"⚫ {studio}")
            else:
                if studios.get('matches') or studios.get('mismatches'):
                    for studio in (studios.get('matches', []) + studios.get('mismatches', [])):
                        st.write(f"⚫ {studio} (not selected)")
                else:
                    st.write("⚫ No studios")
                    
        # Team match
        if 'team' in details:
            team = details['team']
            st.write(f"Team ({prod_details['components']['team']}/5):")
            if team['selected']:
                if team['matches']:
                    first = True
                    for member in team['matches']:
                        if first:
                            st.write(f"🟢 {member} (+3)")
                            first = False
                        else:
                            st.write(f"🟢 {member} (+1)")
                if team['mismatches']:
                    for member in team['mismatches']:
                        st.write(f"⚫ {member}")
            else:
                if team.get('matches') or team.get('mismatches'):
                    for member in (team.get('matches', []) + team.get('mismatches', [])):
                        st.write(f"⚫ {member} (not selected)")
                else:
                    st.write("⚫ No team members")
        
        # Format match section
        format_details = details['format']
        st.markdown(f"\n**Format Match** ({format_details['score']}/{format_details['max']} points)")
        
        # Episode count match
        if 'episodes' in details:
            episodes = details['episodes']
            st.write(f"Episodes ({format_details['components']['episodes']}/4):")
            if episodes['selected']:
                if episodes['match']:
                    st.write(f"🟢 Both {episodes['count1']} (+4)")
                else:
                    st.write(f"⚫ {episodes['count1']} vs {episodes['count2']}")
            else:
                st.write(f"⚫ {episodes['count1']} (not selected)")
                        
        # Order type match
        if 'order_type' in details:
            order = details['order_type']
            st.write(f"Order Type ({format_details['components']['order_type']}/1):")
            if order['selected']:
                if order['match']:
                    st.write(f"🟢 Both {order['name1']} (+1)")
                else:
                    st.write(f"⚫ {order['name1']} vs {order['name2']}")
            else:
                st.write(f"⚫ {order['name1']} (not selected)")
        
        # Setting match section
        setting_score = content_details['components']['setting']
        st.markdown(f"\n**Setting Match** ({setting_score}/7 points)")
        
        # Time period match
        if 'time_period' in details:
            time = details['time_period']
            st.write(f"Time Period ({setting_score * 4/7}/4):")
            if time['selected']:
                if time['match']:
                    st.write(f"🟢 Both {time['name1']} (+4)")
                else:
                    st.write(f"⚫ {time['name1']} vs {time['name2']}")
            else:
                st.write(f"⚫ {time['name1']} (not selected)")
                        
        # Location match
        if 'location' in details:
            location = details['location']
            st.write(f"Location ({setting_score * 3/7}/3):")
            if location['selected']:
                if location['match']:
                    st.write(f"🟢 Both {location['name1']} (+3)")
                else:
                    st.write(f"⚫ {location['name1']} vs {location['name2']}")
            else:
                st.write(f"⚫ {location['name1']} (not selected)")
        
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
