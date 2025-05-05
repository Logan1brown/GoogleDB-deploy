"""Base template for match breakdown components.

This module provides the core template methods for rendering match breakdowns,
which can be extended by specific views like comp builder and show details.

IMPORTANT: This is a template that defines the structure and common patterns.
Specific views should extend these methods with their own display logic.
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.dashboard.components.match_details import FieldMatch, ArrayFieldMatch
from src.dashboard.utils.style_config import FONTS

@dataclass
class ScoreDisplay:
    """Helper class for displaying scores."""
    score: float
    max_score: float
    show_score: bool = True
    
    def format(self) -> str:
        """Format the score for display."""
        return f" ({self.score}/{self.max_score})" if self.show_score else ""

def render_section_header(header: str, score: Optional[ScoreDisplay] = None) -> None:
    """Template method for rendering a section header with optional score."""
    header_text = f"<p style='font-family: {FONTS['primary']['family']}; font-size: {FONTS['primary']['sizes']['body']}px; font-weight: 600; margin-bottom: 0.5em;'>{header}"
    if score:
        header_text += score.format()
    header_text += "</p>"
    st.markdown(header_text, unsafe_allow_html=True)




def render_match_details_section(details: Dict) -> None:
    """Template method for rendering match details section with columns."""
    # Get the CompScore object
    comp_score = details['comp_score']
    
    # Get section scores from CompScore
    content_score = comp_score.content_score()
    content_max = 82  # From CompScore total
    
    production_score = comp_score.production_score()
    production_max = 13  # From CompScore total
    
    format_score = comp_score.format_score()
    format_max = 5  # From CompScore total
    
    # Content Match section
    st.markdown(f"<p style='font-family: {FONTS['primary']['family']}; font-size: {FONTS['primary']['sizes']['header']}px; font-weight: 600; margin-bottom: 0.5em;'>Content Match ({content_score:.1f}/{content_max:.1f})</p>", unsafe_allow_html=True)
    st.write("")
    
    # Content fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre and subgenres
        # Primary genre
        render_field_match("Genre", FieldMatch(
            name1=details.get('genre_name', 'Unknown'),
            name2=details.get('selected_genre_name', 'Unknown'),
            selected=details.get('selected_genre_name') is not None,
            match=details.get('genre_match', False),
            score=comp_score.genre_base,
            max_score=9.0
        ))
        
        # Subgenres
        render_array_field_match("Subgenres", ArrayFieldMatch(
            name1='Multiple' if details.get('subgenre_names') else 'None',
            name2='Multiple' if details.get('selected_subgenre_names') else 'None',
            selected=bool(details.get('selected_subgenre_names')),
            match=bool(details.get('subgenre_matches')),
            score=comp_score.genre_overlap,
            max_score=8.0,
            values1=details.get('subgenre_names', []),
            values2=details.get('selected_subgenre_names', []),
            matches=details.get('subgenre_matches', [])
        ))
        
        # Source type
        render_field_match("Source Type", FieldMatch(
            name1=details.get('source_type_name', 'Unknown'),
            name2=details.get('selected_source_type_name', 'Unknown'),
            selected=details.get('selected_source_type_name') is not None,
            match=details.get('source_type_match', False),
            score=comp_score.source_type,
            max_score=10.0
        ))
        
        # Character types
        render_array_field_match("Character Types", ArrayFieldMatch(
            name1='Multiple' if details.get('character_type_names') else 'None',
            name2='Multiple' if details.get('selected_character_type_names') else 'None',
            selected=bool(details.get('selected_character_type_names')),
            match=bool(details.get('character_type_matches')),
            score=comp_score.character_types,
            max_score=14.0,
            values1=details.get('character_type_names', []),
            values2=details.get('selected_character_type_names', []),
            matches=details.get('character_type_matches', [])
        ))
    
    with col2:
        # Plot elements
        render_array_field_match("Plot Elements", ArrayFieldMatch(
            name1='Multiple' if details.get('plot_element_names') else 'None',
            name2='Multiple' if details.get('selected_plot_element_names') else 'None',
            selected=bool(details.get('selected_plot_element_names')),
            match=bool(details.get('plot_element_matches')),
            score=comp_score.plot_elements,
            max_score=12.0,
            values1=details.get('plot_element_names', []),
            values2=details.get('selected_plot_element_names', []),
            matches=details.get('plot_element_matches', [])
        ))
        
        # Theme elements
        render_array_field_match("Theme Elements", ArrayFieldMatch(
            name1='Multiple' if details.get('theme_element_names') else 'None',
            name2='Multiple' if details.get('selected_theme_element_names') else 'None',
            selected=bool(details.get('selected_theme_element_names')),
            match=bool(details.get('theme_element_matches')),
            score=comp_score.theme_elements,
            max_score=13.0,
            values1=details.get('theme_element_names', []),
            values2=details.get('selected_theme_element_names', []),
            matches=details.get('theme_element_matches', [])
        ))
        
        # Tone
        render_field_match("Tone", FieldMatch(
            name1=details.get('tone_name', 'Unknown'),
            name2=details.get('selected_tone_name', 'Unknown'),
            selected=details.get('selected_tone_name') is not None,
            match=details.get('tone_match', False),
            score=comp_score.tone,
            max_score=9.0
        ))
        
        # Time Setting
        render_field_match("Time Setting", FieldMatch(
            name1=details.get('time_setting_name', 'Unknown'),
            name2=details.get('selected_time_setting_name', 'Unknown'),
            selected=details.get('selected_time_setting_name') is not None,
            match=details.get('time_setting_match', False),
            score=comp_score.time_setting,
            max_score=3.5
        ))
        
        # Location Setting
        render_field_match("Location", FieldMatch(
            name1=details.get('location_setting_name', 'Unknown'),
            name2=details.get('selected_location_setting_name', 'Unknown'),
            selected=details.get('selected_location_setting_name') is not None,
            match=details.get('location_match', False),
            score=comp_score.location,
            max_score=3.5
        ))
    
    st.write("")
    
    # Production match section
    st.markdown(f"<p style='font-family: {FONTS['primary']['family']}; font-size: {FONTS['primary']['sizes']['header']}px; font-weight: 600; margin-bottom: 0.5em;'>Production Match ({production_score:.1f}/{production_max:.1f})</p>", unsafe_allow_html=True)
    st.write("")
    
    # Production fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Network
        render_field_match("Network", FieldMatch(
            name1=details.get('network_name', 'Unknown'),
            name2=details.get('selected_network_name', 'Unknown'),
            selected=details.get('selected_network_name') is not None,
            match=details.get('network_match', False),
            score=comp_score.network,
            max_score=5.0
        ))
        
        # Studio
        render_field_match("Studio", FieldMatch(
            name1=details.get('studio_name', 'Unknown'),
            name2=details.get('selected_studio_name', 'Unknown'),
            selected=details.get('selected_studio_name') is not None,
            match=details.get('studio_match', False),
            score=comp_score.studio,
            max_score=4.0
        ))
    
    with col2:
        # Team
        render_array_field_match("Team", ArrayFieldMatch(
            name1='Multiple' if details.get('team_member_names') else 'None',
            name2='Multiple' if details.get('selected_team_member_names') else 'None',
            selected=bool(details.get('selected_team_member_names')),
            match=bool(details.get('team_member_matches')),
            score=comp_score.team,
            max_score=4.0,
            values1=details.get('team_member_names', []),
            values2=details.get('selected_team_member_names', []),
            matches=details.get('team_member_matches', [])
        ))
    
    st.write("")
    
    # Format match section
    st.markdown(f"<p style='font-family: {FONTS['primary']['family']}; font-size: {FONTS['primary']['sizes']['header']}px; font-weight: 600; margin-bottom: 0.5em;'>Format Match ({format_score:.1f}/{format_max:.1f})</p>", unsafe_allow_html=True)
    st.write("")
    
    # Format fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Episodes
        render_field_match("Episodes", FieldMatch(
            name1=str(details.get('episode_count', 'Unknown')),
            name2=str(details.get('selected_episode_count', 'Unknown')),
            selected=details.get('selected_episode_count') is not None,
            match=details.get('episode_count_match', False),
            score=comp_score.episodes,
            max_score=4.0
        ))
        
    with col2:
        # Order type
        render_field_match("Order Type", FieldMatch(
            name1=details.get('order_type_name', 'Unknown'),
            name2=details.get('selected_order_type_name', 'Unknown'),
            selected=details.get('selected_order_type_name') is not None,
            match=details.get('order_type_match', False),
            score=comp_score.order_type,
            max_score=1.0
        ))
    


def render_matches_section(matches: List[Dict], details_manager, criteria: Dict) -> None:
    """Template method for rendering the matches section."""
    st.markdown(f"<p style='font-family: {FONTS['primary']['family']}; font-size: {FONTS['primary']['sizes']['title']}px; font-weight: 600; margin-bottom: 1em;'>Top Matches</p>", unsafe_allow_html=True)
    
    # Show top 10 matches in expanders
    for match in matches[:10]:
        comp_score = match['comp_score']
        with st.expander(
            f"#### #{match['id']}: {match['title']} (Match: {comp_score.total():.1f})", 
            expanded=match == matches[0]
        ):
            # Get display names for all fields
            details = {
                'comp_score': comp_score,
                # Content fields
                'genre_name': details_manager.get_field_name('genre', match.get('genre_id')),
                'selected_genre_name': details_manager.get_field_name('genre', criteria.get('genre_id')),
                'genre_match': match.get('genre_id') == criteria.get('genre_id'),
                'subgenre_names': details_manager.get_field_names('subgenres', match.get('subgenres', [])),
                'selected_subgenre_names': details_manager.get_field_names('subgenres', criteria.get('subgenres', [])),
                'subgenre_matches': [name for name in details_manager.get_field_names('subgenres', match.get('subgenres', [])) 
                                    if name in details_manager.get_field_names('subgenres', criteria.get('subgenres', []))],
                'source_type_name': details_manager.get_field_name('source_type', match.get('source_type_id')),
                'selected_source_type_name': details_manager.get_field_name('source_type', criteria.get('source_type_id')),
                'source_type_match': match.get('source_type_id') == criteria.get('source_type_id'),
                'character_type_names': details_manager.get_field_names('character_types', match.get('character_type_ids', [])),
                'selected_character_type_names': details_manager.get_field_names('character_types', criteria.get('character_type_ids', [])),
                'character_type_matches': [name for name in details_manager.get_field_names('character_types', match.get('character_type_ids', [])) 
                                         if name in details_manager.get_field_names('character_types', criteria.get('character_type_ids', []))],
                'plot_element_names': details_manager.get_field_names('plot_elements', match.get('plot_element_ids', [])),
                'selected_plot_element_names': details_manager.get_field_names('plot_elements', criteria.get('plot_element_ids', [])),
                'plot_element_matches': [name for name in details_manager.get_field_names('plot_elements', match.get('plot_element_ids', [])) 
                                       if name in details_manager.get_field_names('plot_elements', criteria.get('plot_element_ids', []))],
                'theme_element_names': details_manager.get_field_names('thematic_elements', match.get('thematic_element_ids', [])),
                'selected_theme_element_names': details_manager.get_field_names('thematic_elements', criteria.get('thematic_element_ids', [])),
                'theme_element_matches': [name for name in details_manager.get_field_names('thematic_elements', match.get('thematic_element_ids', [])) 
                                        if name in details_manager.get_field_names('thematic_elements', criteria.get('thematic_element_ids', []))],
                'tone_name': details_manager.get_field_name('tone', match.get('tone_id')),
                'selected_tone_name': details_manager.get_field_name('tone', criteria.get('tone_id')),
                'tone_match': match.get('tone_id') == criteria.get('tone_id'),
                # Setting
                'time_setting_name': details_manager.get_field_name('time_setting', match.get('time_setting_id')),
                'selected_time_setting_name': details_manager.get_field_name('time_setting', criteria.get('time_setting_id')),
                'time_setting_match': match.get('time_setting_id') == criteria.get('time_setting_id'),
                'location_setting_name': details_manager.get_field_name('location_setting', match.get('location_setting_id')),
                'selected_location_setting_name': details_manager.get_field_name('location_setting', criteria.get('location_setting_id')),
                'location_match': match.get('location_setting_id') == criteria.get('location_setting_id'),
                # Production
                'network_name': details_manager.get_field_name('network', match.get('network_id')),
                'selected_network_name': details_manager.get_field_name('network', criteria.get('network_id')),
                'network_match': match.get('network_id') == criteria.get('network_id'),
                'studio_name': details_manager.get_field_name('studio', match.get('studio_id')),
                'selected_studio_name': details_manager.get_field_name('studio', criteria.get('studio_id')),
                'studio_match': match.get('studio_id') == criteria.get('studio_id'),
                'team_member_names': match.get('team_member_names', []),
                'selected_team_member_names': criteria.get('team_member_names', []),
                'team_member_matches': [name for name in match.get('team_member_names', []) 
                                      if name in criteria.get('team_member_names', [])],
                # Format
                'episode_count': match.get('episode_count'),
                'selected_episode_count': criteria.get('episode_count'),
                'episode_count_match': match.get('episode_count') == criteria.get('episode_count'),
                'order_type_name': details_manager.get_field_name('order_type', match.get('order_type_id')),
                'selected_order_type_name': details_manager.get_field_name('order_type', criteria.get('order_type_id')),
                'order_type_match': match.get('order_type_id') == criteria.get('order_type_id')
            }
            render_match_details_section(details)

def render_field_base(label: str, score: Optional[ScoreDisplay] = None) -> None:
    """Base template for rendering a field with optional score."""
    header = f"**{label}**"
    if score:
        header += score.format()
    st.markdown(header)

def render_match_indicator(value: str, matched: bool = True, selected: bool = True) -> None:
    """Template method for rendering a match indicator."""
    # Only show green if both selected and matched
    if matched and selected:
        st.markdown(f"🟢 {value}")
    else:
        st.markdown(f"⚫ {value}")

def render_field_match(label: str, match: FieldMatch, show_score: bool = True) -> None:
    """Render a single field match using base template methods."""
    score = ScoreDisplay(match.score, match.max_score, show_score)
    render_section_header(label, score)
    # Just show the value with appropriate bullet point
    render_match_indicator(match.name1, matched=match.match, selected=match.selected)

def render_array_field_match(label: str, match: ArrayFieldMatch) -> None:
    """Render a multi-value field match using base template methods."""
    score = ScoreDisplay(match.score, match.max_score, True)
    render_section_header(label, score)
    # Just show values with appropriate bullet points
    if not match.values1:
        st.markdown("⚫ None")
    else:
        for value in match.values1:
            render_match_indicator(value, value in match.matches, match.selected)

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
    title_score = ScoreDisplay(details['total'], details['max'], True)
    header = f"{title} (Match:{title_score.format()}"
    if success_score is not None:
        success_display = ScoreDisplay(success_score, 100, True)
        header += f", Success:{success_display.format()}"
    header += ")"
    
    def render_content():
        # Content Match section
        content_score = ScoreDisplay(details['content']['score'], details['content']['max'])
        render_section_header("Content Match", content_score)
        
        # Create two columns for content
        col1, col2 = render_two_columns()
        
        with col1:
            # Genre details
            genre = details['genre']
            genre_score = ScoreDisplay(
                content_details['components']['genre']['base'],
                content_details['components']['genre']['base']
            )
            render_field_base("Genre", genre_score)
            
            if genre['selected']:
                # Primary genre
                render_match_indicator(
                    f"Primary: {genre['primary']}",
                    genre['primary_match'],
                    True
                )
                
                # Subgenres
                subgenre_score = ScoreDisplay(
                    content_details['components']['genre']['overlap'],
                    content_details['components']['genre']['overlap']
                )
                render_field_base("Subgenres", subgenre_score)
                
                if genre.get('subgenres'):
                    render_array_field_base(
                        genre['subgenres'],
                        genre.get('subgenre_matches', []),
                        True
                    )
            else:
                render_match_indicator(genre['primary'], False, False)
                if genre.get('subgenres'):
                    render_array_field_base(
                        genre['subgenres'],
                        [],
                        False
                    )
            
            # Character types
            if 'characters' in details:
                chars = details['characters']
                char_score = ScoreDisplay(
                    content_details['components']['character_types']['score'],
                    content_details['components']['character_types']['max']
                )
                render_field_base("Character Types", char_score)
                
                if chars['selected']:
                    render_array_field_base(
                        chars.get('matches', []) + chars.get('mismatches', []),
                        chars.get('matches', []),
                        True
                    )
                else:
                    render_array_field_base(
                        chars.get('matches', []) + chars.get('mismatches', []),
                        [],
                        False
                    )
                    
            # Plot elements
            if 'plot' in details:
                plots = details['plot']
                plot_score = ScoreDisplay(
                    details['content']['components']['plot_elements']['score'],
                    details['content']['components']['plot_elements']['max']
                )
                render_field_base("Plot Elements", plot_score)
                if plots['selected']:
                    render_array_field_base(
                        plots.get('matches', []) + plots.get('mismatches', []),
                        plots.get('matches', []),
                        True
                    )
                else:
                    render_array_field_base(
                        plots.get('matches', []) + plots.get('mismatches', []),
                        [],
                        False
                    )
                
        with col2:
            # Source type
            source = details['source']
            source_score = ScoreDisplay(
                details['content']['components']['source_type']['score'],
                details['content']['components']['source_type']['max']
            )
            render_field_base("Source Type", source_score)
            render_match_indicator(source['name1'], source['match'], source['selected'])
            
            # Theme elements
            if 'themes' in details:
                themes = details['themes']
                theme_score = ScoreDisplay(
                    details['content']['components']['theme_elements']['score'],
                    details['content']['components']['theme_elements']['max']
                )
                render_field_base("Theme Elements", theme_score)
                render_array_field_base(
                    themes.get('matches', []) + themes.get('mismatches', []),
                    themes.get('matches', []),
                    themes['selected']
                )
            
            # Tone
            tone = details['tone']
            tone_score = ScoreDisplay(
                details['content']['components']['tone']['score'],
                details['content']['components']['tone']['max']
            )
            render_field_base("Tone", tone_score)
            render_match_indicator(tone['name1'], tone['match'], tone['selected'])
        
        # Production match section
        production_score = ScoreDisplay(
            details['production']['score'],
            details['production']['max']
        )
        render_section_header("Production Match", production_score)
        
        col1, col2 = render_two_columns()
        
        with col1:
            # Network
            network = details['network']
            network_score = ScoreDisplay(
                details['production']['components']['network']['score'],
                details['production']['components']['network']['max']
            )
            render_field_base("Network", network_score)
            render_match_indicator(network['name1'], network['match'], network['selected'])
            
            # Studio
            if 'studio' in details:
                studio = details['studio']
                studio_score = ScoreDisplay(
                    details['production']['components']['studio']['score'],
                    details['production']['components']['studio']['max']
                )
                render_field_base("Studio", studio_score)
                render_array_field_base(
                    studio.get('matches', []) + studio.get('mismatches', []),
                    studio.get('matches', []),
                    studio['selected']
                )
        
        with col2:
            # Team members
            if 'team' in details:
                team = details['team']
                team_score = ScoreDisplay(
                    details['production']['components']['team']['score'],
                    details['production']['components']['team']['max']
                )
                render_field_base("Team", team_score)
                render_array_field_base(
                    team.get('matches', []) + team.get('mismatches', []),
                    team.get('matches', []),
                    team['selected']
                )
        
        # Setting match section
        setting_score = ScoreDisplay(
            details['setting']['total_score'],
            details['setting']['max_score']
        )
        render_section_header("Setting Match", setting_score)
        
        col1, col2 = render_two_columns()
        
        with col1:
            # Time setting
            time = details['setting']['time']
            render_match_indicator(time['name1'], time['match'], time['selected'])
        
        with col2:
            # Location setting
            location = details['setting']['location']
            render_match_indicator(location['name1'], location['match'], location['selected'])
        
        # Format match section
        format_score = ScoreDisplay(
            details['format']['total_score'],
            details['format']['max_score']
        )
        render_section_header("Format Match", format_score)
        
        col1, col2 = render_two_columns()
        
        with col1:
            # Episodes
            episodes = details['format']['episodes']
            render_match_indicator(episodes['name1'], episodes['match'], episodes['selected'])
        
        with col2:
            # Order type
            order = details['format']['order_type']
            render_match_indicator(order['name1'], order['match'], order['selected'])
        
        if description:
            st.markdown(f"\n**Description:**\n{description}")
    
    # Render with or without expander
    if use_expander:
        with st.expander(header, expanded=expanded):
            render_content()
    else:
        st.markdown(header)
        render_content()
