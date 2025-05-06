"""Base template for match breakdown components.

This module provides the core template methods for rendering match breakdowns,
which can be extended by specific views like comp builder and show details.

IMPORTANT: This is a template that defines the structure and common patterns.
Specific views should extend these methods with their own display logic.
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dashboard.components.match_details import FieldMatch, ArrayFieldMatch
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




def render_match_details_section(details: Dict, success_score: Optional[float] = None, description: Optional[str] = None) -> None:
    """Template method for rendering match details section with columns."""
    # Score details passed directly
    
    # Display description if available
    if description:
        st.markdown(f"<p style='font-family: {FONTS['primary']['family']}; font-size: {FONTS['primary']['sizes']['body']}px; margin-bottom: 1em;'>{description}</p>", unsafe_allow_html=True)
        st.write("")
    
    # Display success score if available
    if success_score is not None:
        success_display = ScoreDisplay(success_score, 100, True)
        st.markdown(f"Success Score: {success_display.format()}", unsafe_allow_html=True)
        st.write("")
    
    # Content match section
    render_section_header("Content Match", ScoreDisplay(
        score=details['content']['score'],
        max_score=details['content']['max']
    ))
    
    st.write("")
    
    # Content fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre and subgenres
        render_field_match("Genre", details['genre'])
        render_array_field_match("Subgenres", details['subgenres'])
        
        # Source type
        render_field_match("Source Type", details['source'])
        
        # Character types
        render_array_field_match("Character Types", details['characters'])
        
        # Time Setting
        render_field_match("Time Setting", details['time_setting'])
    
    with col2:
        # Plot elements
        render_array_field_match("Plot Elements", details['plot_elements'])
        
        # Theme elements
        render_array_field_match("Theme Elements", details['theme_elements'])
        
        # Tone
        render_field_match("Tone", details['tone'])
    
    st.write("")
    
    # Production match section
    render_section_header("Production Match", ScoreDisplay(
        score=details['production']['score'],
        max_score=details['production']['max']
    ))
    
    # Production fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Network
        render_field_match("Network", details['network'])
        
        # Studios
        render_array_field_match("Studios", details['studio'])
    
    with col2:
        # Team
        render_array_field_match("Team Members", details['team'])
    
    st.write("")
    
    # Format match section
    render_section_header("Format Match", ScoreDisplay(
        score=details['format']['score'],
        max_score=details['format']['max']
    ))
    
    # Format fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Episodes
        render_field_match("Episode Count", details['episodes'])
        
    with col2:
        # Order type
        render_field_match("Order Type", details['order_type'])
    


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
                'studio_names': details_manager.get_field_names('studios', match.get('studios', [])),
                'selected_studio_names': details_manager.get_field_names('studios', criteria.get('studio_ids', [])),
                'studio_matches': [name for name in details_manager.get_field_names('studios', match.get('studios', [])) 
                                  if name in details_manager.get_field_names('studios', criteria.get('studio_ids', []))],
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

def render_field_match(label: str, match: 'FieldMatch', show_score: bool = True) -> None:
    """Render a single field match using base template methods."""
    score = ScoreDisplay(match.score, match.max_score, show_score)
    render_section_header(label, score)
    # Just show the value with appropriate bullet point
    render_match_indicator(match.name1, matched=match.match, selected=match.selected)

def render_array_field_match(label: str, match: 'ArrayFieldMatch') -> None:
    """Render a multi-value field match using base template methods."""
    score = ScoreDisplay(match.score, match.max_score, True)
    render_section_header(label, score)
    # Just show values with appropriate bullet points
    if not match.values1:
        render_match_indicator("None", False, match.selected)
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
    # Build title with match score only
    scores = details.get('scores', {})
    total = scores.get('total', {})
    title_score = ScoreDisplay(total.get('score', 0), total.get('max', 0), True)
    header = f"{title} (Match:{title_score.format()})"
    
    def render_content():
        # Show success score if available
        if success_score is not None:
            success_display = ScoreDisplay(success_score, 100, True)
            st.markdown(f"Success Score: {success_display.format()}", unsafe_allow_html=True)
            st.write("")
            
        # Content Match section
        content = scores.get('content', {})
        content_score = ScoreDisplay(content.get('score', 0), content.get('max', 0))
        render_section_header("Content Match", content_score)
        
        # Create two columns for content
        col1, col2 = render_two_columns()
        
        with col1:
            # Genre details
            genre = content.get('breakdown', {}).get('genre', {})
            genre_score = ScoreDisplay(
                genre.get('score', 0),
                genre.get('max', 0)
            )
            render_field_base("Genre", genre_score)
            
            if genre.get('selected', False):
                # Primary genre
                render_match_indicator(
                    f"Primary: {genre.get('name1', 'Unknown')}",
                    genre.get('match', False),
                    True
                )
                
                # Subgenres
                subgenres = genre.get('subgenres', {})
                subgenre_score = ScoreDisplay(
                    subgenres.get('score', 0),
                    subgenres.get('max', 0)
                )
                render_field_base("Subgenres", subgenre_score)
                
                if subgenres.get('selected', False):
                    render_array_field_base(
                        subgenres.get('values1', []),
                        subgenres.get('matches', []),
                        True
                    )
            else:
                render_match_indicator(genre.get('name1', 'Unknown'), False, False)
                if genre.get('subgenres', {}).get('values1'):
                    render_array_field_base(
                        genre.get('subgenres', {}).get('values1', []),
                        [],
                        False
                    )
            
            # Character types
            characters = content.get('breakdown', {}).get('characters', {})
            if characters:
                char_score = ScoreDisplay(
                    characters.get('score', 0),
                    characters.get('max', 0)
                )
                render_field_base("Character Types", char_score)
                
                if characters.get('selected', False):
                    render_array_field_base(
                        characters.get('values1', []),
                        characters.get('matches', []),
                        True
                    )
                else:
                    render_array_field_base(
                        characters.get('values1', []),
                        [],
                        False
                    )
                    
            # Plot elements
            plot = content.get('breakdown', {}).get('plot', {})
            if plot:
                plot_score = ScoreDisplay(
                    plot.get('score', 0),
                    plot.get('max', 0)
                )
                render_field_base("Plot Elements", plot_score)
                if plot.get('selected', False):
                    render_array_field_base(
                        plot.get('values1', []),
                        plot.get('matches', []),
                        True
                    )
                else:
                    render_array_field_base(
                        plot.get('values1', []),
                        [],
                        False
                    )
                
        with col2:
            # Source type
            source = content.get('breakdown', {}).get('source', {})
            source_score = ScoreDisplay(
                source.get('score', 0),
                source.get('max', 0)
            )
            render_field_base("Source Type", source_score)
            render_match_indicator(
                source.get('name1', 'Unknown'),
                source.get('match', False),
                source.get('selected', False)
            )
            
            # Theme elements
            themes = content.get('breakdown', {}).get('themes', {})
            if themes:
                theme_score = ScoreDisplay(
                    themes.get('score', 0),
                    themes.get('max', 0)
                )
                render_field_base("Theme Elements", theme_score)
                render_array_field_base(
                    themes.get('values1', []),
                    themes.get('matches', []),
                    themes.get('selected', False)
                )
            
            # Time Setting
            time = content.get('breakdown', {}).get('time_setting', {})
            time_score = ScoreDisplay(
                time.get('score', 0),
                time.get('max', 0)
            )
            render_field_base("Time Setting", time_score)
            render_match_indicator(
                time.get('name1', 'Unknown'),
                time.get('match', False),
                time.get('selected', False)
            )
            
            # Location Setting
            location = content.get('breakdown', {}).get('location_setting', {})
            location_score = ScoreDisplay(
                location.get('score', 0),
                location.get('max', 0)
            )
            render_field_base("Location Setting", location_score)
            render_match_indicator(
                location.get('name1', 'Unknown'),
                location.get('match', False),
                location.get('selected', False)
            )
            
            # Tone
            tone = content.get('breakdown', {}).get('tone', {})
            tone_score = ScoreDisplay(
                tone.get('score', 0),
                tone.get('max', 0)
            )
            render_field_base("Tone", tone_score)
            render_match_indicator(
                tone.get('name1', 'Unknown'),
                tone.get('match', False),
                tone.get('selected', False)
            )
        
        # Production match section
        production = scores.get('production', {})
        production_score = ScoreDisplay(
            production.get('score', 0),
            production.get('max', 0)
        )
        render_section_header("Production Match", production_score)
        
        col1, col2 = render_two_columns()
        
        with col1:
            # Network
            network = production.get('breakdown', {}).get('network', {})
            network_score = ScoreDisplay(
                network.get('score', 0),
                network.get('max', 0)
            )
            render_field_base("Network", network_score)
            render_match_indicator(
                network.get('name1', 'Unknown'),
                network.get('match', False),
                network.get('selected', False)
            )
            
            # Studio
            studio = production.get('breakdown', {}).get('studio', {})
            if studio:
                studio_score = ScoreDisplay(
                    studio.get('score', 0),
                    studio.get('max', 0)
                )
                render_field_base("Studio", studio_score)
                render_array_field_base(
                    studio.get('values1', []),
                    studio.get('matches', []),
                    studio.get('selected', False)
                )
        
        with col2:
            # Team members
            team = production.get('breakdown', {}).get('team', {})
            if team:
                team_score = ScoreDisplay(
                    team.get('score', 0),
                    team.get('max', 0)
                )
                render_field_base("Team", team_score)
                render_array_field_base(
                    team.get('values1', []),
                    team.get('matches', []),
                    team.get('selected', False)
                )
        
        # Format match section
        format_section = scores.get('format', {})
        format_score = ScoreDisplay(
            format_section.get('score', 0),
            format_section.get('max', 0)
        )
        render_section_header("Format Match", format_score)
        
        col1, col2 = render_two_columns()
        
        with col1:
            # Episodes
            episodes = format_section.get('breakdown', {}).get('episodes', {})
            render_match_indicator(
                episodes.get('name1', 'Unknown'),
                episodes.get('match', False),
                episodes.get('selected', False)
            )
        
        with col2:
            # Order type
            order = format_section.get('breakdown', {}).get('order_type', {})
            render_match_indicator(
                order.get('name1', 'Unknown'),
                order.get('match', False),
                order.get('selected', False)
            )
        
        if description:
            st.markdown(f"\n**Description:**\n{description}")
    
    # Render with or without expander
    if use_expander:
        with st.expander(header, expanded=expanded):
            render_content()
    else:
        st.markdown(header)
        render_content()
