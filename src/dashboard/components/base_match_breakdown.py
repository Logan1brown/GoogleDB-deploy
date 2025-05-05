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
    header = f"#### {header}"
    if score:
        header += score.format()
    st.markdown(header)




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
    st.markdown("### Content Match ({:.1f}/{:.1f})".format(content_score, content_max))
    st.write("")
    
    # Content fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre
        render_field_match("Genre", details['genre'])
        
        # Source type
        render_field_match("Source Type", details['source'])
        
        # Character types
        render_array_field_match("Character Types", details['characters'])
    
    with col2:
        # Plot elements
        render_array_field_match("Plot Elements", details['plot'])
        
        # Theme elements
        render_array_field_match("Theme Elements", details['themes'])
        
        # Tone
        render_field_match("Tone", details['tone'])
        
        # Setting
        st.markdown("**Setting:**")
        render_field_match("Time", details['time_setting'])
        render_field_match("Location", details['location'])
    
    st.write("")
    
    # Production match section
    st.markdown(f"### Production Match ({production_score:.1f}/{production_max:.1f})")
    st.write("")
    
    # Production fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Network
        render_field_match("Network", details['network'])
        
        # Studio
        render_field_match("Studio", details['studio'])
    
    with col2:
        # Team
        render_array_field_match("Team", details['team'])
    
    st.write("")
    
    # Format match section
    st.markdown(f"### Format Match ({format_score:.1f}/{format_max:.1f})")
    st.write("")
    
    # Format fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Episodes
        render_field_match("Episodes", details['episodes'])
        
        # Order Type
        render_field_match("Order Type", details['order_type'])

def render_matches_section(matches: List[Dict], details_manager, criteria: Dict) -> None:
    """Template method for rendering the matches section."""
    st.markdown("### Top Matches")
    
    # Show top 10 matches in expanders
    for match in matches[:10]:
        comp_score = match['comp_score']
        with st.expander(
            f"#{match['id']}: {match['title']} (Match: {comp_score.total():.1f})", 
            expanded=match == matches[0]
        ):
            details = details_manager.create_match_details(match, criteria)
            # Add CompScore object and raw match data to details
            details['comp_score'] = comp_score
            details.update({
                'comp_score': match['comp_score'],
                'genre': details_manager._process_genre_match(match, criteria),
                'subgenres': details_manager._process_array_field_match(
                    'genre', match.get('subgenres', []), criteria.get('subgenres', []),
                    {'first': details_manager.scoring['content']['components']['genre']['overlap'], 'second': 0}
                ),
                'source': details_manager._process_single_field_match(
                    'source_type', match.get('source_type_id'), criteria.get('source_type_id'),
                    details_manager.scoring['content']['components']['source_type']['match']
                ),
                'characters': details_manager._process_array_field_match(
                    'character_types', match.get('character_type_ids', []), criteria.get('character_type_ids', []),
                    details_manager.scoring['content']['components']['character_types']
                ),
                'plot': details_manager._process_array_field_match(
                    'plot_elements', match.get('plot_element_ids', []), criteria.get('plot_element_ids', []),
                    details_manager.scoring['content']['components']['plot_elements']
                ),
                'themes': details_manager._process_array_field_match(
                    'thematic_elements', match.get('thematic_element_ids', []), criteria.get('thematic_element_ids', []),
                    details_manager.scoring['content']['components']['theme_elements']
                ),
                'tone': details_manager._process_single_field_match(
                    'tone', match.get('tone_id'), criteria.get('tone_id'),
                    details_manager.scoring['content']['components']['tone']['match']
                ),
                'network': details_manager._process_single_field_match(
                    'network', match.get('network_id'), criteria.get('network_id'),
                    details_manager.scoring['production']['components']['network']['match']
                ),
                'studio': details_manager._process_production_field_match(
                    'studios', match.get('studios', []), criteria.get('studio_ids', []),
                    details_manager.scoring['production']['components']['studio']
                ),
                'team': details_manager._process_array_field_match(
                    'team_members', match.get('team_member_ids', []), criteria.get('team_member_ids', []),
                    details_manager.scoring['production']['components']['team']
                ),
                'time_setting': details_manager._process_single_field_match(
                    'time_setting', match.get('time_setting_id'), criteria.get('time_setting_id'),
                    details_manager.scoring['content']['components']['setting']['time']
                ),
                'location': details_manager._process_single_field_match(
                    'location_setting', match.get('location_setting_id'), criteria.get('location_setting_id'),
                    details_manager.scoring['content']['components']['setting']['location']
                ),
                'episodes': details_manager._process_single_field_match(
                    'episodes', match.get('episode_count'), criteria.get('episode_count'),
                    details_manager.scoring['format']['components']['episodes']['within_2']
                ),
                'order_type': details_manager._process_single_field_match(
                    'order_type', match.get('order_type_id'), criteria.get('order_type_id'),
                    details_manager.scoring['format']['components']['order_type']['match']
                )
            })
            render_match_details_section(details)

def render_field_base(label: str, score: Optional[ScoreDisplay] = None) -> None:
    """Base template for rendering a field with optional score."""
    header = f"**{label}**"
    if score:
        header += score.format()
    st.markdown(header)

def render_match_indicator(value: str, matched: bool = True, selected: bool = True) -> None:
    """Template method for rendering a match indicator."""
    if not selected:
        st.markdown(f"⚫ {value} (not selected)")
        return
    
    if matched:
        st.markdown(f"🟢 {value}")
    else:
        st.markdown(f"⚫ {value}")

def render_array_field_base(values: List[str], matches: List[str], selected: bool = True) -> None:
    """Base template for rendering an array field."""
    if not selected:
        if not values:
            st.markdown("⚫ None")
        else:
            st.markdown(f"⚫ {', '.join(values)} (not selected)")
        return
    
    for value in values:
        render_match_indicator(value, value in matches, selected)

def render_field_match(label: str, match: FieldMatch, show_score: bool = True) -> None:
    """Render a single field match using base template methods."""
    score = ScoreDisplay(match.score, match.max_score, show_score)
    render_section_header(label, score)
    
    st.markdown("**Selected:**")
    render_match_indicator(match.name2, matched=match.match, selected=True)
    
    st.markdown("**Matched:**")
    render_match_indicator(match.name1, matched=match.match)

def render_array_field_match(label: str, match: ArrayFieldMatch) -> None:
    """Render a multi-value field match using base template methods."""
    score = ScoreDisplay(match.score, match.max_score, True)
    render_section_header(label, score)
    
    st.markdown("**Selected:**")
    if not match.names2:
        render_match_indicator("None", matched=False, selected=False)
    else:
        for name in match.names2:
            render_match_indicator(name, matched=name in match.matches, selected=True)
    
    st.markdown("**Matched:**")
    if not match.names1:
        render_match_indicator("None", matched=False)
    else:
        for name in match.names1:
            render_match_indicator(name, matched=name in match.matches)

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
