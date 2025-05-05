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
    st.markdown("## Content Match ({:.1f}/{:.1f})".format(content_score, content_max))
    st.markdown("### Score Breakdown")
    st.markdown(f"- Genre: {comp_score.genre_base + comp_score.genre_overlap:.1f}/17.0")
    st.markdown(f"- Source Type: {comp_score.source_type:.1f}/10.0")
    st.markdown(f"- Character Types: {comp_score.character_types:.1f}/14.0")
    st.markdown(f"- Plot Elements: {comp_score.plot_elements:.1f}/12.0")
    st.markdown(f"- Theme Elements: {comp_score.theme_elements:.1f}/13.0")
    st.markdown(f"- Tone: {comp_score.tone:.1f}/9.0")
    st.markdown(f"- Setting: {comp_score.time_setting + comp_score.location:.1f}/7.0")
    
    # Production match section
    st.markdown(f"## Production Match ({production_score:.1f}/{production_max:.1f})")
    st.markdown("### Score Breakdown")
    st.markdown(f"- Network: {comp_score.network:.1f}/5.0")
    st.markdown(f"- Studio: {comp_score.studio:.1f}/4.0")
    st.markdown(f"- Team: {comp_score.team:.1f}/4.0")
    
    # Format match section
    st.markdown(f"## Format Match ({format_score:.1f}/{format_max:.1f})")
    st.markdown("### Score Breakdown")
    st.markdown(f"- Episodes: {comp_score.episodes:.1f}/4.0")
    st.markdown(f"- Order Type: {comp_score.order_type:.1f}/1.0")
    
    st.markdown("### Detailed Matches")
    col1, col2 = st.columns(2)
    
    with col1:
        # Content Match section
        st.markdown("### Content Details")
        
        # Genre and subgenres
        render_field_match("Genre", FieldMatch(
            name1=details.get('genre_name', 'Unknown'),
            name2=details.get('selected_genre_name', 'Unknown'),
            selected=True,
            match=comp_score.genre_base > 0,
            score=comp_score.genre_base + comp_score.genre_overlap,
            max_score=17  # 9 base + 8 overlap
        ))
        
        # Character types
        render_array_field_match("Character Types", ArrayFieldMatch(
            name1='Multiple' if details.get('character_type_names') else 'None',
            name2='Multiple' if details.get('selected_character_type_names') else 'None',
            selected=bool(details.get('selected_character_type_names')),
            match=comp_score.character_types > 0,
            score=comp_score.character_types,
            max_score=14,  # 10.5 first + 3.5 second
            values1=details.get('character_type_names', []),
            values2=details.get('selected_character_type_names', []),
            matches=details.get('matching_character_types', [])
        ))
        
        # Plot elements
        render_array_field_match("Plot Elements", ArrayFieldMatch(
            name1='Multiple' if details.get('plot_element_names') else 'None',
            name2='Multiple' if details.get('selected_plot_element_names') else 'None',
            selected=bool(details.get('selected_plot_element_names')),
            match=comp_score.plot_elements > 0,
            score=comp_score.plot_elements,
            max_score=12,  # 9 first + 3 second
            values1=details.get('plot_element_names', []),
            values2=details.get('selected_plot_element_names', []),
            matches=details.get('matching_plot_elements', [])
        ))
        
        # Production match section
        st.markdown("### Production Details")
        
        # Network
        render_field_match("Network", FieldMatch(
            name1=details.get('network_name', 'Unknown'),
            name2=details.get('selected_network_name', 'Unknown'),
            selected=True,
            match=comp_score.network > 0,
            score=comp_score.network,
            max_score=5
        ))
        
        # Studio
        render_array_field_match("Studio", ArrayFieldMatch(
            name1='Multiple' if details.get('studio_names') else 'None',
            name2='Multiple' if details.get('selected_studio_names') else 'None',
            selected=bool(details.get('selected_studio_names')),
            match=comp_score.studio > 0,
            score=comp_score.studio,
            max_score=4,  # 2 primary + 2 max additional
            values1=details.get('studio_names', []),
            values2=details.get('selected_studio_names', []),
            matches=details.get('matching_studios', [])
        ))
    
    with col2:
        # Source type
        render_field_match("Source", FieldMatch(
            name1=details.get('source_type_name', 'Unknown'),
            name2=details.get('selected_source_type_name', 'Unknown'),
            selected=True,
            match=comp_score.source_type > 0,
            score=comp_score.source_type,
            max_score=10
        ))
        
        # Theme elements
        render_array_field_match("Theme Elements", ArrayFieldMatch(
            name1='Multiple' if details.get('theme_element_names') else 'None',
            name2='Multiple' if details.get('selected_theme_element_names') else 'None',
            selected=bool(details.get('selected_theme_element_names')),
            match=comp_score.theme_elements > 0,
            score=comp_score.theme_elements,
            max_score=13,  # 9.75 first + 3.25 second
            values1=details.get('theme_element_names', []),
            values2=details.get('selected_theme_element_names', []),
            matches=details.get('matching_theme_elements', [])
        ))
        
        # Tone
        render_field_match("Tone", FieldMatch(
            name1=details.get('tone_name', 'Unknown'),
            name2=details.get('selected_tone_name', 'Unknown'),
            selected=True,
            match=comp_score.tone > 0,
            score=comp_score.tone,
            max_score=9
        ))
        
        # Team
        render_array_field_match("Team", ArrayFieldMatch(
            name1='Multiple' if details.get('team_member_names') else 'None',
            name2='Multiple' if details.get('selected_team_member_names') else 'None',
            selected=bool(details.get('selected_team_member_names')),
            match=comp_score.team > 0,
            score=comp_score.team,
            max_score=4,  # 3 first + 2 max additional
            values1=details.get('team_member_names', []),
            values2=details.get('selected_team_member_names', []),
            matches=details.get('matching_team_members', [])
        ))
        
        # Setting
        st.markdown("### Setting Details")
        render_field_match("Time", FieldMatch(
            name1=details.get('time_setting_name', 'Unknown'),
            name2=details.get('selected_time_setting_name', 'Unknown'),
            selected=True,
            match=comp_score.time_setting > 0,
            score=comp_score.time_setting,
            max_score=4
        ))
        render_field_match("Location", FieldMatch(
            name1=details.get('location_setting_name', 'Unknown'),
            name2=details.get('selected_location_setting_name', 'Unknown'),
            selected=True,
            match=comp_score.location > 0,
            score=comp_score.location,
            max_score=3
        ))
        
        # Format
        st.markdown("### Format Details")
        render_field_match("Episodes", FieldMatch(
            name1=str(details.get('episode_count', 'Unknown')),
            name2=str(details.get('selected_episode_count', 'Unknown')),
            selected=True,
            match=comp_score.episodes > 0,
            score=comp_score.episodes,
            max_score=4
        ))
        render_field_match("Order Type", FieldMatch(
            name1=details.get('order_type_name', 'Unknown'),
            name2=details.get('selected_order_type_name', 'Unknown'),
            selected=True,
            match=comp_score.order_type > 0,
            score=comp_score.order_type,
            max_score=1
        ))

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
                'genre_name': match.get('genre_name'),
                'selected_genre_name': criteria.get('genre_name'),
                'character_type_names': match.get('character_type_names', []),
                'selected_character_type_names': criteria.get('character_type_names', []),
                'matching_character_types': match.get('matching_character_types', []),
                'plot_element_names': match.get('plot_element_names', []),
                'selected_plot_element_names': criteria.get('plot_element_names', []),
                'matching_plot_elements': match.get('matching_plot_elements', []),
                'source_type_name': match.get('source_type_name'),
                'selected_source_type_name': criteria.get('source_type_name'),
                'theme_element_names': match.get('theme_element_names', []),
                'selected_theme_element_names': criteria.get('theme_element_names', []),
                'matching_theme_elements': match.get('matching_theme_elements', []),
                'tone_name': match.get('tone_name'),
                'selected_tone_name': criteria.get('tone_name'),
                'network_name': match.get('network_name'),
                'selected_network_name': criteria.get('network_name'),
                'studio_names': match.get('studio_names', []),
                'selected_studio_names': criteria.get('studio_names', []),
                'matching_studios': match.get('matching_studios', []),
                'team_member_names': match.get('team_member_names', []),
                'selected_team_member_names': criteria.get('team_member_names', []),
                'matching_team_members': match.get('matching_team_members', []),
                'time_setting_name': match.get('time_setting_name'),
                'selected_time_setting_name': criteria.get('time_setting_name'),
                'location_setting_name': match.get('location_setting_name'),
                'selected_location_setting_name': criteria.get('location_setting_name'),
                'episode_count': match.get('episode_count'),
                'selected_episode_count': criteria.get('episode_count'),
                'order_type_name': match.get('order_type_name'),
                'selected_order_type_name': criteria.get('order_type_name')
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
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**{label}:**")
    with col2:
        if show_score:
            st.markdown(f"**Score:** {match.score:.1f}/{match.max_score:.1f}")
        if match.selected:
            st.markdown(f"**Selected:** {match.name2}")
            st.markdown(f"**Match:** {'✓' if match.match else '✗'}")
            st.markdown(f"**Value:** {match.name1}")
        else:
            st.markdown("*Not selected*")

def render_array_field_match(label: str, match: ArrayFieldMatch) -> None:
    """Render a multi-value field match using base template methods."""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**{label}:**")
    with col2:
        st.markdown(f"**Score:** {match.score:.1f}/{match.max_score:.1f}")
        if match.selected:
            st.markdown("**Selected:**")
            for value in match.values2:
                st.markdown(f"* {value}")
            if match.matches:
                st.markdown("**Matches:**")
                for value in match.matches:
                    st.markdown(f"* {value}")
            st.markdown("**Show has:**")
            for value in match.values1:
                st.markdown(f"* {value}")
        else:
            st.markdown("*Not selected*")

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
