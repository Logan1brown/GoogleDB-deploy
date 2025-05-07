"""Base template for match breakdown components.

This module provides the core template methods for rendering match breakdowns,
which can be extended by specific views like comp builder and show details.

IMPORTANT: This is a template that defines the structure and common patterns.
Specific views should extend these methods with their own display logic.
"""

import streamlit as st
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.dashboard.utils.style_config import FONTS
from src.dashboard.components.match_details import FieldMatch, ArrayFieldMatch

@dataclass
class ScoreDisplay:
    """Helper class for displaying scores."""
    score: float
    max_score: float
    show_score: bool = True
    
    def format(self) -> str:
        """Format score for display.
        
        Returns:
            String in format (X/Y pts) or empty string if show_score is False
        """
        if not self.show_score:
            return ""
        return f" ({int(self.score)}/{int(self.max_score)} pts)"

def render_section_header(header: str, score: Optional[ScoreDisplay] = None) -> None:
    """Template method for rendering a section header with optional score."""
    header_text = f"<p style='font-family: {FONTS['primary']['family']}; font-size: {FONTS['primary']['sizes']['body']}px; font-weight: 600; margin-bottom: 0.5em;'>{header}"
    if score:
        header_text += score.format()
    header_text += "</p>"
    st.markdown(header_text, unsafe_allow_html=True)

def render_match_details_section(details: Dict, success_score: Optional[float] = None, description: Optional[str] = None) -> None:
    """Template method for rendering match details section with columns.
    
    This is a base template that defines the structure. Each section (content, production, format)
    should follow this structure:
    1. Section header with total score
    2. Two-column layout
    3. Fields rendered using either render_field_match or render_array_field_match
    
    Args:
        details: Dictionary containing section scores and field displays
        success_score: Optional overall success score
        description: Optional description text
    """
    if not details:
        return
        
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
    content = details.get('content', {})
    if content:
        render_section_header("Content Match", ScoreDisplay(
            score=content.get('score', 0),
            max_score=content.get('max_score', 0),
            show_score=True
        ))
        
        st.write("")
        col1, col2 = st.columns(2)
        
        components = content.get('components', {})
        with col1:
            # Single-value fields
            for field, label in [
                ('genre', 'Genre'),
                ('source_type', 'Source Type'),
                ('time_setting', 'Time Setting')
            ]:
                if display := components.get(field, {}).get('display'):
                    render_field_match(label, display)
            
            # Multi-value fields
            if display := components.get('subgenres', {}).get('display'):
                render_array_field_match("Subgenres", display)
                
            if display := components.get('character_types', {}).get('display'):
                render_array_field_match("Character Types", display)
        
        with col2:
            # Multi-value fields
            for field, label in [
                ('plot_elements', 'Plot Elements'),
                ('thematic_elements', 'Theme Elements')
            ]:
                if display := components.get(field, {}).get('display'):
                    render_array_field_match(label, display)
            
            # Single-value fields
            # Note: location_setting in UI maps to location in comp_score
            for field, label in [
                ('tone', 'Tone'),
                ('location_setting', 'Location')  # This matches our field name in match_details.py
            ]:
                if display := components.get(field, {}).get('display'):
                    render_field_match(label, display)
        
        st.write("")
    
    # Production match section
    production = details.get('production', {})
    if production:
        render_section_header("Production Match", ScoreDisplay(
            score=production.get('score', 0),
            max_score=production.get('max_score', 0),
            show_score=True
        ))
        
        st.write("")
        col1, col2 = st.columns(2)
        
        components = production.get('components', {})
        with col1:
            # Single-value field
            if display := components.get('network', {}).get('display'):
                render_field_match("Network", display)
            
            # Multi-value field
            if display := components.get('studio', {}).get('display'):
                render_array_field_match("Studios", display)
        
        with col2:
            # Multi-value field
            if display := components.get('team', {}).get('display'):
                render_array_field_match("Team Members", display)
        
        st.write("")
    
    # Format match section
    format_section = details.get('format', {})
    if format_section:
        render_section_header("Format Match", ScoreDisplay(
            score=format_section.get('score', 0),
            max_score=format_section.get('max_score', 0),
            show_score=True
        ))
        
        st.write("")
        col1, col2 = st.columns(2)
        
        components = format_section.get('components', {})
        with col1:
            if display := components.get('episodes', {}).get('display'):
                render_field_match("Episodes", display)
        
        with col2:
            if display := components.get('order_type', {}).get('display'):
                render_field_match("Order Type", display)

def render_matches_section(matches: List[Dict], details_manager, criteria: Dict) -> None:
    """Template method for rendering the matches section.
    
    This is a base template that defines the structure. Specific views should:
    1. Get match details from their details manager
    2. Pass those details to render_match_details_section
    
    Args:
        matches: List of match data
        details_manager: Manager for getting match details
        criteria: Match criteria
    """
    if not matches:
        return
        
    st.markdown(f"<p style='font-family: {FONTS['primary']['family']}; font-size: {FONTS['primary']['sizes']['title']}px; font-weight: 600; margin-bottom: 1em;'>Top Matches</p>", unsafe_allow_html=True)
    
    # Show top 10 matches in expanders
    for i, match in enumerate(matches[:10]):
        # Get match details from the details manager
        details = details_manager.create_match_details(match, criteria)
        
        # Skip if no details available
        if not details:
            continue
            
        # Get total scores from details
        total = details.get('total', {})
        total_score = total.get('score', 0)
        max_score = total.get('max_score', 0)
        
        with st.expander(
            f"#### #{match['id']}: {match['title']} (Match: {total_score:.1f}/{max_score:.1f})",
            expanded=i == 0  # Expand first match
        ):
            # Use the template method to render details
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
    
    # Show all values with appropriate bullet points
    for value in match.values1:
        render_match_indicator(
            value,
            matched=value in match.matches,
            selected=match.selected
        )

