"""View component for rendering match details.

This module uses the base template methods from base_match_breakdown.py
to render match details in a compact format for the comp builder.
"""

import streamlit as st
from typing import Dict, List
from .base_match_breakdown import (
    ScoreDisplay,
    render_field_base,
    render_match_indicator,
    render_array_field_base,
    render_content_section,
    render_production_section,
    render_setting_format_section
)

def render_field_match(label: str, match: Dict, show_score: bool = True) -> None:
    """Render a single field match using base template methods."""
    if show_score:
        score = ScoreDisplay(match['score'], match['max_score'])
    else:
        score = None
    
    render_field_base(label, score)
    render_match_indicator(
        value=match['name1'],
        matched=match['match'],
        selected=match['selected']
    )

def render_array_field_match(label: str, match: Dict, show_score: bool = True) -> None:
    """Render a multi-value field match using base template methods."""
    if show_score:
        score = ScoreDisplay(match['score'], match['max_score'])
    else:
        score = None
        
    render_field_base(label, score)
    
    # Special handling for team members to avoid duplicate display
    if label == 'Team':
        # Show selected team members first
        for value in match['values2']:
            render_match_indicator(
                value=value,
                matched=value in match['matches'],
                selected=True
            )
        
        # Show remaining team members
        remaining = [v for v in match['values1'] if v not in match['values2']]
        for value in remaining:
            render_match_indicator(
                value=value,
                matched=False,
                selected=False
            )
        return
    
    # Standard handling for other array fields
    render_array_field_base(
        values=match['values1'],
        matches=match['matches'],
        selected=match['selected']
    )

def render_match_details_section(details: Dict) -> None:
    """Render match details section with columns."""
    # Content Match section
    col1, col2 = render_content_section(details)
    
    with col1:
        render_field_match("Genre", details['genre'])
        render_array_field_match("Subgenres", details['subgenres'])
        render_array_field_match("Character Types", details['characters'])
        render_array_field_match("Plot Elements", details['plot'])
    
    with col2:
        render_field_match("Source", details['source'])
        render_array_field_match("Theme Elements", details['themes'])
        render_field_match("Tone", details['tone'])
    
    # Production match section
    col1, col2 = render_production_section(details)
    
    with col1:
        render_field_match("Network", details['network'])
        render_array_field_match("Studio", details['studio'])
    
    with col2:
        render_array_field_match("Team", details['team'])
    
    # Setting and format sections
    col1, col2 = render_setting_format_section(details)
    
    with col1:
        render_field_match("Time", details['setting']['time'], show_score=False)
        render_field_match("Location", details['setting']['location'], show_score=False)
    
    with col2:
        render_field_match("Episodes", details['format']['episodes'], show_score=False)
        render_field_match("Order Type", details['format']['order_type'], show_score=False)

def render_match_details(matches: List[Dict], details_manager, criteria: Dict) -> None:
    """Render match details for top matches."""
    st.markdown("### Top Matches")
    
    # Show top 10 matches in expanders
    for match in matches[:10]:
        with st.expander(f"#{match['id']}: {match['title']} (Match: {match['comp_score'].total():.1f})", expanded=match == matches[0]):
            details = details_manager.create_match_details(match, criteria)
            render_match_details_section(details)
