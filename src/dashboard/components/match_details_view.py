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
    render_matches_section
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

def render_match_details(matches: List[Dict], details_manager, criteria: Dict) -> None:
    """Render match details for top matches."""
    render_matches_section(matches, details_manager, criteria)
