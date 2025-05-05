"""View component for rendering match details.

This module uses the base template methods from base_match_breakdown.py
to render match details in a compact format for the comp builder.
"""

from typing import Dict, List
from . import get_render_match_details_section

def render_match_details(matches: List[Dict], details_manager, criteria: Dict) -> None:
    """Render match details for top matches."""
    render_matches_section = get_render_match_details_section()
    render_matches_section(matches, details_manager, criteria)
