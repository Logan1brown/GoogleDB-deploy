"""Dashboard components module."""

# Imports moved inside functions to prevent early initialization
def get_match_details_manager():
    from .match_details import MatchDetailsManager
    return MatchDetailsManager

def get_render_match_details():
    from .match_details_view import render_match_details
    return render_match_details

def get_render_match_details_section():
    from .base_match_breakdown import render_match_details_section
    return render_match_details_section
