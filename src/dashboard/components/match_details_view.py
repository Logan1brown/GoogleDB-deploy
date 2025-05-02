"""View component for rendering match details."""

import streamlit as st
from typing import Dict
from .match_details import FieldMatch, ArrayFieldMatch

def render_field_match(label: str, match: FieldMatch, show_score: bool = True) -> None:
    """Render a single field match."""
    st.markdown(f"**{label}** {f'({match.score}/{match.max_score})' if show_score else ''}")
    
    if not match.selected:
        st.markdown(f"⚫ {match.name1} (not selected)")
        return
        
    if match.match:
        st.markdown(f"🟢 {match.name1}")
    else:
        st.markdown(f"🔴 {match.name1} ≠ {match.name2}")

def render_array_field_match(label: str, match: ArrayFieldMatch, show_score: bool = True) -> None:
    """Render a multi-value field match."""
    st.markdown(f"**{label}** {f'({match.score}/{match.max_score})' if show_score else ''}")
    
    if not match.selected:
        if not match.values1:
            st.markdown(f"⚫ No {label.lower()}")
        else:
            values = ', '.join(match.values1)
            st.markdown(f"⚫ {values} (not selected)")
        return
        
    if match.match:
        st.markdown("🟢 Matching values:")
        for value in match.matches:
            st.markdown(f"  • {value}")
            
        non_matches = set(match.values1) - set(match.matches)
        if non_matches:
            st.markdown("Additional values:")
            for value in non_matches:
                st.markdown(f"  • {value}")
    else:
        if not match.values1:
            st.markdown(f"🔴 No {label.lower()}")
        else:
            values = ', '.join(match.values1)
            st.markdown(f"🔴 No matches: {values}")

def render_setting_match(setting: Dict[str, FieldMatch]) -> None:
    """Render time and location setting matches."""
    st.markdown(f"**Setting Match ({setting['total_score']}/{setting['max_score']} points)**")
    
    with st.expander("Time & Location", expanded=True):
        render_field_match("Time", setting['time'], show_score=False)
        render_field_match("Location", setting['location'], show_score=False)

def render_format_match(format_details: Dict[str, FieldMatch]) -> None:
    """Render episode count and order type matches."""
    st.markdown(f"**Format Match ({format_details['total_score']}/{format_details['max_score']} points)**")
    
    with st.expander("Episodes & Order", expanded=True):
        render_field_match("Episodes", format_details['episodes'], show_score=False)
        render_field_match("Order Type", format_details['order_type'], show_score=False)

def render_match_details(show_id: int, details: Dict) -> None:
    """Render full match details for a show."""
    st.markdown(f"### #{show_id}")
    
    # Content match section
    st.markdown("#### Content Match")
    render_field_match("Genre", details['genre'])
    render_array_field_match("Character Types", details['characters'])
    render_array_field_match("Plot Elements", details['plot'])
    render_field_match("Source", details['source'])
    render_array_field_match("Theme Elements", details['themes'])
    render_field_match("Tone", details['tone'])
    
    # Production match section
    st.markdown("#### Production Match")
    render_array_field_match("Studio", details['studio'])
    render_array_field_match("Team", details['team'])
    
    # Setting and format sections
    render_setting_match(details['setting'])
    render_format_match(details['format'])
