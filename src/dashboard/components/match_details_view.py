"""View component for rendering match details."""

import streamlit as st
from typing import Dict, List
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
    
    # Special handling for team members to avoid duplicate display
    if label == 'Team':
        # Show selected team members first
        for value in match.values2:
            bullet = '🟢' if value in match.matches else '⚫'
            st.write(f'{bullet} {value}')
        
        # Show remaining team members
        remaining = [v for v in match.values1 if v not in match.values2]
        for value in remaining:
            st.write(f'⚫ {value}')
        return
        
    # Standard handling for other array fields
    if not match.selected:
        if not match.values1:
            st.markdown(f"⚫ No {label.lower()}")
        else:
            values = ', '.join(match.values1)
            st.markdown(f"⚫ {values} (not selected)")
        return
    
    # Show selected values first with green/black bullets
    for value in match.values2:
        bullet = '🟢' if value in match.matches else '⚫'
        st.write(f'{bullet} {value}')
    
    # Show remaining values from source show that weren't selected
    remaining = [v for v in match.values1 if v not in match.values2]
    if remaining:
        st.write('⚫ ' + ', '.join(remaining))

def render_match_details_section(details: Dict) -> None:
    """Render match details section with columns."""
    # Content match section
    st.markdown("#### Content Match")
    render_field_match("Genre", details['genre'])
    render_array_field_match("Subgenres", details['subgenres'])
    render_array_field_match("Character Types", details['characters'])
    render_array_field_match("Plot Elements", details['plot'])
    render_field_match("Source", details['source'])
    render_array_field_match("Theme Elements", details['themes'])
    render_field_match("Tone", details['tone'])
    
    # Production match section
    st.markdown("#### Production Match")
    render_field_match("Network", details['network'])
    render_array_field_match("Studio", details['studio'])
    render_array_field_match("Team", details['team'])
    
    # Setting and format sections in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### Setting Match ({details['setting']['total_score']}/{details['setting']['max_score']} points)")
        render_field_match("Time", details['setting']['time'], show_score=False)
        render_field_match("Location", details['setting']['location'], show_score=False)
    
    with col2:
        st.markdown(f"#### Format Match ({details['format']['total_score']}/{details['format']['max_score']} points)")
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
