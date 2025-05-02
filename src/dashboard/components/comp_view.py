"""View component for comp builder functionality."""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.components.base_match_breakdown import render_base_match_breakdown


def get_field_name(id: int, options: List[Tuple[int, str]], default='Unknown') -> str:
    """Get name for an ID from options, with default fallback."""
    return next((name for opt_id, name in options if opt_id == id), default)

def get_field_names(ids: List[int], options: List[Tuple[int, str]], default='Unknown') -> List[str]:
    """Get names for a list of IDs from options."""
    names = {id: name for id, name in options}
    return [names.get(id, default) for id in ids]

def get_id_for_name(name: str, options: List[Tuple[int, str]]) -> Optional[int]:
    """Get ID for a display name from options."""
    return next((id for id, display_name in options if display_name == name), None)

def get_ids_for_names(names: List[str], options: List[Tuple[int, str]]) -> List[int]:
    """Get IDs for a list of display names from options."""
    return [id for name in names if (id := get_id_for_name(name, options)) is not None]


def update_single_id_field(details: Dict, field: str, show_id: Optional[int], selected_id: Optional[int], 
                        options: List[Tuple[int, str]], score_key: Optional[str] = None) -> None:
    """Update details for single-ID fields (genre, network, tone)."""
    if field not in details:
        details[field] = {}
    show_name = get_field_name(show_id, options)
    selected_name = get_field_name(selected_id, options)
    details[field].update({
        'name1': selected_name,
        'name2': show_name, 
        'match': show_id == selected_id and show_id is not None,
        'selected': selected_id is not None,
        **(({'score': scores[score_key]} if score_key else {}))
    })

def update_array_field(details: Dict, field: str, show_ids: List[int], selected_ids: List[int], 
                      options: List[Tuple[int, str]], raw_names: Optional[List[str]] = None) -> None:
    """Update details for array-type fields with fallback to raw names."""
    if field not in details:
        details[field] = {}
    names = {id: name for id, name in options}
    matches = [names.get(id, 'Unknown') for id in selected_ids if id in show_ids]
    mismatches = []
    for idx, id in enumerate(show_ids):
        if id not in selected_ids:
            name = names.get(id) or (raw_names[idx] if raw_names and idx < len(raw_names) else 'Unknown')
            mismatches.append(name)
    details[field].update({
        'matches': matches,
        'mismatches': mismatches,
        'selected': bool(selected_ids)
    })

def update_setting_field(details: Dict, show_time_id: Optional[int], show_loc_id: Optional[int], 
                        selected_time_id: Optional[int], selected_loc_id: Optional[int],
                        time_options: List[Tuple[int, str]], loc_options: List[Tuple[int, str]]) -> None:
    """Update details for setting which combines time and location."""
    if 'setting' not in details:
        details['setting'] = {}
    show_time = get_field_name(show_time_id, time_options)
    show_location = get_field_name(show_loc_id, loc_options)
    details['setting'].update({
        'time': show_time,
        'location': show_location,
        'selected': (selected_time_id is not None or selected_loc_id is not None)
    })

def update_production_field(details: Dict, field: str, show_ids: List[int], selected_ids: List[int], 
                          options: List[Tuple[int, str]], scores: Dict[str, float], score_key: str) -> None:
    """Update details for production fields (studios, team) with special scoring."""
    if field not in details:
        details[field] = {}
    show_names = get_field_names(show_ids, options)
    selected_names = get_field_names(selected_ids, options)
    details[field].update({
        'name1': ', '.join(selected_names) or 'Unknown',
        'name2': ', '.join(show_names) or 'Unknown',
        'match': bool(set(show_ids) & set(selected_ids)),
        'selected': bool(selected_ids),
        'score': scores[score_key]
    })

def update_match_details(details: Dict, match: Dict, criteria: Dict, display_options: Dict, scores: Dict) -> None:
    """Update all match details using appropriate field handlers."""
    # Single ID fields
    update_single_id_field(details, 'genre', match.get('genre_id'), criteria.get('genre_id'), 
                          display_options['genre'], 'genre_score')
    update_single_id_field(details, 'source', match.get('source_type_id'), criteria.get('source_type_id'),
                          display_options['source_type'], 'source_score')
    update_single_id_field(details, 'tone', match.get('tone_id'), criteria.get('tone_id'), 
                          display_options['tone'])
                          
    # Array fields
    update_array_field(details, 'characters', match.get('character_type_ids', []), 
                      criteria.get('character_type_ids', []), display_options['character_types'],
                      match.get('character_type_names'))
    update_array_field(details, 'plot', match.get('plot_element_ids', []),
                      criteria.get('plot_element_ids', []), display_options['plot_elements'],
                      match.get('plot_element_names'))
    update_array_field(details, 'themes', match.get('theme_element_ids', []),
                      criteria.get('theme_element_ids', []), display_options['thematic_elements'],
                      match.get('theme_element_names'))
                      
    # Setting
    update_setting_field(details, match.get('time_setting_id'), match.get('location_setting_id'),
                        criteria.get('time_setting_id'), criteria.get('location_setting_id'),
                        display_options['time_setting'], display_options['location_setting'])
                        
    # Studios/Team
    update_production_field(details, 'studio', match.get('studios', []), criteria.get('studios', []),
                          display_options['studios'], scores, 'studio_score')
    update_production_field(details, 'team', match.get('team_member_ids', []), criteria.get('team_member_ids', []),
                          display_options['team_members'], scores, 'team_score')

def render_comp_builder(state: Dict) -> None:
    """Render the comp builder interface."""
    comp_analyzer = CompAnalyzer()
    st.title("Comp Builder")
    criteria_col, results_col = st.columns([1, 2])
    with criteria_col:
        render_criteria_section(comp_analyzer, state)
    with results_col:
        render_results_section(comp_analyzer, state)


def render_criteria_section(comp_analyzer: CompAnalyzer, state: Dict) -> None:
    """Render criteria selection panel."""
    # Get field options
    field_options = comp_analyzer.get_field_options()
    display_options = {field: comp_analyzer.get_field_display_options(field) 
                      for field in field_options.keys()}
    with st.expander("Content Match Criteria (82 pts)", expanded=True):
        # Content criteria
        st.markdown("### Content")
        
        # Genre
        genre_name = st.selectbox("Genre", 
            options=[name for _, name in display_options['genre'] if name and name.strip()],
            key="genre_id", index=None, placeholder="Select genre...")
        state["criteria"]["genre_id"] = get_id_for_name(genre_name, display_options['genre']) if genre_name else None
        
        # Subgenres
        subgenre_names = st.multiselect("Subgenres",
            options=[name for _, name in display_options['subgenres'] if name and name.strip()],
            key="subgenres", placeholder="Select subgenres...")
        state["criteria"]["subgenres"] = get_ids_for_names(subgenre_names, display_options['subgenres'])
        
        # Source Type
        source_name = st.selectbox("Source Type",
            options=[name for _, name in display_options['source_type'] if name and name.strip()],
            key="source_type_id", index=None, placeholder="Select source type...")
        state["criteria"]["source_type_id"] = get_id_for_name(source_name, display_options['source_type']) if source_name else None
        
        # Array-type fields
        # Character Types (special case - uses character_type_ids in database)
        names = st.multiselect('Character Types',
            options=[name for _, name in display_options['character_types'] if name and name.strip()],
            key="character_type_ids", placeholder="Select character types...")
        state["criteria"]["character_type_ids"] = get_ids_for_names(names, display_options['character_types'])
        
        # Other array fields
        for field, label in [('plot_elements', 'Plot Elements'),
                           ('thematic_elements', 'Theme Elements')]:
            names = st.multiselect(label,
                options=[name for _, name in display_options[field] if name and name.strip()],
                key=f"{field}_ids", placeholder=f"Select {label.lower()}...")
            state["criteria"][f"{field}_ids"] = get_ids_for_names(names, display_options[field])
        
        # Single-select fields
        for field, label in [('tone', 'Tone'),
                           ('time_setting', 'Time Setting'),
                           ('location_setting', 'Location')]:
            name = st.selectbox(label,
                options=[name for _, name in display_options[field] if name and name.strip()],
                key=field, index=None, placeholder=f"Select {label.lower()}...")
            state["criteria"][f"{field}_id"] = get_id_for_name(name, display_options[field]) if name else None
        
    # Production criteria
    with st.expander("Production Match Criteria (13 pts)", expanded=True):
        st.markdown("### Production")
        
        # Network
        network_name = st.selectbox("Network",
            options=[name for _, name in display_options['network'] if name and name.strip()],
            key="network_id", index=None, placeholder="Select network...")
        state["criteria"]["network_id"] = get_id_for_name(network_name, display_options['network']) if network_name else None
        
        # Studios
        studio_names = st.multiselect("Studios",
            options=[name for _, name in display_options['studios'] if name and name.strip()],
            key="studio_ids", placeholder="Select studios...")
        state["criteria"]["studios"] = get_ids_for_names(studio_names, display_options['studios'])
    
    # Format criteria
    with st.expander("Format Match Criteria (5 pts)", expanded=True):
        st.markdown("### Format")
        
        # Episode Count
        eps = st.number_input("Episode Count", min_value=1, max_value=100, value=None,
            key="episode_count", help="Episode count proximity (2 within ±2, 1.5 within ±4, 1 within ±6)")
        state["criteria"]["episode_count"] = eps if eps is not None and eps > 0 else None
        
        # Order Type
        order_name = st.selectbox("Order Type",
            options=[name for _, name in display_options['order_type']],
            key="order_type_id", index=None, placeholder="Select order type...")
        state["criteria"]["order_type_id"] = get_id_for_name(order_name, display_options['order_type']) if order_name else None


def create_results_df(results: List[Dict]) -> pd.DataFrame:
    """Create results DataFrame with consistent formatting."""
    return pd.DataFrame([
        {
            'Show': r.get('title', 'Unknown'),
            'Success': int(r.get('success_score', 0)),
            'Total Score': int(r['comp_score'].total),
            'Content': int(r['comp_score'].content_score),
            'Production': int(r['comp_score'].production_score),
            'Format': int(r['comp_score'].format_score)
        } for r in results
    ])

def apply_table_styling() -> None:
    """Apply consistent styling to results table."""
    st.markdown(f"""
        <style>
        .stDataFrame table {{ color: {COLORS['text']['primary']}; }}
        .stDataFrame th {{ 
            background-color: {COLORS['accent']}; 
            color: white;
            font-family: {FONTS['primary']['family']};
            font-size: {FONTS['primary']['sizes']['body']}px;
        }}
        .stDataFrame td {{ 
            font-family: {FONTS['primary']['family']};
            font-size: {FONTS['primary']['sizes']['small']}px;
        }}
        [data-testid="stMetricValue"] {{
            overflow: visible !important;
            width: auto !important;
            margin-right: 20px !important;
        }}
        </style>
    """, unsafe_allow_html=True)

def render_results_section(comp_analyzer: CompAnalyzer, state: Dict) -> None:
    """Render comp results panel."""
    if not state.get("criteria"):
        st.info("Select criteria on the left to find similar shows.")
        return
    # Get display options for lookups
    display_options = {field: comp_analyzer.get_field_display_options(field)
                      for field in comp_analyzer.get_field_options().keys()}
    
    # Get results
    results = comp_analyzer.find_by_criteria(state["criteria"])
    if not results:
        st.info("No shows found matching your criteria. Try adjusting the filters.")
        return
        
    # Filter valid results
    results = [r for r in results if r.get('comp_score') is not None]
    if not results:
        st.info("No valid matches found. Try adjusting your criteria.")
        return
        
    # Display results table
    st.markdown("### Similar Shows")
    df = create_results_df(results)
    apply_table_styling()
    st.dataframe(df, use_container_width=True)
    
    # Show match details
    if results:
        st.markdown("### Match Details")
        for i, match in enumerate(results[:10], 1):
            with st.expander(f"#{i}: {match['title']}", expanded=(i==1)):
                try:
                    # Get comp score data
                    comp_score = match.get('comp_score')
                    if not comp_score:
                        continue
                        
                    # Get scores and details
                    scores = comp_score.to_display_dict()
                    details = comp_score.get_match_details()
                    
                    # Update details with current match data
                    update_match_details(details, match, state.get('criteria', {}), 
                                       display_options, scores)
                    
                    # Render match breakdown
                    render_base_match_breakdown(
                        title=match['title'],
                        scores=scores,
                        details=details,
                        matches=match,
                        success_score=match.get('success_score'),
                        expanded=i==1,
                        use_expander=False,
                        description=match.get('description')
                    )
                except Exception as e:
                    st.error(f"Error processing match {i}: {str(e)}")
    # Get results
    results = comp_analyzer.find_by_criteria(state["criteria"])
    if not results:
        st.info("No shows found matching your criteria. Try adjusting the filters.")
        return
        
    # Filter valid results
    results = [r for r in results if r.get('comp_score') is not None]
    if not results:
        st.info("No valid matches found. Try adjusting your criteria.")
        return
        
    # Display results table
    st.markdown("### Similar Shows")
    df = create_results_df(results)
    apply_table_styling()
    st.dataframe(df, use_container_width=True)
    
    # Show match details
    if results:
        st.markdown("### Match Details")
        for i, match in enumerate(results[:10], 1):
            with st.expander(f"#{i}: {match['title']}", expanded=(i==1)):
                try:
                    # Get comp score data
                    comp_score = match.get('comp_score')
                    if not comp_score:
                        continue
                        
                    # Get scores and details
                    scores = comp_score.to_display_dict()
                    details = comp_score.get_match_details()
                    
                    # Update details with current match data
                    update_match_details(details, match, state.get('criteria', {}), 
                                       display_options, scores)
                    
                    # Render match breakdown
                    render_base_match_breakdown(
                        title=match['title'],
                        scores=scores,
                        details=details,
                        matches=match,
                        success_score=match.get('success_score'),
                        expanded=i==1,
                        use_expander=False,
                        description=match.get('description')
                    )
                except Exception as e:
                    st.error(f"Error processing match {i}: {str(e)}")
