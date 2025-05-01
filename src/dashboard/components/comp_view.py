"""View component for comp builder functionality."""

import streamlit as st
from typing import Dict, List, Tuple, Optional
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)

from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.components.base_match_breakdown import render_base_match_breakdown


def get_id_for_name(name: str, options: List[Tuple[int, str]]) -> Optional[int]:
    """Get the ID for a given display name from field options.
    
    Args:
        name: Display name to look up
        options: List of (id, name) tuples from field_options
        
    Returns:
        ID if found, None if not found
    """
    for id, display_name in options:
        if display_name == name:
            return id
    return None


def get_ids_for_names(names: List[str], options: List[Tuple[int, str]]) -> List[int]:
    """Get IDs for a list of display names from field options.
    
    Args:
        names: List of display names to look up
        options: List of (id, name) tuples from field_options
        
    Returns:
        List of found IDs (skips any names not found)
    """
    ids = []
    for name in names:
        id = get_id_for_name(name, options)
        if id is not None:
            ids.append(id)
    return ids


def render_comp_builder(state: Dict) -> None:
    """Render the comp builder interface.
    
    Args:
        state: Page state dictionary containing criteria selections
    """
    # Initialize analyzer
    comp_analyzer = CompAnalyzer()
    
    # Layout
    st.title("Comp Builder")
    criteria_col, results_col = st.columns([1, 2])
    
    with criteria_col:
        render_criteria_section(comp_analyzer, state)
        
    with results_col:
        render_results_section(comp_analyzer, state)


def render_criteria_section(comp_analyzer: CompAnalyzer, state: Dict) -> None:
    """Render criteria selection panel.
    
    Args:
        comp_analyzer: CompAnalyzer instance for getting field options
        state: Page state dictionary to store selections
    """
    with st.expander("Content Match Criteria (70 pts)", expanded=True):
        # Get field options
        field_options = comp_analyzer.get_field_options()
        
        # Content criteria
        st.markdown("### Content")
        genre_name = st.selectbox(
            "Genre",
            options=[name for _, name in field_options['genres']],
            format_func=lambda x: x,
            key="genre_id",
            index=None,
            placeholder="Select genre..."
        )
        state["criteria"]["genre_id"] = get_id_for_name(genre_name, field_options['genres']) if genre_name else None
        
        # Deduplicate and sort subgenre options
        unique_subgenres = sorted(set(name for _, name in field_options['subgenre_names']))
        
        subgenre_names = st.multiselect(
            "Subgenres",
            options=unique_subgenres,
            format_func=lambda x: x,
            key="subgenres",
            placeholder="Select subgenres..."
        )
        
        # Convert to IDs using a dictionary to ensure 1:1 mapping
        subgenre_map = {name: id for id, name in field_options['subgenre_names']}
        subgenre_ids = [subgenre_map[name] for name in subgenre_names if name in subgenre_map]
        state["criteria"]["subgenres"] = subgenre_ids
        
        source_name = st.selectbox(
            "Source Type", 
            options=[name for _, name in field_options['source_types']],
            format_func=lambda x: x,
            key="source_type_id",
            index=None,
            placeholder="Select source type..."
        )
        state["criteria"]["source_type_id"] = get_id_for_name(source_name, field_options['source_types']) if source_name else None
        
        # Deduplicate and sort character type options
        unique_char_types = sorted(set(name for _, name in field_options['character_types']))
        
        char_names = st.multiselect(
            "Character Types",
            options=unique_char_types,
            format_func=lambda x: x,
            key="character_type_ids",
            placeholder="Select character types..."
        )
        char_type_map = {name: id for id, name in field_options['character_types']}
        state["criteria"]["character_type_ids"] = [char_type_map[name] for name in char_names if name in char_type_map]
        
        # Deduplicate and sort plot element options
        unique_plot_elements = sorted(set(name for _, name in field_options['plot_elements']))
        
        plot_names = st.multiselect(
            "Plot Elements",
            options=unique_plot_elements,
            format_func=lambda x: x,
            key="plot_element_ids",
            placeholder="Select plot elements..."
        )
        
        # Convert to IDs using a dictionary to ensure 1:1 mapping
        plot_element_map = {name: id for id, name in field_options['plot_elements']}
        plot_element_ids = [plot_element_map[name] for name in plot_names if name in plot_element_map]
        plot_element_names = [name for name in plot_names if name in plot_element_map]
        
        # Debug: Show final mapping
        if plot_names:
            st.write("Plot elements selected:")
            for name in plot_names:
                if name in plot_element_map:
                    st.write(f"{name} (ID: {plot_element_map[name]})")
            
        state["criteria"]["plot_element_ids"] = plot_element_ids
        state["criteria"]["plot_element_names"] = plot_element_names
        
        # Deduplicate and sort theme element options
        unique_theme_elements = sorted(set(name for _, name in field_options['thematic_elements']))
        
        theme_names = st.multiselect(
            "Theme Elements",
            options=unique_theme_elements,
            format_func=lambda x: x,
            key="theme_element_ids",
            placeholder="Select theme elements..."
        )
        theme_element_map = {name: id for id, name in field_options['thematic_elements']}
        state["criteria"]["theme_element_ids"] = [theme_element_map[name] for name in theme_names if name in theme_element_map]
        
        tone_name = st.selectbox(
            "Tone",
            options=[name for _, name in field_options['tones']],
            format_func=lambda x: x,
            key="tone",
            index=None,
            placeholder="Select tone..."
        )
        state["criteria"]["tone_id"] = get_id_for_name(tone_name, field_options['tones']) if tone_name else None
        
        time_name = st.selectbox(
            "Time Setting",
            options=[name for _, name in field_options['time_settings']],
            format_func=lambda x: x,
            key="time_setting",
            index=None,
            placeholder="Select time setting..."
        )
        state["criteria"]["time_setting_id"] = get_id_for_name(time_name, field_options['time_settings']) if time_name else None
        
        loc_name = st.selectbox(
            "Location",
            options=[name for _, name in field_options['locations']],
            format_func=lambda x: x,
            key="location",
            index=None,
            placeholder="Select location..."
        )
        state["criteria"]["location_setting_id"] = get_id_for_name(loc_name, field_options['locations']) if loc_name else None
        
        # Production criteria
        st.markdown("### Production")
        
        # Get current network ID from state
        current_network_id = state.get("criteria", {}).get("network_id")
        
        # Find index of current network if it exists
        network_options = [name for _, name in field_options['networks']]
        current_network_name = None
        if current_network_id is not None:
            for id, name in field_options['networks']:
                if id == current_network_id:
                    current_network_name = name
                    break
        current_index = network_options.index(current_network_name) if current_network_name else None
        
        network_name = st.selectbox(
            "Network",
            options=network_options,
            format_func=lambda x: x,
            key="network_select",  # Change key to not conflict with state
            index=current_index,
            placeholder="Select network..."
        )
        
        # Update state with new network ID
        if network_name:
            network_id = get_id_for_name(network_name, field_options['networks'])
            state["criteria"]["network_id"] = network_id
            # Also update session state to ensure persistence
            if "network_id" in st.session_state:
                st.session_state.network_id = network_id
        
        # Deduplicate and sort studio options
        unique_studios = sorted(set(name for _, name in field_options['studios']))
        
        studio_names = st.multiselect(
            "Studios",
            options=unique_studios,
            format_func=lambda x: x,
            key="studio_ids",
            placeholder="Select studios..."
        )
        studio_map = {name: id for id, name in field_options['studios']}
        state["criteria"]["studios"] = [studio_map[name] for name in studio_names if name in studio_map]
        
        # Format criteria
        st.markdown("### Format")
        
        eps = st.number_input(
            "Episode Count",
            min_value=1,
            max_value=100,
            value=None,
            key="episode_count",
            help="Episode count proximity (2 within ±2, 1.5 within ±4, 1 within ±6)"
        )
        # Handle None value from number_input
        state["criteria"]["episode_count"] = eps if eps is not None and eps > 0 else None
        
        order_name = st.selectbox(
            "Order Type",
            options=[name for _, name in field_options['order_types']],
            format_func=lambda x: x,
            key="order_type_id",
            index=None,
            placeholder="Select order type..."
        )
        state["criteria"]["order_type_id"] = get_id_for_name(order_name, field_options['order_types']) if order_name else None


def render_results_section(comp_analyzer: CompAnalyzer, state: Dict) -> None:
    """Render comp results panel.
    
    Args:
        comp_analyzer: CompAnalyzer instance for getting similar shows
        state: Page state dictionary containing criteria
    """
    # Get field options for lookups
    field_options = comp_analyzer.get_field_options()
    if state.get("criteria"):
        # Get results based on criteria
        results = comp_analyzer.find_by_criteria(state["criteria"])
        
        if not results:
            st.info("No shows found matching your criteria. Try adjusting the filters.")
            return
        
        if not results:
            st.info("No matching shows found. Try adjusting your criteria.")
            return
            
        # Results are already sorted by weighted combo of comp_score and success_score
        
        # Filter out results with missing scores
        results = [r for r in results if r.get('comp_score') is not None]
        
        # Already sorted by match score (primary) and success score (tiebreaker) in comp_analyzer.py
        
        if not results:
            st.info("No valid matches found. Try adjusting your criteria.")
            return
            
        # Add results table to grid
        st.markdown("### Similar Shows")
        
        # Create DataFrame for results
        import pandas as pd
        df = pd.DataFrame([
            {
                'Show': r.get('title', 'Unknown'),
                'Success': int(r.get('success_score', 0)),
                'Total Score': int(r['comp_score'].total),
                'Content': int(r['comp_score'].content_score),
                'Production': int(r['comp_score'].production_score),
                'Format': int(r['comp_score'].format_score)
            } for r in results
        ])
        
        # Add custom CSS for metrics
        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"] {
                overflow: visible !important;
                width: auto !important;
                margin-right: 20px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Display as Streamlit table with custom CSS
        st.markdown(
            f"""
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
            </style>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(df, use_container_width=True)
        
        # Show detailed breakdowns for top 10 matches
        if results:
            st.markdown("### Match Details")
            
            # Take top 10 matches
            top_matches = results[:10]
            
            # Create expandable section for each match
            for i, match in enumerate(top_matches, 1):
                with st.expander(f"#{i}: {match['title']}", expanded=(i==1)):
                    try:
                        # Get comp score components
                        comp_score = match.get('comp_score', None)
                        if not comp_score:
                            continue
                            
                        logging.debug(f"Processing match {i}: {match['title']}")
                        logging.debug(f"Raw match data: {match}")
                        logging.debug(f"Comp score data: {comp_score}")
                        
                        # Debug prints
                        st.write('Raw match data:')
                        st.write(match)
                        st.write('Comp score data:')
                        st.write(comp_score)
                        
                        logging.debug("Getting display scores and details...")
                        scores = comp_score.to_display_dict()
                        logging.debug(f"Display scores: {scores}")
                        
                        details = comp_score.get_match_details()
                        logging.debug(f"Match details: {details}")
                        
                        # Add additional show-specific details
                    except Exception as e:
                        logging.error("Error processing match:")
                        logging.error(traceback.format_exc())
                        st.error(f"Error processing match {i}: {str(e)}\n{traceback.format_exc()}")
                        continue
                    # Update selected status based on criteria
                    criteria = state.get('criteria', {})
                    
                    if details.get('genre'):
                        # Get genre info with safe defaults
                        genre_name = match.get('genre_name') or 'Unknown'
                        subgenres = [sg for sg in match.get('subgenre_names', []) if sg]  # Filter out None values
                        
                        # Get IDs
                        show_genre_id = match.get('genre_id')
                        selected_genre_id = criteria.get('genre_id')
                        show_subgenre_ids = match.get('subgenres', [])
                        selected_subgenre_ids = criteria.get('subgenres', [])
                        
                        # Create ID -> name mappings from field options
                        genre_names = {id: name for id, name in field_options['genres']}
                        subgenre_names = {id: name for id, name in field_options['subgenre_names']}
                        
                        # Get genre names from field options
                        show_genre = genre_names.get(show_genre_id, 'Unknown')
                        selected_genre = genre_names.get(selected_genre_id, 'Unknown')
                        
                        # Split into matches and mismatches using IDs
                        matching_ids = [sg_id for sg_id in show_subgenre_ids if sg_id in selected_subgenre_ids]
                        mismatched_ids = [sg_id for sg_id in show_subgenre_ids if sg_id not in selected_subgenre_ids]
                        
                        # Convert IDs to names using field options
                        matches = [subgenre_names.get(sg_id, 'Unknown') for sg_id in matching_ids]
                        mismatches = [subgenre_names.get(sg_id, 'Unknown') for sg_id in mismatched_ids]
                        
                        details['genre'].update({
                            'primary': selected_genre,  # Show selected genre
                            'shared_subgenres': matches,  # Use matches as shared subgenres
                            'subgenre_points': comp_score.genre_overlap or 0,
                            'subgenre_matches': matches,
                            'subgenre_mismatches': mismatches,
                            'selected': True  # Genre is always selected since it's required
                        })
                    
                    # Add source details
                    if details.get('source'):
                        source_name = match.get('source_type_name', 'Unknown')
                        details['source'].update({
                            'type1': source_name,
                            'type2': source_name,  # Same for both since this is criteria-based comparison
                            'selected': criteria.get('source_type_id') is not None
                        })
                    
                    # Add character details
                    if details.get('characters'):
                        # Get IDs
                        show_char_ids = match.get('character_type_ids', [])
                        selected_char_ids = criteria.get('character_type_ids', [])
                        
                        # Create ID -> name mapping from field options
                        char_names = {id: name for id, name in field_options['character_types']}
                        
                        # Find matches and mismatches using IDs only
                        matches = [char_names[id] for id in selected_char_ids if id in show_char_ids]
                        mismatches = [char_names[id] for id in show_char_ids if id not in selected_char_ids]
                        
                        details['characters'].update({
                            'matches': matches,
                            'mismatches': mismatches,
                            'selected': bool(selected_char_ids)
                        })
                    
                    # Add plot details
                    if details.get('plot'):
                        # Get IDs
                        show_plot_ids = match.get('plot_element_ids', [])
                        selected_plot_ids = criteria.get('plot_element_ids', [])
                        
                        # Create ID -> name mapping from field options
                        plot_names = {id: name for id, name in field_options['plot_elements']}
                        
                        # Find matches and mismatches using IDs only
                        matches = [plot_names[id] for id in selected_plot_ids if id in show_plot_ids]
                        mismatches = [plot_names[id] for id in show_plot_ids if id not in selected_plot_ids]
                        
                        details['plot'].update({
                            'matches': matches,
                            'mismatches': mismatches,
                            'selected': bool(selected_plot_ids)
                        })
                    
                    # Add theme details
                    if details.get('themes'):
                        # Get IDs
                        show_theme_ids = match.get('thematic_element_ids', [])
                        selected_theme_ids = criteria.get('thematic_element_ids', [])
                        
                        # Create ID -> name mapping from field options
                        theme_names = {id: name for id, name in field_options['thematic_elements']}
                        
                        # Find matches and mismatches using IDs only
                        matches = [theme_names[id] for id in selected_theme_ids if id in show_theme_ids]
                        mismatches = [theme_names[id] for id in show_theme_ids if id not in selected_theme_ids]
                        
                        details['themes'].update({
                            'matches': matches,
                            'mismatches': mismatches,
                            'selected': bool(selected_theme_ids)
                        })
                    
                    # Add tone details
                    if details.get('tone'):
                        # Get IDs
                        show_tone_id = match.get('tone_id')
                        selected_tone_id = criteria.get('tone_id')
                        
                        # Create ID -> name mapping from field options
                        tone_names = {id: name for id, name in field_options['tones']}
                        
                        # Get tone names from field options
                        show_tone_name = tone_names.get(show_tone_id, 'Unknown')
                        selected_tone_name = tone_names.get(selected_tone_id, 'Unknown')
                        
                        details['tone'].update({
                            'tone1': selected_tone_name,
                            'tone2': show_tone_name,
                            'selected': selected_tone_id is not None
                        })
                    
                    # Add setting details
                    if details.get('setting'):
                        # Get IDs
                        show_time_id = match.get('time_setting_id')
                        show_location_id = match.get('location_setting_id')
                        selected_time_id = criteria.get('time_setting_id')
                        selected_location_id = criteria.get('location_setting_id')
                        
                        # Create ID -> name mappings from field options
                        time_names = {id: name for id, name in field_options['time_settings']}
                        location_names = {id: name for id, name in field_options['locations']}
                        
                        # Get setting names from field options
                        show_time = time_names.get(show_time_id, 'Unknown')
                        show_location = location_names.get(show_location_id, 'Unknown')
                        
                        details['setting'].update({
                            'time': show_time,
                            'location': show_location,
                            'selected': (selected_time_id is not None or selected_location_id is not None)
                        })
                        
                    # Initialize network details if not present
                    if 'network' not in details:
                        details['network'] = {}
                        
                    # Get IDs
                    show_network_id = match.get('network_id')
                    selected_network_id = criteria.get('network_id')
                    
                    # Create ID -> name mapping from field options
                    network_names = {id: name for id, name in field_options['networks']}
                    
                    # Get network names from field options
                    show_network = network_names.get(show_network_id, 'Unknown')
                    selected_network = network_names.get(selected_network_id, 'Unknown')
                    
                    # Network match if IDs match
                    network_match = show_network_id == selected_network_id and show_network_id is not None
                    details['network'].update({
                        'name1': selected_network,
                        'name2': show_network,
                        'match': network_match,
                        'selected': selected_network_id is not None,  # True if any network was selected
                        'score': scores['network_score']
                    })
                    
                    # Add studio details
                    if details.get('studio'):
                        # Get IDs
                        show_studio_id = match.get('studio_id')
                        selected_studio_id = criteria.get('studio_id')
                        
                        # Create ID -> name mapping from field options
                        studio_names = {id: name for id, name in field_options['studios']}
                        
                        # Get studio names from field options
                        show_studio = studio_names.get(show_studio_id, 'Unknown')
                        selected_studio = studio_names.get(selected_studio_id, 'Unknown')
                        
                        details['studio'].update({
                            'name1': selected_studio,
                            'name2': show_studio,
                            'match': show_studio_id == selected_studio_id,
                            'selected': selected_studio_id is not None
                        })
                    
                    # Add team details
                    if details.get('team'):
                        # Get IDs
                        show_team_ids = match.get('team_member_ids', [])
                        selected_team_ids = criteria.get('team_member_ids', [])
                        
                        # Create ID -> name mapping from field options
                        team_names = {id: name for id, name in field_options['team_members']}
                        
                        # Get shared members using IDs
                        shared_ids = [id for id in show_team_ids if id in selected_team_ids]
                        shared_members = [team_names.get(id, 'Unknown') for id in shared_ids]
                        
                        details['team'].update({
                            'shared_members': shared_members,
                            'selected': bool(selected_team_ids)
                        })
                        
                    # Add format details
                    if details.get('format'):
                        # Get episode counts
                        show_episode_count = match.get('episode_count')
                        selected_episode_count = criteria.get('episode_count')
                        
                        # Get order type IDs
                        show_order_id = match.get('order_type_id')
                        selected_order_id = criteria.get('order_type_id')
                        
                        # Create ID -> name mapping from field options
                        order_names = {id: name for id, name in field_options['order_types']}
                        
                        # Get order type names from field options
                        show_order = order_names.get(show_order_id, 'Unknown')
                        selected_order = order_names.get(selected_order_id, 'Unknown')
                        
                        details['format'].update({
                            'episode_count1': selected_episode_count,
                            'episode_count2': show_episode_count,
                            'order_type1': selected_order,
                            'order_type2': show_order,
                            'selected': (selected_episode_count is not None or 
                                        selected_order_id is not None)
                        })
                    
                    # Create match details dictionary
                    match_details = {
                        'scores': scores,
                        'details': details,
                        'matches': {
                            'genre': match.get('genre_name', 'Unknown'),
                            'subgenres': match.get('subgenre_names', []),
                            'source': match.get('source_type_name', 'Unknown'),
                            'characters': match.get('character_type_names', []),
                            'plot': match.get('plot_element_names', []),
                            'themes': match.get('theme_element_names', []),
                            'tone': match.get('tone_name', 'Unknown'),
                            'time': match.get('time_setting_name', 'Unknown'),
                            'location': match.get('location_setting_name', 'Unknown'),
                            'episodes': match.get('episode_count', 'Unknown'),
                            'order': match.get('order_type_name', 'Unknown'),
                            'team': match.get('team_names', [])
                        }
                    }
                    
                    # Display match details
                    st.write('### Match Details')
                    st.write(match_details)
                    
                    # Use base match breakdown
                    # Call base match breakdown component
                    render_base_match_breakdown(
                        title=match['title'],
                        scores=scores,
                        details=details,
                        matches=match,
                        success_score=match.get('success_score'),
                        expanded=i==1,  # Only expand first match
                        use_expander=False,  # We're already in an expander
                        description=match.get('description')
                    )
    else:
        st.info("Select criteria on the left to find similar shows.")
