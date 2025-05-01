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
        
        subgenre_names = st.multiselect(
            "Subgenres",
            options=[name for _, name in field_options['subgenre_names']],
            format_func=lambda x: x,
            key="subgenres",
            placeholder="Select subgenres..."
        )
        
        # Convert selected subgenres to IDs
        subgenre_ids = get_ids_for_names(subgenre_names, field_options['subgenre_names'])
        
        # Show selected subgenres and their IDs
        if subgenre_names:
            st.write("")
            st.write("Selected subgenres:")
            for name in subgenre_names:
                for id, opt_name in field_options['subgenre_names']:
                    if opt_name == name:
                        st.write(f"- {name} (ID: {id})")
            st.write("")
        
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
        
        char_names = st.multiselect(
            "Character Types",
            options=[name for _, name in field_options['character_types']],
            format_func=lambda x: x,
            key="character_type_ids",
            placeholder="Select character types..."
        )
        state["criteria"]["character_type_ids"] = get_ids_for_names(char_names, field_options['character_types'])
        
        plot_names = st.multiselect(
            "Plot Elements",
            options=[name for _, name in field_options['plot_elements']],
            format_func=lambda x: x,
            key="plot_element_ids",
            placeholder="Select plot elements..."
        )
        state["criteria"]["plot_element_ids"] = get_ids_for_names(plot_names, field_options['plot_elements'])
        
        theme_names = st.multiselect(
            "Theme Elements",
            options=[name for _, name in field_options['thematic_elements']],
            format_func=lambda x: x,
            key="theme_element_ids",
            placeholder="Select theme elements..."
        )
        state["criteria"]["theme_element_ids"] = get_ids_for_names(theme_names, field_options['thematic_elements'])
        
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
        
        network_name = st.selectbox(
            "Network",
            options=[name for _, name in field_options['networks']],
            format_func=lambda x: x,
            key="network_id",
            index=None,
            placeholder="Select network..."
        )
        state["criteria"]["network_id"] = get_id_for_name(network_name, field_options['networks']) if network_name else None
        
        studio_names = st.multiselect(
            "Studios",
            options=[name for _, name in field_options['studios']],
            format_func=lambda x: x,
            key="studios",
            placeholder="Select studios..."
        )
        state["criteria"]["studios"] = get_ids_for_names(studio_names, field_options['studios'])
        
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
        
        # Sort by success score
        results.sort(key=lambda x: x.get('success_score', 0), reverse=True)
        
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
                'Success': f"{r.get('success_score', 0):.1f}%",
                'Total Score': f"{r['comp_score'].total:.1f}%",
                'Content': f"{r['comp_score'].content_score:.1f}%",
                'Production': f"{r['comp_score'].production_score:.1f}%",
                'Format': f"{r['comp_score'].format_score:.1f}%"
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
                        
                        # Get selected subgenre IDs and names
                        selected_subgenre_ids = criteria.get('subgenres', [])
                        
                        # Get show's subgenre IDs and names
                        show_subgenre_ids = match.get('subgenres', [])
                        show_subgenre_names = match.get('subgenre_names', [])
                        
                        # Create ID to name mapping for the show's subgenres
                        show_subgenre_map = dict(zip(show_subgenre_ids, show_subgenre_names))
                        
                        # Split into matches and mismatches using IDs first
                        matching_ids = [sg_id for sg_id in show_subgenre_ids if sg_id in selected_subgenre_ids]
                        mismatched_ids = [sg_id for sg_id in show_subgenre_ids if sg_id not in selected_subgenre_ids]
                        
                        # Convert IDs to names for display
                        matches = [show_subgenre_map[sg_id] for sg_id in matching_ids]
                        mismatches = [show_subgenre_map[sg_id] for sg_id in mismatched_ids]
                        
                        details['genre'].update({
                            'primary': genre_name,
                            'shared_subgenres': subgenres,
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
                        char_types = match.get('character_type_names', [])
                        details['characters'].update({
                            'matches': char_types if comp_score.character_types > 0 else [],
                            'mismatches': [] if comp_score.character_types > 0 else char_types,
                            'selected': bool(criteria.get('character_type_ids'))
                        })
                    
                    # Add plot details
                    if details.get('plot'):
                        # Get all plot elements
                        plot_elements = match.get('plot_element_names', [])
                        # Calculate how many matches we have based on score
                        num_matches = int(comp_score.plot_elements / 2.4)  # 2.4 points per match
                        # Split into matches and mismatches
                        matches = plot_elements[:num_matches] if num_matches > 0 else []
                        mismatches = plot_elements[num_matches:] if num_matches < len(plot_elements) else []
                        details['plot'].update({
                            'matches': matches,
                            'mismatches': mismatches,
                            'selected': bool(criteria.get('plot_element_ids'))
                        })
                    
                    # Add theme details
                    if details.get('themes'):
                        theme_elements = match.get('theme_element_names', [])
                        details['themes'].update({
                            'matches': theme_elements if comp_score.theme_elements > 0 else [],
                            'mismatches': [] if comp_score.theme_elements > 0 else theme_elements,
                            'selected': bool(criteria.get('theme_element_ids'))
                        })
                    
                    # Add tone details
                    if details.get('tone'):
                        tone_name = match.get('tone_name', 'Unknown')
                        details['tone'].update({
                            'tone1': tone_name,
                            'tone2': tone_name,  # Same for both since this is criteria-based comparison
                            'selected': criteria.get('tone_id') is not None
                        })
                    
                    # Add setting details
                    if details.get('setting'):
                        details['setting'].update({
                            'time': match.get('time_setting_name', 'Unknown'),
                            'location': match.get('location_setting_name', 'Unknown'),
                            'selected': (criteria.get('time_setting_id') is not None or 
                                       criteria.get('location_setting_id') is not None)
                        })
                        
                    # Add network details
                    if details.get('network'):
                        network_name = match.get('network_name', 'Unknown')
                        details['network'].update({
                            'name1': network_name,
                            'name2': network_name,  # Same for criteria-based comparison
                            'match': bool(comp_score.network),
                            'selected': criteria.get('network_id') is not None
                        })
                    
                    # Add studio details
                    if details.get('studio'):
                        studio_name = match.get('studio_name', 'Unknown')
                        details['studio'].update({
                            'name1': studio_name,
                            'name2': studio_name,  # Same for criteria-based comparison
                            'match': bool(comp_score.studio),
                            'selected': criteria.get('studio_id') is not None
                        })
                    
                    # Add team details
                    if details.get('team'):
                        team_members = match.get('team_members', [])
                        details['team'].update({
                            'shared_members': team_members,
                            'selected': bool(criteria.get('team_member_ids'))
                        })
                        
                    # Add format details
                    if details.get('format'):
                        episode_count = match.get('episode_count')
                        order_type = match.get('order_type_name', 'Unknown')
                        match_order = match.get('order_type_name', 'Unknown')
                        criteria_order = next((name for id, name in field_options['order_types'] 
                                             if id == criteria.get('order_type_id')), 'Unknown')
                        
                        details['format'].update({
                            'episode_count1': episode_count,
                            'episode_count2': episode_count,  # Same for criteria-based comparison
                            'order_type1': match_order,
                            'order_type2': criteria_order,
                            'selected': bool(criteria.get('episode_count')) or \
                                       criteria.get('order_type_id') is not None
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
