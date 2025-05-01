"""View component for comp builder functionality."""

import streamlit as st
from typing import Dict, List, Tuple, Optional

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
        state["criteria"]["subgenres"] = get_ids_for_names(subgenre_names, field_options['subgenre_names'])
        
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
        state["criteria"]["episode_count"] = eps if eps > 0 else None
        
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
                    # Get comp score components
                    comp_score = match.get('comp_score', None)
                    if not comp_score:
                        continue
                        
                    # Debug prints
                    st.write('Raw match data:')
                    st.write(match)
                    st.write('Comp score data:')
                    st.write(comp_score)
                    
                    # Get display scores and details
                    scores = comp_score.to_display_dict()
                    
                    # Get base match details
                    details = comp_score.get_match_details()
                    
                    # Add additional show-specific details
                    details['genre'].update({
                        'primary': match.get('genre_name', 'Unknown'),
                        'shared_subgenres': match.get('subgenre_names', []),
                        'subgenre_points': comp_score.genre_overlap
                    })
                    
                    # Add source details
                    details['source'].update({
                        'name': match.get('source_type_name', 'Unknown')
                    })
                    
                    # Add character details
                    details['characters'].update({
                        'types': match.get('character_type_names', [])
                    })
                    
                    # Add plot details
                    details['plot'].update({
                        'elements': match.get('plot_element_names', [])
                    })
                    
                    # Add theme details
                    details['themes'].update({
                        'elements': match.get('theme_element_names', [])
                    })
                    
                    # Add tone details
                    details['tone'].update({
                        'name': match.get('tone_name', 'Unknown')
                    })
                    
                    # Add setting details
                    details['setting'].update({
                        'time': match.get('time_setting_name', 'Unknown'),
                        'location': match.get('location_setting_name', 'Unknown')
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
                        success_score=match.get('success_score'),
                        expanded=i==1,  # Only expand first match
                        use_expander=False,  # We're already in an expander
                        description=match.get('description')
                    )
    else:
        st.info("Select criteria on the left to find similar shows.")
