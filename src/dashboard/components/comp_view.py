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
        
        # Sort by success score
        results.sort(key=lambda x: x.get('success_score', 0), reverse=True)
        
        # Add results table to grid
        st.markdown("### Similar Shows")
        
        # Create DataFrame for results
        import pandas as pd
        df = pd.DataFrame([
            {
                'Show': r['title'],
                'Success': f"{r['success_score']:.1f}%",
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
                    
                    # Build scores dict
                    scores = {
                        'genre_score': comp_score.genre_base + comp_score.genre_overlap,
                        'source_score': comp_score.source_type,
                        'character_score': comp_score.character_types,
                        'plot_score': comp_score.plot_elements,
                        'theme_score': comp_score.theme_elements,
                        'tone_score': comp_score.tone,
                        'time_score': comp_score.time_setting,
                        'location_score': comp_score.location,
                        'team_score': comp_score.team,
                        'episode_score': comp_score.episodes,
                        'order_score': comp_score.order_type,
                        'network_score': comp_score.network,
                        'studio_score': comp_score.studio,
                        'date_score': 0,  # We don't track this yet
                        'content_total': (
                            comp_score.genre_base +
                            comp_score.genre_overlap +
                            comp_score.source_type +
                            comp_score.character_types +
                            comp_score.plot_elements +
                            comp_score.theme_elements +
                            comp_score.tone +
                            comp_score.time_setting +
                            comp_score.location
                        ),
                        'format_total': comp_score.episodes + comp_score.order_type,
                        'setting_total': comp_score.time_setting + comp_score.location
                    }
                    scores['total'] = scores['content_total'] + scores['team_score'] + scores['format_total']
                    
                    # Build details dict
                    details = {
                        'genre': {
                            'primary_match': comp_score.genre_base > 0,
                            'primary': match.get('genre_name', 'Unknown'),
                            'shared_subgenres': match.get('subgenre_names', []),
                            'subgenre_points': comp_score.genre_overlap
                        },
                        'source': {
                            'match': comp_score.source_type > 0,
                            'type1': match.get('source_type_name', 'Unknown'),
                            'type2': match.get('source_type_name', 'Unknown')  # Same since this is base criteria
                        },
                        'character_types': {
                            'matches': match.get('character_type_names', []),
                            'mismatches': []  # No mismatches since this is base criteria
                        },
                        'plot_elements': {
                            'matches': match.get('plot_element_names', []),
                            'mismatches': []  # No mismatches since this is base criteria
                        },
                        'theme_elements': {
                            'matches': match.get('theme_element_names', []),
                            'mismatches': []  # No mismatches since this is base criteria
                        },
                        'tone': {
                            'match': comp_score.tone > 0,
                            'tone1': match.get('tone_name', 'Unknown'),
                            'tone2': match.get('tone_name', 'Unknown')  # Same since this is base criteria
                        },
                        'time_setting': {
                            'match': comp_score.time_setting > 0,
                            'time1': match.get('time_setting_name', 'Unknown'),
                            'time2': match.get('time_setting_name', 'Unknown')  # Same since this is base criteria
                        },
                        'location': {
                            'match': comp_score.location > 0,
                            'location1': match.get('location_setting_name', 'Unknown'),
                            'location2': match.get('location_setting_name', 'Unknown')  # Same since this is base criteria
                        },
                        'format': {
                            'eps_per_season1': str(match.get('episode_count', 'Unknown')),
                            'eps_per_season2': str(match.get('episode_count', 'Unknown')),  # Same since this is base criteria
                            'order_type1': match.get('order_type_name', 'Unknown'),
                            'order_type2': match.get('order_type_name', 'Unknown')  # Same since this is base criteria
                        },
                        'network': {
                            'match': comp_score.network > 0,
                            'name1': match.get('network_name', 'Unknown'),
                            'name2': match.get('network_name', 'Unknown')  # Same since this is base criteria
                        },
                        'studio': {
                            'match': comp_score.studio > 0,
                            'name1': match.get('studio_name', 'Unknown'),
                            'name2': match.get('studio_name', 'Unknown')  # Same since this is base criteria
                        },
                        'team': {
                            'shared_members': [(name, 'Team Member') for name in match.get('team_names', [])]
                        }
                    }
                    
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
