"""View component for comp builder functionality."""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
from src.dashboard.templates.defaults.table import create_table_defaults
from src.dashboard.utils.style_config import COLORS, FONTS


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
        col1, col2 = st.columns(2)
        
        with col1:
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
            
        with col2:
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
        col1, col2 = st.columns(2)
        
        with col1:
            network_name = st.selectbox(
                "Network",
                options=[name for _, name in field_options['networks']],
                format_func=lambda x: x,
                key="network_id",
                index=None,
                placeholder="Select network..."
            )
            state["criteria"]["network_id"] = get_id_for_name(network_name, field_options['networks']) if network_name else None
            
        with col2:
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
        col1, col2 = st.columns(2)
        
        with col1:
            state["criteria"]["episode_count"] = st.number_input(
                "Episode Count",
                min_value=1,
                max_value=100,
                value=10,
                key="episode_count",
                help="Episode count proximity (2 within ±2, 1.5 within ±4, 1 within ±6)"
            )
            
        with col2:
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
        
        # Create results table
        table_fig = go.Figure(template=create_table_defaults())
        table_fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "Show",
                        "Success",
                        "Total Score",
                        "Content",
                        "Production",
                        "Format"
                    ],
                    font=dict(
                        family=FONTS["primary"]["family"],
                        size=FONTS["primary"]["sizes"]["body"],
                        color=COLORS["text"]["primary"]
                    ),
                    fill_color=COLORS["background"]["secondary"],
                    line_color=COLORS["border"]
                ),
                cells=dict(
                    values=[
                        [r["title"] for r in results],
                        [f"{r['success_score']:.1f}%" for r in results],
                        [f"{r['comp_score'].total:.1f}%" for r in results],
                        [f"{r['comp_score'].content_score:.1f}%" for r in results],
                        [f"{r['comp_score'].production_score:.1f}%" for r in results],
                        [f"{r['comp_score'].format_score:.1f}%" for r in results]
                    ],
                    font=dict(
                        family=FONTS["primary"]["family"],
                        size=FONTS["primary"]["sizes"]["small"],
                        color=COLORS["text"]["primary"]
                    ),
                    fill_color=COLORS["background"]["primary"],
                    line_color=COLORS["border"],
                    align=["left", "right", "right", "right", "right"]
                )
            )
        )
        
        # Add hover effect
        table_fig.update_traces(
            cells=dict(
                fill=dict(
                    color=[COLORS["background"]["primary"]],
                    line=dict(color=COLORS["border"], width=1)
                )
            )
        )
        
        st.plotly_chart(table_fig, use_container_width=True)
        
        # Show detailed breakdowns for top 10 matches
        if results:
            st.markdown("### Match Details")
            
            # Take top 10 matches
            top_matches = results[:10]
            
            # Create expandable section for each match
            for i, match in enumerate(top_matches, 1):
                with st.expander(f"#{i}: {match['title']} - {match['success_score']:.1f}% Success, {match['comp_score'].total:.1f}% Match", expanded=(i==1)):
                    # Success metrics
                    st.markdown("#### Success Metrics")
                    success_col1, success_col2 = st.columns(2)
                    with success_col1:
                        st.markdown(
                            f"**Episodes:** {match.get('tmdb_total_episodes', 'Unknown')}\n\n"
                            f"**Seasons:** {match.get('tmdb_seasons', 'Unknown')}"
                        )
                    with success_col2:
                        st.markdown(
                            f"**Longevity Score:** {match.get('longevity_score', 0):.1f}%\n\n"
                            f"**Overall Success:** {match.get('success_score', 0):.1f}%"
                        )
                    
                    # Match score summary
                    st.markdown("#### Match Score Breakdown")
                    st.markdown(
                        f"Content Match: {match['content_score']:.1f}% (70 pts max)\n\n"
                        f"Production Match: {match['production_score']:.1f}% (13 pts max)\n\n"
                        f"Format Match: {match['format_score']:.1f}% (3 pts max)"
                    )
                    
                    # Show details
                    st.markdown("#### Show Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(
                            f"**Genre:** {match['genre_name']}\n\n"
                            f"**Source:** {match['source_type']}\n\n"
                            f"**Network:** {match['network_name']}\n\n"
                            f"**Studios:** {', '.join(match['studio_names'])}"
                        )
                    
                    with col2:
                        st.markdown(
                            f"**Episodes:** {match['episode_count']}\n\n"
                            f"**Order Type:** {match['order_type']}\n\n"
                            f"**Time Setting:** {match['time_setting']}\n\n"
                            f"**Location:** {match['location']}"
                        )
                    
                    # Show matching elements
                    st.markdown("#### Matching Elements")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if match.get('character_types'):
                            st.markdown(f"**Character Types:** {', '.join(match['character_types'])}")
                        if match.get('plot_elements'):
                            st.markdown(f"**Plot Elements:** {', '.join(match['plot_elements'])}")
                    
                    with col2:
                        if match.get('thematic_elements'):
                            st.markdown(f"**Themes:** {', '.join(match['thematic_elements'])}")
                        if match.get('team_roles'):
                            st.markdown(f"**Key Roles:** {', '.join(match['team_roles'])}")
    else:
        st.info("Select criteria on the left to find similar shows.")
