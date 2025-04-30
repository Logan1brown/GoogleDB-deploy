"""View component for comp builder functionality."""

import streamlit as st
from typing import Dict, List, Tuple, Optional

from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
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
        
        state["criteria"]["episode_count"] = st.number_input(
            "Episode Count",
            min_value=1,
            max_value=100,
            value=10,
            key="episode_count",
            help="Episode count proximity (2 within ±2, 1.5 within ±4, 1 within ±6)"
        )
        
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
                    # Success and match scores
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Success Score", f"{match.get('success_score', 0):.1f}/100")
                    with metric_col2:
                        st.metric("Match Score", f"{match['comp_score'].total:.1f}/100")
                    
                    # Score breakdown
                    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; font-weight: 600; color: {COLORS["text"]["primary"]}; margin: 20px 0;">Score Breakdown</p>', unsafe_allow_html=True)
                    
                    # Only show scoring factors that contribute points
                    if match.get('tmdb_seasons', 0) >= 2:
                        st.write(f"**Renewed for Season {match.get('tmdb_seasons')}** _(+40 points)_")
                    if match.get('tmdb_total_episodes', 0) >= 20:
                        st.write("**20+ Episodes** _(+30 points)_")
                    if match.get('longevity_score', 0) > 0:
                        st.write(f"**Longevity Bonus** _(+{match.get('longevity_score', 0):.1f} points)_")
                    
                    # Match score components
                    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; font-weight: 600; color: {COLORS["text"]["primary"]}; margin: 20px 0;">Match Components</p>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Content", f"{match['comp_score'].content_score:.1f}/70")
                    with col2:
                        st.metric("Production", f"{match['comp_score'].production_score:.1f}/13")
                    with col3:
                        st.metric("Format", f"{match['comp_score'].format_score:.1f}/3")
                    
                    # Show details
                    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; font-weight: 600; color: {COLORS["text"]["primary"]}; margin: 20px 0;">Show Details</p>', unsafe_allow_html=True)
                    
                    # Helper function to format value with match highlighting
                    def format_value(value, is_match):
                        return f"{'🟢' if is_match else '⚫'} {value}"
                    
                    # Content details
                    content_col1, content_col2 = st.columns(2)
                    with content_col1:
                        # Genre
                        st.markdown("**Genre**")
                        is_match = state.get('genre_name') == match['genre_name']
                        st.markdown(format_value(match['genre_name'], is_match), unsafe_allow_html=True)
                        
                        # Subgenres
                        st.markdown("**Subgenres**")
                        subgenres = match.get('subgenre_names', [])
                        selected_subgenres = state.get('subgenre_names', [])
                        if subgenres:
                            subgenre_texts = [format_value(name, name in selected_subgenres) for name in subgenres if name]
                            st.markdown(', '.join(subgenre_texts), unsafe_allow_html=True)
                        else:
                            st.write('None')
                        
                        # Source Type
                        st.markdown("**Source Type**")
                        is_match = state.get('source_type_name') == match['source_type_name']
                        st.markdown(format_value(match['source_type_name'], is_match), unsafe_allow_html=True)
                        
                        # Character Types
                        if match.get('character_types'):
                            st.markdown("**Character Types**")
                            selected_chars = state.get('character_types', [])
                            char_texts = [format_value(char, char in selected_chars) for char in match['character_types']]
                            st.markdown(' • '.join(char_texts), unsafe_allow_html=True)
                        
                        # Plot Elements
                        if match.get('plot_elements'):
                            st.markdown("**Plot Elements**")
                            selected_plots = state.get('plot_elements', [])
                            plot_texts = [format_value(plot, plot in selected_plots) for plot in match['plot_elements']]
                            st.markdown(' • '.join(plot_texts), unsafe_allow_html=True)
                        
                        # Themes
                        if match.get('thematic_elements'):
                            st.markdown("**Themes**")
                            selected_themes = state.get('thematic_elements', [])
                            theme_texts = [format_value(theme, theme in selected_themes) for theme in match['thematic_elements']]
                            st.markdown(' • '.join(theme_texts), unsafe_allow_html=True)
                    
                    with content_col2:
                        # Network
                        st.markdown("**Network**")
                        is_match = state.get('network_name') == match['network_name']
                        st.markdown(format_value(match['network_name'], is_match), unsafe_allow_html=True)
                        
                        # Studios
                        st.markdown("**Studios**")
                        selected_studios = state.get('studio_names', [])
                        studio_texts = [format_value(studio, studio in selected_studios) for studio in match['studio_names']]
                        st.markdown(', '.join(studio_texts), unsafe_allow_html=True)
                        
                        # Episodes
                        st.markdown("**Episodes**")
                        st.write(match['episode_count'])
                        
                        # Order Type
                        st.markdown("**Order Type**")
                        is_match = state.get('order_type_name') == match['order_type_name']
                        st.markdown(format_value(match['order_type_name'], is_match), unsafe_allow_html=True)
                        
                        # Time Setting
                        st.markdown("**Time Setting**")
                        is_match = state.get('time_setting_name') == match['time_setting_name']
                        st.markdown(format_value(match['time_setting_name'], is_match), unsafe_allow_html=True)
                        
                        # Location
                        st.markdown("**Location**")
                        is_match = state.get('location_setting_name') == match['location_setting_name']
                        st.markdown(format_value(match['location_setting_name'], is_match), unsafe_allow_html=True)
                        
                        # Key Roles
                        if match.get('team_roles'):
                            st.markdown("**Key Roles**")
                            selected_roles = state.get('team_roles', [])
                            role_texts = [format_value(role, role in selected_roles) for role in match['team_roles']]
                            st.markdown(' • '.join(role_texts), unsafe_allow_html=True)
    else:
        st.info("Select criteria on the left to find similar shows.")
