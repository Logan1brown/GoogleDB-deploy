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
                    # Success Score and Breakdown
                    st.markdown("### Success Score")
                    success_score = match.get('success_score', 0)
                    st.metric("Success Score", f"{success_score:.1f}/100")
                    
                    # Season achievements
                    season_score = 0
                    if pd.notna(match.get('tmdb_seasons')):
                        seasons = int(match['tmdb_seasons'])
                        if seasons >= 2:
                            season_score += 40
                            st.write(f"Season {seasons} renewal: +40 points")
                            extra_seasons = seasons - 2
                            if extra_seasons > 0:
                                bonus = min(extra_seasons * 20, 40)
                                season_score += bonus
                                st.write(f"Additional seasons: +{bonus} points")
                    
                    # Episode volume 
                    episode_score = 0
                    avg_eps = match.get('tmdb_avg_eps', 0)
                    if pd.notna(avg_eps):
                        avg_eps = float(avg_eps)
                        if avg_eps >= 10:
                            episode_score = 40
                            st.write("High episode volume: +40 points")
                        elif avg_eps >= 8:
                            episode_score = 20
                            st.write("Standard episode volume: +20 points")
                    
                    # Status modifier
                    status_score = 0
                    status = match.get('status_name')
                    if status and status == 'Returning Series':
                        status_score = success_score * 0.2  # 20% bonus
                        st.write("Active show bonus: Score x 1.2")
                    
                    # Display final breakdown
                    if season_score > 0 or episode_score > 0 or status_score > 0:
                        st.write("")
                        st.write("Final breakdown:")
                        if season_score > 0:
                            st.write(f"Season achievements: +{season_score:.1f}")
                        if episode_score > 0:
                            st.write(f"Episode volume: +{episode_score:.1f}")
                        if status_score > 0:
                            st.write(f"Status modifier: +{status_score:.1f}")
                    
                    # Get comp score components
                    comp_score = match.get('comp_score', {})
                    
                    # Content Match Section
                    content_score = sum([
                        float(comp_score.get('genre_base', 0)),
                        float(comp_score.get('genre_overlap', 0)),
                        float(comp_score.get('source_type', 0)),
                        float(comp_score.get('character_types', 0)),
                        float(comp_score.get('plot_elements', 0)),
                        float(comp_score.get('theme_elements', 0)),
                        float(comp_score.get('tone', 0)),
                        float(comp_score.get('time_setting', 0)),
                        float(comp_score.get('location', 0))
                    ])
                    st.markdown(f"### Content Match _({content_score:.1f}/70 points)_")
                    st.write("")
                    
                    # Content Match Details
                    col1, col2 = st.columns(2)
                    with col1:
                        # Genre
                        genre_score = float(comp_score.get('genre_base', 0)) + float(comp_score.get('genre_overlap', 0))
                        st.markdown(f"Genre ({genre_score:.1f}/17)")
                        st.write(f"⚫ {match.get('genre_name', 'None')}")
                        
                        # Subgenres
                        st.markdown("Subgenres")
                        subgenres = match.get('subgenre_names', [])
                        if subgenres:
                            for subgenre in subgenres:
                                st.write(f"⚫ {subgenre}")
                        else:
                            st.write("None")
                        
                        # Source Type
                        source_score = float(comp_score.get('source_type', 0))
                        st.markdown(f"Source Type ({source_score:.1f}/8)")
                        st.write(f"⚫ {match.get('source_type_name', 'None')}")
                        
                        # Character Types
                        char_score = float(comp_score.get('character_types', 0))
                        st.markdown(f"Character Types ({char_score:.1f}/14)")
                        char_types = match.get('character_type_names', [])
                        if char_types:
                            for char_type in char_types:
                                st.write(f"⚫ {char_type}")
                    
                    with col2:
                        # Plot Elements
                        plot_score = float(comp_score.get('plot_elements', 0))
                        st.markdown(f"Plot Elements ({plot_score:.1f}/12)")
                        plot_elements = match.get('plot_element_names', [])
                        if plot_elements:
                            for element in plot_elements:
                                st.write(f"⚫ {element}")
                        
                        # Theme Elements
                        theme_score = float(comp_score.get('theme_elements', 0))
                        st.markdown(f"Theme Elements ({theme_score:.1f}/13)")
                        theme_elements = match.get('thematic_element_names', [])
                        if theme_elements:
                            for element in theme_elements:
                                st.write(f"⚫ {element}")
                        
                        # Tone
                        tone_score = float(comp_score.get('tone', 0))
                        st.markdown(f"Tone ({tone_score:.1f}/8)")
                        st.write(f"⚫ {match.get('tone_name', 'None')}")
                    
                    st.write("")


                    
                    st.write("")
                    
                    # Production Match Section
                    production_score = sum([
                        float(comp_score.get('network', 0)),
                        float(comp_score.get('studio', 0)),
                        float(comp_score.get('team', 0))
                    ])
                    st.markdown(f"### Production Match _({production_score:.1f}/13 points)_")
                    st.write("")
                    
                    # Production Match Details
                    col1, col2 = st.columns(2)
                    with col1:
                        # Network
                        network_score = float(comp_score.get('network', 0))
                        st.markdown(f"Network ({network_score:.1f}/5)")
                        st.write(f"⚫ {match.get('network_name', 'None')}")
                        
                        # Studios
                        studio_score = float(comp_score.get('studio', 0))
                        st.markdown(f"Studios ({studio_score:.1f}/3)")
                        studios = match.get('studio_names', [])
                        if studios:
                            for studio in studios:
                                st.write(f"⚫ {studio}")
                    
                    with col2:
                        # Team
                        team_score = float(comp_score.get('team', 0))
                        st.markdown(f"Team ({team_score:.1f}/5)")
                        st.write("")
                    
                    st.write("")
                    
                    # Format Match Section
                    format_score = sum([
                        float(comp_score.get('episodes', 0)),
                        float(comp_score.get('order_type', 0))
                    ])
                    st.markdown(f"### Format Match _({format_score:.1f}/3 points)_")
                    st.write("")
                    
                    # Format Match Details
                    col1, col2 = st.columns(2)
                    with col1:
                        # Episodes
                        eps_score = float(comp_score.get('episodes', 0))
                        st.markdown(f"Episodes ({eps_score:.1f}/2)")
                        eps = match.get('tmdb_avg_eps', 0)
                        st.write(f"⚫ {eps}")
                    
                    with col2:
                        # Order Type
                        order_score = float(comp_score.get('order_type', 0))
                        st.markdown(f"Order Type ({order_score:.1f}/1)")
                        st.write(f"⚫ {match.get('order_type_name', 'None')}")
                    
                    st.write("")
                    
                    # Setting Match Section
                    setting_score = sum([
                        float(comp_score.get('time_setting', 0)),
                        float(comp_score.get('location', 0))
                    ])
                    st.markdown(f"### Setting Match _({setting_score:.1f}/7 points)_")
                    st.write("")
                    
                    # Setting Match Details
                    col1, col2 = st.columns(2)
                    with col1:
                        # Time Setting
                        time_score = float(comp_score.get('time_setting', 0))
                        st.markdown(f"Time Setting ({time_score:.1f}/4)")
                        st.write(f"⚫ {match.get('time_setting_name', 'None')}")
                    
                    with col2:
                        # Location
                        location_score = float(comp_score.get('location', 0))
                        st.markdown(f"Location ({location_score:.1f}/3)")
                        st.write(f"⚫ {match.get('location_setting_name', 'None')}")
                    
                    st.write("")
                    
                    st.write("")
                    
                    st.write("")

                    if match.get('longevity_score', 0) > 0:
                        st.markdown(f"**Longevity Bonus** _(+{match.get('longevity_score', 0):.1f} points)_")
                        st.write("")
                    
                    # Key Roles
                    if match.get('team_roles'):
                        st.markdown("**Key Roles**")
                        selected_roles = state.get('team_roles', [])
                        role_texts = [format_value(role, role in selected_roles) for role in match['team_roles']]
                        st.markdown(' • '.join(role_texts), unsafe_allow_html=True)
    else:
        st.info("Select criteria on the left to find similar shows.")
