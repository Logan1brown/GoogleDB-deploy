"""View component for comp builder functionality."""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict

from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
from src.dashboard.templates.defaults.table import create_table_defaults
from src.dashboard.utils.style_config import COLORS, FONTS


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
        # Genre (20 points)
        state["criteria"]["genre_id"] = st.selectbox(
            "Genre",
            options=comp_analyzer.get_field_options("genre_id"),
            key="comp_genre",
            help="20 points - Full match on primary genre"
        )
        
        # Source Type (10 points)
        state["criteria"]["source_type_id"] = st.selectbox(
            "Source Type", 
            options=comp_analyzer.get_field_options("source_type_id"),
            key="comp_source",
            help="10 points - Match on source material type"
        )
        
        # Characters (15 points)
        state["criteria"]["character_types"] = st.multiselect(
            "Character Types",
            options=comp_analyzer.get_field_options("character_types"),
            key="comp_characters",
            help="15 points - Matching character archetypes and dynamics"
        )
        
        # Plot Elements (10 points)
        state["criteria"]["plot_elements"] = st.multiselect(
            "Plot Elements",
            options=comp_analyzer.get_field_options("plot_elements"),
            key="comp_plot",
            help="10 points - Key plot devices and story structures"
        )
        
        # Themes (5 points)
        state["criteria"]["thematic_elements"] = st.multiselect(
            "Thematic Elements",
            options=comp_analyzer.get_field_options("thematic_elements"),
            key="comp_themes",
            help="5 points - Core themes and messages"
        )
        
        # Tone (5 points)
        state["criteria"]["tone"] = st.selectbox(
            "Tone",
            options=comp_analyzer.get_field_options("tone"),
            key="comp_tone",
            help="5 points - Overall tone and mood"
        )
        
        # Setting (5 points)
        col1, col2 = st.columns(2)
        with col1:
            state["criteria"]["time_setting"] = st.selectbox(
                "Time Setting",
                options=comp_analyzer.get_field_options("time_setting"),
                key="comp_time",
                help="2.5 points - Time period setting"
            )
        with col2:
            state["criteria"]["location"] = st.selectbox(
                "Location",
                options=comp_analyzer.get_field_options("location"),
                key="comp_location",
                help="2.5 points - Geographic setting"
            )

    with st.expander("Production Match Criteria (13 pts)"):
        # Network (5 points)
        state["criteria"]["network_id"] = st.selectbox(
            "Network",
            options=comp_analyzer.get_field_options("network_id"),
            key="comp_network",
            help="5 points - Broadcasting/streaming network"
        )
        
        # Studios (5 points)
        state["criteria"]["studios"] = st.multiselect(
            "Studios",
            options=comp_analyzer.get_field_options("studios"),
            key="comp_studios",
            help="5 points - Production studios involved"
        )
        
        # Team Roles (3 points)
        state["criteria"]["team_roles"] = st.multiselect(
            "Key Creative Roles",
            options=comp_analyzer.get_field_options("team_roles"),
            key="comp_team",
            help="3 points - Key creative team roles"
        )

    with st.expander("Format Match Criteria (3 pts)"):
        col1, col2 = st.columns(2)
        
        # Episodes (2 points)
        with col1:
            state["criteria"]["episode_count"] = st.number_input(
                "Episode Count",
                min_value=1,
                max_value=100,
                value=13,
                key="comp_episodes",
                help="2 points - Number of episodes per season"
            )
        
        # Order Type (1 point)
        with col2:
            state["criteria"]["order_type"] = st.selectbox(
                "Order Type",
                options=comp_analyzer.get_field_options("order_type"),
                key="comp_order",
                help="1 point - Series order type"
            )


def render_results_section(comp_analyzer: CompAnalyzer, state: Dict) -> None:
    """Render comp results panel.
    
    Args:
        comp_analyzer: CompAnalyzer instance for getting similar shows
        state: Page state dictionary containing criteria
    """
    if state.get("criteria"):
        # Get results
        results = comp_analyzer.get_similar_shows(state["criteria"])
        
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
