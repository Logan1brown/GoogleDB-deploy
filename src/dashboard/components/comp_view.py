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
        # Genre (17 points)
        state["criteria"]["genre_id"] = st.selectbox(
            "Genre",
            options=comp_analyzer.get_field_options("genre_id"),
            key="comp_genre",
            help="17 points - Genre match (9 base + 8 subgenre)"
        )
        
        # Source Type (8 points)
        state["criteria"]["source_type_id"] = st.selectbox(
            "Source Type", 
            options=comp_analyzer.get_field_options("source_type_id"),
            key="comp_source",
            help="8 points - Direct match on source type"
        )
        
        # Character Types (14 points)
        state["criteria"]["character_type_ids"] = st.multiselect(
            "Character Types",
            options=comp_analyzer.get_field_options("character_type_ids"),
            key="comp_characters",
            help="14 points - Character type matches (5 primary + 1.8 per additional up to 5)"
        )
        
        # Plot Elements (12 points)
        state["criteria"]["plot_element_ids"] = st.multiselect(
            "Plot Elements",
            options=comp_analyzer.get_field_options("plot_element_ids"),
            key="comp_plot",
            help="12 points - Plot element matches (2.4 per match up to 5)"
        )
        
        # Theme Elements (13 points)
        state["criteria"]["theme_element_ids"] = st.multiselect(
            "Theme Elements",
            options=comp_analyzer.get_field_options("theme_element_ids"),
            key="comp_themes",
            help="13 points - Theme matches (2.6 per match up to 5)"
        )
        
        # Tone (8 points)
        state["criteria"]["tone"] = st.selectbox(
            "Tone",
            options=comp_analyzer.get_field_options("tone"),
            key="comp_tone",
            help="8 points - Direct match on tone"
        )
        
        # Setting (7 points)
        col1, col2 = st.columns(2)
        with col1:
            state["criteria"]["time_setting"] = st.selectbox(
                "Time Period",
                options=comp_analyzer.get_field_options("time_setting"),
                key="comp_time",
                help="4 points - Time period match"
            )
        with col2:
            state["criteria"]["location"] = st.selectbox(
                "Location",
                options=comp_analyzer.get_field_options("location"),
                key="comp_location",
                help="3 points - Location match"
            )
        
    with st.expander("Production Match Criteria (13 pts)"):
        # Network (5 points)
        state["criteria"]["network_id"] = st.selectbox(
            "Network",
            options=comp_analyzer.get_field_options("network_id"),
            key="comp_network",
            help="5 points - Direct network match"
        )
        
        # Studios (3 points)
        state["criteria"]["studio_ids"] = st.multiselect(
            "Studios",
            options=comp_analyzer.get_field_options("studio_ids"),
            key="comp_studios",
            help="3 points - Studio matches (2 primary + 0.5 per additional up to 2)"
        )
        
        # Team Members (5 points)
        state["criteria"]["team_member_ids"] = st.multiselect(
            "Team Members",
            options=comp_analyzer.get_field_options("team_member_ids"),
            key="comp_team",
            help="5 points - Team member matches (1 point per match up to 5)"
        )

    with st.expander("Format Match Criteria (3 pts)"):
        col1, col2 = st.columns(2)
        
        # Episodes (2 points)
        with col1:
            state["criteria"]["episode_count"] = st.number_input(
                "Episode Count",
                min_value=1,
                max_value=100,
                value=10,
                key="comp_episodes",
                help="2 points - Episode count proximity (2 within ±2, 1.5 within ±4, 1 within ±6)"
            )
        
        # Order Type (1 point)
        with col2:
            state["criteria"]["order_type"] = st.selectbox(
                "Order Type",
                options=comp_analyzer.get_field_options("order_type"),
                key="comp_order",
                help="1 point - Direct order type match"
            )


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
