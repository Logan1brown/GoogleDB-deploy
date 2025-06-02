"""Show Optimizer Helper Functions.

This module contains helper functions for the Show Optimizer UI components.
These functions handle common patterns for field rendering and visualization.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable


def get_id_for_name(name: str, options: List[Tuple[int, str]]) -> Optional[int]:
    """Get ID for a display name.
    
    Args:
        name: Display name to look up
        options: List of (id, name) tuples
        
    Returns:
        ID if found, None otherwise
    """
    if not name:
        return None
    for id, opt_name in options:
        if opt_name == name:
            return id
    return None


def get_ids_for_names(names: List[str], options: List[Tuple[int, str]], field_name: str = None, optimizer = None) -> List[int]:
    """Get IDs for display names.
    
    Args:
        names: List of display names to look up
        options: List of (id, name) tuples
        field_name: Name of the field (used to identify team members)
        optimizer: Optional ShowOptimizer instance to use for team member lookups
        
    Returns:
        List of IDs for the given names
    """
    if not names:
        return []
        
    # For team members, get all IDs for each name (similar to comp builder)
    if field_name == 'team_members' and optimizer and hasattr(optimizer, 'field_manager') and optimizer.field_manager is not None:
        try:
            # Get all IDs for selected names
            all_ids = []
            for name in names:
                # Find the option with this name
                opt = next((opt for opt in optimizer.field_manager.get_options('team_members') 
                          if opt.name == name), None)
                if opt and hasattr(opt, 'all_ids'):
                    all_ids.extend(opt.all_ids)
                elif opt:
                    all_ids.append(opt.id)
            return all_ids
        except (AttributeError, TypeError):
            # Fallback to basic ID lookup if field_manager.get_options fails
            pass
    
    # For other fields, just take the first ID for each name
    result = []
    for name in names:
        id = get_id_for_name(name, options)
        if id is not None:
            result.append(id)
    return result


def render_content_criteria(state: Dict, update_callback: Callable) -> None:
    """Render content criteria section.
    
    Args:
        state: State dictionary containing criteria and display options
        update_callback: Callback function for updating criteria
    """
    criteria = state.get('criteria', {})
    display_options = state.get('display_options', {})
    
    with st.expander("Content Criteria (82 pts)", expanded=True):
        st.markdown("### Content")
        
        # Genre selection
        genre_name = st.selectbox("Genre",
            options=[name for _, name in display_options.get('genre', []) if name and name.strip()],
            key="optimizer_genre", index=None, placeholder="Select genre...",
            on_change=lambda: update_callback("genre", 
                                get_id_for_name(st.session_state.optimizer_genre, display_options.get('genre', [])) 
                                if st.session_state.optimizer_genre else None))
        
        # Subgenre selection (if available)
        if 'subgenres' in display_options:
            subgenre_names = st.multiselect("Subgenres",
                options=[name for _, name in display_options['subgenres'] if name and name.strip()],
                key="optimizer_subgenres", placeholder="Select subgenres...",
                on_change=lambda: update_callback("subgenres", 
                                get_ids_for_names(st.session_state.optimizer_subgenres, 
                                                display_options['subgenres'])))
        
        # Source type selection
        source_name = st.selectbox("Source Type",
            options=[name for _, name in display_options.get('source_type', []) if name and name.strip()],
            key="optimizer_source_type", index=None, placeholder="Select source type...",
            on_change=lambda: update_callback("source_type", 
                                get_id_for_name(st.session_state.optimizer_source_type, display_options.get('source_type', [])) 
                                if st.session_state.optimizer_source_type else None))
        
        # Character types selection
        char_names = st.multiselect("Character Types",
            options=[name for _, name in display_options.get('character_types', []) if name and name.strip()],
            key="optimizer_character_types", placeholder="Select character types...",
            on_change=lambda: update_callback("character_types", 
                                get_ids_for_names(st.session_state.optimizer_character_types, 
                                                display_options.get('character_types', []))))
        
        # Thematic elements selection
        theme_names = st.multiselect("Thematic Elements",
            options=[name for _, name in display_options.get('thematic_elements', []) if name and name.strip()],
            key="optimizer_thematic_elements", placeholder="Select thematic elements...",
            on_change=lambda: update_callback("thematic_elements", 
                                get_ids_for_names(st.session_state.optimizer_thematic_elements, 
                                                display_options.get('thematic_elements', []))))
        
        # Plot elements selection
        plot_names = st.multiselect("Plot Elements",
            options=[name for _, name in display_options.get('plot_elements', []) if name and name.strip()],
            key="optimizer_plot_elements", placeholder="Select plot elements...",
            on_change=lambda: update_callback("plot_elements", 
                                get_ids_for_names(st.session_state.optimizer_plot_elements, 
                                                display_options.get('plot_elements', []))))
        
        # Tone selection
        tone_names = st.multiselect("Tone",
            options=[name for _, name in display_options.get('tone', []) if name and name.strip()],
            key="optimizer_tone", placeholder="Select tone...",
            on_change=lambda: update_callback("tone", 
                                get_ids_for_names(st.session_state.optimizer_tone, 
                                                display_options.get('tone', []))))
        
        # Time setting selection
        time_names = st.multiselect("Time Setting",
            options=[name for _, name in display_options.get('time_setting', []) if name and name.strip()],
            key="optimizer_time_setting", placeholder="Select time setting...",
            on_change=lambda: update_callback("time_setting", 
                                get_ids_for_names(st.session_state.optimizer_time_setting, 
                                                display_options.get('time_setting', []))))
        
        # Location setting selection
        location_names = st.multiselect("Location Setting",
            options=[name for _, name in display_options.get('location_setting', []) if name and name.strip()],
            key="optimizer_location_setting", placeholder="Select location setting...",
            on_change=lambda: update_callback("location_setting", 
                                get_ids_for_names(st.session_state.optimizer_location_setting, 
                                                display_options.get('location_setting', []))))


def render_production_criteria(state: Dict, update_callback: Callable) -> None:
    """Render production criteria section.
    
    Args:
        state: State dictionary containing criteria and display options
        update_callback: Callback function for updating criteria
    """
    criteria = state.get('criteria', {})
    display_options = state.get('display_options', {})
    
    with st.expander("Production Criteria (13 pts)", expanded=False):
        st.markdown("### Production")
        
        # Network selection
        network_name = st.selectbox("Network",
            options=[name for _, name in display_options.get('network', []) if name and name.strip()],
            key="optimizer_network", index=None, placeholder="Select network...",
            on_change=lambda: update_callback("network", 
                                get_id_for_name(st.session_state.optimizer_network, display_options.get('network', [])) 
                                if st.session_state.optimizer_network else None))
        
        # Studios selection
        studio_names = st.multiselect("Studios",
            options=[name for _, name in display_options.get('studios', []) if name and name.strip()],
            key="optimizer_studios", placeholder="Select studios...",
            on_change=lambda: update_callback("studios", 
                                get_ids_for_names(st.session_state.optimizer_studios, 
                                                display_options.get('studios', []))))
        
        # Team members selection
        team_names = st.multiselect("Team Members",
            options=[name for _, name in display_options.get('team_members', []) if name and name.strip()],
            key="optimizer_team_members", placeholder="Select team members...",
            on_change=lambda: update_callback("team_members", 
                                get_ids_for_names(st.session_state.optimizer_team_members, 
                                                display_options.get('team_members', []))))


def render_format_criteria(state: Dict, update_callback: Callable) -> None:
    """Render format criteria section.
    
    Args:
        state: State dictionary containing criteria and display options
        update_callback: Callback function for updating criteria
    """
    criteria = state.get('criteria', {})
    display_options = state.get('display_options', {})
    
    with st.expander("Format Criteria (5 pts)", expanded=False):
        st.markdown("### Format")
        
        # Episode count
        eps = st.number_input("Episode Count", min_value=0, value=None, placeholder="Enter episode count...",
                            key="optimizer_episode_count",
                            on_change=lambda: update_callback("episode_count", 
                                                          st.session_state.optimizer_episode_count 
                                                          if st.session_state.optimizer_episode_count and 
                                                             st.session_state.optimizer_episode_count > 0 
                                                          else None))
        
        # Order Type
        if 'order_type' in display_options:
            order_name = st.selectbox("Order Type",
                options=[name for _, name in display_options['order_type'] if name and name.strip()],
                key="optimizer_order_type", index=None, placeholder="Select order type...",
                on_change=lambda: update_callback("order_type", 
                                    get_id_for_name(st.session_state.optimizer_order_type, display_options['order_type']) 
                                    if st.session_state.optimizer_order_type else None))
   

# Helper functions for rendering tab content

# Visualization helper functions
def render_metric_card(title: str, value: str, subtitle: str = None):
    """Render a metric card with title, value, and optional subtitle.
    
    Args:
        title: Card title
        value: Main value to display
        subtitle: Optional subtitle
    """
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
        <p style="font-size: 14px; color: #666; margin-bottom: 0;">{title}</p>
        <h3 style="font-size: 24px; margin: 5px 0;">{value}</h3>
        {f'<p style="font-size: 12px; color: #888; margin-top: 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_info_card(title: str, content: str):
    """Render an info card with title and content.
    
    Args:
        title: Card title
        content: Card content
    """
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
        <p style="font-size: 14px; font-weight: bold; margin-bottom: 5px;">{title}</p>
        <p style="font-size: 14px; margin-top: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_success_metrics(summary: Any):
    """Render success probability metrics.
    
    Args:
        summary: Optimization summary
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check for overall_success_probability attribute
        if hasattr(summary, 'overall_success_probability'):
            probability = summary.overall_success_probability
            confidence = summary.confidence if hasattr(summary, 'confidence') else 'medium'
            
            render_metric_card(
                "Success Probability", 
                f"{probability:.0%}", 
                f"Confidence: {confidence.capitalize()}"
            )
        else:
            st.info("Overall success probability not available.")
    
    # Component scores
    with col2:
        # Check if component_scores exists
        if hasattr(summary, 'component_scores') and summary.component_scores:
            audience_score = summary.component_scores.get("audience", None)
            if audience_score:
                # Check if audience_score has score attribute
                if hasattr(audience_score, 'score'):
                    score = audience_score.score
                    confidence = audience_score.confidence if hasattr(audience_score, 'confidence') else 'medium'
                    
                    render_metric_card(
                        "Audience Appeal", 
                        f"{score:.0%}", 
                        f"Confidence: {confidence.capitalize()}"
                    )
        
    with col3:
        # Check if component_scores exists
        if hasattr(summary, 'component_scores') and summary.component_scores:
            critic_score = summary.component_scores.get("critics", None)
            if critic_score:
                # Check if critic_score has score attribute
                if hasattr(critic_score, 'score'):
                    score = critic_score.score
                    confidence = critic_score.confidence if hasattr(critic_score, 'confidence') else 'medium'
                    
                    render_metric_card(
                        "Critical Reception", 
                        f"{score:.0%}", 
                        f"Confidence: {confidence.capitalize()}"
                    )


def render_success_factors(success_factors: List):
    """Render success factors chart.
    
    Args:
        success_factors: List of success factors
    """
    if not success_factors:
        st.info("No significant success factors identified.")
        return
        
    # Create a dataframe for the factors
    factor_data = []
    for factor in success_factors:
        factor_data.append({
            "Type": factor.criteria_type.replace("_", " ").title(),
            "Name": factor.criteria_name,
            "Impact": factor.impact_score,
            "Confidence": factor.confidence.capitalize()
        })
        
    factor_df = pd.DataFrame(factor_data)
    
    # Create a bar chart
    chart = alt.Chart(factor_df).mark_bar().encode(
        x=alt.X('Impact:Q', title='Impact on Success'),
        y=alt.Y('Name:N', title=None, sort='-x'),
        color=alt.Color('Impact:Q', scale=alt.Scale(
            domain=[-0.5, 0, 0.5],
            range=['#f77', '#ddd', '#7d7']
        )),
        tooltip=['Type', 'Name', 'Impact', 'Confidence']
    ).properties(
        height=30 * len(factor_data)
    )
    
    st.altair_chart(chart, use_container_width=True)


def render_network_compatibility(networks: List):
    """Render network compatibility table.
    
    Args:
        networks: List of network matches
    """
    if not networks:
        st.info("No network compatibility data available.")
        return
        
    # Create a dataframe for the networks
    network_data = []
    for network in networks:
        network_data.append({
            "Network": network.network_name,
            "Success Probability": network.success_probability,
            "Compatibility": network.compatibility_score,
            "Sample Size": network.sample_size,
            "Confidence": network.confidence.capitalize()
        })
        
    network_df = pd.DataFrame(network_data)
    
    # Display as a table
    st.dataframe(
        network_df,
        column_config={
            "Success Probability": st.column_config.ProgressColumn(
                "Success Probability",
                format="%.0f%%",
                min_value=0,
                max_value=1
            ),
            "Compatibility": st.column_config.ProgressColumn(
                "Compatibility",
                format="%.0f%%",
                min_value=0,
                max_value=1
            )
        },
        hide_index=True
    )


def group_recommendations(recommendations: List) -> Dict[str, List]:
    """Group recommendations by type.
    
    Args:
        recommendations: List of recommendations
        
    Returns:
        Dictionary of recommendation types to lists of recommendations
    """
    grouped = {
        "add": [],
        "replace": [],
        "remove": [],
        "consider": []
    }
    
    for rec in recommendations:
        if rec.recommendation_type in grouped:
            grouped[rec.recommendation_type].append(rec)
            
    return grouped


def render_recommendation_group(rec_type: str, recommendations: List, on_click_handler=None, limit: int = 3):
    """Render a group of recommendations with appropriate UI elements.
    
    Args:
        rec_type: Type of recommendation (add, replace, remove, consider)
        recommendations: List of recommendations of this type
        on_click_handler: Function to call when recommendation button is clicked
        limit: Maximum number of recommendations to show
    """
    if not recommendations:
        return
        
    # Set up headers and UI based on recommendation type
    if rec_type == "add":
        st.subheader("Consider Adding")
        button_prefix = "Add"
        use_button = True
        use_info_card = True
    elif rec_type == "replace":
        st.subheader("Consider Replacing")
        button_prefix = "Replace with"
        use_button = True
        use_info_card = True
    elif rec_type == "remove":
        st.subheader("Consider Removing")
        button_prefix = "Remove"
        use_button = True
        use_info_card = False
    elif rec_type == "consider":
        st.subheader("Additional Insights")
        use_button = False
        use_info_card = True
    else:
        return
    
    # Display recommendations
    for rec in recommendations[:limit]:
        if use_button:
            col1, col2 = st.columns([1, 3])
            with col1:
                if on_click_handler:
                    st.button(
                        f"{button_prefix} {rec.suggested_name}",
                        key=f"{rec_type}_{rec.criteria_type}_{rec.suggested_value or rec.current_value}",
                        on_click=on_click_handler,
                        args=(rec,)
                    )
            with col2:
                if use_info_card:
                    render_info_card(
                        f"{rec.criteria_type.replace('_', ' ').title()}: {rec.suggested_name}",
                        rec.explanation
                    )
                else:
                    # For remove recommendations, use warning style
                    st.markdown(f"""
                    <div style="border: 1px solid #f77; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: #fff8f8;">
                        <p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #c00;">{rec.criteria_type.replace('_', ' ').title()}: {rec.suggested_name}</p>
                        <p style="font-size: 14px; margin-top: 0;">{rec.explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Just show info card with no button
            render_info_card(
                f"{rec.criteria_type.replace('_', ' ').title()}: {rec.suggested_name}",
                rec.explanation
            )
