"""Show Optimizer Helper Functions.

This module contains helper functions for the Show Optimizer UI components.
These functions handle common patterns for field rendering and visualization.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from src.data_processing.show_optimizer.optimizer_config import OptimizerConfig
from src.dashboard.utils.style_config import render_metric_card, render_info_card, COLORS


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
        
        # Subgenre selection
        subgenre_names = st.multiselect("Subgenres",
            options=[name for _, name in display_options.get('subgenres', []) if name and name.strip()],
            key="optimizer_subgenres", placeholder="Select subgenres...",
            on_change=lambda: update_callback("subgenres", 
                            get_ids_for_names(st.session_state.optimizer_subgenres, 
                                            display_options.get('subgenres', []))))
        
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

# Visualization helper functions are imported from style_config.py


def render_success_metrics(summary: Any):
    """Render success probability metrics using pre-formatted data from OptimizerView.
    
    Args:
        summary: Optimization summary with formatted_data attribute
    """
    try:
        # Get config for consistent display
        config = OptimizerConfig()
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Check if we have pre-formatted component scores
        has_formatted_data = (hasattr(summary, 'formatted_data') and 
                            'component_scores' in summary.formatted_data)
        
        if not has_formatted_data:
            # No formatted data available - show N/A for all metrics
            with col1:
                render_metric_card("Success Probability", "N/A", "Data unavailable")
            with col2:
                render_metric_card("Audience Appeal", "N/A", "Data unavailable")
            with col3:
                render_metric_card("Critical Reception", "N/A", "Data unavailable")
            with col4:
                render_metric_card("Longevity", "N/A", "Data unavailable")
            return
        
        # Success Probability
        with col1:
            if 'success_probability' in summary.formatted_data:
                # Use pre-formatted success probability data
                success_data = summary.formatted_data['success_probability']
                render_metric_card(
                    "Success Probability", 
                    success_data.get('display', 'N/A'), 
                    success_data.get('subtitle', 'Data unavailable')
                )
            else:
                render_metric_card("Success Probability", "N/A", "Data unavailable")
        
        # Audience Score
        with col2:
            if 'audience' in summary.formatted_data['component_scores']:
                # Use pre-formatted audience score data
                audience_data = summary.formatted_data['component_scores']['audience']
                score = audience_data.get('score')
                sample_size = audience_data.get('sample_size', 'N/A')
                
                if score is not None:
                    render_metric_card(
                        "Audience Appeal", 
                        f"{score:.0%}", 
                        f"Sample: {sample_size}"
                    )
                else:
                    render_metric_card("Audience Appeal", "N/A", "Data unavailable")
            else:
                render_metric_card("Audience Appeal", "N/A", "Data unavailable")
        
        # Critics Score
        with col3:
            if 'critics' in summary.formatted_data['component_scores']:
                # Use pre-formatted critics score data
                critics_data = summary.formatted_data['component_scores']['critics']
                score = critics_data.get('score')
                sample_size = critics_data.get('sample_size', 'N/A')
                
                if score is not None:
                    render_metric_card(
                        "Critical Reception", 
                        f"{score:.0%}", 
                        f"Sample: {sample_size}"
                    )
                else:
                    render_metric_card("Critical Reception", "N/A", "Data unavailable")
            else:
                render_metric_card("Critical Reception", "N/A", "Data unavailable")
        
        # Longevity Score
        with col4:
            if 'longevity' in summary.formatted_data['component_scores']:
                # Use pre-formatted longevity score data
                longevity_data = summary.formatted_data['component_scores']['longevity']
                score = longevity_data.get('score')
                sample_size = longevity_data.get('sample_size', 'N/A')
                
                if score is not None:
                    render_metric_card(
                        "Longevity", 
                        f"{score:.0%}", 
                        f"Sample: {sample_size}"
                    )
                else:
                    render_metric_card("Longevity", "N/A", "Data unavailable")
            else:
                render_metric_card("Longevity", "N/A", "Data unavailable")
    
    except Exception as e:
        OptimizerConfig.debug(f"Error rendering success metrics: {str(e)}", category='recommendation', force=True)
        st.error("Unable to display success metrics due to an error.")
        # Render empty cards as fallback
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Success Probability", "N/A", "Error in data")
        with col2:
            render_metric_card("Audience Appeal", "N/A", "Error in data")
        with col3:
            render_metric_card("Critical Reception", "N/A", "Error in data")
        with col4:
            render_metric_card("Longevity", "N/A", "Error in data")


def render_success_factors(formatted_factors: List[Dict[str, Any]]):
    """Render success factors chart using pre-formatted data from OptimizerView.
    
    Args:
        formatted_factors: List of pre-formatted success factor dictionaries
    """
    if not formatted_factors:
        st.info("No significant success factors identified.")
        return
        
    # Create a dataframe from the pre-formatted data
    factor_df = pd.DataFrame(formatted_factors)
    
    # Create a bar chart with sample size information
    chart = alt.Chart(factor_df).mark_bar().encode(
        x=alt.X('Impact:Q', title='Impact on Success'),
        y=alt.Y('Name:N', title=None, sort='-x'),
        color=alt.Color('Impact:Q', scale=alt.Scale(
            domain=[-0.5, 0, 0.5],
            range=['#f77', '#ddd', '#7d7']
        )),
        tooltip=['Type', 'Name', 'ImpactDisplay', 'Confidence', 'SampleDisplay']
    ).properties(
        height=30 * len(formatted_factors)
    )
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Add a note about sample sizes
    st.caption("Note: Sample size indicates the number of shows matching each criteria. Larger samples provide more reliable insights.")
    
    # Display a table with detailed information including sample sizes
    with st.expander("View detailed success factors data"):
        # Create a more readable table for display
        display_df = factor_df[['Name', 'Type', 'ImpactDisplay', 'Sample', 'Confidence']].copy()
        display_df.columns = ['Factor', 'Category', 'Impact Score', 'Sample Size', 'Confidence']
        st.dataframe(display_df, use_container_width=True)
        
    # Display matching show titles in an expander
    for factor in formatted_factors:
        matching_titles = factor.get('_matching_titles', [])
        if matching_titles:
            with st.expander(f"Shows matching '{factor['Name']}' ({len(matching_titles)} shows)"):
                # Display titles in a scrollable container with fixed height
                titles_html = "<div style='max-height: 300px; overflow-y: auto;'><ul>"
                for title in matching_titles:
                    titles_html += f"<li>{title}</li>"
                titles_html += "</ul></div>"
                st.markdown(titles_html, unsafe_allow_html=True)


def group_recommendations(recommendations: List) -> Dict[str, List]:
    """Group recommendations by type.
    
    Args:
        recommendations: List of recommendations
        
    Returns:
        Dictionary of recommendation types to lists of recommendations
    """
    # Initialize with keys from OptimizerConfig.RECOMMENDATION_TYPES
    config = OptimizerConfig()
    grouped = {rec_type: [] for rec_type in config.RECOMMENDATION_TYPES.keys()}
    
    # Add backward compatibility for 'remove' type which might be used in UI
    if 'remove' not in grouped:
        grouped['remove'] = []
    
    for rec in recommendations:
        # Check if this is a valid Recommendation object with required attributes
        if not hasattr(rec, 'recommendation_type') and not hasattr(rec, 'rec_type'):
            # Skip invalid recommendations
            continue
            
        # Get recommendation type, defaulting to rec_type if recommendation_type doesn't exist
        rec_type = getattr(rec, 'recommendation_type', getattr(rec, 'rec_type', None))
        
        if rec_type and rec_type in grouped:
            grouped[rec_type].append(rec)
        elif rec_type:
            if rec_type not in grouped:
                grouped[rec_type] = []
            grouped[rec_type].append(rec)
            
    return grouped


def render_recommendations(formatted_recommendations: Dict[str, Any], on_click_handler=None):
    """Render recommendations using pre-formatted data from OptimizerView.
    
    Args:
        formatted_recommendations: Dictionary with formatted recommendation data
        on_click_handler: Function to call when recommendation button is clicked
    """
    try:
        # Check if we have any recommendations
        if not formatted_recommendations or not formatted_recommendations.get("grouped"):
            st.info("No recommendations available.")
            return
            
        # Get the grouped recommendations
        grouped = formatted_recommendations.get("grouped", {})
        
        # Render each group that has recommendations
        for rec_type, recs in grouped.items():
            if recs:  # Only render groups with recommendations
                render_recommendation_group(rec_type, recs, on_click_handler)
                
        # Check if we actually rendered any recommendations
        if all(len(recs) == 0 for recs in grouped.values()):
            st.info("No recommendations available.")
            
    except Exception as e:
        OptimizerConfig.debug(f"Error rendering recommendations: {str(e)}", category='recommendation', force=True)
        st.error("Unable to display recommendations due to an error.")
        # No fallback UI needed as the error message is sufficient


def render_recommendation_group(rec_type: str, recommendations: List[Dict[str, Any]], on_click_handler=None, limit: int = 3):
    """Render a group of recommendations with appropriate UI elements.
    
    Args:
        rec_type: Type of recommendation (add, replace, remove, consider, etc.)
        recommendations: List of pre-formatted recommendations of this type
        on_click_handler: Function to call when recommendation button is clicked
        limit: Maximum number of recommendations to show
    """
    if not recommendations:
        return
    
    # Get recommendation type display names from config
    config = OptimizerConfig()
    
    # Set up headers and UI based on recommendation type
    if rec_type == "add":
        st.subheader(config.RECOMMENDATION_TYPES.get('add', "Consider Adding"))
        button_prefix = "Add"
        use_button = True
        use_info_card = True
    # 'replace' type has been consolidated into 'change'
    elif rec_type == "remove":
        st.subheader("Consider Removing")
        button_prefix = "Remove"
        use_button = True
        use_info_card = False
    # 'relax' type has been consolidated into 'remove'
    elif rec_type == "change":
        st.subheader(config.RECOMMENDATION_TYPES.get('change', "Successful Pattern"))
        button_prefix = "Change to"
        use_button = True
        use_info_card = True
    elif rec_type == "consider":
        st.subheader(config.RECOMMENDATION_TYPES.get('consider', "Additional Insights"))
        use_button = False
        use_info_card = True
    elif rec_type == "fallback":
        st.subheader(config.RECOMMENDATION_TYPES.get('fallback', "Fallback Recommendation"))
        button_prefix = "Add"
        use_button = True
        use_info_card = True
    elif rec_type == "network_keep":
        st.subheader(config.RECOMMENDATION_TYPES.get('network_keep', "Network Strengths"))
        button_prefix = "Keep"
        use_button = False
        use_info_card = True
    elif rec_type == "network_change":
        st.subheader(config.RECOMMENDATION_TYPES.get('network_change', "Network Adjustments"))
        button_prefix = "Adjust"
        use_button = True
        use_info_card = True
    else:
        # For any other recommendation type, use a generic header
        st.subheader(f"{rec_type.replace('_', ' ').title()} Recommendations")
        button_prefix = "Apply"
        use_button = True
        use_info_card = True
    
    # Display recommendations
    for rec in recommendations[:limit]:
        if use_button:
            col1, col2 = st.columns([1, 3])
            with col1:
                if on_click_handler:
                    st.button(
                        f"{button_prefix} {rec['suggested_name']}",
                        key=f"{rec_type}_{rec['criteria_type']}_{rec['suggested_value'] or rec['current_value']}",
                        on_click=on_click_handler,
                        args=(rec,)
                    )
            with col2:
                if use_info_card:
                    render_info_card(
                        rec['title'],
                        rec['description']
                    )
                else:
                    # For remove recommendations, use warning style
                    st.markdown(f"""
                    <div style="border: 1px solid #f77; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: #fff8f8;">
                        <p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #c00;">{rec['title']}</p>
                        <p style="font-size: 14px; margin-top: 0;">{rec['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Just show info card with no button
            render_info_card(
                rec['title'],
                rec['description']
            )
