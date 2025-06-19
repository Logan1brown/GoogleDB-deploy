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


def render_recommendations(formatted_recommendations: Dict[str, Any]):
    """Render recommendations using pre-formatted data from OptimizerView.
    
    Args:
        formatted_recommendations: Dictionary with formatted recommendation data
    """
    try:
        # Check if there are any recommendations to display
        if not formatted_recommendations:
            st.info("No recommendations available.")
            return
        
        # Extract recommendation groups
        grouped = formatted_recommendations.get("grouped", {})
        network_specific = formatted_recommendations.get("network_specific", [])
        all_recs = formatted_recommendations.get("all", [])
        
        # Add minimal debug output if in debug mode
        if st.session_state.get('debug_mode', False):
            st.write(f"DEBUG: Recommendation groups: {[k for k, v in grouped.items() if v]}")
            st.write(f"DEBUG: Total recommendations: {len(all_recs)}")
            st.write(f"DEBUG: Network-specific recommendations: {len(network_specific)}")
        
        
        # Debug the raw recommendations (only log to debug system, not UI)
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Raw formatted recommendations keys: {list(formatted_recommendations.keys())}", category='recommendation')
            OptimizerConfig.debug(f"All recommendations count: {len(formatted_recommendations.get('all', []))}", category='recommendation', force=True)
            if "grouped" in formatted_recommendations:
                OptimizerConfig.debug(f"Grouped keys: {list(formatted_recommendations['grouped'].keys())}", category='recommendation')
                for group_key, group_items in formatted_recommendations['grouped'].items():
                    OptimizerConfig.debug(f"Group '{group_key}' has {len(group_items)} items", category='recommendation', force=True)
                    if group_items and len(group_items) > 0:
                        OptimizerConfig.debug(f"First item in group '{group_key}': {group_items[0]}", category='recommendation')
                
        # If there are recommendations but no grouped recommendations, create a default group
        if all_recs and (not grouped or all(len(recs) == 0 for recs in grouped.values())):
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Creating default 'add' group with {len(all_recs)} recommendations", category='recommendation', force=True)
            grouped = {"add": all_recs}
            
        # If still no recommendations to display after trying to create a default group
        if not grouped or all(len(recs) == 0 for recs in grouped.values()):
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("No recommendations available in any group", category='recommendation', force=True)
                OptimizerConfig.debug(f"Grouped keys: {list(grouped.keys())}", category='recommendation', force=True)
                for k, v in grouped.items():
                    OptimizerConfig.debug(f"Group '{k}' has {len(v)} items", category='recommendation', force=True)
            st.info("No recommendations available.")
            return
                
        # Check if there are any recommendations in any group
        has_recommendations = False
        for rec_type, recs in grouped.items():
            if recs:
                has_recommendations = True
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Found recommendations in group '{rec_type}': {len(recs)}", category='recommendation', force=True)
                break
                
        if not has_recommendations:
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("No recommendations found in any group", category='recommendation', force=True)
                OptimizerConfig.debug(f"Grouped keys: {list(grouped.keys())}", category='recommendation', force=True)
                for k, v in grouped.items():
                    OptimizerConfig.debug(f"Group '{k}' has {len(v)} items", category='recommendation', force=True)
            st.info("No recommendations available for your current criteria.")
            return
            
        # Get the grouped recommendations
        grouped = formatted_recommendations.get("grouped", {})
        
        # Debug the recommendations
        if OptimizerConfig.DEBUG_MODE or st.session_state.get('debug_mode', False):
            OptimizerConfig.debug(f"Rendering recommendations: {len(grouped)} groups", category='recommendation')
            st.write(f"DEBUG: Formatted recommendations structure")
            st.write(f"Total recommendations: {len(all_recs)}")
            st.write(f"Non-empty groups: {[k for k, v in grouped.items() if v]}")
            st.write(f"Network-specific recommendations: {len(network_specific)}")
            
            # Special debug for 'remove' recommendations
            remove_recs = grouped.get('remove', [])
            if remove_recs:
                st.write(f"DEBUG: Found {len(remove_recs)} 'remove' recommendations")
                for rec in remove_recs:
                    st.write(f"DEBUG: Remove recommendation: {rec.get('title', 'unknown')}")
            else:
                st.write(f"DEBUG: No 'remove' recommendations found in grouped dictionary")
                # Check if there are any 'remove' recommendations in all_recs
                remove_in_all = [rec for rec in all_recs if rec.get('category') == 'remove']
                if remove_in_all:
                    st.write(f"DEBUG: Found {len(remove_in_all)} 'remove' recommendations in all_recs but not in grouped")
                    for rec in remove_in_all:
                        st.write(f"DEBUG: Remove in all_recs: {rec.get('title', 'unknown')}")
                        # Add it to the grouped dictionary
                        if 'remove' not in grouped:
                            grouped['remove'] = []
                        grouped['remove'].append(rec)
                        
                        # Also add it to the appropriate criteria_type group when rendering
                        criteria_type = rec.get('criteria_type', 'unknown')
                        if criteria_type not in by_criteria_type:
                            by_criteria_type[criteria_type] = []
                        by_criteria_type[criteria_type].append(rec)
            for rec_type, recs in grouped.items():
                OptimizerConfig.debug(f"Group {rec_type}: {len(recs)} recommendations", category='recommendation')
                if recs:
                    st.write(f"DEBUG: Group '{rec_type}' has {len(recs)} recommendations")
                    if len(recs) > 0:
                        st.write(f"DEBUG: First item in group '{rec_type}': {recs[0].get('title', 'No title')} - Category: {recs[0].get('category', 'unknown')}")
        
        # Track if we've rendered any general recommendations
        general_recs_rendered = False
        network_recs_rendered = False
        
        # First, render all non-network recommendations
        # Check if we have any general recommendations
        general_recommendations = []
        for rec_type, recs in grouped.items():
            # Skip empty groups and network-specific recommendations
            if not recs or rec_type.startswith('network_'):
                continue
                
            # Make sure 'remove' recommendations are included
            if rec_type == 'remove' and OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Including {len(recs)} 'remove' recommendations in general_recommendations")
                
            general_recommendations.extend(recs)
                
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Found {len(general_recommendations)} general recommendations", category='recommendation', force=True)
            for rec in general_recommendations[:3]:
                OptimizerConfig.debug(f"General recommendation: {rec.get('title', 'unknown')}", category='recommendation')
                OptimizerConfig.debug(f"  - Category: {rec.get('category', 'unknown')}", category='recommendation')
                OptimizerConfig.debug(f"  - Impact: {rec.get('impact', 0)}", category='recommendation')
                OptimizerConfig.debug(f"  - Criteria Type: {rec.get('criteria_type', 'unknown')}", category='recommendation')
                
        if general_recommendations:
            # Render general recommendations
            st.subheader("General Recommendations")
            
            # Debug output for recommendations
            if OptimizerConfig.DEBUG_MODE or st.session_state.get('debug_mode', False):
                st.write(f"DEBUG: Processing {len(general_recommendations)} general recommendations")
            
            # Sort all recommendations by impact score regardless of category
            general_recommendations.sort(key=lambda x: abs(x.get('_impact_raw', 0)), reverse=True)
            
            # Group by criteria_type only, regardless of recommendation type
            # This ensures all recommendations for the same criteria type are grouped together
            by_criteria_type = {}
            
            # Count recommendations by type before grouping
            rec_type_counts = {'add': 0, 'change': 0, 'remove': 0}
            for rec in general_recommendations:
                rec_type = rec.get('category', 'unknown')
                if rec_type in rec_type_counts:
                    rec_type_counts[rec_type] += 1
            
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Recommendation counts by type: {rec_type_counts}", category='recommendation', force=True)
                
            # Process each recommendation for grouping
            for rec in general_recommendations:
                criteria_type = rec.get('criteria_type', 'unknown')
                rec_type = rec.get('category', 'unknown')
                
                # Debug output for each recommendation being processed
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Processing recommendation for grouping: {criteria_type}/{rec.get('title', 'unknown')} - Type: {rec_type}", category='recommendation')
                    if rec_type == 'remove':
                        OptimizerConfig.debug(f"FOUND REMOVE REC FOR GROUPING: {criteria_type}/{rec.get('title', 'unknown')}", category='recommendation', force=True)
                
                # Group by criteria_type only, not by recommendation type
                # This ensures 'add', 'change', and 'remove' recommendations for the same criteria
                # are displayed together
                if criteria_type not in by_criteria_type:
                    by_criteria_type[criteria_type] = []
                by_criteria_type[criteria_type].append(rec)
            
            # Debug output for criteria types
            if OptimizerConfig.DEBUG_MODE or st.session_state.get('debug_mode', False):
                st.write(f"DEBUG: Found {len(by_criteria_type)} criteria types: {list(by_criteria_type.keys())}")
            
            # Render recommendations grouped by criteria type
            for criteria_type, criteria_recs in by_criteria_type.items():
                # We're now using the actual criteria_type directly since we're grouping only by criteria_type
                actual_criteria_type = criteria_type
                
                # Sort recommendations within each criteria type by impact (absolute value)
                criteria_recs.sort(key=lambda x: abs(x.get('_impact_raw', 0)), reverse=True)
                
                # Debug output for criteria type and recommendations
                if OptimizerConfig.DEBUG_MODE or st.session_state.get('debug_mode', False):
                    # Count recommendations by type within this criteria group
                    rec_types = {}
                    for rec in criteria_recs:
                        rec_type = rec.get('category', 'unknown')
                        if rec_type not in rec_types:
                            rec_types[rec_type] = 0
                        rec_types[rec_type] += 1
                    
                    # Display the count of each recommendation type
                    st.write(f"DEBUG: Rendering criteria_type '{actual_criteria_type}' with {len(criteria_recs)} recommendations")
                    st.write(f"DEBUG: Recommendation types: {rec_types}")
                    
                    # Special debug for 'remove' recommendations
                    if 'remove' in rec_types and rec_types['remove'] > 0:
                        st.write(f"DEBUG: Found {rec_types['remove']} 'remove' recommendations for '{actual_criteria_type}'")
                        for rec in [r for r in criteria_recs if r.get('category') == 'remove'][:2]:
                            st.write(f"DEBUG: Remove rec - {rec.get('title')}: {rec.get('description')}")
                    
                
                # Render each recommendation (limit to top 3 per criteria type)
                for rec in criteria_recs[:3]:
                            # Determine if this is a positive or negative recommendation based on impact
                            is_negative = rec.get('impact', 0) < 0
                            
                            # Get the title and description directly without modification
                            title = rec.get('title', '')
                            description = rec.get('description', '')
                            
                            # Ensure the title reflects the correct recommendation type
                            rec_type = rec.get('category', 'add')
                            if rec_type == 'change' and 'Adding' in description:
                                # Replace 'Adding' with 'Changing to' in the description
                                description = description.replace('Adding', 'Changing to')
                            
                            if is_negative:
                                # Use warning style for negative recommendations
                                st.markdown(f"""
                                <div style="border: 1px solid #f77; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: #fff8f8;">
                                    <p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #c00;">{title}</p>
                                    <p style="font-size: 12px; margin: 0; color: #333;">{description}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Use info card style for positive recommendations
                                render_info_card(title, description)
                        
            general_recs_rendered = True
        # This else clause is no longer needed as we're handling all recommendation types above
        # But we'll keep it as a fallback just in case
        else:
            # Debug what's happening
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Falling back to old rendering method", category='recommendation', force=True)
                OptimizerConfig.debug(f"Groups: {list(grouped.keys())}", category='recommendation', force=True)
                for k, v in grouped.items():
                    OptimizerConfig.debug(f"Group '{k}' has {len(v)} items", category='recommendation', force=True)
                    
            # Use the old rendering method as a fallback
            for rec_type, recs in grouped.items():
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Checking group '{rec_type}' with {len(recs)} recommendations", category='recommendation')
                    
                if recs and not rec_type.startswith('network_'):
                    if OptimizerConfig.DEBUG_MODE:
                        OptimizerConfig.debug(f"Rendering group '{rec_type}' with {len(recs)} recommendations", category='recommendation')
                        if recs:
                            OptimizerConfig.debug(f"First recommendation in group: {recs[0]}", category='recommendation')
                            
                    render_recommendation_group(rec_type, recs)
                    general_recs_rendered = True
        
        # Then render network-specific recommendations
        network_specific_recs = []
        for rec_type, recs in grouped.items():
            if recs and rec_type.startswith('network_'):
                network_specific_recs.extend(recs)
                
        if network_specific_recs:
            # Group by network name for better organization
            by_network = {}
            for rec in network_specific_recs:
                # Extract network name from category or title
                network_name = rec.get('category', '').replace('network_', '').replace('_', ' ').title()
                if not network_name and 'title' in rec:
                    # Try to extract from title
                    title = rec.get('title', '')
                    if ':' in title:
                        network_name = title.split(':', 1)[0].strip()
                        
                if not network_name:
                    network_name = 'Unknown Network'
                    
                if network_name not in by_network:
                    by_network[network_name] = []
                by_network[network_name].append(rec)
                
            # Sort by impact score within each network
            for network, recs in by_network.items():
                recs.sort(key=lambda x: abs(x.get('_impact_raw', 0)), reverse=True)
                
            # Render network-specific recommendations
            st.subheader("Network-Specific Recommendations")
            for network, recs in by_network.items():
                st.write(f"**{network} Recommendations:**")
                for rec in recs[:3]:  # Limit to top 3 per network
                    # Determine if this is a positive or negative recommendation based on impact
                    is_negative = rec.get('impact', 0) < 0
                    
                    # Get the title and description directly without modification
                    title = rec.get('title', '')
                    description = rec.get('description', '')
                    
                    if is_negative:
                        # Use warning style for negative recommendations
                        st.markdown(f"""
                        <div style="border: 1px solid #f77; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: #fff8f8;">
                            <p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #c00;">{title}</p>
                            <p style="font-size: 12px; margin: 0; color: #333;">{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Use info card style for positive recommendations
                        render_info_card(title, description)
                        
            network_recs_rendered = True
        
        # Check if we actually rendered any recommendations
        if not general_recs_rendered:
            st.info("No general recommendations available for your current criteria.")
            
        if not network_recs_rendered:
            st.info("No network-specific recommendations available for your current criteria.")
            
    except Exception as e:
        OptimizerConfig.debug(f"Error rendering recommendations: {str(e)}", category='recommendation', force=True)
        st.error("Unable to display recommendations due to an error.")
        import traceback
        OptimizerConfig.debug(traceback.format_exc(), category='recommendation', force=True)


def render_recommendation_group(rec_type: str, recommendations: List[Dict[str, Any]], limit: int = 3):
    """
Render a group of recommendations with appropriate UI elements.
    
Args:
        rec_type: Type of recommendation (add, change, remove, consider, etc.)
        recommendations: List of pre-formatted recommendations of this type
        limit: Maximum number of recommendations to show
    """
    if OptimizerConfig.DEBUG_MODE:
        st.write(f"DEBUG: render_recommendation_group called with type '{rec_type}' and {len(recommendations)} recommendations")
        if recommendations:
            st.write(f"DEBUG: First recommendation keys: {list(recommendations[0].keys())}")
    
    if not recommendations:
        if OptimizerConfig.DEBUG_MODE:
            st.write(f"DEBUG: No recommendations for group '{rec_type}'")
        return
    
    # Get recommendation type display names from config
    config = OptimizerConfig()
    
    # Set header based on recommendation type
    if rec_type == 'add':
        header = "Add to Your Concept"
    elif rec_type == 'remove':
        header = "Consider Removing"
    elif rec_type == 'replace':
        header = "Consider Replacing"
    elif rec_type == 'change':
        header = "Consider Changing"
    elif rec_type.startswith('network_'):
        network_name = rec_type.replace('network_', '').replace('_', ' ').title()
        header = f"{network_name} Network Recommendations"
    elif rec_type == "network_keep":
        header = config.RECOMMENDATION_TYPES.get('network_keep', "Network Strengths")
    elif rec_type == "network_change":
        header = config.RECOMMENDATION_TYPES.get('network_change', "Network Adjustments")
    else:
        # For any other recommendation type, use a generic header
        header = f"{rec_type.replace('_', ' ').title()} Recommendations"
        
    # Display the header
    st.subheader(header)
    
    # Display recommendations
    for rec in recommendations[:limit]:
        # Debug the recommendation structure if in debug mode
        if OptimizerConfig.DEBUG_MODE:
            st.write(f"DEBUG: Rendering recommendation: {rec}")
        
        # Use info card style for all recommendations, but only show the description
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: #f9f9f9;">
            <p style="font-size: 14px; margin: 0; color: #333;">{rec.get('description', '')}</p>
        </div>
        """, unsafe_allow_html=True)
