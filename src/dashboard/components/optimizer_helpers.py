"""Show Optimizer Helper Functions.

This module contains helper functions for the Show Optimizer UI components.
These functions handle common patterns for field rendering and visualization.
"""

import streamlit as st
import pandas as pd
import altair as alt
import traceback
from typing import List, Dict, Tuple, Any, Callable, Optional, Union

# Import style config for rendering
from src.dashboard.utils.style_config import render_metric_card, render_info_card, COLORS
from src.data_processing.show_optimizer.optimizer_config import OptimizerConfig
from src.data_processing.show_optimizer.show_optimizer import ShowOptimizer
from src.data_processing.show_optimizer.optimizer_data_contracts import CriteriaDict, ConfidenceInfo

# Access the ShowOptimizer instance from session state
def get_optimizer():
    """Get the ShowOptimizer instance from session state.
    
    Returns:
        ShowOptimizer instance from session state
    """
    return st.session_state['show_optimizer']


def render_content_criteria(state: Dict, update_callback: Callable) -> None:
    """Render content criteria section.
    
    Args:
        state: State dictionary containing criteria and display options
        update_callback: Callback function for updating criteria
    """
    criteria = state['criteria']
    display_options = state['display_options']
    
    with st.expander("Content Criteria (82 pts)", expanded=True):
        st.markdown("### Content")
        
        # Genre selection
        genre_name = st.selectbox("Genre",
            options=[name for _, name in display_options['genre'] if name and name.strip()],
            key="optimizer_genre", index=None, placeholder="Select genre...",
            on_change=lambda: update_callback("genre", 
                                get_optimizer().field_manager.get_id_for_name('genre', st.session_state.optimizer_genre) 
                                if st.session_state.optimizer_genre else None))
        
        # Subgenre selection
        subgenre_names = st.multiselect("Subgenres",
            options=[name for _, name in display_options['subgenres'] if name and name.strip()],
            key="optimizer_subgenres", placeholder="Select subgenres...",
            on_change=lambda: update_callback("subgenres", 
                            get_optimizer().field_manager.get_ids_for_names('subgenres', st.session_state.optimizer_subgenres)))
        
        # Source type selection
        source_name = st.selectbox("Source Type",
            options=[name for _, name in display_options['source_type'] if name and name.strip()],
            key="optimizer_source_type", index=None, placeholder="Select source type...",
            on_change=lambda: update_callback("source_type", 
                                get_optimizer().field_manager.get_id_for_name('source_type', st.session_state.optimizer_source_type) 
                                if st.session_state.optimizer_source_type else None))
        
        # Character types selection
        char_names = st.multiselect("Character Types",
            options=[name for _, name in display_options['character_types'] if name and name.strip()],
            key="optimizer_character_types", placeholder="Select character types...",
            on_change=lambda: update_callback("character_types", 
                                get_optimizer().field_manager.get_ids_for_names('character_types', st.session_state.optimizer_character_types)))
        
        # Thematic elements selection
        theme_names = st.multiselect("Thematic Elements",
            options=[name for _, name in display_options['thematic_elements'] if name and name.strip()],
            key="optimizer_thematic_elements", placeholder="Select thematic elements...",
            on_change=lambda: update_callback("thematic_elements", 
                                get_optimizer().field_manager.get_ids_for_names('thematic_elements', st.session_state.optimizer_thematic_elements)))
        
        # Plot elements selection
        plot_names = st.multiselect("Plot Elements",
            options=[name for _, name in display_options['plot_elements'] if name and name.strip()],
            key="optimizer_plot_elements", placeholder="Select plot elements...",
            on_change=lambda: update_callback("plot_elements", 
                                get_optimizer().field_manager.get_ids_for_names('plot_elements', st.session_state.optimizer_plot_elements)))
        
        # Tone selection
        tone_names = st.multiselect("Tone",
            options=[name for _, name in display_options['tone'] if name and name.strip()],
            key="optimizer_tone", placeholder="Select tone...",
            on_change=lambda: update_callback("tone", 
                                get_optimizer().field_manager.get_ids_for_names('tone', st.session_state.optimizer_tone)))
        
        # Time setting selection
        time_names = st.multiselect("Time Setting",
            options=[name for _, name in display_options['time_setting'] if name and name.strip()],
            key="optimizer_time_setting", placeholder="Select time setting...",
            on_change=lambda: update_callback("time_setting", 
                                get_optimizer().field_manager.get_ids_for_names('time_setting', st.session_state.optimizer_time_setting)))
        
        # Location setting selection
        location_names = st.multiselect("Location Setting",
            options=[name for _, name in display_options['location_setting'] if name and name.strip()],
            key="optimizer_location_setting", placeholder="Select location setting...",
            on_change=lambda: update_callback("location_setting", 
                                get_optimizer().field_manager.get_ids_for_names('location_setting', st.session_state.optimizer_location_setting)))


def render_production_criteria(state: Dict, update_callback: Callable) -> None:
    """Render production criteria section.
    
    Args:
        state: State dictionary containing criteria and display options
        update_callback: Callback function for updating criteria
    """
    criteria = state['criteria']
    display_options = state['display_options']
    
    with st.expander("Production Criteria (13 pts)", expanded=False):
        st.markdown("### Production")
        
        # Network selection
        network_name = st.selectbox("Network",
            options=[name for _, name in display_options['network'] if name and name.strip()],
            key="optimizer_network", index=None, placeholder="Select network...",
            on_change=lambda: update_callback("network", 
                                get_optimizer().field_manager.get_id_for_name('network', st.session_state.optimizer_network) 
                                if st.session_state.optimizer_network else None))
        
        # Studios selection
        studio_names = st.multiselect("Studios",
            options=[name for _, name in display_options['studios'] if name and name.strip()],
            key="optimizer_studios", placeholder="Select studios...",
            on_change=lambda: update_callback("studios", 
                                get_optimizer().field_manager.get_ids_for_names('studios', st.session_state.optimizer_studios)))
        
        # Team members selection
        team_names = st.multiselect("Team Members",
            options=[name for _, name in display_options.get('team_members', []) if name and name.strip()],
            key="optimizer_team_members", placeholder="Select team members...",
            on_change=lambda: update_callback("team_members", 
                                get_optimizer().field_manager.get_ids_for_names('team_members', st.session_state.optimizer_team_members)))


def render_format_criteria(state: Dict, update_callback: Callable) -> None:
    """Render format criteria section.
    
    Args:
        state: State dictionary containing criteria and display options
        update_callback: Callback function for updating criteria
    """
    criteria = state['criteria']
    display_options = state['display_options']
    
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
                                    get_optimizer().field_manager.get_id_for_name('order_type', st.session_state.optimizer_order_type) 
                                    if st.session_state.optimizer_order_type else None))
   

# Helper functions for rendering tab content

# Visualization helper functions are imported from style_config.py


def render_success_metrics(summary: Any):
    """Render success metrics using pre-formatted data from OptimizerView.
    
    Displays:
    1. Overall Success Score and Success Probability in top row
    2. Component scores (Audience, Critics, Longevity) in bottom row
    
    Args:
        summary: Optimization summary with formatted_data attribute
    """
    try:
        # Get config for consistent display
        config = OptimizerConfig()
        
        # Check if we have pre-formatted component scores
        has_formatted_data = hasattr(summary, 'formatted_data')
        
        if not has_formatted_data:
            # No formatted data available - show N/A for all metrics
            st.write("### Success Metrics")
            col1, col2 = st.columns(2)
            with col1:
                render_metric_card("Overall Success Score", "N/A", "Data unavailable")
            with col2:
                render_metric_card("Success Probability", "N/A", "Data unavailable")
                
            st.write("### Component Scores")
            col1, col2, col3 = st.columns(3)
        formatted_data = summary.formatted_data
        
        # Check formatted_data structure
        st.write(f"DEBUG: formatted_data keys: {list(formatted_data.keys())}")
        st.write(f"DEBUG: formatted_data type: {type(formatted_data)}")
        st.write(f"DEBUG: summary type: {type(summary)}")
        
        # Check if component_scores exists
        if 'component_scores' not in formatted_data:
            st.error("Unable to display success metrics: Missing component scores data")
            return
            
        # Display any error message at the top if present
        if 'confidence_info' in formatted_data and isinstance(formatted_data['confidence_info'], dict) and 'error' in formatted_data['confidence_info']:
            error_message = formatted_data['confidence_info']['error']
            st.warning(f"Analysis encountered an issue: {error_message}")
            # Continue to display metrics with error information
            
        # Top row: Overall Success Score and Success Probability
        st.write("### Success Metrics")
        col1, col2 = st.columns(2)
        
        # Overall Success Score
        with col1:
            if 'success' in formatted_data['component_scores']:
                # Use pre-formatted success score data
                success_data = formatted_data['component_scores']['success']
                
                # Check if there's an error in the component score
                if 'error' in success_data:
                    render_metric_card(
                        "Overall Success Score", 
                        "Error", 
                        success_data['error'][:50] + "..." if len(success_data['error']) > 50 else success_data['error']
                    )
                elif success_data['score'] is not None:
                    # Success score is 0-100 scale
                    success_score = success_data['score'] * 100 if success_data['score'] <= 1 else success_data['score']
                    render_metric_card(
                        "Overall Success Score", 
                        f"{success_score:.0f}", 
                        f"Scale: 0-100 | Sample: {success_data['sample_size']}"
                    )
                else:
                    render_metric_card("Overall Success Score", "N/A", "Success score not available")
            else:
                render_metric_card("Overall Success Score", "N/A", "Success score not available")
        
        # Success Probability
        with col2:
            if 'success_probability' in formatted_data and formatted_data['success_probability']:
                # Use pre-formatted success probability data
                success_data = formatted_data['success_probability']
                render_metric_card(
                    "Success Probability", 
                    success_data['display'], 
                    success_data['subtitle']
                )
            else:
                render_metric_card("Success Probability", "N/A", "Success probability not available")
        
        # Bottom row: Component Scores
        st.write("### Component Scores")
        col1, col2, col3 = st.columns(3)
        
        # Audience Score
        with col1:
            if 'audience' in formatted_data['component_scores']:
                # Use pre-formatted audience score data
                audience_data = formatted_data['component_scores']['audience']
                
                # Check if there's an error in the component score
                if 'error' in audience_data:
                    render_metric_card(
                        "Audience Appeal", 
                        "Error", 
                        audience_data['error'][:50] + "..." if len(audience_data['error']) > 50 else audience_data['error']
                    )
                elif audience_data['score'] is not None:
                    render_metric_card(
                        "Audience Appeal", 
                        f"{audience_data['score']:.0%}", 
                        f"Sample: {audience_data['sample_size']}"
                    )
                else:
                    render_metric_card("Audience Appeal", "N/A", "Audience data not available")
            else:
                render_metric_card("Audience Appeal", "N/A", "Audience data not available")
        
        # Critics Score
        with col2:
            if 'critics' in formatted_data['component_scores']:
                # Use pre-formatted critics score data
                critics_data = formatted_data['component_scores']['critics']
                
                # Check if there's an error in the component score
                if 'error' in critics_data:
                    render_metric_card(
                        "Critical Reception", 
                        "Error", 
                        critics_data['error'][:50] + "..." if len(critics_data['error']) > 50 else critics_data['error']
                    )
                elif critics_data['score'] is not None:
                    render_metric_card(
                        "Critical Reception", 
                        f"{critics_data['score']:.0%}", 
                        f"Sample: {critics_data['sample_size']}"
                    )
                else:
                    render_metric_card("Critical Reception", "N/A", "Critics data not available")
            else:
                render_metric_card("Critical Reception", "N/A", "Critics data not available")
        
        # Longevity Score
        with col3:
            if 'longevity' in formatted_data['component_scores']:
                # Use pre-formatted longevity score data
                longevity_data = formatted_data['component_scores']['longevity']
                
                # Check if there's an error in the component score
                if 'error' in longevity_data:
                    render_metric_card(
                        "Longevity", 
                        "Error", 
                        longevity_data['error'][:50] + "..." if len(longevity_data['error']) > 50 else longevity_data['error']
                    )
                elif longevity_data['score'] is not None:
                    score = longevity_data['score']
                    sample_size = longevity_data['sample_size']
                    
                    render_metric_card(
                        "Longevity", 
                        f"{score:.0%}", 
                        f"Sample: {sample_size}"
                    )
                else:
                    render_metric_card("Longevity", "N/A", "Longevity data not available")
            else:
                render_metric_card("Longevity", "N/A", "Longevity data not available")
    
    except Exception as e:
        # Show the actual error message to help with debugging
        error_msg = str(e)
        st.error(f"Error rendering success metrics: {error_msg}")

def render_success_factors(formatted_factors: List[Dict[str, Union[str, float, int]]]):
    """Render success factors chart using pre-formatted data from OptimizerView.
    
    Args:
        formatted_factors: List of pre-formatted success factor dictionaries with fields like name, impact, etc.
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
        # Access matching_titles using the correct key '_matching_titles'
        # This matches the key used in _format_success_factors in optimizer_view.py
        matching_titles = factor['_matching_titles']
        if matching_titles:
            # Access Name using the correct key 'Name' from _format_success_factors
            with st.expander(f"Shows matching '{factor['Name']}' ({len(matching_titles)} shows)"):
                # Display titles in a scrollable container with fixed height
                titles_html = "<div style='max-height: 300px; overflow-y: auto;'><ul>"
                titles_html += ''.join(f"<li>{title}</li>" for title in matching_titles)
                titles_html += "</ul></div>"
                st.markdown(titles_html, unsafe_allow_html=True)


# group_recommendations function has been removed
# All recommendation grouping is now handled by OptimizerView._format_recommendations


def render_recommendations(formatted_recommendations: Dict[str, Union[List[Dict[str, Union[str, float, int, bool]]], Dict[str, List[Dict[str, Union[str, float, int, bool]]]]]]):
    """Render recommendations using pre-formatted data from OptimizationSummary.
    
    Args:
        formatted_recommendations: Dictionary with formatted recommendation data grouped by type
        This is the result of _format_recommendations which separates the unified list into
        general and network-specific recommendations based on is_network_specific tags
    """
    try:
        # Check if there are any recommendations to display
        if not formatted_recommendations:
            st.info("No recommendations available.")
            return
            
        # Extract recommendation groups from OptimizationSummary structure
        general = formatted_recommendations.get("general", [])
        network_specific = formatted_recommendations.get("network_specific", [])
        
        # Create a grouped structure for compatibility with the rest of the function
        grouped = {
            "add": {"items": [], "header": "Add to Your Concept"},
            "change": {"items": [], "header": "Consider Changing"},
            "remove": {"items": [], "header": "Consider Removing"}
        }
        
        # Populate the grouped structure with recommendations from general
        # This is used for checking if we have any recommendations at all
        for rec in general:
            rec_type = rec.get('recommendation_type', 'add')
            if rec_type in grouped:
                grouped[rec_type]['items'].append(rec)
                
        if OptimizerConfig.DEBUG_MODE:
            rec_type_counts = {'add': 0, 'change': 0, 'remove': 0}
            
            # Count recommendation types from general recommendations
            for rec in general:
                rec_type = rec.get('recommendation_type', 'add')  # Default to 'add' if not specified
                if rec_type in rec_type_counts:
                    rec_type_counts[rec_type] += 1
                else:
                    pass
            
            # Count recommendation types from network-specific recommendations
            for rec in network_specific:
                rec_type = rec.get('recommendation_type', 'add')  # Default to 'add' if not specified
                if rec_type in rec_type_counts:
                    rec_type_counts[rec_type] += 1
                else:
                    pass
                    
            OptimizerConfig.debug(f"Recommendation type counts: {rec_type_counts} (general: {len(general)}, network-specific: {len(network_specific)})", category='recommendation')
                
        # Check if there are any recommendations in any group
        has_recommendations = any(group_data['items'] for group_data in grouped.values())
                
        if not has_recommendations:

            st.info("No recommendations available for your current criteria.")
            return
            
        # We already created the grouped structure earlier, so we can use it directly
        # grouped structure is already populated from general recommendations
                
        # Extract general recommendations directly from the input
        general_recommendations = general
        

        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Found {len(general_recommendations)} general recommendations", category='recommendation')
                
        if general_recommendations:
            # Render general recommendations
            st.subheader("General Recommendations")
                        
            # Sort all recommendations by impact score regardless of category
            general_recommendations.sort(key=lambda x: abs(x['_impact_raw']), reverse=True)
            
            # Group by criteria_type only, regardless of recommendation type
            by_criteria_type = {}
            
            # Process each recommendation for grouping
            for rec in general_recommendations:
                # Get criteria type - fallback to field if criteria_type is not available
                criteria_type = rec.get('criteria_type', rec.get('field', 'Other'))
                
                # Debug output for criteria_type

                
                # Initialize this criteria type group if needed
                if criteria_type not in by_criteria_type:
                    by_criteria_type[criteria_type] = []
                by_criteria_type[criteria_type].append(rec)
                        
            # Render recommendations grouped by criteria type
            for criteria_type, criteria_recs in by_criteria_type.items():
                # Sort recommendations within each criteria type by impact (absolute value)
                criteria_recs.sort(key=lambda x: abs(x['_impact_raw']), reverse=True)
                
                # Debug output only for recommendations

                                        
                # Render each recommendation (limit to top 20 per criteria type)
                for rec in criteria_recs[:20]:
                    # Get the title - ensure we have a fallback
                    title = rec.get('title', rec.get('name', 'Recommendation'))
                    
                    # Get the content - could be 'explanation' or 'description' depending on where it was formatted
                    content = rec.get('explanation', rec.get('description', 'No details available.'))
                                   
                    # Use info card style for all recommendations
                    render_info_card(title, content)
                        
        else:
            # If no general recommendations were found
            st.info("No general recommendations available for your current criteria.")
            
        # Network-specific recommendations are shown only in the Network Analysis tab
            
    except Exception as e:
        OptimizerConfig.debug(f"Error rendering recommendations: {str(e)}", category='recommendation')
        OptimizerConfig.debug(traceback.format_exc(), category='recommendation')
        st.error("Unable to display recommendations due to an error.")
        

def render_network_recommendations(network_recs: List[Dict]) -> None:
    """Render network-specific recommendations, showing only those that suggest alternatives.
    
    This function takes network-specific recommendations, groups them by network,
    and displays only those that suggest an alternative option rather than keeping
    the current selection.
    
    Args:
        network_recs: List of network-specific recommendation dictionaries
    """
    if not network_recs:
        # Display a more informative message when no network-specific recommendations are available
        st.info("No network-specific recommendations are currently available.")
        
        # Explain why recommendations might not be available
        with st.expander("Why aren't there network-specific recommendations?"):
            st.write("""
            Network-specific recommendations are generated when there are significant differences in how criteria perform on specific networks 
            compared to their overall performance. Some reasons you might not see recommendations:
            
            1. **Limited data variance**: The success rates across different networks may be too similar to generate meaningful recommendations.
            
            2. **Insufficient network-specific data**: There may not be enough shows on specific networks that match your criteria.
            
            3. **Try adding more criteria**: Adding more specific criteria to your concept may help identify network-specific patterns.
            
            4. **Explore different combinations**: Try different combinations of criteria to see if network-specific patterns emerge.
            """)
            
        # Show the network compatibility table as a reminder
        st.write("Review the network compatibility table above to see which networks best match your current concept.")
        return
    
    try:
        # Debug count of total recommendations
        st.write(f"DEBUG UI: Processing {len(network_recs)} network-specific recommendations")
        
        # Group recommendations by network for better organization
        network_grouped_recs = {}
        for rec in network_recs:
            # Extract network name from metadata if available
            network_name = None
            if 'metadata' in rec and 'network_name' in rec['metadata']:
                network_name = rec['metadata']['network_name']
            else:
                # Fall back to extracting from description
                if 'description' in rec and 'Network ' in rec['description']:
                    parts = rec['description'].split('Network ')
                    if len(parts) > 1:
                        network_parts = parts[1].split(' has')
                        if network_parts:
                            network_name = network_parts[0]
            
            if not network_name:
                network_name = "Unknown Network"
                
            if network_name not in network_grouped_recs:
                network_grouped_recs[network_name] = []
            network_grouped_recs[network_name].append(rec)
        
        # Display recommendations grouped by network
        for network_name, recs in network_grouped_recs.items():
            # Debug count of recommendations per network
            st.write(f"DEBUG UI: Network '{network_name}' has {len(recs)} recommendations")
            
            # Filter out recommendations that say "you should change this" without specifying an alternative
            useful_recs = []
            filtered_out = []
            
            for rec in recs:
                # Keep recommendations that say "keep this"
                keep_recommendation = False
                rec_type = rec.get('recommendation_type', 'unknown')
                
                # Check if this is a "keep" recommendation based on title or description
                if 'title' in rec and rec['title'] and ('keep' in rec['title'].lower() or 'maintain' in rec['title'].lower()):
                    keep_recommendation = True
                    
                if 'description' in rec and rec['description'] and ('keep' in rec['description'].lower() or 'maintain' in rec['description'].lower()):
                    keep_recommendation = True
                
                # Check if this is a recommendation that suggests a specific alternative
                has_specific_alternative = False
                
                # Check metadata for best_alternative
                if 'metadata' in rec and rec['metadata']:
                    if 'best_alternative' in rec['metadata'] and rec['metadata']['best_alternative'] is not None:
                        has_specific_alternative = True
                
                # Check if suggested_value is different from current_value and not None
                if ('suggested_value' in rec and 'current_value' in rec and 
                    rec['suggested_value'] is not None and 
                    rec['suggested_value'] != rec['current_value']):
                    has_specific_alternative = True
                    
                # Include recommendations that either say "keep this" or provide a specific alternative
                if keep_recommendation or has_specific_alternative:
                    useful_recs.append(rec)
                else:
                    filtered_out.append(rec)
                    st.write(f"DEBUG UI: Filtered out {rec_type} recommendation - Keep: {keep_recommendation}, Has Alt: {has_specific_alternative}, Suggested: {rec.get('suggested_value')}, Current: {rec.get('current_value')}")
            
            st.write(f"DEBUG UI: Kept {len(useful_recs)} recommendations, filtered out {len(filtered_out)} recommendations for network '{network_name}'")

            
            # Only show networks that have useful recommendations
            if useful_recs:
                st.write(f"### {network_name}")
                for rec in useful_recs:
                    # Determine color based on impact score
                    if 'impact_score' in rec:
                        color = COLORS['success'] if rec['impact_score'] > 0 else COLORS['warning']
                    else:
                        color = None
                        
                    # Use the title directly from the recommendation
                    # The title should already be properly formatted by OptimizerView._format_recommendations
                    title = rec.get('title', 'Network-Specific Insight')
                        
                    # Display the recommendation using an info card
                    render_info_card(
                        title=title,
                        content=rec.get('description', 'No description available.'),
                        color=color
                    )
    except Exception as e:
        OptimizerConfig.debug(f"Error rendering network recommendations: {str(e)}", category='recommendation')
        OptimizerConfig.debug(traceback.format_exc(), category='recommendation')
        st.error("Unable to display network recommendations due to an error.")

