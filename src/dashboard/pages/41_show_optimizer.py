"""Show Optimizer page for the TV Series Database Dashboard.

This page provides an interface for optimizing show concepts based on
historical success patterns and network preferences.
"""

import streamlit as st
import pandas as pd
import sys
import os
from typing import Dict, Any

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required
from src.dashboard.state.session import get_page_state, update_page_state
from src.dashboard.utils.style_config import COLORS, FONTS, CHART_DEFAULTS, render_metric_card, render_info_card, get_match_level_color
from src.dashboard.components.optimizer_view import OptimizerView
from src.data_processing.show_optimizer.optimizer_config import OptimizerConfig
from src.dashboard.components.optimizer_helpers import (
    render_success_metrics, render_recommendations,
    render_content_criteria, render_production_criteria, render_format_criteria,
    render_success_factors
)
from src.data_processing.show_optimizer.optimizer_profiler import display_profiling_results
from src.data_processing.show_optimizer.apply_profiler import apply_profiling


# Use OptimizerConfig.get_match_level_description instead of a local function


@auth_required()
def show():
    # Apply profiling to key methods
    if st.session_state.get('debug_mode', False):
        try:
            apply_profiling()
        except Exception as e:
            st.write(f"Error applying profiling: {str(e)}")
    """Main page content."""
    # Sidebar for debug controls
    with st.sidebar:
        with st.expander("Developer Options", expanded=False):
            debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.get('debug_mode', False))
            st.session_state['debug_mode'] = debug_mode
            
            if debug_mode:
                st.write("Debug mode is ON")
                if st.button("Reset Profiling Data"):
                    from src.data_processing.show_optimizer.optimizer_profiler import reset_profiling
                    reset_profiling()
                    st.rerun()
            else:
                st.write("Debug mode is OFF")
        
    # Page title using style from style_config
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Show Optimizer</p>', unsafe_allow_html=True)
    
    # Add description
    st.write("Build and analyze a show concept to optimize its success potential.")
    st.write("Select criteria to define your concept, then view success metrics, network compatibility, and recommendations.")
    
    # Get page state
    state = get_page_state("show_optimizer")
    
    # Initialize state if needed
    if "field_options" not in state:
        state["field_options"] = {}
    if "display_options" not in state:
        state["display_options"] = {}
    if "criteria" not in state:
        state["criteria"] = {}
    if "results" not in state:
        state["results"] = False
    
    # Initialize state if needed
    
    # Debug mode is controlled from the sidebar only
    
    # Initialize the optimizer
    optimizer_view = OptimizerView()
    
    # Initialize optimizer components if needed
    if not optimizer_view.initialized:
        if not optimizer_view.initialize(state):
            st.error("Failed to initialize Show Optimizer. Please refresh the page and try again.")
            return
    
    # Create columns for criteria and results
    col1, col2 = st.columns([1, 2])
    
    # Render criteria in the first column
    with col1:
        # Get criteria and display options from state
        criteria = state.get('criteria', {})
        display_options = state.get('display_options', {})
        
        # Check if field options are available in state
        if not display_options:
            st.error("Unable to load field options from the database.")
            st.info("This may be due to a temporary connection issue or database maintenance.")
            return
        
        st.subheader("Build Your Show Concept")
        
        # Define update function for criteria changes
        def update_criteria_and_analyze(field_name, value):
            # Ensure criteria exists in state
            if 'criteria' not in state:
                state['criteria'] = {}
                
            # Update the specific field in the criteria
            if value is None or (isinstance(value, list) and len(value) == 0):
                # Remove the field if value is None or empty list
                if field_name in state['criteria']:
                    del state['criteria'][field_name]
            else:
                # Set the field value
                state['criteria'][field_name] = value
            
            # Update the state in session state
            update_page_state("show_optimizer", state)
            
            # Initialize session state variables if not already set
            if 'optimizer_summary' not in st.session_state:
                st.session_state['optimizer_summary'] = None
            
            # Keep the session_state.optimizer_criteria in sync with state['criteria']
            st.session_state.optimizer_criteria = state['criteria'].copy()
            
            # Run the analysis with the updated criteria
            # Always run the analysis, even if criteria is empty
            # This ensures the UI updates when criteria are deselected
            optimizer_view.run_analysis(state)
        
        # Render criteria sections using helper functions
        render_content_criteria(state, update_criteria_and_analyze)
        render_production_criteria(state, update_criteria_and_analyze)
        render_format_criteria(state, update_criteria_and_analyze)
        
        # Save criteria to session state after running analysis
        criteria = state.get('criteria', {})
        st.session_state.optimizer_criteria = criteria.copy()
        
        # Analysis is already run on line 130 after updating criteria
        # No need to run it again here
    
    # If we have results, render tabs in the second column
    with col2:
        if state.get('summary') or st.session_state.get("optimizer_summary"):
            # Get summary from state or session state
            summary = state.get('summary') or st.session_state.get("optimizer_summary")
            
            if summary:
                # Create a clear header for the results section
                st.subheader("Analysis Results")
                
                # Create tabs - EXACTLY like in Show Detail page
                tab1, tab2, tab3 = st.tabs(["Success Metrics", "Network Analysis", "Recommendations"])
                
                # Tab 1: Success Metrics
                with tab1:
                    # Use our improved helper function to render success metrics
                    render_success_metrics(summary)
                    
                    # Display network recommendations if available
                    
                    # Display matching show titles
                    st.subheader("Matching Shows")
                    
                    # Display matching shows with match level differentiation
                    if hasattr(summary, 'matching_shows') and summary.matching_shows is not None and not summary.matching_shows.empty:
                        # Get match level information
                        match_level = getattr(summary, 'match_level', 1)
                        match_level_name = OptimizerConfig.get_match_level_description(match_level)
                        sample_size = len(summary.matching_shows)
                        
                        # Use get_match_level_color function from style_config to get colors for match levels
                        
                        # Use OptimizerConfig.get_match_level_description for consistent descriptions
                        level_descriptions = {}
                        for level in range(1, 10):  # Support up to 9 levels dynamically
                            level_descriptions[level] = OptimizerConfig.get_match_level_description(level)
                        
                        # Get match counts by level if available
                        match_counts_by_level = {}
                        if hasattr(summary, 'confidence_info') and isinstance(summary.confidence_info, dict) and 'match_counts_by_level' in summary.confidence_info:
                            # Get match counts from confidence_info if available
                            match_counts_by_level = summary.confidence_info['match_counts_by_level']
                        elif hasattr(summary, 'match_counts_by_level'):
                            match_counts_by_level = summary.match_counts_by_level
                        if 'match_level' in summary.matching_shows.columns:
                            # Count shows by match level
                            level_counts = {}
                            # Get all unique match levels in the data
                            unique_levels = summary.matching_shows['match_level'].unique()
                            for level in unique_levels:
                                count = len(summary.matching_shows[summary.matching_shows['match_level'] == level])
                                if count > 0:
                                    level_counts[level] = count
                        
                        # Check if we have confidence_info with validation details
                        if hasattr(summary, 'confidence_info') and summary.confidence_info:
                            confidence_info = summary.confidence_info
                            if 'original_match_level' in confidence_info and 'match_level' in confidence_info:
                                original_level = confidence_info['original_match_level']
                                actual_level = confidence_info['match_level']
                                
                                if original_level != actual_level and original_level == 1:
                                    # Show a warning that some shows were downgraded from exact match
                                    st.warning(
                                        "Some shows were initially categorized as exact matches but were downgraded "
                                        "because they didn't match all selected criteria (like character types)."
                                    )
                        
                        # Create a summary of match levels
                        match_level_summary = []
                        for level in range(1, 6):
                            if level in match_counts_by_level and match_counts_by_level[level] > 0:
                                level_name = level_descriptions.get(level, f"Level {level}")
                                count = match_counts_by_level[level]
                                match_level_summary.append(f"{count} {level_name}")
                        
                        # Count shows by match level for verification
                        level_counts = {}
                        if 'match_level' in summary.matching_shows.columns:
                            # Get all unique match levels in the data
                            unique_levels = summary.matching_shows['match_level'].unique()
                            for level in unique_levels:
                                count = len(summary.matching_shows[summary.matching_shows['match_level'] == level])
                                if count > 0:
                                    level_counts[level] = count
                        else:
                            # Use match_counts_by_level if available
                            level_counts = match_counts_by_level
                        
                        # Generate fully programmatic descriptions based on match level
                        # Use OptimizerConfig for match level descriptions
                        level_descriptions = {}
                        for level in level_counts.keys():
                            level_descriptions[level] = OptimizerConfig.get_match_level_description(level)
                        
                        # Format the level counts with descriptions
                        formatted_counts = {f"{level_descriptions.get(level, f'Level {level}')}": count 
                                           for level, count in level_counts.items()}
                        
                        st.write(f"Shows by match level: {formatted_counts}")

                        
                        if 'title' in summary.matching_shows.columns:
                            # Create a list of unique shows by title
                            # First, convert to records for easier manipulation
                            show_records = summary.matching_shows.to_dict('records')
                            
                            # Use a dictionary to keep track of the best match level for each title
                            unique_shows_by_title = {}
                            
                            # Process each show
                            for show in show_records:
                                title = show.get('title', 'Unknown Title')
                                # Debug output if match_level is missing
                                if 'match_level' not in show and OptimizerConfig.DEBUG_MODE:
                                    st.write(f"Debug: match_level missing for show: {title}")
                                current_level = show.get('match_level', 1)  # Use 1 as fallback only for comparison
                                
                                # If we haven't seen this title before, or this match level is better
                                if title not in unique_shows_by_title or current_level < unique_shows_by_title[title]['match_level']:
                                    unique_shows_by_title[title] = show
                            
                            # Convert back to a list of unique shows
                            unique_shows = list(unique_shows_by_title.values())
                            
                            # Sort by match level (1 first, then 2, etc.)
                            unique_shows.sort(key=lambda x: x.get('match_level', match_level))
                            
                            # Limit to 100 shows
                            unique_shows = unique_shows[:100]
                            
                            # Count shows by match level for verification
                            level_counts = {}
                            for show in unique_shows:
                                # Each show must have a match_level from the matcher
                                level = show['match_level']
                                level_counts[level] = level_counts.get(level, 0) + 1
                            
                            # Format the level counts with descriptions
                            formatted_counts = {OptimizerConfig.get_match_level_description(level): count 
                                              for level, count in level_counts.items()}
                            
                            st.write(f"Shows by match level: {formatted_counts}")
                                                        
                            # Group shows by match level for better display
                            shows_by_level = {}
                            for show in unique_shows:
                                # Each show must have a match_level from the matcher
                                level = show['match_level']
                                if level not in shows_by_level:
                                    shows_by_level[level] = []
                                shows_by_level[level].append(show)
                            
                            # Use the match_level_colors defined earlier with style guide colors
                            
                            # Display shows grouped by match level
                            for level in sorted(shows_by_level.keys()):
                                shows = shows_by_level[level]
                                level_desc = OptimizerConfig.get_match_level_description(level)
                                
                                # Add separator between match levels if there are multiple levels
                                if len(shows_by_level) > 1 and level > 1:
                                    st.write(f"---")
                                                                
                                # Sort shows by success_score within each match level
                                if 'success_score' in shows[0] if shows else {}:
                                    # Sort by success_score if available
                                    if all('success_score' in show for show in shows):
                                        shows.sort(key=lambda x: x['success_score'], reverse=True)
                                
                                # Display each show with appropriate formatting based on match level
                                for show in shows:
                                    title = show['title']
                                    
                                    # Get color based on match level using style_config function
                                    color = get_match_level_color(level)
                                    
                                    # Format based on match level
                                    if level == 1:
                                        # All criteria matched - bold with high success color
                                        st.markdown(f"<span style='color: {color}; font-weight: bold;'>{title}</span>", unsafe_allow_html=True)
                                    elif level == 2:
                                        # Missing 1 criterion - medium success color
                                        st.markdown(f"<span style='color: {color};'>{title}</span>", unsafe_allow_html=True)
                                    else:
                                        # Missing 2+ criteria - use appropriate color from the match level colors
                                        st.markdown(f"<span style='color: {color};'>{title}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No matching shows available for the selected criteria.")
                
                # Tab 2: Network Analysis
                with tab2:
                    # Always show debug header to confirm we're in the right tab
                    st.write("## Network Compatibility Analysis")
                    
                    # Force debug output to always appear
                    st.write("Debug Status: " + ("ON" if st.session_state.get('debug_mode', False) else "OFF"))
                    
                    # Display network compatibility data
                    debug_mode = st.session_state.get('debug_mode', False)
                    
                    # Always show some basic debug info regardless of debug mode
                    st.write(f"Basic debug: summary object exists: {summary is not None}")
                    if summary is not None:
                        st.write(f"Basic debug: summary has top_networks: {hasattr(summary, 'top_networks')}")
                        if hasattr(summary, 'top_networks'):
                            st.write(f"Basic debug: top_networks length: {len(summary.top_networks)}")
                    
                    # More detailed debug info when debug mode is on
                    if debug_mode:
                        st.write("---")
                        st.write("### Detailed Debug Information")
                        st.write("Debug: Checking network compatibility data")
                        st.write(f"Debug: summary has top_networks attribute: {hasattr(summary, 'top_networks')}")
                        if hasattr(summary, 'top_networks'):
                            st.write(f"Debug: top_networks length: {len(summary.top_networks)}")
                            if len(summary.top_networks) > 0:
                                st.write(f"Debug: First network match type: {type(summary.top_networks[0]).__name__}")
                                st.write(f"Debug: First network match dir: {dir(summary.top_networks[0])}")
                                st.write(f"Debug: First network match network_id: {getattr(summary.top_networks[0], 'network_id', 'Not found')}")
                                st.write(f"Debug: First network match network_name: {getattr(summary.top_networks[0], 'network_name', 'Not found')}")
                        
                        st.write(f"Debug: summary has formatted_data attribute: {hasattr(summary, 'formatted_data')}")
                        if hasattr(summary, 'formatted_data'):
                            st.write(f"Debug: formatted_data keys: {list(summary.formatted_data.keys())}")
                            if 'networks' in summary.formatted_data:
                                st.write(f"Debug: networks data length: {len(summary.formatted_data['networks'])}")
                                if len(summary.formatted_data['networks']) > 0:
                                    st.write(f"Debug: First formatted network: {summary.formatted_data['networks'][0]}")
                            else:
                                st.write("Debug: 'networks' key not found in formatted_data")
                        st.write(f"Debug: summary has recommendations attribute: {hasattr(summary, 'recommendations')}")
                        if hasattr(summary, 'recommendations'):
                            st.write(f"Debug: recommendations length: {len(summary.recommendations) if summary.recommendations else 0}")
                            st.write(f"Debug: recommendations types: {[getattr(rec, 'recommendation_type', getattr(rec, 'rec_type', 'unknown')) for rec in summary.recommendations[:5]] if summary.recommendations else []}")
                                
                        # Debug network-specific recommendations in formatted_data
                        if hasattr(summary, 'formatted_data') and 'recommendations' in summary.formatted_data:
                            st.write(f"Debug: formatted_data['recommendations'] keys: {list(summary.formatted_data['recommendations'].keys())}")
                            if 'network_specific' in summary.formatted_data['recommendations']:
                                st.write(f"Debug: network_specific recommendations length: {len(summary.formatted_data['recommendations']['network_specific'])}")
                                if summary.formatted_data['recommendations']['network_specific']:
                                    st.write(f"Debug: First network-specific recommendation: {summary.formatted_data['recommendations']['network_specific'][0]}")
                        else:
                            st.write("Debug: No recommendations in formatted_data")
                        st.write("---")
                    
                    # Check for network compatibility data
                    if hasattr(summary, 'top_networks') and summary.top_networks:
                        # Add a note about match level
                        match_level = getattr(summary, 'match_level', 0)
                        if match_level > 1:
                            st.info(f"Network compatibility is based on flexible matching (level {match_level}). Results may vary with exact matches.")
                        
                        # Display the formatted network data directly
                        if hasattr(summary, 'formatted_data') and 'networks' in summary.formatted_data and summary.formatted_data['networks']:
                            st.subheader("Network Compatibility")
                            # Convert to DataFrame for display
                            network_df = pd.DataFrame(summary.formatted_data['networks'])
                            
                            # Remove sorting columns before display
                            display_columns = [col for col in network_df.columns if not col.startswith('_')]
                            
                            # Display the network compatibility table
                            st.dataframe(
                                network_df[display_columns],
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Display network-specific recommendations if available
                            if hasattr(summary, 'formatted_data') and 'recommendations' in summary.formatted_data:
                                # Network-specific recommendations are stored directly in the recommendations dictionary
                                network_recs = summary.formatted_data['recommendations'].get('network_specific', [])
                                if network_recs:
                                    st.subheader("Network-Specific Recommendations")
                                    st.write("These recommendations highlight criteria where specific networks have significantly different success rates.")
                                    
                                    # Group recommendations by network for better organization
                                    network_grouped_recs = {}
                                    for rec in network_recs:
                                        # Extract network name from description
                                        network_name = None
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
                                        st.write(f"### {network_name}")
                                        for rec in recs:
                                            # Determine color based on impact score
                                            if 'impact_score' in rec:
                                                color = COLORS['success'] if rec['impact_score'] > 0 else COLORS['warning']
                                            else:
                                                color = None
                                                
                                            # Create a title that includes the impact
                                            if 'impact_score' in rec and rec['impact_score'] != 0:
                                                impact_direction = "Positive" if rec['impact_score'] > 0 else "Negative"
                                                impact_percent = abs(rec['impact_score'] * 100)
                                                title = f"{impact_direction} Impact: {impact_percent:.1f}%"
                                            else:
                                                title = "Network-Specific Insight"
                                                
                                            # Display the recommendation using an info card
                                            render_info_card(
                                                title=title,
                                                content=rec.get('description', 'No description available.'),
                                                color=color
                                            )
                                else:
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
                            else:
                                st.write("Network-specific recommendations will appear here after analyzing your show concept.")
                                st.write("Complete your concept criteria and click 'Analyze' to see recommendations.")
                        else:
                            # This should not happen if top_networks exists but formatted_data['networks'] doesn't
                            st.info("Network data was found but could not be formatted properly.")
                            if st.session_state.get('debug_mode', False):
                                st.write(f"Debug: top_networks exists with {len(summary.top_networks)} items but formatted_data['networks'] is empty or missing")
                                # Show the first network match to help diagnose
                                if summary.top_networks:
                                    st.write("Debug: First network match details:")
                                    first_match = summary.top_networks[0]
                                    st.write(f"  Network ID: {first_match.network_id}")
                                    st.write(f"  Network Name: {first_match.network_name}")
                                    st.write(f"  Compatibility Score: {first_match.compatibility_score}")
                                    st.write(f"  Success Probability: {first_match.success_probability}")
                    else:
                        st.info("No network compatibility data available.")
                        if st.session_state.get('debug_mode', False):
                            st.write("Debug: No top_networks attribute or it is empty")
                
                # Tab 3: Recommendations
                with tab3:
                    # Display success factors if available
                    if hasattr(summary, 'formatted_data') and 'success_factors' in summary.formatted_data:
                        st.subheader("Success Factors")
                        # Display success factors using our helper function with pre-formatted data
                        render_success_factors(summary.formatted_data['success_factors'])
                    elif hasattr(summary, 'success_factors') and summary.success_factors:
                        # Fallback to old method if formatted data is not available
                        st.subheader("Success Factors")
                        render_success_factors(summary.success_factors)
                    
                    # Display recommendations if available
                    if hasattr(summary, 'formatted_data') and 'recommendations' in summary.formatted_data:
                        # Get general and network-specific recommendations
                        general_recs = summary.formatted_data['recommendations'].get('general', [])
                        network_recs = summary.formatted_data['recommendations'].get('network_specific', [])
                        
                        # Group recommendations by type for better organization
                        grouped_recs = {}
                        
                        # Process general recommendations
                        if general_recs:
                            st.subheader("General Recommendations")
                            for rec in general_recs:
                                rec_type = rec.get('recommendation_type', 'other')
                                if rec_type not in grouped_recs:
                                    grouped_recs[rec_type] = []
                                grouped_recs[rec_type].append(rec)
                            
                            # Use our improved helper function to render recommendations with pre-formatted data
                            render_recommendations({"grouped": grouped_recs}, on_click_handler=None)
                        else:
                            st.info("No general recommendations available for your current criteria.")
                        
                        # Process network-specific recommendations
                        if network_recs:
                            st.subheader("Network-Specific Recommendations")
                            st.write("These recommendations highlight criteria where specific networks have significantly different success rates.")
                            
                            # Group recommendations by network for better organization
                            network_grouped_recs = {}
                            for rec in network_recs:
                                # Extract network name from description
                                network_name = None
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
                                st.write(f"### {network_name}")
                                for rec in recs:
                                    # Determine color based on impact score
                                    if 'impact_score' in rec:
                                        color = COLORS['success'] if rec['impact_score'] > 0 else COLORS['warning']
                                    else:
                                        color = None
                                        
                                    # Create a title that includes the impact
                                    if 'impact_score' in rec and rec['impact_score'] != 0:
                                        impact_direction = "Positive" if rec['impact_score'] > 0 else "Negative"
                                        impact_percent = abs(rec['impact_score'] * 100)
                                        title = f"{impact_direction} Impact: {impact_percent:.1f}%"
                                    else:
                                        title = "Network-Specific Insight"
                                        
                                    # Display the recommendation using an info card
                                    render_info_card(
                                        title=title,
                                        content=rec.get('description', 'No description available.'),
                                        color=color
                                    )
                        else:
                            st.info("No network-specific recommendations available for your current criteria.")
                    elif hasattr(summary, 'recommendations') and summary.recommendations:
                        # Fallback to old method if formatted data is not available
                        st.subheader("Recommendations")
                        render_recommendations({"grouped": group_recommendations(summary.recommendations)}, on_click_handler=None)
                    else:
                        st.info("No recommendations available for the selected criteria.")
                
                # No need for a second reset button since we already have one at the top of the page
        
    # Save state back to session
    update_page_state("show_optimizer", state)
    
    # End of page
    st.markdown("---")
    
    # Add profiling results at the bottom of the page
    if st.session_state.get('debug_mode', False):
        with st.expander("Performance Profiling", expanded=False):
            display_profiling_results()

if __name__ == "__main__":
    show()
