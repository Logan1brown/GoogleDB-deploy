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
from src.dashboard.utils.style_config import COLORS, FONTS, CHART_DEFAULTS
from src.dashboard.components.optimizer_view import OptimizerView
from src.data_processing.show_optimizer.optimizer_config import OptimizerConfig
from src.dashboard.components.optimizer_helpers import (
    render_success_metrics, render_network_compatibility, render_recommendations,
    render_content_criteria, render_production_criteria, render_format_criteria,
    render_success_factors
)

@auth_required()
def show():
    """Main page content."""
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
            
            # Also update session state for compatibility
            if "optimizer_criteria" not in st.session_state:
                st.session_state.optimizer_criteria = {}
            
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
        
        # Save criteria to session state before running analysis
        criteria = state.get('criteria', {})
        st.session_state.optimizer_criteria = criteria.copy()
        
        # Run analysis automatically when criteria changes
        # Always run the analysis, even if criteria is empty
        optimizer_view.run_analysis(state)
    
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
                    
                    # Longevity score is now displayed in the main metrics section
                    
                    # Display network recommendations if available
                    
                    # Display matching show titles
                    st.subheader("Matching Shows")
                    
                    # Display matching shows with match level differentiation
                    if hasattr(summary, 'matching_shows') and summary.matching_shows is not None and not summary.matching_shows.empty:
                        # Get match level information
                        match_level = getattr(summary, 'match_level', 1)
                        match_level_name = getattr(summary, 'match_level_name', 'Exact Match')
                        sample_size = len(summary.matching_shows)
                        
                        # Define match level names and colors
                        match_level_names = {
                            1: "Exact Match",
                            2: "Close Match",
                            3: "Partial Match",
                            4: "Loose Match"
                        }
                        
                        match_level_colors = {
                            1: "#000000",  # Black for exact match
                            2: "#000000",  # Black for close match
                            3: "#666666",  # Grey for partial match
                            4: "#999999"   # Light grey for loose match
                        }
                        
                        # Get match counts by level if available
                        match_counts_by_level = {}
                        if hasattr(summary, 'confidence_info') and isinstance(summary.confidence_info, dict) and 'match_counts_by_level' in summary.confidence_info:
                            # Get match counts from confidence_info if available
                            match_counts_by_level = summary.confidence_info['match_counts_by_level']
                        elif hasattr(summary, 'match_counts_by_level'):
                            match_counts_by_level = summary.match_counts_by_level
                        elif 'match_level' in summary.matching_shows.columns:
                            # Calculate counts from the DataFrame
                            for level in range(1, 5):
                                count = len(summary.matching_shows[summary.matching_shows['match_level'] == level])
                                if count > 0:
                                    match_counts_by_level[level] = count
                        
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
                        for level in range(1, 5):
                            if level in match_counts_by_level and match_counts_by_level[level] > 0:
                                level_name = match_level_names.get(level, f"Level {level}")
                                count = match_counts_by_level[level]
                                match_level_summary.append(f"{count} {level_name}")
                        
                        # Display match counts with clearer explanation
                        exact_match_count = match_counts_by_level.get(1, 0)
                        
                        if exact_match_count > 0 and exact_match_count < sample_size:
                            # We have both exact and supplemental matches
                            st.write(f"Found {exact_match_count} exact matches plus {sample_size - exact_match_count} supplemental matches")
                            if exact_match_count < 5:
                                st.info(f"Found only {exact_match_count} exact matches, so supplemental matches were added to provide better analysis.")
                        elif match_level_summary:
                            st.write(f"Found {sample_size} shows: {', '.join(match_level_summary)}")
                        else:
                            st.write(f"Found {sample_size} shows with similar criteria ({match_level_name})")
                        
                        # Add legend for the colors with clearer explanation
                        st.write("**Bold** = Exact match (matches ALL selected criteria including character types)")
                        st.write("Normal = Close match (matches most criteria)")
                        st.write("Grey = Supplemental match (matches core criteria only)")
                        
                        # Add note about supplemental matches if needed
                        if exact_match_count > 0 and exact_match_count < sample_size:
                            st.write("Note: Supplemental matches may not include all selected criteria but help provide a more robust analysis.")
                        
                        # Add explanation about character types if they're in the criteria
                        if 'character_types' in state.get('criteria', {}) and state['criteria']['character_types']:
                            char_types = state['criteria']['character_types']
                            if char_types:
                                # Convert character types to strings before joining
                                char_types_str = [str(ct) for ct in char_types]
                                
                                # Look up character type names if possible
                                char_type_names = []
                                if 'field_options' in state and 'character_types' in state['field_options']:
                                    # Create a mapping of ID to name
                                    ct_map = {opt.id: opt.name for opt in state['field_options']['character_types']}
                                    char_type_names = [ct_map.get(int(ct), f"Type {ct}") for ct in char_types_str]
                                
                                # Use names if available, otherwise use IDs
                                display_types = char_type_names if char_type_names else char_types_str
                                st.info(f"Shows are matched based on character types: {', '.join(display_types)}. "
                                       f"Only exact matches (bold) are guaranteed to have these character types.")

                        
                        if 'title' in summary.matching_shows.columns:
                            
                            # Create a list of unique shows by title
                            # First, convert to records for easier manipulation
                            show_records = summary.matching_shows.to_dict('records')
                            
                            # Use a dictionary to keep track of the best match level for each title
                            unique_shows_by_title = {}
                            
                            # Process each show
                            for show in show_records:
                                title = show.get('title', 'Unknown Title')
                                current_level = show.get('match_level', match_level)
                                
                                # If we haven't seen this title before, or this match level is better
                                if title not in unique_shows_by_title or current_level < unique_shows_by_title[title]['match_level']:
                                    unique_shows_by_title[title] = show
                            
                            # Convert back to a list of unique shows
                            unique_shows = list(unique_shows_by_title.values())
                            
                            # Sort by match level (1 first, then 2, etc.)
                            unique_shows.sort(key=lambda x: x.get('match_level', match_level))
                            
                            # Limit to 100 shows
                            unique_shows = unique_shows[:100]
                            
                            # Display the unique shows
                            st.write(f"Displaying {len(unique_shows)} unique shows")
                            
                            # Count shows by match level for verification
                            level_counts = {}
                            for show in unique_shows:
                                level = show.get('match_level', match_level)
                                level_counts[level] = level_counts.get(level, 0) + 1
                            
                            st.write(f"Shows by match level: {level_counts}")
                            
                            # Sort shows by match level to ensure exact matches appear first
                            unique_shows.sort(key=lambda x: x.get('match_level', 4))
                            
                            # Group shows by match level for better display
                            shows_by_level = {}
                            for show in unique_shows:
                                level = show.get('match_level', match_level)
                                if level not in shows_by_level:
                                    shows_by_level[level] = []
                                shows_by_level[level].append(show)
                            
                            # Display shows grouped by match level
                            for level in sorted(shows_by_level.keys()):
                                shows = shows_by_level[level]
                                color = match_level_colors.get(level, "#000000")
                                
                                # Add separator between match levels if there are multiple levels
                                if len(shows_by_level) > 1 and level > 1:
                                    st.write(f"---")
                                
                                # Sort shows by success_score within each match level
                                if 'success_score' in shows[0] if shows else {}:
                                    shows.sort(key=lambda x: x.get('success_score', 0), reverse=True)
                                
                                # Display each show with appropriate formatting
                                for show in shows:
                                    title = show.get('title', 'Unknown Title')
                                    
                                    # Format based on match level
                                    if level == 1:
                                        # Exact match - bold black
                                        st.markdown(f"**{title}**")
                                    elif level == 2:
                                        # Close match - normal black
                                        st.markdown(title)
                                    else:
                                        # Other match levels - use appropriate color
                                        st.markdown(f"<span style='color: {color};'>{title}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No matching shows available for the selected criteria.")
                
                # Tab 2: Network Analysis
                with tab2:
                    # Display network compatibility using our helper function
                    if hasattr(summary, 'network_compatibility'):
                        # Add a note about match level
                        match_level = getattr(summary, 'match_level', 0)
                        if match_level > 1:
                            st.info(f"Network compatibility is based on flexible matching (level {match_level}). Results may vary with exact matches.")
                        
                        # Use the helper function to render network compatibility
                        render_network_compatibility(summary.network_compatibility)
                    else:
                        st.info("No network compatibility data available.")
                
                # Tab 3: Recommendations
                with tab3:
                    # Display success factors if available
                    if hasattr(summary, 'success_factors') and summary.success_factors:
                        st.subheader("Success Factors")
                        # Display success factors using our helper function
                        render_success_factors(summary.success_factors)
                    
                    # Display recommendations if available
                    if hasattr(summary, 'recommendations') and summary.recommendations:
                        st.subheader("Recommendations")
                        # Use our improved helper function to render recommendations
                        render_recommendations(summary.recommendations, on_click_handler=None)
                    else:
                        st.info("No recommendations available for the selected criteria.")
                
                # No need for a second reset button since we already have one at the top of the page
        
    # Save state back to session
    update_page_state("show_optimizer", state)

if __name__ == "__main__":
    show()
