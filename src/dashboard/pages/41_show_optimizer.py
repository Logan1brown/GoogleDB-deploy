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
                        
                        # Define match level names and colors based on percentage criteria
                        match_level_names = {
                            1: "100% Criteria Match",
                            2: "75% Criteria Match",
                            3: "50% Criteria Match",
                            4: "25% Criteria Match",
                            5: "Minimal Criteria Match"
                        }
                        
                        match_level_colors = {
                            1: "#000000",  # Black for 100% match
                            2: "#000000",  # Black for 75% match
                            3: "#666666",  # Grey for 50% match
                            4: "#999999",  # Light grey for 25% match
                            5: "#AAAAAA"   # Very light grey for minimal match
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
                        for level in range(1, 6):
                            if level in match_counts_by_level and match_counts_by_level[level] > 0:
                                level_name = match_level_names.get(level, f"Level {level}")
                                count = match_counts_by_level[level]
                                match_level_summary.append(f"{count} {level_name}")
                        
                        # Count shows by match level for verification
                        level_counts = {}
                        if 'match_level' in summary.matching_shows.columns:
                            for level in range(1, 6):
                                count = len(summary.matching_shows[summary.matching_shows['match_level'] == level])
                                if count > 0:
                                    level_counts[level] = count
                        else:
                            # Use match_counts_by_level if available
                            level_counts = match_counts_by_level
                        
                        # Generate fully programmatic descriptions based on match level
                        level_descriptions = {}
                        for level in level_counts.keys():
                            # Match level directly corresponds to criteria differences + 1
                            # Level 1 = 0 differences, Level 2 = 1 difference, etc.
                            diff = level - 1
                            
                            if diff == 0:
                                level_descriptions[level] = "All criteria matched"
                            elif diff == 1:
                                level_descriptions[level] = f"Missing {diff} criterion"
                            else:
                                level_descriptions[level] = f"Missing {diff} criteria"
                        
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
                            
                            # No need to display the unique shows count
                            
                            # Count shows by match level for verification
                            level_counts = {}
                            for show in unique_shows:
                                level = show.get('match_level', match_level)
                                level_counts[level] = level_counts.get(level, 0) + 1
                            
                            # Format the level counts with descriptions
                            formatted_counts = {f"{level_descriptions.get(level, f'Level {level}')}": count 
                                              for level, count in level_counts.items()}
                            
                            st.write(f"Shows by match level: {formatted_counts}")
                            
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
                                        # 100% criteria match - bold black
                                        st.markdown(f"**{title}**")
                                    elif level == 2:
                                        # 75% criteria match - normal black
                                        st.markdown(title)
                                    else:
                                        # 50% or 25% criteria match - use appropriate color
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
    
    # Add cache status section at the bottom of the page
    st.markdown("---")
    with st.expander("Cache Status"):
        # Display cache status
        if optimizer_view.optimizer:
            optimizer_view.optimizer.display_cache_status(show_details=True)
            
            # Add button to invalidate cache
            if st.button("Invalidate Cache"):
                optimizer_view.optimizer.invalidate_cache(components=True, data=True)
                st.success("Cache invalidated successfully. Please refresh the page to see the changes.")
                st.rerun()

if __name__ == "__main__":
    show()
