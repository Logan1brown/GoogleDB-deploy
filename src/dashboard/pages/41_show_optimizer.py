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
from src.dashboard.components.optimizer_helpers import (
    render_success_metrics, render_network_compatibility, group_recommendations,
    render_recommendation_group, render_content_criteria, render_production_criteria, render_format_criteria,
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
            if state['criteria']:
                optimizer_view._run_analysis(state)
        
        # Render criteria sections using helper functions
        render_content_criteria(state, update_criteria_and_analyze)
        render_production_criteria(state, update_criteria_and_analyze)
        render_format_criteria(state, update_criteria_and_analyze)
        
        # Save criteria to session state before running analysis
        criteria = state.get('criteria', {})
        st.session_state.optimizer_criteria = criteria.copy()
        
        # Run analysis automatically when criteria changes
        if criteria:
            optimizer_view._run_analysis(state)
    
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
                    # Display component scores in metrics
                        
                    # Display success metrics using available attributes
                    # Create columns for metrics - now 4 columns for all scores
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        if hasattr(summary, 'overall_success_probability'):
                            probability = summary.overall_success_probability
                            if probability is not None:
                                st.metric("Success Probability", f"{probability:.0%}")
                            else:
                                st.metric("Success Probability", "N/A")
                                
                            if hasattr(summary, 'confidence'):
                                st.caption(f"Confidence: {summary.confidence.capitalize()}")
                        else:
                            st.info("Overall success probability not available.")
                    
                    with metric_col2:
                        if hasattr(summary, 'component_scores') and summary.component_scores and "audience" in summary.component_scores:
                            audience_score = summary.component_scores["audience"]
                            if hasattr(audience_score, 'score'):
                                if audience_score.score is not None:
                                    st.metric("Audience Appeal", f"{audience_score.score:.0%}")
                                    if hasattr(audience_score, 'sample_size'):
                                        st.caption(f"Sample size: {audience_score.sample_size} shows")
                                else:
                                    st.metric("Audience Appeal", "N/A")
                                    st.caption("Insufficient data")
                            else:
                                st.error("Audience score object missing 'score' attribute")
                    
                    with metric_col3:
                        if hasattr(summary, 'component_scores') and summary.component_scores and "critics" in summary.component_scores:
                            critics_score = summary.component_scores["critics"]
                            if hasattr(critics_score, 'score'):
                                if critics_score.score is not None:
                                    st.metric("Critics Score", f"{critics_score.score:.0%}")
                                    if hasattr(critics_score, 'sample_size'):
                                        st.caption(f"Sample size: {critics_score.sample_size} shows")
                                else:
                                    st.metric("Critics Score", "N/A")
                                    st.caption("Insufficient data")
                            else:
                                st.error("Critics score object missing 'score' attribute")
                    
                    with metric_col4:
                        if hasattr(summary, 'component_scores') and summary.component_scores and "longevity" in summary.component_scores:
                            longevity_score = summary.component_scores["longevity"]
                            if hasattr(longevity_score, 'score'):
                                if longevity_score.score is not None:
                                    st.metric("Longevity Score", f"{longevity_score.score:.0%}")
                                    if hasattr(longevity_score, 'sample_size'):
                                        st.caption(f"Sample size: {longevity_score.sample_size} shows")
                                else:
                                    st.metric("Longevity Score", "N/A")
                                    st.caption("Insufficient data")
                            else:
                                st.error("Longevity score object missing 'score' attribute")
                    
                    # Display matching show titles
                    st.subheader("Matching Shows")
                    
                    # Display matching shows with match level differentiation
                    if hasattr(summary, 'matching_shows') and summary.matching_shows is not None and not summary.matching_shows.empty:
                        # Get match level from confidence info if available
                        match_level = 1  # Default to exact match
                        if hasattr(summary, 'confidence_info') and summary.confidence_info:
                            match_level = summary.confidence_info.get('match_level', 1)
                        
                        # Display sample size and match level
                        sample_size = len(summary.matching_shows)
                        
                        # Define match level names and colors
                        match_level_names = {
                            1: "Exact Match",
                            2: "Close Match",
                            3: "Partial Match",
                            4: "Minimal Match"
                        }
                        match_level_colors = {
                            1: "#000000",  # Bold black
                            2: "#333333",  # Normal black
                            3: "#666666",  # Dark gray
                            4: "#999999"   # Light gray
                        }
                        match_level_name = match_level_names.get(match_level, "Flexible Match")
                        
                        # Get match counts by level from confidence info
                        match_counts_by_level = {}
                        if hasattr(summary, 'confidence_info') and summary.confidence_info:
                            match_counts_by_level = summary.confidence_info.get('match_counts_by_level', {})
                        
                        # Create a summary of match levels
                        match_level_summary = []
                        for level in range(1, 5):
                            if level in match_counts_by_level and match_counts_by_level[level] > 0:
                                level_name = match_level_names.get(level, f"Level {level}")
                                count = match_counts_by_level[level]
                                match_level_summary.append(f"{count} {level_name}")
                        
                        # Display the match level distribution
                        if match_level_summary:
                            st.write(f"Found {sample_size} shows: {', '.join(match_level_summary)}")
                        else:
                            st.write(f"Found {sample_size} shows with similar criteria ({match_level_name})")
                        
                        # Add legend for the colors
                        st.write("**Bold** = Exact match, Normal = Close match, Grey = Partial match")
                        
                        # Display up to 100 shows with color coding based on individual match level
                        for i, (_, show) in enumerate(summary.matching_shows.head(100).iterrows()):
                            if i >= 100:
                                break
                                
                            title = show.get('title', 'Unknown Title')
                            
                            # Get this show's individual match level
                            show_match_level = show.get('match_level', match_level)  # Fall back to global match level if not present
                            color = match_level_colors.get(show_match_level, "#000000")
                            
                            # Format based on this show's match level
                            if show_match_level == 1:
                                # Exact match - bold black
                                st.markdown(f"**{title}**")
                            else:
                                # Other match levels - use appropriate color
                                st.markdown(f"<span style='color: {color};'>{title}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No matching shows available for the selected criteria.")
                
                # Tab 2: Network Analysis
                with tab2:
                    # Display network compatibility directly without helper
                    if hasattr(summary, 'network_compatibility') and summary.network_compatibility:
                        networks = summary.network_compatibility
                        
                        # Add a note about match level
                        match_level = getattr(summary, 'match_level', 0)
                        if match_level > 1:
                            st.info(f"Network compatibility is based on flexible matching (level {match_level}). Results may vary with exact matches.")
                        
                        # Create a dataframe for the networks
                        network_data = []
                        for network in networks:
                            # Format the success probability and compatibility score for display
                            success_prob = f"{network.success_probability:.0%}" if network.success_probability is not None else "N/A"
                            compat_score = f"{network.compatibility_score:.0%}" if network.compatibility_score is not None else "N/A"
                            
                            network_data.append({
                                "Network": network.network_name,
                                "Success Probability": success_prob,
                                "Compatibility": compat_score,
                                "Sample Size": network.sample_size
                            })
                        
                        # Sort by compatibility score (descending)
                        network_df = pd.DataFrame(network_data)
                        if not network_df.empty:
                            st.write("Top networks for your concept:")
                            st.dataframe(network_df)
                            
                            # Add explanation of what the scores mean
                            st.caption("**Compatibility**: How well your concept aligns with the network's content strategy")
                            st.caption("**Success Probability**: Estimated chance of success if your show airs on this network")
                            st.caption("**Sample Size**: Number of similar shows used in this analysis")
                    elif hasattr(summary, 'top_networks') and summary.top_networks:
                        # Create a simple table of top networks
                        network_df = pd.DataFrame(summary.top_networks)
                        st.dataframe(network_df)
                    else:
                        st.info("No network compatibility data available for the selected criteria.")
                
                # Tab 3: Recommendations
                with tab3:
                    # Display success factors if available
                    if hasattr(summary, 'success_factors') and summary.success_factors:
                        st.subheader("Success Factors")
                        # Display success factors
                        render_success_factors(summary.success_factors)
                        
                        # Group by type
                        grouped = {
                            "add": [],
                            "replace": [],
                            "remove": [],
                            "consider": []
                        }
                        
                        # Use summary.recommendations instead of undefined 'recommendations' variable
                        for rec in summary.recommendations:
                            if hasattr(rec, 'recommendation_type') and rec.recommendation_type in grouped:
                                grouped[rec.recommendation_type].append(rec)
                        
                        # Display each group
                        if grouped["add"]:
                            st.subheader("Consider Adding")
                            for rec in grouped["add"][:3]:
                                st.write(f"**{rec.criteria_type}:** {rec.suggested_name}")
                                if hasattr(rec, 'impact_score'):
                                    st.write(f"Impact: {rec.impact_score:.2f}")
                                st.write("---")
                        
                        if grouped["replace"]:
                            st.subheader("Consider Replacing")
                            for rec in grouped["replace"][:3]:
                                # Get the human-readable name for the current value if available
                                current_name = rec.current_value
                                if hasattr(rec, 'current_name') and rec.current_name:
                                    current_name = rec.current_name
                                
                                # Get the field name from the criteria type
                                field_name = rec.criteria_type.replace('_', ' ').title()
                                
                                st.write(f"Replace **{field_name}**: '{current_name}' with '{rec.suggested_name}'")
                                if hasattr(rec, 'impact_score'):
                                    st.write(f"Impact: +{rec.impact_score:.2f} (success probability)")
                                st.write("---")
                        
                        if grouped["remove"]:
                            st.subheader("Consider Removing")
                            for rec in grouped["remove"][:3]:
                                st.write(f"**{rec.criteria_type}:** {rec.current_value}")
                                if hasattr(rec, 'impact_score'):
                                    st.write(f"Impact: {rec.impact_score:.2f}")
                                st.write("---")
                        
                        if grouped["consider"]:
                            st.subheader("Additional Insights")
                            for rec in grouped["consider"][:3]:
                                st.write(f"**{rec.criteria_type}:** {rec.suggested_name}")
                                if hasattr(rec, 'explanation'):
                                    st.write(rec.explanation)
                                if hasattr(rec, 'impact_score'):
                                    st.write(f"Impact: {rec.impact_score:.2f}")
                                st.write("---")
                    else:
                        st.info("No recommendations available for the selected criteria.")
                
                # No need for a second reset button since we already have one at the top of the page
        
    # Save state back to session
    update_page_state("show_optimizer", state)

if __name__ == "__main__":
    show()
