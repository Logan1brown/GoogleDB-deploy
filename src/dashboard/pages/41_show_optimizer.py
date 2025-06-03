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
                            st.metric("Success Probability", f"{probability:.0%}")
                            if hasattr(summary, 'confidence'):
                                st.caption(f"Confidence: {summary.confidence.capitalize()}")
                        else:
                            st.info("Overall success probability not available.")
                    
                    with metric_col2:
                        if hasattr(summary, 'component_scores') and summary.component_scores and "audience" in summary.component_scores:
                            audience_score = summary.component_scores["audience"]
                            if hasattr(audience_score, 'score'):
                                st.metric("Audience Appeal", f"{audience_score.score:.0%}")
                                if hasattr(audience_score, 'sample_size'):
                                    st.caption(f"Sample size: {audience_score.sample_size} shows")
                            else:
                                st.error("Audience score object missing 'score' attribute")
                    
                    with metric_col3:
                        if hasattr(summary, 'component_scores') and summary.component_scores and "critics" in summary.component_scores:
                            critics_score = summary.component_scores["critics"]
                            if hasattr(critics_score, 'score'):
                                st.metric("Critics Score", f"{critics_score.score:.0%}")
                                if hasattr(critics_score, 'sample_size'):
                                    st.caption(f"Sample size: {critics_score.sample_size} shows")
                            else:
                                st.error("Critics score object missing 'score' attribute")
                    
                    with metric_col4:
                        if hasattr(summary, 'component_scores') and summary.component_scores and "longevity" in summary.component_scores:
                            longevity_score = summary.component_scores["longevity"]
                            if hasattr(longevity_score, 'score'):
                                st.metric("Longevity Score", f"{longevity_score.score:.0%}")
                                if hasattr(longevity_score, 'sample_size'):
                                    st.caption(f"Sample size: {longevity_score.sample_size} shows")
                            else:
                                st.error("Longevity score object missing 'score' attribute")
                    
                    # Display matching show titles
                    st.subheader("Matching Shows")
                    st.write("Shows matching all selected criteria:")
                    
                    # Display matching titles from the summary object
                    if hasattr(summary, 'matching_titles') and summary.matching_titles:
                        # Show sample size
                        st.caption(f"Sample size: {len(summary.matching_titles)} shows")
                        
                        # Display titles in a scrollable container
                        titles_html = "<div style='max-height: 300px; overflow-y: auto;'><ul>"
                        for title in summary.matching_titles:
                            titles_html += f"<li>{title}</li>"
                        titles_html += "</ul></div>"
                        st.markdown(titles_html, unsafe_allow_html=True)
                    else:
                        st.info("No matching shows available for the selected criteria.")
                
                # Tab 2: Network Analysis
                with tab2:
                    # Display network compatibility directly without helper
                    if hasattr(summary, 'network_compatibility') and summary.network_compatibility:
                        networks = summary.network_compatibility
                        
                        # Create a dataframe for the networks
                        network_data = []
                        for network in networks:
                            network_data.append({
                                "Network": network.network_name,
                                "Success Probability": network.success_probability,
                                "Compatibility": network.compatibility_score,
                                "Sample Size": network.sample_size
                            })
                            
                        network_df = pd.DataFrame(network_data)
                        st.dataframe(network_df)
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
                                st.write(f"Replace **{rec.current_value}** with **{rec.suggested_name}**")
                                if hasattr(rec, 'impact_score'):
                                    st.write(f"Impact: {rec.impact_score:.2f}")
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
