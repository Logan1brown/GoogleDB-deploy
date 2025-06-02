"""Show Optimizer Page.

This page provides an interface for optimizing show concepts based on
historical success patterns and network preferences.
"""

import streamlit as st
import logging
import pandas as pd
import sys
import os
import traceback
from typing import Dict, Any

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required
from src.dashboard.utils.style_config import COLORS, FONTS, CHART_DEFAULTS
from src.dashboard.components.optimizer_view import OptimizerView
from src.data_processing.show_optimizer.show_optimizer import ShowOptimizer
from src.dashboard.state.session import get_page_state, update_page_state

# Configure logging
logger = logging.getLogger(__name__)

@auth_required()
def show():
    """Main page content."""
    # Page title using style from style_config
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Show Optimizer</p>', unsafe_allow_html=True)
    
    # Add description
    st.write("Optimize your show concept based on historical success patterns and network preferences. This tool analyzes your concept against our database to calculate success probability, identify best-matching networks, and recommend optimizations.")
    
    try:
        # Get page state
        state = get_page_state("show_optimizer")
        if not state:
            state = {}
        
        # Initialize state if needed
        if "criteria" not in state:
            state["criteria"] = {}
        if "field_options" not in state:
            state["field_options"] = {}
        if "display_options" not in state:
            state["display_options"] = {}
        if "results" not in state:
            state["results"] = False
        if "summary" not in state:
            state["summary"] = None
        
        # Initialize the optimizer
        optimizer_view = OptimizerView()
        
        # Render the criteria section
        optimizer_view.render_criteria(state)
        
        # If we have results, render tabs directly in the page
        if state.get('summary') or st.session_state.get("optimizer_summary"):
            # Get summary from state or session state
            summary = state.get('summary') or st.session_state.get("optimizer_summary")
            
            if summary:
                # Create a clear header for the results section
                st.header("Analysis Results")
                
                # Create tabs directly in the page
                success_tab, network_tab, recommendations_tab = st.tabs(["Success Metrics", "Network Analysis", "Recommendations"])
                
                # Import helper functions for rendering tab content
                from src.dashboard.components.optimizer_helpers import (
                    render_success_metrics, 
                    render_network_compatibility, 
                    group_recommendations
                )
                
                # Tab 1: Success Metrics
                with success_tab:
                    if hasattr(summary, 'success_metrics') and summary.success_metrics:
                        render_success_metrics(summary)
                    else:
                        st.info("No success metrics available for the selected criteria.")
                
                # Tab 2: Network Analysis
                with network_tab:
                    st.subheader("Network Compatibility")
                    if hasattr(summary, 'network_compatibility') and summary.network_compatibility:
                        render_network_compatibility(summary.network_compatibility)
                    elif hasattr(summary, 'top_networks') and summary.top_networks:
                        # Create a simple table of top networks
                        import pandas as pd
                        # Check if networks have compatibility_score or score attribute
                        if hasattr(summary.top_networks[0], 'compatibility_score'):
                            networks_df = pd.DataFrame([(n.network_name, n.compatibility_score) for n in summary.top_networks], 
                                                       columns=["Network", "Compatibility Score"])
                        else:
                            networks_df = pd.DataFrame([(n.network_name, n.score) for n in summary.top_networks], 
                                                       columns=["Network", "Compatibility Score"])
                        st.dataframe(networks_df.sort_values("Compatibility Score", ascending=False))
                    else:
                        st.info("No network analysis available for the selected criteria.")
                
                # Tab 3: Recommendations
                with recommendations_tab:
                    if hasattr(summary, 'recommendations') and summary.recommendations:
                        # Group recommendations by type
                        grouped_recs = group_recommendations(summary.recommendations)
                        
                        # Display each group
                        for group_name, recs in grouped_recs.items():
                            st.subheader(f"{group_name.capitalize()} Recommendations")
                            for i, rec in enumerate(recs):
                                with st.expander(f"{rec.criteria_type.replace('_', ' ').title()}", expanded=i==0):
                                    # Check if recommendation has impact_score or score attribute
                                    if hasattr(rec, 'impact_score'):
                                        st.markdown(f"**Impact Score:** {rec.impact_score:.2f}")
                                    elif hasattr(rec, 'score'):
                                        st.markdown(f"**Score:** {rec.score:.2f}")
                                    
                                    # Check for description or explanation
                                    if hasattr(rec, 'description') and rec.description:
                                        st.markdown(f"**Description:** {rec.description}")
                                    if hasattr(rec, 'explanation') and rec.explanation:
                                        st.markdown(f"**Explanation:** {rec.explanation}")
                                    
                                    # Display match details if available
                                    if hasattr(rec, 'match_details') and rec.match_details:
                                        st.markdown("**Match Details:**")
                                        for category, details in rec.match_details.items():
                                            st.markdown(f"*{category}:* {details}")
                    else:
                        st.info("No recommendations available. Try adjusting your criteria.")
                
                # Add button to reset criteria
                if st.button("Reset Criteria", key="reset_criteria_button"):
                    # Clear criteria in state
                    if 'criteria' in state:
                        state['criteria'] = {}
                    if 'summary' in state:
                        del state['summary']
                    state['results'] = False
                    
                    # Also clear session state for compatibility
                    if "optimizer_criteria" in st.session_state:
                        st.session_state.optimizer_criteria = {}
                    if "optimizer_summary" in st.session_state:
                        del st.session_state.optimizer_summary
                        
                    st.rerun()
        
        # Save state back to session
        update_page_state("show_optimizer", state)
        
    except Exception as e:
        st.error(f"Error in show optimizer: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    show()
