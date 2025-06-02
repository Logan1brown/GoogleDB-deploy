"""Show Optimizer View Component.

This module contains the main view component for the Show Optimizer feature.
It handles the UI presentation of the optimizer results and user interactions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from src.data_processing.show_optimizer.show_optimizer import ShowOptimizer
from src.data_processing.show_optimizer.criteria_scorer import CriteriaScorer
from src.data_processing.show_optimizer.criteria_analyzer import CriteriaAnalyzer
from src.data_processing.show_optimizer.suggestion_analyzer import SuggestionAnalyzer, OptimizationSummary, Recommendation
from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.success_analysis.success_analyzer import SuccessAnalyzer

from src.dashboard.components.optimizer_helpers import (
    get_id_for_name, get_ids_for_names,
    render_success_metrics, render_metric_card, render_info_card,
    render_success_factors, render_network_compatibility, group_recommendations,
    render_recommendation_group, render_content_criteria, render_production_criteria, render_format_criteria
)

# No logger needed for deployed app

class OptimizerView:
    """Main view component for the Show Optimizer."""
    
    def __init__(self):
        """Initialize the optimizer view."""
        self.optimizer = ShowOptimizer()
        self.initialized = False
            
    def initialize(self, state: Dict) -> bool:
        """Initialize the optimizer components.
        
        Args:
            state: State dictionary for the optimizer
            
        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.initialized:
            try:
                with st.spinner("Initializing Show Optimizer..."):
                    # Bypass the cached initialize method and directly initialize components
                    # This matches how Comp Builder initializes and avoids st.cache_data issues
                    try:
                        # Initialize components directly (similar to ShowOptimizer.initialize but without cache)
                        self.optimizer.criteria_scorer = CriteriaScorer(self.optimizer.shows_analyzer, self.optimizer.success_analyzer)
                        self.optimizer.field_manager = self.optimizer.criteria_scorer.field_manager
                        self.optimizer.criteria_analyzer = CriteriaAnalyzer(self.optimizer.shows_analyzer, self.optimizer.success_analyzer)
                        self.optimizer.suggestion_analyzer = SuggestionAnalyzer(self.optimizer.shows_analyzer, self.optimizer.success_analyzer)
                        self.optimizer.initialized = True
                        self.optimizer.last_update = datetime.now()
                    except Exception as e:
                        st.error(f"Failed to initialize optimizer components: {str(e)}")
                        return False
                    
                    # Verify field_manager is available
                    if not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
                        st.error("Failed to initialize Show Optimizer. Field manager is not available.")
                        st.write("⚠️ Please try refreshing the page or contact support if the problem persists.")
                        return False
                    
                    # Cache field options in state
                    field_names = ["genre", "character_types", "source_type", "thematic_elements", "plot_elements", 
                                   "tone", "time_setting", "location_setting", "network", "studios", "team_members", "order_type"]
                    for field_name in field_names:
                        try:
                            options = self.optimizer.get_field_options(field_name)
                            if options:
                                state["field_options"][field_name] = options
                                state["display_options"][field_name] = [
                                    (option.id, option.name) for option in options
                                ]
                        except Exception:
                            pass    # Skip warning for deployed app
                    
                    self.initialized = True
            except Exception as e:
                # Display user-friendly error
                st.error(f"Failed to initialize Show Optimizer: {str(e)}")
                st.write("⚠️ The application requires database access to function properly.")
                st.write("Please try again later or contact support if the problem persists.")
                return False
                
        return self.initialized
    
    def render_criteria(self, state: Dict[str, Any]) -> None:
        """Render only the criteria section of the optimizer view.

        Args:
            state: Page state dictionary
        """
        # Initialize optimizer components if needed
        if not self.initialized:
            if not self.initialize(state):
                st.error("Failed to initialize Show Optimizer. Please refresh the page and try again.")
                return
        
        # Get criteria and display options from state
        criteria = state.get('criteria', {})
        display_options = state.get('display_options', {})
        
        # Check if field options are available in state
        if not display_options:
            st.error("Unable to load field options from the database.")
            st.info("This may be due to a temporary connection issue or database maintenance.")
            
            if st.button("Retry Initialization", type="primary"):
                self.initialized = False
                self.initialize(state)
                
            st.write("If the problem persists, please try again later or contact support.")
            return
        
        st.header("Build Your Show Concept")
        
        # Create two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Render criteria sections using helper functions
            render_content_criteria(state, self._update_criteria_and_analyze)
            render_production_criteria(state, self._update_criteria_and_analyze)
            render_format_criteria(state, self._update_criteria_and_analyze)
            
            # Save criteria to session state before running analysis
            criteria = state.get('criteria', {})
            st.session_state.optimizer_criteria = criteria.copy()
            
            # Run analysis automatically when criteria changes
            if criteria:
                self._run_analysis(state)
                
        with col2:
            # Results section placeholder
            if criteria:
                if not (state.get('summary') or st.session_state.get("optimizer_summary")):
                    st.info("Select or adjust criteria on the left to analyze your show concept.")
            else:
                st.info("Select criteria on the left to analyze your show concept.")
    
    def render(self, state: Dict[str, Any]) -> None:
        """Render the complete optimizer view.

        Args:
            state: Page state dictionary
        """
        # This method is kept for backward compatibility
        # Initialize optimizer components if needed
        if not self.initialized:
            if not self.initialize(state):
                st.error("Failed to initialize Show Optimizer. Please refresh the page and try again.")
                return
        
        # Get criteria from state
        criteria = state.get("criteria", {})
        
        # Create two columns
        col1, col2 = st.columns([1, 2])
        
        # Render the concept builder
        self._render_concept_builder(state, col1, col2)
    
    def _render_concept_builder(self, state: Dict, col1, col2):
        """Render the concept builder section.
        
        Args:
            state: State dictionary for the optimizer
        """
        # Get criteria and display options from state
        criteria = state.get('criteria', {})
        display_options = state.get('display_options', {})
        
        # Check if field options are available in state
        if not display_options:
            st.error("Unable to load field options from the database.")
            st.info("This may be due to a temporary connection issue or database maintenance.")
            
            if st.button("Retry Initialization", type="primary"):
                self.initialized = False
                self.initialize(state)
                
            st.write("If the problem persists, please try again later or contact support.")
            return
        
        st.header("Build Your Show Concept")
        
        # Use the columns passed as parameters instead of creating new ones
        
        with col1:
            # Render criteria sections using helper functions
            render_content_criteria(state, self._update_criteria_and_analyze)
            render_production_criteria(state, self._update_criteria_and_analyze)
            render_format_criteria(state, self._update_criteria_and_analyze)
            
            # Save criteria to session state before running analysis
            criteria = state.get('criteria', {})
            st.session_state.optimizer_criteria = criteria.copy()
            
            # Run analysis automatically when criteria changes
            if criteria:
                self._run_analysis(state)
                
        with col2:
            # Results section
            
            # If we have criteria, show results
            if criteria:
                # Get results from state or session state if available
                if state.get('summary') or st.session_state.get("optimizer_summary"):
                    # Call our own render method
                    self._render_results(state)
                else:
                    st.info("Select or adjust criteria on the left to analyze your show concept.")
            else:
                st.info("Select criteria on the left to analyze your show concept.")
                
    def _render_results(self, state: Dict):
        """Render the analysis results in a tabbed interface.
        
        Args:
            state: State dictionary containing summary and results
        """
        # Get summary from state or session state
        summary = None
        if 'summary' in state and state['summary']:
            summary = state['summary']
        elif 'optimizer_summary' in st.session_state:
            summary = st.session_state.optimizer_summary
            
        if not summary:
            st.info("No analysis results available. Try adjusting your criteria.")
            return False
        
        # Create a clear header for the results section
        st.header("Analysis Results")
        
        # Create a container for the tabs to ensure proper rendering
        tabs_container = st.container()
        
        # Create tabs for main content - use a simple, direct approach like in StudioView
        with tabs_container:
            success_tab, network_tab, recommendations_tab = st.tabs(["Success Metrics", "Network Analysis", "Recommendations"])
        
        # Helper functions are already imported at the top of the file
        
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
        
        return True
        
    def _run_analysis(self, state: Dict):
        """Run the analysis with the current criteria.
        
        Args:
            state: State dictionary for the optimizer
        """
        # Get criteria from state
        criteria = state.get('criteria', {})
        
        # Check if optimizer is initialized
        if not self.initialized:
            st.error("Show Optimizer is not initialized. Please refresh the page and try again.")
            return
        
        # Check if field_manager is available
        if not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
            try:
                # Try to reinitialize
                with st.spinner("Attempting to reinitialize Show Optimizer..."):
                    self.initialized = self.optimizer.initialize(force_refresh=True)
                if not self.initialized or not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
                    st.error("Could not initialize field manager. Please try again later.")
                    return
            except Exception as e:
                st.error(f"Error connecting to database: {str(e)}")
                return
        
        # Validate and analyze criteria
        if not criteria:
            st.info("Select criteria to analyze your concept.")
            return
            
        try:
            # Log the criteria for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Analyzing criteria: {criteria}")
            
            # Run the analysis
            with st.spinner("Analyzing concept..."):
                # Ensure criteria is properly formatted
                # Convert any list values with single items to scalar values
                # This is a common issue with form submissions
                normalized_criteria = {}
                for key, value in criteria.items():
                    if isinstance(value, list) and len(value) == 1:
                        normalized_criteria[key] = value[0]
                    else:
                        normalized_criteria[key] = value
                
                # Ensure genre is an integer
                if 'genre' in normalized_criteria and not isinstance(normalized_criteria['genre'], int):
                    try:
                        normalized_criteria['genre'] = int(normalized_criteria['genre'])
                    except (ValueError, TypeError):
                        logger.warning(f"Failed to convert genre to int: {normalized_criteria['genre']}")
                
                # Run the analysis with normalized criteria
                summary = self.optimizer.analyze_concept(normalized_criteria)
            
            # Store results in state
            if summary:
                # Update state with results
                state['summary'] = summary
                state['results'] = True
                
                # Also update session state for compatibility
                st.session_state.optimizer_summary = summary
                st.session_state.optimizer_results = True
                
                # Force rerun to show results
                st.rerun()
            else:
                st.error("Analysis failed to produce results. Please try different criteria.")
                logger.error(f"analyze_concept returned None for criteria: {normalized_criteria}")
                state['results'] = False
                if 'summary' in state:
                    del state['summary']
                    
                # Also update session state for compatibility
                st.session_state.optimizer_results = False
                if "optimizer_summary" in st.session_state:
                    del st.session_state.optimizer_summary
                    
        except Exception as e:
            st.error(f"Error analyzing concept: {str(e)}")
            state['results'] = False
            if 'summary' in state:
                del state['summary']
                
            # Also update session state for compatibility
            st.session_state.optimizer_results = False
            if "optimizer_summary" in st.session_state:
                del st.session_state.optimizer_summary
            
    def _update_criteria_and_analyze(self, field_name, value):
        """Update a specific criteria field and run the analysis.
        
        Args:
            field_name: Name of the field to update
            value: New value for the field
        """
        # Get current state from session state
        from src.dashboard.state.session import get_page_state, update_page_state
        state = get_page_state("show_optimizer")
        
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
            self._run_analysis(state)
            
    # The _render_results, _render_concept_analysis, _render_network_analysis, and _render_recommendations methods
    # have been moved to helper functions in optimizer_helpers.py to reduce file size and improve maintainability.
    # This follows the Comp Builder pattern of separating view logic from state management.
