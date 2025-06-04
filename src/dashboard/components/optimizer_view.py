"""Show Optimizer View Component.

This module contains the core functionality for the Show Optimizer feature.
It handles initialization and analysis of show concepts.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any

from src.data_processing.show_optimizer.show_optimizer import ShowOptimizer
from src.data_processing.show_optimizer.criteria_scorer import CriteriaScorer
from src.data_processing.show_optimizer.criteria_analyzer import CriteriaAnalyzer
from src.data_processing.show_optimizer.suggestion_analyzer import SuggestionAnalyzer

class OptimizerView:
    """Core functionality for the Show Optimizer.
    
    This class handles initialization and analysis of show concepts.
    It does not handle any UI rendering, which is done directly in the page file.
    """
    
    def __init__(self):
        """Initialize the optimizer view."""
        self.optimizer = ShowOptimizer()
        self.initialized = False
        
        # Ensure state dictionaries exist
        if "field_options" not in st.session_state:
            st.session_state.field_options = {}
        if "display_options" not in st.session_state:
            st.session_state.display_options = {}
        if "criteria" not in st.session_state:
            st.session_state.criteria = {}
        if "optimizer_results" not in st.session_state:
            st.session_state.optimizer_results = False
            
    def initialize(self, state: Dict) -> bool:
        """Initialize the optimizer components.
        
        Args:
            state: State dictionary for the optimizer
            
        Returns:
            True if initialization was successful, False otherwise
        """
        # Ensure state dictionaries exist
        if "field_options" not in state:
            state["field_options"] = {}
        if "display_options" not in state:
            state["display_options"] = {}
            
        if not self.initialized:
            try:
                with st.spinner("Initializing Show Optimizer..."):
                    # Initialize the optimizer - this will set up the field_manager
                    # Force refresh to ensure we get fresh data
                    if not self.optimizer.initialize(force_refresh=True):
                        st.error("Failed to initialize Show Optimizer components")
                        return False
                        
                    # Verify field_manager is available
                    if not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
                        st.error("Failed to initialize Show Optimizer. Field manager is not available.")
                        st.write("⚠️ Please try refreshing the page or contact support if the problem persists.")
                        return False
                    
                    # Cache field options in state
                    field_names = ["genre", "subgenres", "character_types", "source_type", "thematic_elements", "plot_elements", 
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
                            pass  # Skip warning for deployed app
                    
                    self.initialized = True
            except Exception as e:
                # Display user-friendly error
                st.error(f"Failed to initialize Show Optimizer: {str(e)}")
                st.write("⚠️ The application requires database access to function properly.")
                st.write("Please try again later or contact support if the problem persists.")
                return False
                
        return self.initialized
                
    def _run_analysis(self, state: Dict):
        """Run the analysis with the current criteria.
        
        Args:
            state: State dictionary for the optimizer
        """
        criteria = state.get('criteria', {})
        
        # Check if criteria is empty
        if not criteria:
            st.info("Select criteria to analyze your concept.")
            return
            
        # Check if optimizer is initialized
        if not self.initialized:
            # Try to initialize if not already initialized
            if not self.initialize(state):
                st.error("Show Optimizer is not initialized. Please refresh the page and try again.")
                return
            
        # Optimizer initialization status checked above
        
        # Check if field_manager is available
        if not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
            st.error("Field manager is not available. Please refresh the page.")
            return
        
        # Continue with analysis
            
        try:
            # Run the analysis
            with st.spinner("Analyzing concept..."):
                # Ensure criteria is properly formatted
                # Convert any list values with single items to scalar values
                normalized_criteria = {}
                for key, value in criteria.items():
                    if isinstance(value, list) and len(value) == 1:
                        normalized_criteria[key] = value[0]
                    else:
                        normalized_criteria[key] = value
                
                # Normalize criteria silently
                
                # Ensure genre is an integer
                if 'genre' in normalized_criteria and not isinstance(normalized_criteria['genre'], int):
                    try:
                        normalized_criteria['genre'] = int(normalized_criteria['genre'])
                    except (ValueError, TypeError):
                        pass
                
                # Final criteria ready for analysis
                
                try:
                    # Check if criteria_scorer has data
                    if hasattr(self.optimizer, 'criteria_scorer'):
                        criteria_data = self.optimizer.criteria_scorer.fetch_criteria_data(force_refresh=True)
                        if criteria_data.empty:
                            st.warning("No criteria data available for analysis.")
                            return
                    
                    # Run the analysis with normalized criteria
                    
                    # Try to catch any exceptions during analysis
                    try:
                        # Check if criteria_analyzer is initialized
                        if not (hasattr(self.optimizer, 'criteria_analyzer') and self.optimizer.criteria_analyzer is not None):
                            st.warning("Analysis components are not properly initialized.")
                            return
                            
                        # Run the actual analysis
                        summary = self.optimizer.analyze_concept(normalized_criteria)
                        
                        # Debug info to verify our fixes
                        st.write(f"Debug: Summary type: {type(summary)}")
                        if summary:
                            st.write(f"Debug: Has component_scores: {hasattr(summary, 'component_scores')}")
                            if hasattr(summary, 'component_scores'):
                                st.write(f"Debug: Component scores keys: {list(summary.component_scores.keys())}")
                        
                        # Analysis complete
                        
                    except Exception as e:
                        st.error(f"Error analyzing concept: {str(e)}")
                        summary = None
                except Exception as analysis_error:
                    st.error(f"Error in analysis: {str(analysis_error)}")
                    return
            
            # Store results in state
            if summary:
                # Update state with results
                state['summary'] = summary
                state['results'] = True
                # Store results in state and session state
                
                # Also update session state for compatibility
                st.session_state.optimizer_summary = summary
                st.session_state.optimizer_results = True
                
                # Do not force rerun - let the page flow handle rendering
            else:
                # No valid results were produced
                st.error("Analysis failed to produce results. Please try different criteria.")
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
            
    # All UI rendering and criteria updates are now handled directly in the page file
