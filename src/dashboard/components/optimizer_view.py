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
                    # Initialize components directly
                    try:
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
                
                # Ensure genre is an integer
                if 'genre' in normalized_criteria and not isinstance(normalized_criteria['genre'], int):
                    try:
                        normalized_criteria['genre'] = int(normalized_criteria['genre'])
                    except (ValueError, TypeError):
                        pass
                
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
                
                # Do not force rerun - let the page flow handle rendering
            else:
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
