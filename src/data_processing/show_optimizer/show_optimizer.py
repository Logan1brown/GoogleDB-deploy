"""Show Optimizer: Main coordinator for show optimization.

This component coordinates the Show Optimizer feature, integrating all the
specialized components (FieldManager, CriteriaScorer,
OptimizerView) to provide a unified interface for the UI.

Key responsibilities:

1. Component Coordination:
   - Initialize and manage all optimizer components
   - Coordinate data flow between components
   - Provide a unified interface for the UI

2. Optimization Pipeline:
   - Validate and normalize input criteria
   - Match shows based on criteria
   - Calculate success metrics and scores
   - Analyze patterns and generate recommendations
   - Format results for UI presentation

3. Performance:

   - Support incremental updates for real-time UI
   - Handle partial criteria sets
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from datetime import datetime, timedelta
import streamlit as st

from .optimizer_concept_analyzer import ConceptAnalyzer, OptimizationSummary

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer, NetworkMatch, ComponentScore
from .recommendation_engine import SuccessFactor, RecommendationEngine, Recommendation
from src.dashboard.components.optimizer_view import OptimizerView
from .optimizer_config import OptimizerConfig
from .optimizer_matcher import Matcher




class ShowOptimizer:
    """Main coordinator for show optimization.
    
    This class serves as the central orchestrator for the show optimization process,
    integrating various specialized components to analyze show concepts, provide
    recommendations, and calculate success metrics.
    
    The ShowOptimizer uses a ConceptAnalyzer as its primary analysis engine.
    It implements centralized data fetching and caching for performance optimization.
    """
    
    def __init__(self):
        """Initialize the show optimizer with default configuration.
        
        Sets up core analyzers and prepares component attributes that will be
        initialized later. The actual initialization of components happens in
        the initialize() method which is called when needed.
        """
        # Core analyzers - these are initialized immediately
        self.shows_analyzer = None
        self.success_analyzer = None
        
        # Primary components
        self.field_manager = None
        self.criteria_scorer = None
        self.network_analyzer = None
        self.concept_analyzer = None
        self.optimizer_view = None
        self.recommendation_engine = None
        
    def fetch_and_integrate_data(self) -> Dict[str, Any]:
        """Fetch and integrate data from different sources.
        
        This method centralizes data fetching and integration to ensure all components
        work with the same dataset. It fetches show data from shows_analyzer and success
        metrics from success_analyzer, then integrates them into a unified dataset.
        
        Returns:
            Dictionary containing integrated data sets needed for optimization
        """
        try:
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            
            # Get show data
            shows_df = self.shows_analyzer.fetch_optimizer_data()
            if shows_df.empty:
                st.error("No show data available")
                return {}
            
            # Get reference data (genres, networks, etc.)
            reference_data = self.shows_analyzer.get_reference_data()
            
            # Get success metrics
            success_df = self.success_analyzer.success_data
            
            # Integrate the data
            # First, ensure we have the necessary columns for integration
            # The api_show_comp_data view uses 'id' as the show identifier
            if 'id' in shows_df.columns:
                # Create a copy to avoid modifying the original
                integrated_shows = shows_df.copy()
                
                # Rename 'id' to 'show_id' for consistency
                if 'show_id' not in integrated_shows.columns:
                    integrated_shows['show_id'] = integrated_shows['id']
                
                # Merge with success metrics if available
                if not success_df.empty:
                    # Reset index to make show_id a column again if it's the index
                    if success_df.index.name == 'show_id':
                        success_df = success_df.reset_index()
                    
                    # Only keep success metrics that have matching shows
                    if 'show_id' in success_df.columns and 'show_id' in integrated_shows.columns:
                        success_df = success_df[success_df['show_id'].isin(integrated_shows['show_id'])]
                        
                        # Check for duplicate columns that might cause conflicts
                        # tmdb_status exists in both datasets, so we'll drop it from success_df
                        conflict_columns = ['tmdb_status', 'tmdb_seasons']
                        for col in conflict_columns:
                            if col in success_df.columns and col in integrated_shows.columns:
                                success_df = success_df.drop(columns=[col])
                        
                        # Now merge the datasets
                        if not success_df.empty:
                            # Only merge columns from success_df that are not present in shows_df (except for 'show_id')
                            shows_columns = set(integrated_shows.columns)
                            merge_columns = [col for col in success_df.columns if col not in shows_columns and col != 'id']
                            # Always include 'show_id' for the join
                            columns_to_merge = ['show_id'] + merge_columns
                            integrated_shows = pd.merge(
                                integrated_shows,
                                success_df[columns_to_merge],
                                on='show_id',
                                how='left'
                            )
                else:
                    st.warning("Could not integrate success metrics: missing 'show_id' column")
                
                # Create the integrated data dictionary
                integrated_data = {
                    'shows': integrated_shows,
                    'reference_data': reference_data
                }
                
                if OptimizerConfig.DEBUG_MODE:
                    pass  # Debug message removed to reduce spam
                return integrated_data
            else:
                st.error("Show data missing required columns for integration")
                return {}
            
        except Exception as e:
            st.error(f"Error fetching and integrating data: {str(e)}")
            return {}
            
    def initialize(self) -> bool:
        """Initialize all components and analyzers.
        
        This method initializes all required components for the show optimizer,
        including the field manager, criteria scorer, and concept analyzer.
        It also fetches and integrates the necessary data.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            
            # Initialize analyzers first
            self.shows_analyzer = ShowsAnalyzer()
            self.success_analyzer = SuccessAnalyzer(shows_analyzer=self.shows_analyzer)
            
            # Now fetch and integrate data
            integrated_data = self.fetch_and_integrate_data()
            if not integrated_data:
                st.error("Failed to fetch and integrate data")
                return False
            
            # Initialize field manager
            reference_data = self.shows_analyzer.get_reference_data()
            self.field_manager = FieldManager(reference_data)
            
            # Initialize matcher
            self.matcher = Matcher(self.field_manager)
            
            # Initialize criteria scorer with matcher
            self.criteria_scorer = CriteriaScorer(self.field_manager, matcher=self.matcher)
            
            # Initialize network analyzer
            from .network_analyzer import NetworkAnalyzer
            self.network_analyzer = NetworkAnalyzer(
                criteria_scorer=self.criteria_scorer,
                field_manager=self.field_manager
            )
            
            # Set the network_analyzer in the criteria_scorer
            self.criteria_scorer.network_analyzer = self.network_analyzer
            
            # Initialize recommendation engine
            self.recommendation_engine = RecommendationEngine(
                shows_analyzer=self.shows_analyzer,
                success_analyzer=self.success_analyzer,
                field_manager=self.field_manager,
                criteria_scorer=self.criteria_scorer
            )
            
            # Initialize concept analyzer with all required components
            self.concept_analyzer = ConceptAnalyzer(
                shows_analyzer=self.shows_analyzer,
                success_analyzer=self.success_analyzer,
                field_manager=self.field_manager,
                criteria_scorer=self.criteria_scorer
            )
            
            # Initialize optimizer view for formatting results
            self.optimizer_view = OptimizerView(self)
            
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            return True
            
        except Exception as e:
            st.error(f"Error initializing Show Optimizer: {str(e)}")
            return False
    
    def _ensure_initialized(self) -> Tuple[bool, Optional[str]]:
        """Ensure the optimizer and its components are initialized.
        
        Checks initialization status and required components.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Check if components are initialized
        if self.concept_analyzer is None:
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            if not self.initialize():
                return False, "Failed to initialize Show Optimizer"
        
        # Check required components
        missing_components = []
        
        if self.shows_analyzer is None:
            missing_components.append("ShowsAnalyzer")
        if self.success_analyzer is None:
            missing_components.append("SuccessAnalyzer")
        if self.field_manager is None:
            missing_components.append("FieldManager")
        if self.criteria_scorer is None:
            missing_components.append("CriteriaScorer")
        if self.network_analyzer is None:
            missing_components.append("NetworkAnalyzer")
        if self.concept_analyzer is None:
            missing_components.append("ConceptAnalyzer")
        if self.optimizer_view is None:
            missing_components.append("OptimizerView")
            
        if missing_components:
            error_msg = f"Missing required components: {', '.join(missing_components)}"
            st.error(error_msg)
            return False, error_msg
            
        return True, None
    
    def get_field_options(self, field_name: str) -> List[Any]:
        """Get options for a field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            List of options for the field
        """
        # Ensure initialization and check components
        initialized, error = self._ensure_initialized()
        if not initialized:
            st.warning(f"Cannot get options for {field_name}: {error}")
            return []
        
        # Try to get options with error handling
        try:
            return self.field_manager.get_options(field_name)
        except Exception as e:
            st.error(f"Error getting options for {field_name}: {e}")
            return []
    
    def validate_criteria(self, criteria: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Validate and normalize criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Tuple of (normalized_criteria, validation_errors)
        """
        # Ensure initialization and check components
        initialized, error = self._ensure_initialized()
        if not initialized:
            return {}, {"error": f"Cannot validate criteria: {error}"}
        
        # Try to validate with error handling
        try:
            # Validate criteria using field manager
            validation_errors = self.field_manager.validate_criteria(criteria)
            
            # Convert the list of errors to the expected format
            error_dict = {}
            if validation_errors:
                for i, error in enumerate(validation_errors):
                    error_dict[f"error_{i}"] = error
            
            # Return a copy of the criteria and the error dictionary
            return criteria.copy(), error_dict
        except Exception as e:
            # Error occurred during criteria validation
            st.error(f"Error validating criteria: {e}")
            return {}, {"error": f"Error validating criteria: {str(e)}"}        
    
    def match_shows(self, criteria: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Match shows based on criteria.
        
        Args:
            criteria: Dictionary of criteria for matching shows
            
        Returns:
            Tuple of (matching_shows_dataframe, total_matches_count)
        """
        # Ensure initialization and check components
        initialized, error = self._ensure_initialized()
        if not initialized:
            st.error(f"Cannot match shows: {error}")
            return pd.DataFrame(), 0
            
        # Fetch integrated data
        try:
            integrated_data = self.fetch_and_integrate_data()
            if not integrated_data or 'shows' not in integrated_data or integrated_data['shows'].empty:
                st.error("No show data available for matching")
                return pd.DataFrame(), 0
                
            # Perform matching using field manager
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            matching_shows, total_matches = self.field_manager.match_shows(criteria, integrated_data['shows'])
            
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            return matching_shows, total_matches
        except Exception as e:
            st.error(f"Error matching shows: {str(e)}")
            return pd.DataFrame(), 0
    
    def _create_fallback_summary(self, error_message: str) -> OptimizationSummary:
        """Create a fallback summary when analysis fails.
        
        Args:
            error_message: The error message to include in the summary
            
        Returns:
            A fallback OptimizationSummary object with error information
        """
        # Create a minimal but valid OptimizationSummary with error information
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        return OptimizationSummary(
            overall_success_probability=None,  # None indicates unknown probability
            confidence="Low",
            top_networks=[],
            component_scores={},
            recommendations=[
                Recommendation(
                    title="Error Occurred",
                    description=f"Analysis failed: {error_message}",
                    impact_score=1.0,  # High impact
                    confidence_score=0.0,  # Low confidence
                    type="error"
                )
            ],
            success_factors=[],
            matching_titles=[],
            match_level=0,
            match_quality=0.0,
            confidence_score=0.0,
            matching_shows=None,
            match_count=0,
            match_counts_by_level={},
            confidence_info={'error': error_message, 'level': 'none'}
        )
    
    def _handle_error(self, error_message: str) -> None:
        """Handle errors by displaying an error message.
        
        Args:
            error_message: Error message to display
        """
        st.error(f"Analysis error: {error_message}")
    
    def _prepare_analysis_context(self, criteria: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Prepare the context for analysis operations.
        
        Args:
            criteria: The criteria to prepare context for
            
        Returns:
            Tuple of (normalized_criteria, integrated_data, success)
        """
        # Normalize criteria
        normalized_criteria = criteria.copy()
        
        # Fetch integrated data
        integrated_data = self.fetch_and_integrate_data()
        if not integrated_data or 'shows' not in integrated_data or integrated_data['shows'].empty:
            st.error("No integrated data available for analysis")
            return normalized_criteria, {}, False
        
        return normalized_criteria, integrated_data, True
    
    def analyze_concept(self, criteria: Dict[str, Any]) -> OptimizationSummary:
        """Analyze a show concept and generate optimization recommendations.
        
        Args:
            criteria: Dictionary of criteria for the show concept
            
        Returns:
            OptimizationSummary containing analysis results or error information
        """
        try:
            # Ensure initialization and check components
            initialized, error = self._ensure_initialized()
            if not initialized:
                self._handle_error(f"Cannot analyze concept: {error}")
                return self.concept_analyzer._handle_analysis_error(f"Initialization failed: {error}")
                
            # Prepare analysis context
            normalized_criteria, integrated_data, success = self._prepare_analysis_context(criteria)
            if not success:
                self._handle_error("No show data available")
                return self.concept_analyzer._handle_analysis_error("No show data available")
                
            # Perform analysis
            # Note: ConceptAnalyzer will print its own analysis message
            analysis_result = self.concept_analyzer.analyze_concept(
                criteria=normalized_criteria,
                integrated_data=integrated_data
            )
            
            # Format the analysis result using OptimizerView before returning
            if analysis_result:
                if OptimizerConfig.DEBUG_MODE:
                    pass  # Debug message removed to reduce spam
                # Format the results using OptimizerView
                if self.optimizer_view:
                    formatted_summary = self.optimizer_view.format_optimization_summary(analysis_result)
                    return formatted_summary
                else:
                    # Fall back to unformatted results if OptimizerView is not available
                    return analysis_result
            else:
                st.warning("Analysis produced no results")
                return self.concept_analyzer._handle_analysis_error("No analysis results")
                
        except Exception as e:
            st.error(f"Error in analyze_concept: {str(e)}")
            return self.concept_analyzer._handle_analysis_error(f"Analysis error: {str(e)}")
    
    def get_network_tiers(self, criteria: Dict[str, Any], 
                        min_confidence: str = 'low') -> Dict[str, List[NetworkMatch]]:
        """Get network tiers for the given criteria.
        
        Args:
            criteria: Dictionary of criteria for network matching
            min_confidence: Minimum confidence level for network matches ('low', 'medium', 'high')
            
        Returns:
            Dictionary mapping tier names to lists of NetworkMatch objects
        """
        # Ensure initialization and check components
        initialized, error = self._ensure_initialized()
        if not initialized:
            st.error(f"Cannot analyze network tiers: {error}")
            return {}
            
        try:
            # Get integrated data
            integrated_data = self.fetch_and_integrate_data()
            if not integrated_data or 'shows' not in integrated_data or integrated_data['shows'].empty:
                st.error("No show data available for network analysis")
                return {}
                
            # Validate criteria
            normalized_criteria, errors = self.validate_criteria(criteria)
            if errors:
                st.warning(f"Some criteria could not be validated: {', '.join(errors.keys())}")
                # Continue with normalized criteria
            
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            
            # Use network_analyzer to get network tiers
            network_tiers = self.network_analyzer.get_network_tiers(
                criteria=normalized_criteria,
                integrated_data=integrated_data,
                min_confidence=min_confidence
            )
                
            return network_tiers
        except Exception as e:
            st.error(f"Error analyzing network tiers: {str(e)}")
            return {}
    
    def get_success_factors(self, criteria: Dict[str, Any], 
                          limit: int = 5) -> List[SuccessFactor]:
        """Get success factors for the given criteria.
        
        Args:
            criteria: Dictionary of criteria for success factor analysis
            limit: Maximum number of factors to return
            
        Returns:
            List of SuccessFactor objects sorted by importance
        """
        # Ensure initialization and check components
        initialized, error = self._ensure_initialized()
        if not initialized:
            st.error(f"Cannot analyze success factors: {error}")
            return []
            
        try:
            # Get integrated data
            integrated_data = self.fetch_and_integrate_data()
            if not integrated_data or 'shows' not in integrated_data or integrated_data['shows'].empty:
                st.error("No show data available for success factor analysis")
                return []
                
            # Validate criteria
            normalized_criteria, errors = self.validate_criteria(criteria)
            if errors:
                st.warning(f"Some criteria could not be validated: {', '.join(errors.keys())}")
                # Continue with normalized criteria
            
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            
            # Use concept_analyzer to get success factors
            success_factors = self.concept_analyzer.get_success_factors(
                criteria=normalized_criteria,
                integrated_data=integrated_data,
                limit=limit
            )
                
            return success_factors
        except Exception as e:
            st.error(f"Error analyzing success factors: {str(e)}")
            return []
    
    def get_recommendations(self, criteria: Dict[str, Any]) -> List[Recommendation]:
        """Get recommendations for the given criteria.
        
        Args:
            criteria: Dictionary of criteria for recommendation generation
            
        Returns:
            List of Recommendation objects sorted by importance
        """
        # Ensure initialization and check components
        initialized, error = self._ensure_initialized()
        if not initialized:
            st.error(f"Cannot generate recommendations: {error}")
            return []
            
        try:
            # Get integrated data
            integrated_data = self.fetch_and_integrate_data()
            if not integrated_data or 'shows' not in integrated_data or integrated_data['shows'].empty:
                st.error("No show data available for generating recommendations")
                return []
                
            # Validate criteria
            normalized_criteria, errors = self.validate_criteria(criteria)
            if errors:
                st.warning(f"Some criteria could not be validated: {', '.join(errors.keys())}")
                # Continue with normalized criteria
            
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            
            # Get success factors and network compatibility from concept_analyzer
            success_factors = self.concept_analyzer.identify_success_factors(
                criteria=normalized_criteria,
                integrated_data=integrated_data
            )
            
            top_networks = self.concept_analyzer.get_network_compatibility(
                criteria=normalized_criteria,
                integrated_data=integrated_data
            )
            
            # Generate recommendations using concept_analyzer
            recommendations = self.concept_analyzer.generate_recommendations(
                normalized_criteria, success_factors, top_networks
            )
                
            return recommendations
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def get_component_scores(self, criteria: Dict[str, Any]) -> Dict[str, ComponentScore]:
        """Get component scores for the given criteria.
        
        Args:
            criteria: Dictionary of criteria for component scoring
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        # Ensure initialization and check components
        initialized, error = self._ensure_initialized()
        if not initialized:
            st.error(f"Cannot calculate component scores: {error}")
            return {}
            
        try:
            # Get integrated data
            integrated_data = self.fetch_and_integrate_data()
            if not integrated_data or 'shows' not in integrated_data or integrated_data['shows'].empty:
                st.error("No show data available for component scoring")
                return {}
                
            # Validate criteria
            normalized_criteria, errors = self.validate_criteria(criteria)
            if errors:
                st.warning(f"Some criteria could not be validated: {', '.join(errors.keys())}")
                # Continue with normalized criteria
            
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            
            # Use concept_analyzer to analyze components
            component_scores = self.concept_analyzer.analyze_components(
                criteria=normalized_criteria,
                integrated_data=integrated_data
            )
                
            return component_scores
        except Exception as e:
            st.error(f"Error calculating component scores: {str(e)}")
            return {}
    
    def get_overall_success_rate(self, criteria: Dict[str, Any]) -> Tuple[float, str]:
        """Get overall success rate for the given criteria.
        
        Args:
            criteria: Dictionary of criteria for success rate calculation
            
        Returns:
            Tuple of (success_rate_float, confidence_level_string)
        """
        # Ensure initialization and check components
        initialized, error = self._ensure_initialized()
        if not initialized:
            st.error(f"Cannot calculate success rate: {error}")
            return 0.0, 'none'
            
        try:
            # Get integrated data
            integrated_data = self.fetch_and_integrate_data()
            if not integrated_data or 'shows' not in integrated_data or integrated_data['shows'].empty:
                st.error("No show data available for success rate calculation")
                return 0.0, 'none'
                
            # Validate criteria
            normalized_criteria, errors = self.validate_criteria(criteria)
            if errors:
                st.warning(f"Some criteria could not be validated: {', '.join(errors.keys())}")
                # Continue with normalized criteria
            
            if OptimizerConfig.DEBUG_MODE:
                pass  # Debug message removed to reduce spam
            
            # Use concept_analyzer to calculate success rate
            success_rate, confidence = self.concept_analyzer.calculate_success_rate(
                criteria=normalized_criteria,
                integrated_data=integrated_data
            )
                
            return success_rate, confidence
        except Exception as e:
            st.error(f"Error calculating success rate: {str(e)}")
            return 0.0, 'none'
