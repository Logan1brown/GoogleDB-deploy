"""Show Optimizer: Main coordinator for show optimization.

This component coordinates the Show Optimizer feature, integrating all the
specialized components (FieldManager, CriteriaScorer, CriteriaAnalyzer,
SuggestionAnalyzer) to provide a unified interface for the UI.

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
   - Cache results for performance
   - Support incremental updates for real-time UI
   - Handle partial criteria sets
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
import logging
from datetime import datetime, timedelta
import streamlit as st

from .optimizer_concept_analyzer import ConceptAnalyzer

from .optimizer_cache import OptimizerCache

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer, NetworkMatch, ComponentScore
from .criteria_analyzer import CriteriaAnalyzer, SuccessFactor
from .suggestion_analyzer import SuggestionAnalyzer, Recommendation, OptimizationSummary
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


class ShowOptimizer:
    """Main coordinator for show optimization.
    
    This class serves as the central orchestrator for the show optimization process,
    integrating various specialized components to analyze show concepts, provide
    recommendations, and calculate success metrics.
    
    The ShowOptimizer uses a ConceptAnalyzer as its primary analysis engine with
    fallbacks to legacy components during the transition period. It implements
    centralized data fetching and caching for performance optimization.
    """
    
    def __init__(self):
        """Initialize the show optimizer with default configuration.
        
        Sets up core analyzers and prepares component attributes that will be
        initialized later. The actual initialization of components happens in
        the initialize() method which is called when needed.
        """
        # Core analyzers - these are initialized immediately
        self.shows_analyzer = ShowsAnalyzer()
        self.success_analyzer = SuccessAnalyzer(self.shows_analyzer)
        
        # Component attributes - initialized during initialize()
        self.field_manager = None
        self.criteria_scorer = None
        self.network_analyzer = None
        self.concept_analyzer = None
        
        # Legacy components (to be removed after transition)
        self.criteria_analyzer = None
        self.suggestion_analyzer = None
        
        # State tracking
        self.initialized = False
        self.last_update = None
        
        # Cache management
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        self.integrated_data = None
        self.data_last_update = None
        
    def fetch_and_integrate_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch and integrate data from different sources with caching.
        
        This method centralizes data fetching and integration to ensure all components
        work with the same dataset. It fetches show data from shows_analyzer and success
        metrics from success_analyzer, then integrates them into a unified dataset.
        
        The method implements caching to avoid redundant data fetching operations.
        Data is only refreshed when force_refresh is True or when the cache has expired.
        
        Args:
            force_refresh: Whether to force a refresh of the data regardless of cache
            
        Returns:
            Dictionary containing integrated data sets needed for optimization
        """
        try:
            current_time = datetime.now()
            
            # Check if we need to refresh the data
            if (self.integrated_data is None or 
                force_refresh or 
                self.data_last_update is None or 
                (current_time - self.data_last_update).total_seconds() > self.cache_duration):
                
                st.write("Fetching and integrating show data...")
                
                # Get show data
                shows_df = self.shows_analyzer.get_shows_data()
                if shows_df.empty:
                    st.error("No show data available")
                    return {}
                
                # Get reference data (genres, networks, etc.)
                reference_data = self.shows_analyzer.get_reference_data()
                
                # Get success metrics
                success_df = self.success_analyzer.get_success_metrics()
                
                # Integrate the data
                # First, ensure we have the necessary columns for integration
                if 'show_id' in shows_df.columns:
                    # Create a copy to avoid modifying the original
                    integrated_shows = shows_df.copy()
                    
                    # Merge with success metrics if available
                    if not success_df.empty and 'show_id' in success_df.columns:
                        # Only keep success metrics that have matching shows
                        success_df = success_df[success_df['show_id'].isin(shows_df['show_id'])]
                    else:
                        st.warning("Could not integrate success metrics: missing 'show_id' column")
                    
                    # Create the integrated data dictionary
                    self.integrated_data = {
                        'shows': integrated_shows,
                        'reference_data': reference_data,
                        'success_metrics': success_df
                    }
                    
                    # Update the last update timestamp
                    self.data_last_update = current_time
                    
                    st.write("Data integration complete")
                else:
                    st.error("Could not integrate data: missing 'show_id' column in shows data")
                    return {}
            else:
                st.write("Using cached integrated data")
            
            return self.integrated_data
            
        except Exception as e:
            st.error(f"Error fetching and integrating data: {str(e)}")
            return {}
            
    def initialize(self, force_refresh: bool = False) -> bool:
        """Initialize all components.
        
        This method initializes all required components for the show optimizer,
        including the field manager, criteria scorer, and concept analyzer.
        It also fetches and integrates the necessary data.
        
        Args:
            force_refresh: Whether to force a refresh of the data
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Check if we need to refresh
            current_time = datetime.now()
            if (not self.initialized or 
                force_refresh or 
                self.last_update is None or 
                (current_time - self.last_update).total_seconds() > self.cache_duration):
                
                st.write("Initializing Show Optimizer components...")
                
                # Fetch and integrate data first
                integrated_data = self.fetch_and_integrate_data(force_refresh)
                if not integrated_data:
                    st.error("Failed to fetch and integrate data")
                    return False
                
                # Initialize components
                self.criteria_scorer = CriteriaScorer(self.shows_analyzer, self.success_analyzer)
                self.field_manager = self.criteria_scorer.field_manager  # Share the same field manager
                
                # Initialize new components
                self.network_analyzer = self.criteria_scorer.network_analyzer  # Share the network analyzer
                self.concept_analyzer = ConceptAnalyzer(self.shows_analyzer, self.success_analyzer)
                
                # Legacy components (to be removed after transition)
                self.criteria_analyzer = CriteriaAnalyzer(self.shows_analyzer, self.success_analyzer)
                self.suggestion_analyzer = SuggestionAnalyzer(self.shows_analyzer, self.success_analyzer)
                
                self.initialized = True
                self.last_update = current_time
                
                st.write("Show Optimizer initialized successfully")
                
            return self.initialized
            
        except Exception as e:
            st.error(f"Error initializing Show Optimizer: {str(e)}")
            return False
    
    def _ensure_initialized(self) -> bool:
        """Ensure the optimizer is initialized.
        
        This helper method checks if the optimizer is initialized and attempts
        to initialize it if not. It provides a convenient way to ensure the
        optimizer is ready before performing operations.
        
        This method is called at the beginning of all public analysis methods
        to guarantee that the optimizer is properly initialized before any
        operations are performed.
        
        Returns:
            True if the optimizer is initialized, False otherwise
        """
        if not self.initialized:
            st.write("Show Optimizer not initialized. Initializing now...")
            return self.initialize()
        return True
    
    def get_field_options(self, field_name: str) -> List[Any]:
        """Get options for a field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            List of options for the field
        """
        if not self._ensure_initialized():
            st.warning(f"ShowOptimizer not initialized when getting options for {field_name}")
            return []
            
        if self.field_manager is None:
            st.error(f"Field manager is None when trying to get options for {field_name}")
            # Force a re-initialization attempt
            try:
                # Try to initialize again with force_refresh=True
                self.initialize(force_refresh=True)
                
                # If still None after re-initialization, return empty list
                if self.field_manager is None:
                    logger.error("Field manager still None after forced re-initialization")
                    return []
            except Exception as e:
                logger.error(f"Error during re-initialization: {e}", exc_info=True)
                return []
        
        # Try to get options with error handling
        try:
            return self.field_manager.get_options(field_name)
        except Exception as e:
            logger.error(f"Error getting options for {field_name}: {e}", exc_info=True)
            return []
    
    def validate_criteria(self, criteria: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Validate and normalize criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Tuple of (normalized_criteria, validation_errors)
        """
        import streamlit as st
        
        if not self.initialized and not self.initialize():
            logger.warning("ShowOptimizer not initialized when validating criteria")
            return {}, {"error": "Show Optimizer not initialized. Please try refreshing the page."}
            
        if self.field_manager is None:
            logger.error("Field manager is None when trying to validate criteria")
            # Force a re-initialization attempt
            try:
                # Try to initialize again with force_refresh=True
                self.initialize(force_refresh=True)
                
                # If still None after re-initialization, return error
                if self.field_manager is None:
                    logger.error("Field manager still None after forced re-initialization")
                    return {}, {"error": "Unable to initialize field manager. This may be due to database connection issues."}
            except Exception as e:
                logger.error(f"Error during re-initialization: {e}", exc_info=True)
                return {}, {"error": f"Error initializing components: {str(e)}"}
        
        # Try to validate with error handling
        try:
            # The field_manager.validate_criteria returns a list of error strings, not a tuple
            # Validate criteria using field manager
            validation_errors = self.field_manager.validate_criteria(criteria)
            # Check for validation errors
            
            # Convert the list of errors to the expected format
            error_dict = {}
            if validation_errors:
                for i, error in enumerate(validation_errors):
                    error_dict[f"error_{i}"] = error
            
            # Return a copy of the criteria and the error dictionary
            return criteria.copy(), error_dict
        except Exception as e:
            # Error occurred during criteria validation
            logger.error(f"Error validating criteria: {e}", exc_info=True)
            return {}, {"error": f"Error validating criteria: {str(e)}"}
        
    
    def match_shows(self, criteria: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Match shows based on criteria.
        
        This method matches shows from the integrated data based on the provided criteria.
        It uses the field manager to perform the matching and returns the matching shows
        along with the total number of matches.
        
        Args:
            criteria: Dictionary of criteria for matching shows
            
        Returns:
            Tuple of (matching_shows_dataframe, total_matches_count)
        """
        if not self._ensure_initialized():
            st.error("Failed to initialize Show Optimizer components for matching")
            return pd.DataFrame(), 0
            
        if self.field_manager is None:
            st.error("Field manager is not initialized")
            return pd.DataFrame(), 0
            
        # Get integrated data
        try:
            # Use our integrated data instead of fetching it again
            integrated_data = self.fetch_and_integrate_data()
            if not integrated_data or 'shows' not in integrated_data or integrated_data['shows'].empty:
                st.error("No show data available for matching")
                return pd.DataFrame(), 0
                
            # Use the field manager to match shows
            st.write("Matching shows based on criteria...")
            matching_shows, total_matches = self.field_manager.match_shows(criteria, integrated_data['shows'])
            
            st.write(f"Found {total_matches} matching shows")
            return matching_shows, total_matches
        except Exception as e:
            st.error(f"Error matching shows: {str(e)}")
            return pd.DataFrame(), 0
    
    def _create_fallback_summary(self, error_message: str) -> OptimizationSummary:
        """Create a fallback summary when analysis fails.
        
        This helper method creates a fallback OptimizationSummary object with
        default values and an error message when analysis fails. It ensures that
        even in error conditions, a valid OptimizationSummary object is returned
        to prevent cascading failures in the UI.
        
        The method includes two levels of error handling:
        1. Primary try/except to create a detailed fallback summary
        2. Secondary fallback with minimal information if the first attempt fails
        
        Args:
            error_message: The error message to include in the summary
            
        Returns:
            A fallback OptimizationSummary object with error information
        """
        try:
            # Create a minimal but valid OptimizationSummary with error information
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            return OptimizationSummary(
                overall_score=0.0,
                success_probability="Low",
                success_factors=[],
                component_scores={},
                network_tiers=[],
                recommendations=[
                    Recommendation(
                        title="Error Occurred",
                        description=f"Analysis failed: {error_message}",
                        impact="High",
                        effort="Unknown",
                        priority="High"
                    )
                ],
                matched_shows=pd.DataFrame(),
                matched_count=0,
                analysis_date=formatted_time,
                error_message=error_message
            )
        except Exception as e:
            st.error(f"Failed to create fallback summary: {str(e)}")
            # Return an absolute minimal summary as last resort
            return OptimizationSummary(
                overall_score=0.0,
                success_probability="Unknown",
                success_factors=[],
                component_scores={},
                network_tiers=[],
                recommendations=[],
                matched_shows=pd.DataFrame(),
                matched_count=0,
                analysis_date="Unknown",
                error_message=f"Critical error: {str(e)}"
            )
    
    def _prepare_analysis_context(self, criteria: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Prepare the context for analysis operations.
        
        This helper method prepares the necessary context for analysis operations,
        including normalizing criteria and fetching integrated data. It centralizes
        the common preparation steps used by multiple analysis methods.
        
        Args:
            criteria: The criteria to prepare context for
            
        Returns:
            Tuple of (normalized_criteria, integrated_data, success)
            where success is a boolean indicating if preparation was successful
        """
        # Normalize criteria
        normalized_criteria = criteria.copy()
        
        # Fetch integrated data
        integrated_data = self.fetch_and_integrate_data()
        if not integrated_data:
            st.error("No integrated data available for analysis")
            return normalized_criteria, {}, False
        
        return normalized_criteria, integrated_data, True
    
    def analyze_concept(self, criteria: Dict[str, Any]) -> OptimizationSummary:
        """Analyze a show concept and generate optimization recommendations.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            OptimizationSummary with success probability, recommendations, etc.
        """
        try:
            if not self._ensure_initialized():
                return self._create_fallback_summary("Initialization failed")
            if self.concept_analyzer is None or self.suggestion_analyzer is None:
                st.error("Required analyzer components are not initialized")
                return self._create_fallback_summary("Missing required components")
            normalized_criteria, integrated_data, success = self._prepare_analysis_context(criteria)
            if not success:
                return self._create_fallback_summary("No show data available")
            st.write("Analyzing show concept...")
            try:
                analysis_result = self.concept_analyzer.analyze_concept(
                    criteria=normalized_criteria,
                    integrated_data=integrated_data
                )
                if analysis_result:
                    formatted_result = self.suggestion_analyzer.format_optimization_summary(analysis_result)
                    st.write("Analysis completed successfully")
                    return formatted_result
                else:
                    st.warning("Analysis produced no results")
                    return self._create_fallback_summary("No analysis results")
            except Exception as analysis_error:
                st.error(f"Error during concept analysis: {str(analysis_error)}")
                return self._create_fallback_summary(f"Analysis error: {str(analysis_error)}")
        except Exception as e:
            st.error(f"Unexpected error in analyze_concept: {str(e)}")
            return self._create_fallback_summary(f"Unexpected error: {str(e)}")
    
    def get_network_tiers(self, criteria: Dict[str, Any], 
                        min_confidence: str = 'low') -> Dict[str, List[NetworkMatch]]:
        """Get network tiers for the given criteria.
        
        This method analyzes the criteria and returns a dictionary of network tiers
        (premium, standard, niche) with matching networks for each tier.
        
        Args:
            criteria: Dictionary of criteria for network matching
            min_confidence: Minimum confidence level for network matches ('low', 'medium', 'high')
            
        Returns:
            Dictionary mapping tier names to lists of NetworkMatch objects
        """
        if not self._ensure_initialized():
            st.error("Failed to initialize Show Optimizer components for network tier analysis")
            return {}
            
        # Check if required components are initialized
        if self.network_analyzer is None and self.criteria_analyzer is None:
            st.error("Network analysis components are not initialized")
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
            
            st.write("Analyzing network tiers...")
            
            # Use network_analyzer if available, otherwise fall back to criteria_analyzer
            if self.network_analyzer is not None:
                network_tiers = self.network_analyzer.get_network_tiers(
                    criteria=normalized_criteria,
                    integrated_data=integrated_data,
                    min_confidence=min_confidence
                )
            else:
                # Legacy fallback
                network_tiers = self.criteria_analyzer.find_matching_networks(normalized_criteria, min_confidence)
                
            return network_tiers
        except Exception as e:
            st.error(f"Error analyzing network tiers: {str(e)}")
            return {}
    
    def get_success_factors(self, criteria: Dict[str, Any], 
                          limit: int = 5) -> List[SuccessFactor]:
        """Get success factors for the given criteria.
        
        This method analyzes the criteria and identifies key factors that contribute
        to the success of shows with similar characteristics.
        
        Args:
            criteria: Dictionary of criteria for success factor analysis
            limit: Maximum number of factors to return
            
        Returns:
            List of SuccessFactor objects sorted by importance
        """
        if not self._ensure_initialized():
            st.error("Failed to initialize Show Optimizer components for success factor analysis")
            return []
            
        # Check if required components are initialized
        if self.concept_analyzer is None and self.criteria_analyzer is None:
            st.error("Success factor analysis components are not initialized")
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
            
            st.write("Analyzing success factors...")
            
            # Use concept_analyzer if available, otherwise fall back to criteria_analyzer
            if self.concept_analyzer is not None:
                success_factors = self.concept_analyzer.identify_success_factors(
                    criteria=normalized_criteria,
                    integrated_data=integrated_data,
                    limit=limit
                )
            else:
                # Legacy fallback
                success_factors = self.criteria_analyzer.identify_success_factors(normalized_criteria, limit)
                
            return success_factors
        except Exception as e:
            st.error(f"Error analyzing success factors: {str(e)}")
            return []
    
    def get_recommendations(self, criteria: Dict[str, Any]) -> List[Recommendation]:
        """Get recommendations for the given criteria.
        
        This method generates recommendations for improving the show concept based on
        success factors and network compatibility analysis.
        
        Args:
            criteria: Dictionary of criteria for recommendation generation
            
        Returns:
            List of Recommendation objects sorted by importance
        """
        if not self._ensure_initialized():
            st.error("Failed to initialize Show Optimizer components for recommendation generation")
            return []
            
        # Check if required components are initialized
        if self.concept_analyzer is None and self.suggestion_analyzer is None:
            st.error("Recommendation components are not initialized")
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
            
            st.write("Generating recommendations...")
            
            # Use concept_analyzer if available, otherwise use legacy approach
            if self.concept_analyzer is not None and self.suggestion_analyzer is not None:
                # Get success factors and network compatibility from concept_analyzer
                success_factors = self.concept_analyzer.identify_success_factors(
                    criteria=normalized_criteria,
                    integrated_data=integrated_data
                )
                
                top_networks = self.concept_analyzer.get_network_compatibility(
                    criteria=normalized_criteria,
                    integrated_data=integrated_data
                )
                
                # Generate recommendations using suggestion_analyzer
                recommendations = self.suggestion_analyzer.generate_recommendations(
                    normalized_criteria, success_factors, top_networks
                )
            else:
                # Legacy fallback
                success_factors = self.criteria_analyzer.identify_success_factors(normalized_criteria)
                top_networks = self.criteria_analyzer.rank_networks_by_compatibility(normalized_criteria)
                recommendations = self.suggestion_analyzer.generate_recommendations(
                    normalized_criteria, success_factors, top_networks
                )
                
            return recommendations
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def get_component_scores(self, criteria: Dict[str, Any]) -> Dict[str, ComponentScore]:
        """Get component scores for the given criteria.
        
        This method calculates scores for different components of the show concept
        (e.g., premise, characters, setting) based on the provided criteria.
        
        Args:
            criteria: Dictionary of criteria for component scoring
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        if not self._ensure_initialized():
            st.error("Failed to initialize Show Optimizer components for component scoring")
            return {}
            
        # Check if required components are initialized
        if self.concept_analyzer is None and self.criteria_analyzer is None:
            st.error("Component scoring components are not initialized")
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
            
            st.write("Calculating component scores...")
            
            # Use concept_analyzer if available, otherwise fall back to criteria_analyzer
            if self.concept_analyzer is not None:
                component_scores = self.concept_analyzer.analyze_components(
                    criteria=normalized_criteria,
                    integrated_data=integrated_data
                )
            else:
                # Legacy fallback
                component_scores = self.criteria_analyzer.analyze_components(normalized_criteria)
                
            return component_scores
        except Exception as e:
            st.error(f"Error calculating component scores: {str(e)}")
            return {}
    
    def get_overall_success_rate(self, criteria: Dict[str, Any]) -> Tuple[float, str]:
        """Get overall success rate for the given criteria.
        
        This method calculates the overall success probability for a show concept
        based on the provided criteria, along with a confidence level for the prediction.
        
        Args:
            criteria: Dictionary of criteria for success rate calculation
            
        Returns:
            Tuple of (success_rate_float, confidence_level_string)
        """
        if not self._ensure_initialized():
            st.error("Failed to initialize Show Optimizer components for success rate calculation")
            return 0.0, 'none'
            
        # Check if required components are initialized
        if self.concept_analyzer is None and self.criteria_analyzer is None:
            st.error("Success rate calculation components are not initialized")
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
            
            st.write("Calculating overall success rate...")
            
            # Use concept_analyzer if available, otherwise fall back to criteria_analyzer
            if self.concept_analyzer is not None:
                success_rate, confidence = self.concept_analyzer.calculate_success_rate(
                    criteria=normalized_criteria,
                    integrated_data=integrated_data
                )
            else:
                # Legacy fallback
                success_rate, confidence = self.criteria_analyzer.get_overall_success_rate(normalized_criteria)
                
            return success_rate, confidence
        except Exception as e:
            st.error(f"Error calculating success rate: {str(e)}")
            return 0.0, 'none'
