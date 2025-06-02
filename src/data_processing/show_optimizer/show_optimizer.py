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

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer, NetworkMatch, ComponentScore
from .criteria_analyzer import CriteriaAnalyzer, SuccessFactor
from .suggestion_analyzer import SuggestionAnalyzer, Recommendation, OptimizationSummary
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


class ShowOptimizer:
    """Main coordinator for show optimization."""
    
    def __init__(self):
        """Initialize the show optimizer."""
        self.shows_analyzer = ShowsAnalyzer()
        self.success_analyzer = SuccessAnalyzer(self.shows_analyzer)
        self.field_manager = None
        self.criteria_scorer = None
        self.criteria_analyzer = None
        self.suggestion_analyzer = None
        self.initialized = False
        self.last_update = None
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        
    def initialize(_self, force_refresh: bool = False) -> bool:
        """Initialize all components.
        
        Args:
            force_refresh: Whether to force a refresh of the data
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Check if we need to refresh
            current_time = datetime.now()
            if (not _self.initialized or 
                force_refresh or 
                _self.last_update is None or 
                (current_time - _self.last_update).total_seconds() > _self.cache_duration):
                
                logger.info("Initializing Show Optimizer components")
                
                # Initialize components
                _self.criteria_scorer = CriteriaScorer(_self.shows_analyzer, _self.success_analyzer)
                _self.field_manager = _self.criteria_scorer.field_manager  # Share the same field manager
                _self.criteria_analyzer = CriteriaAnalyzer(_self.shows_analyzer, _self.success_analyzer)
                _self.suggestion_analyzer = SuggestionAnalyzer(_self.shows_analyzer, _self.success_analyzer)
                
                _self.initialized = True
                _self.last_update = current_time
                
            return _self.initialized
            
        except Exception as e:
            logger.error(f"Error initializing Show Optimizer: {e}", exc_info=True)
            return False
    
    def get_field_options(self, field_name: str) -> List[Any]:
        """Get options for a field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            List of options for the field
        """
        if not self.initialized and not self.initialize():
            logger.warning(f"ShowOptimizer not initialized when getting options for {field_name}")
            return []
            
        if self.field_manager is None:
            logger.error(f"Field manager is None when trying to get options for {field_name}")
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
            return self.field_manager.validate_criteria(criteria)
        except Exception as e:
            logger.error(f"Error validating criteria: {e}", exc_info=True)
            return {}, {"error": f"Error validating criteria: {str(e)}"}
        
    
    def match_shows(self, criteria: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Match shows based on criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Tuple of (matching_shows, total_matches)
        """
        if not self.initialized and not self.initialize():
            return pd.DataFrame(), 0
            
        # Get show data
        try:
            comp_df, _ = self.shows_analyzer.fetch_comp_data()
            return self.field_manager.match_shows(criteria, comp_df)
        except Exception as e:
            logger.error(f"Error matching shows: {e}", exc_info=True)
            return pd.DataFrame(), 0
    
    def analyze_concept(self, criteria: Dict[str, Any]) -> OptimizationSummary:
        """Analyze a show concept and generate optimization recommendations.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            OptimizationSummary with success probability, recommendations, etc.
        """
        try:
            if not self.initialized and not self.initialize():
                logger.error("Failed to initialize Show Optimizer components")
                return None
                
            # Check if required components are initialized
            if self.suggestion_analyzer is None:
                logger.error("SuggestionAnalyzer is not initialized")
                return None
                
            # Validate criteria
            normalized_criteria, errors = self.validate_criteria(criteria)
            if errors:
                logger.warning(f"Validation errors in criteria: {errors}")
                # Continue with normalized criteria
            
            # Analyze concept
            return self.suggestion_analyzer.analyze_show_concept(normalized_criteria)
        except Exception as e:
            logger.error(f"Error analyzing concept: {e}", exc_info=True)
            return None
    
    def get_network_tiers(self, criteria: Dict[str, Any], 
                        min_confidence: str = 'low') -> Dict[str, List[NetworkMatch]]:
        """Get network tiers for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            min_confidence: Minimum confidence level
            
        Returns:
            Dictionary mapping tier names to lists of NetworkMatch objects
        """
        try:
            if not self.initialized and not self.initialize():
                logger.error("Failed to initialize Show Optimizer components")
                return {}
                
            # Check if required components are initialized
            if self.criteria_analyzer is None:
                logger.error("CriteriaAnalyzer is not initialized")
                return {}
                
            # Validate criteria
            normalized_criteria, errors = self.validate_criteria(criteria)
            if errors:
                logger.warning(f"Validation errors in criteria: {errors}")
                # Continue with normalized criteria
            
            return self.criteria_analyzer.find_matching_networks(normalized_criteria, min_confidence)
        except Exception as e:
            logger.error(f"Error getting network tiers: {e}", exc_info=True)
            return {}
    
    def get_success_factors(self, criteria: Dict[str, Any], 
                          limit: int = 5) -> List[SuccessFactor]:
        """Get success factors for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            limit: Maximum number of factors to return
            
        Returns:
            List of SuccessFactor objects
        """
        try:
            if not self.initialized and not self.initialize():
                logger.error("Failed to initialize Show Optimizer components")
                return []
                
            # Check if required components are initialized
            if self.criteria_analyzer is None:
                logger.error("CriteriaAnalyzer is not initialized")
                return []
                
            # Validate criteria
            normalized_criteria, errors = self.validate_criteria(criteria)
            if errors:
                logger.warning(f"Validation errors in criteria: {errors}")
                # Continue with normalized criteria
            
            return self.criteria_analyzer.identify_success_factors(normalized_criteria, limit)
        except Exception as e:
            logger.error(f"Error getting success factors: {e}", exc_info=True)
            return []
    
    def get_recommendations(self, criteria: Dict[str, Any]) -> List[Recommendation]:
        """Get recommendations for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            List of Recommendation objects
        """
        if not self.initialized and not self.initialize():
            return []
            
        # Validate criteria
        normalized_criteria, errors = self.validate_criteria(criteria)
        if errors:
            logger.warning(f"Validation errors in criteria: {errors}")
            # Continue with normalized criteria
        
        # Get success factors and top networks
        success_factors = self.criteria_analyzer.identify_success_factors(normalized_criteria)
        top_networks = self.criteria_analyzer.rank_networks_by_compatibility(normalized_criteria)
        
        return self.suggestion_analyzer.generate_recommendations(
            normalized_criteria, success_factors, top_networks
        )
    
    def get_component_scores(self, criteria: Dict[str, Any]) -> Dict[str, ComponentScore]:
        """Get component scores for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        if not self.initialized and not self.initialize():
            return {}
            
        # Validate criteria
        normalized_criteria, errors = self.validate_criteria(criteria)
        if errors:
            logger.warning(f"Validation errors in criteria: {errors}")
            # Continue with normalized criteria
        
        return self.criteria_analyzer.analyze_components(normalized_criteria)
    
    def get_overall_success_rate(self, criteria: Dict[str, Any]) -> Tuple[float, str]:
        """Get overall success rate for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Tuple of (success_rate, confidence)
        """
        if not self.initialized and not self.initialize():
            return 0.0, 'none'
            
        # Validate criteria
        normalized_criteria, errors = self.validate_criteria(criteria)
        if errors:
            logger.warning(f"Validation errors in criteria: {errors}")
            # Continue with normalized criteria
        
        return self.criteria_analyzer.get_overall_success_rate(normalized_criteria)
