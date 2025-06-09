"""CriteriaScorer: Calculates component scores for shows based on provided data.

Responsible for calculating various component scores (success, audience, critics, longevity)
based on integrated data provided by orchestrator components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
import numpy as np
import ast
import streamlit as st
import time
from functools import lru_cache
from datetime import datetime, timedelta

from .field_manager import FieldManager
from .optimizer_config import OptimizerConfig
from .score_calculators import ComponentScore, ScoreCalculationError, NetworkMatch, NetworkScoreCalculator
from .score_calculators import SuccessScoreCalculator, AudienceScoreCalculator, CriticsScoreCalculator, LongevityScoreCalculator

SCORE_CALCULATORS_CLASSES = {
    'success': SuccessScoreCalculator,
    'audience': AudienceScoreCalculator,
    'critics': CriticsScoreCalculator,
    'longevity': LongevityScoreCalculator,
}

__all__ = ['CriteriaScorer', 'ComponentScore', 'ScoreCalculationError']



class CriteriaScorer:
    """Calculates component scores for show criteria based on provided data.
    
    This class is responsible for calculating various component scores including success,
    audience, critics, and longevity scores. It delegates the actual score calculations to
    specialized calculator classes while providing orchestration and result aggregation.
    """
    
    def __init__(self, field_manager: FieldManager, matcher=None):
        """Initialize the criteria scorer.
        
        Args:
            field_manager: FieldManager instance for field mapping and validation
            matcher: Optional Matcher instance for finding matches
        """
        self.field_manager = field_manager
        self.matcher = matcher  # Matcher instance for finding matches
        self.network_analyzer = None  # Will be set by ShowOptimizer after initialization
        
    def calculate_success_rate(self, shows: pd.DataFrame, threshold: Optional[float] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Public method to calculate the success rate for a set of shows.
        
        Delegates to the private _calculate_success_rate method.
        
        Args:
            shows: DataFrame of shows
            threshold: Optional success threshold
            
        Returns:
            Tuple of success rate and confidence information
        """
        return self._calculate_success_rate(shows, threshold)
        
    def _calculate_success_rate(self, shows: pd.DataFrame, threshold: Optional[float] = None, confidence_info: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate the success rate for a set of shows with confidence information.
        
        Delegates success rate calculation to the SuccessScoreCalculator.
        
        Args:
            shows: DataFrame of shows
            threshold: Optional success threshold
            confidence_info: Optional confidence information
            
        Returns:
            Tuple of success rate and confidence information
        """
        if shows is None or shows.empty:
            if confidence_info is None:
                confidence_info = {'level': 'none', 'score': 0.0}
            return None, confidence_info
        
        # Delegate to SuccessScoreCalculator
        calculator = SuccessScoreCalculator()
        
        # Use the calculator's validate_and_prepare_data method
        is_valid, validated_data, validation_info = calculator.validate_and_prepare_data(
            shows, 
            required_columns=['success_score'],
            optional_columns=[],
            data_column='success_score'
        )
        
        if not is_valid or validated_data is None or validated_data.empty:
            if confidence_info is None:
                confidence_info = {'level': 'none', 'score': 0.0, 'error': validation_info.get('error', 'Unknown error')}
            else:
                confidence_info['error'] = validation_info.get('error', 'Unknown error')
            return None, confidence_info
        
        # Use the provided threshold or default from config
        if threshold is None:
            threshold = OptimizerConfig.SUCCESS['threshold']
        
        # Let the calculator handle the actual calculation
        component_score = calculator.calculate(validated_data, threshold=threshold)
        
        if component_score is None:
            if confidence_info is None:
                confidence_info = {'level': 'none', 'score': 0.0, 'error': 'Failed to calculate success score'}
            else:
                confidence_info['error'] = 'Failed to calculate success score'
            return None, confidence_info
        
        # Extract success rate and metadata from component score
        success_rate = component_score.score
        
        # Update confidence information
        if confidence_info is None:
            confidence_info = {}
        
        # Merge metadata from component score into confidence info
        for key, value in component_score.metadata.items():
            confidence_info[key] = value
        
        return success_rate, confidence_info

   
    def _batch_calculate_success_rates(self, criteria_list: List[Dict[str, Any]], matching_shows_list: Optional[List[pd.DataFrame]] = None) -> List[Optional[float]]:
        """
        Calculate success rates for a batch of criteria using provided matching shows.

        Args:
            criteria_list: List of criteria dictionaries
            matching_shows_list: Optional list of DataFrames containing shows matching each criteria. If not provided, raises an error (explicit matching is required).
        Returns:
            List of success rates (one for each criteria/matching shows pair)
        """
        if matching_shows_list is None:
            st.error("No matching_shows_list provided to _batch_calculate_success_rates. This method requires explicit matching.")
            return [None] * len(criteria_list)
        if len(criteria_list) != len(matching_shows_list):
            st.error(f"Mismatch between criteria list ({len(criteria_list)}) and matching shows list ({len(matching_shows_list)})")
            return [None] * len(criteria_list)
        
        results = []
        
        # Create a single SuccessScoreCalculator instance to reuse
        calculator = SuccessScoreCalculator()
        
        for i, (criteria, matching_shows) in enumerate(zip(criteria_list, matching_shows_list)):
            try:
                if matching_shows is None or matching_shows.empty:
                    results.append(None)
                    continue
                    
                # Use the calculator to calculate the success rate
                component_score = calculator.calculate(matching_shows)
                
                if component_score is None:
                    results.append(None)
                else:
                    # Extract just the success rate value
                    results.append(component_score.score)
                    
            except Exception as e:
                st.error(f"Error calculating success rate for criteria {criteria}: {str(e)}")
                results.append(None)
                
        return results
    
    @lru_cache(maxsize=32)
    def calculate_criteria_impact(self, base_criteria: Dict[str, Any], base_matching_shows: pd.DataFrame, option_matching_shows_map: Optional[Dict[str, Dict[int, pd.DataFrame]]] = None, field_name: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Calculate the impact of criteria values on success rate.

        Args:
            base_criteria: The base criteria to compare against.
            base_matching_shows: DataFrame of shows matching the base criteria.
            option_matching_shows_map: Optional mapping of field names to option IDs to matching shows. If provided, these pre-matched shows will be used for impact calculation (recommended for batch/impact analysis).
            field_name: Optional name of the field to analyze. If None, analyzes all fields.
            
        Returns:
            A dictionary mapping field names to dictionaries of option IDs to impact scores.
        """
        try:
            # Get the field manager's array field mapping
            array_field_mapping = self.field_manager.get_array_field_mapping()
            array_fields = list(array_field_mapping.keys())
            
            # Use the provided base matching shows
            base_match_count = len(base_matching_shows) if not base_matching_shows.empty else 0
            
            if base_matching_shows.empty:
                raise ValueError("Cannot calculate impact scores with no matching shows")
                
            if base_match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                raise ValueError(f"Cannot calculate impact scores with insufficient sample size ({base_match_count} shows). Minimum required: {OptimizerConfig.CONFIDENCE['minimum_sample']}")
            
            # Calculate base success rate using the provided shows
            base_rate, _ = self._calculate_success_rate(base_matching_shows)
            
            if base_rate is None:
                st.warning("WARNING: Unable to calculate base success rate - success_score data missing")
                return {}
            
            impact_scores = {}
            
            # Determine which fields to process
            fields_to_process = [field_name] if field_name else self.field_manager.FIELD_CONFIGS.keys()
            
            for current_field in fields_to_process:
                # Skip fields already in base criteria
                if current_field in base_criteria:
                    continue
                
                is_array_field = current_field in array_fields
                options = self.field_manager.get_options(current_field)
                
                # Prepare batch criteria for all options
                batch_criteria = []
                option_data = []
                
                for option in options:
                    new_criteria = base_criteria.copy()
                    if is_array_field:
                        # Ensure option.id is treated as a list item for array fields
                        new_criteria[current_field] = [option.id] if not isinstance(option.id, list) else option.id
                    else:
                        new_criteria[current_field] = option.id
                    batch_criteria.append(new_criteria)
                    option_data.append((option.id, option.name))
                
                # Process each option using the provided option_matching_shows_map
                field_impact = {}
                
                if option_matching_shows_map and current_field in option_matching_shows_map:
                    field_options_map = option_matching_shows_map[current_field]
                    
                    for i, (option_id, option_name) in enumerate(option_data):
                        # Skip if we don't have matching shows for this option
                        if option_id not in field_options_map:
                            continue
                            
                        option_shows = field_options_map[option_id]
                        
                        if option_shows is None or option_shows.empty:
                            continue
                            
                        # Calculate success rate for this option
                        option_rate, option_confidence = self._calculate_success_rate(option_shows)
                        
                        if option_rate is not None:
                            match_count = len(option_shows)
                            impact = (option_rate - base_rate) / base_rate if base_rate != 0 else 0
                            
                            field_impact[option_id] = {
                                "impact": impact, 
                                "sample_size": match_count, 
                                "option_name": option_name,
                                "success_rate": option_rate
                            }
                
                if field_impact:
                    impact_scores[current_field] = field_impact
                    
            if not impact_scores and fields_to_process:
                # Check if option_matching_shows_map was provided
                if option_matching_shows_map is None:
                    st.error("Cannot calculate impact scores: option_matching_shows_map is required but was not provided.")
                    return {}
                else:
                    # Only raise if we attempted to process fields but got no results
                    # If fields_to_process was empty (e.g. field_name was already in base_criteria), this is not an error
                    raise ValueError("Could not calculate any impact scores with the given criteria configuration.")
                
            return impact_scores

        except ValueError as ve:
            st.error(str(ve)) # Surface our specific ValueErrors
            raise
        except Exception as e:
            st.error(f"Error calculating criteria impact: {str(e)}")
            raise
    
    def calculate_component_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, confidence_info: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame] = None) -> Dict[str, ComponentScore]:
        """
        Calculate component scores for the given criteria using provided matched shows.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of shows matching the criteria
            confidence_info: Confidence information from matching
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        component_scores: Dict[str, ComponentScore] = {}

        try:
            # First check if we have any matching shows
            if matching_shows is None or matching_shows.empty:
                st.warning(f"No matching shows found for criteria: {criteria}. Cannot calculate component scores.")
                return {}
            
            # Calculate match count
            match_count = len(matching_shows)
            st.write(f"Calculating scores using {match_count} matched shows")
            
            # Using general minimum_sample from OptimizerConfig
            if match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                st.warning(f"Insufficient sample size ({match_count}) for criteria: {criteria} to calculate reliable component scores. Minimum required: {OptimizerConfig.CONFIDENCE['minimum_sample']}")
                # We'll continue with the calculation, but with a warning
            
            # Validate that we have required confidence info fields
            if 'match_level' not in confidence_info:
                st.warning("Incomplete confidence info provided to calculate_component_scores: missing match_level")
                actual_match_level = 1  # Default to exact match level
            else:
                actual_match_level = confidence_info.get('match_level')
                original_match_level = confidence_info.get('original_match_level', actual_match_level)
                
                if actual_match_level != original_match_level:
                    st.write(f"Note: Match level was adjusted from {original_match_level} to {actual_match_level} based on criteria validation")
                    
                # If we have array fields like character_types in our criteria, make sure they're properly matched
                # This helps ensure component scores are calculated based on shows that actually match the criteria
                array_fields = [field for field, value in criteria.items() if isinstance(value, list) and value]
                if array_fields and actual_match_level > 1:
                    st.write(f"Note: Some array criteria like {', '.join(array_fields)} may be relaxed in match level {actual_match_level}")
                
            # Each individual calculator will validate its required columns
            # using the validate_and_prepare_data helper method

        except Exception as e:
            st.error(f"Optimizer Error: Failed to get matching shows for component score calculation. Criteria: {criteria}. Details: {e}")
            return {}

        # Ensure field_manager is initialized (should be in __init__)
        if not hasattr(self, 'field_manager') or self.field_manager is None:
            st.error("Optimizer Critical Error: FieldManager is not initialized in CriteriaScorer. This may affect score calculations.")

        # Initialize and use calculators for each component
        calculators = [
            SuccessScoreCalculator(),
            AudienceScoreCalculator(),
            CriticsScoreCalculator(),
            LongevityScoreCalculator()
        ]
        
        # Calculate scores for each component
        for calculator in calculators:
            try:
                score_component = calculator.calculate(matching_shows.copy())  # Pass a copy to avoid modification issues
                if score_component:  # Ensure a component score object was returned
                    component_scores[calculator.component_name] = score_component
            except Exception as e:
                st.warning(f"Failed to calculate {calculator.component_name} score: {str(e)} - will display N/A")
                # Create a placeholder component score with None value
                component_scores[calculator.component_name] = ComponentScore(
                    component=calculator.component_name,
                    score=None,  # None will be displayed as N/A
                    sample_size=0,
                    confidence='none',
                    details={'status': 'calculation_error', 'error': str(e)}
                )

        if not component_scores:
            st.warning("No component scores could be calculated for the given criteria after attempting all components.")
            return {}

        # Component scores calculation complete
        return component_scores




    def get_criteria_confidence(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence levels for criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary with confidence information
        """
        return self.field_manager.calculate_confidence(criteria)

    def analyze_criteria_confidence(self, criteria: Dict[str, Any]) -> Dict[str, str]:
        """Analyze confidence levels for criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary of confidence levels by component
        """
        return self.field_manager.calculate_confidence(criteria)
        
    def calculate_network_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, integrated_data: Dict[str, pd.DataFrame]) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: Pre-matched shows DataFrame
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores
        """
        criteria_str = str(criteria) if isinstance(criteria, dict) else criteria
        
        if not hasattr(self, '_network_calculator'):
            self._network_calculator = NetworkScoreCalculator(self)
            
        # Always set the matching shows and integrated data
        self._network_calculator.set_matching_shows(matching_shows)
        self._network_calculator.set_integrated_data(integrated_data)
            
        return self._network_calculator.calculate_network_scores(criteria_str)
