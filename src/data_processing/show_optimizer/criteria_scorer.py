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
        
    def calculate_success_rate(self, shows: pd.DataFrame = None, threshold: Optional[float] = None, integrated_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Public method to calculate the success rate for a set of shows.
        
        Delegates to the private _calculate_success_rate method.
        
        Args:
            shows: DataFrame of shows (if None or empty, will use integrated_data['shows'] if provided)
            threshold: Optional success threshold
            integrated_data: Optional dict of integrated DataFrames (e.g., {'shows': DataFrame})
        Returns:
            Tuple of success rate and confidence information
        """
        return self._calculate_success_rate(shows, threshold, integrated_data=integrated_data)

    def _calculate_success_rate(self, shows: pd.DataFrame = None, threshold: Optional[float] = None, confidence_info: Optional[Dict[str, Any]] = None, integrated_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate the success rate for a set of shows with confidence information.
        
        Delegates success rate calculation to the SuccessScoreCalculator.
        
        Args:
            shows: DataFrame of shows (if None or empty, will use integrated_data['shows'] if provided)
            threshold: Optional success threshold
            confidence_info: Optional confidence information
            integrated_data: Optional dict of integrated DataFrames (e.g., {'shows': DataFrame})
        Returns:
            Tuple of success rate and confidence information
        """
        # Use integrated_data['shows'] if shows is None or empty and integrated_data is provided
        if (shows is None or shows.empty) and integrated_data is not None and 'shows' in integrated_data and not integrated_data['shows'].empty:
            shows = integrated_data['shows']
                
        if shows is None or shows.empty:

            if confidence_info is None:
                confidence_info = {'level': 'none', 'score': 0.0}
            return None, confidence_info
        
        # Delegate to SuccessScoreCalculator
        calculator = SuccessScoreCalculator()
        
        # Use the calculator's validate_and_prepare_data method with the same filter condition as in calculate
        def success_filter(df):
            return (df['success_score'].notna()) & (df['success_score'] > OptimizerConfig.SCORE_NORMALIZATION['success_filter_min'])
        
        is_valid, validated_data, validation_info = calculator.validate_and_prepare_data(
            shows, 
            required_columns=['success_score'],
            optional_columns=[],
            data_column='success_score',
            filter_condition=success_filter
        )
        
        # No debug logging for validation results
        
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
        try:
            component_score = calculator.calculate(validated_data, threshold=threshold)
            
            if component_score is None:
                if confidence_info is None:
                    confidence_info = {'level': 'none', 'score': 0.0, 'error': 'Failed to calculate success score'}
                else:
                    confidence_info['error'] = 'Failed to calculate success score'
                return None, confidence_info
            
        except Exception as e:
            if confidence_info is None:
                confidence_info = {'level': 'none', 'score': 0.0, 'error': f'Exception during calculation: {str(e)}'}
            else:
                confidence_info['error'] = f'Exception during calculation: {str(e)}'
            return None, confidence_info
        
        # Extract success rate and metadata from component score
        success_rate = component_score.score
        
        # Update confidence information
        if confidence_info is None:
            confidence_info = {}
        
        # Merge metadata from component score into confidence info
        for key, value in component_score.details.items():
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
            return [None] * len(criteria_list)
        if len(criteria_list) != len(matching_shows_list):
            return [None] * len(criteria_list)
        
        results = []
        
        # Create a single SuccessScoreCalculator instance to reuse
        calculator = SuccessScoreCalculator()
        
        for i, (criteria, matching_shows) in enumerate(zip(criteria_list, matching_shows_list)):
            try:
                if matching_shows is None or matching_shows.empty:
                    results.append(None)
                    continue
                    
                # Use the calculator to calculate the success rate with the proper threshold
                threshold = OptimizerConfig.PERFORMANCE.get('success_threshold', None)
                
                try:
                    # Ensure we pass the correct parameters to calculate
                    component_score = calculator.calculate(matching_shows, threshold=threshold)
                    
                    if component_score is None:
                        results.append(None)
                    else:
                        # Extract just the success rate value
                        results.append(component_score.score)
                except Exception as calc_error:
                    # Handle specific calculation errors
                    results.append(None)
                    
            except Exception as e:
                results.append(None)
                
        return results
    
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
            
            # Let the field manager handle array field identification
            array_field_mapping = self.field_manager.get_array_field_mapping()
            array_fields = list(array_field_mapping.keys())
            
            # Use the provided base matching shows
            base_match_count = len(base_matching_shows) if not base_matching_shows.empty else 0
            
            if base_matching_shows.empty:
                st.write("DEBUG: Cannot calculate impact scores - no matching shows")
                raise ValueError("Cannot calculate impact scores with no matching shows")
                
            if base_match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                # Don't raise an error, just proceed with what we have
                pass
                
            # Calculate base success rate using the provided shows
            base_rate, base_info = self._calculate_success_rate(base_matching_shows)
            
            # Check if base_rate is None and return early if it is
            if base_rate is None:
                return {}
            
            impact_scores = {}
            
            # Determine which fields to process
            fields_to_process = [field_name] if field_name else self.field_manager.FIELD_CONFIGS.keys()
            
            def make_hashable(val):
                """Convert any value to a hashable type for dictionary keys.
                
                Uses field_manager's knowledge of array fields to properly handle list values.
                """
                if isinstance(val, dict):
                    # Convert dict to a sorted tuple of (key, value) pairs
                    return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
                    
                if isinstance(val, list):
                    # For empty lists, return an empty tuple
                    if not val:
                        return tuple()
                    # For lists of primitive values, sort and convert to tuple
                    if all(isinstance(v, (int, float, str, bool)) for v in val):
                        return tuple(sorted(val))
                    # For lists of complex values, convert each item and then sort
                    return tuple(sorted(make_hashable(v) for v in val))
                    
                if isinstance(val, tuple):
                    return tuple(make_hashable(v) for v in val)
                    
                # Ensure we return a hashable type
                if isinstance(val, (str, int, float, bool, type(None))):
                    return val
                    
                # Last resort - convert to string
                return str(val)
            
            # Check if fields_to_process is empty
            if not fields_to_process:
                # If no fields to process, use all available fields
                fields_to_process = list(self.field_manager.FIELD_CONFIGS.keys())
            
            for current_field in fields_to_process:
                # Process both fields in base criteria (for Remove/Change) and not in base criteria (for Add)
                
                # Use field_manager to determine if this is an array field
                is_array_field = self.field_manager.get_field_type(current_field) == 'array'
                options = self.field_manager.get_options(current_field)
                
                
                # Processing field
                
                # Prepare batch criteria for all options
                batch_criteria = []
                option_data = []
                recommendation_types = []
                
                # Check if this field is already in the base criteria
                field_in_base = current_field in base_criteria
                current_value = base_criteria.get(current_field) if field_in_base else None
                
                if field_in_base:
                    # For fields already in criteria, we'll calculate both Remove and Change recommendations
                    
                    # Field in base criteria
                    
                    # 1. First, create a "Remove" recommendation by removing this field
                    remove_criteria = base_criteria.copy()
                    del remove_criteria[current_field]
                    batch_criteria.append(remove_criteria)
                    option_data.append(('remove', 'Remove ' + current_field))
                    recommendation_types.append('remove')
                    
                    # 2. Then, create "Change" recommendations for each alternative option
                    for option in options:
                        # Skip the current value
                        if is_array_field and isinstance(current_value, list) and option.id in current_value:
                            continue
                        elif not is_array_field and option.id == current_value:
                            continue
                            
                        # Create a new criteria with this option
                        new_criteria = base_criteria.copy()
                        
                        # For array fields, we replace the current value with a new array
                        # For scalar fields, we simply replace the value
                        if is_array_field and isinstance(current_value, list):
                            # Replace the array with a new array containing just this option
                            option_key = int(option.id)
                            new_criteria[current_field] = [option_key]
                        else:
                            # Replace the scalar value
                            option_key = int(option.id)
                            new_criteria[current_field] = option_key
                            
                        batch_criteria.append(new_criteria)
                        option_data.append((option_key, self.field_manager.get_name(current_field, option.id)))
                        recommendation_types.append('change')
                else:
                    # For fields not in criteria, create "Add" recommendations for each option
                        
                    for option in options:
                        # Create a new criteria with this option
                        new_criteria = base_criteria.copy()
                        
                        # For array fields, always use list of ints
                        # For scalar fields, use the int directly
                        if is_array_field:
                            option_key = int(option.id)
                            new_criteria[current_field] = [option_key]
                        else:
                            option_key = int(option.id)
                            new_criteria[current_field] = option_key
                            
                        batch_criteria.append(new_criteria)
                        option_data.append((option_key, self.field_manager.get_name(current_field, option.id)))
                        recommendation_types.append('add')
                
                # Process each option using the provided option_matching_shows_map or generate it if not provided
                field_impact = {}
                
                # If option_matching_shows_map is not provided, we need to generate it
                if not option_matching_shows_map or current_field not in option_matching_shows_map:
                    # Generate option matching shows map for this field
                    field_options_map = {}
                    
                    # For each option in option_data, run the matcher to get matching shows
                    for i, (option_id, option_name) in enumerate(option_data):
                        try:
                            # Create criteria for this option
                            option_criteria = base_criteria.copy()
                            
                            # Set the criteria value based on field type
                            is_array_field = self.field_manager.get_field_type(current_field) == 'array'
                            if is_array_field:
                                option_criteria[current_field] = [option_id]
                            else:
                                option_criteria[current_field] = option_id
                                
                            # Get matching shows for this option
                            # Pass None as the data parameter to use the original integrated data in the matcher
                            # This ensures we're not filtering an already filtered dataset
                            option_shows, _, _ = self._get_matching_shows(option_criteria, None)
                            
                            # Check if we got valid shows
                            if option_shows is not None and not option_shows.empty:
                                # Store in field_options_map
                                field_options_map[option_id] = option_shows
                        except Exception as e:
                            # Skip this option
                            continue
                    # Set the option_matching_shows_map for this field
                    if not option_matching_shows_map:
                        option_matching_shows_map = {}
                    option_matching_shows_map[current_field] = field_options_map
                else:
                    # Use the provided option_matching_shows_map
                    field_options_map = option_matching_shows_map[current_field]
                    
                    # Convert all keys in field_options_map to hashable for safe lookup
                    hashable_field_options_map = {}
                    for k, v in field_options_map.items():
                        hashable_key = self.make_hashable(k)
                        hashable_field_options_map[hashable_key] = v
                    
                    # Verify that we have matching shows for at least some options
                    has_valid_options = False
                    for k, v in hashable_field_options_map.items():
                        if v is not None and not (isinstance(v, pd.DataFrame) and v.empty):
                            has_valid_options = True
                            break
                            
                    if not has_valid_options:
                        continue
                
                # Process each option in option_data
                for i, (option_id, option_name) in enumerate(option_data):
                    # Get the option shows from field_options_map if available
                    if option_matching_shows_map and current_field in option_matching_shows_map:
                        # Make the option_id hashable for lookup
                        hashable_key = self.make_hashable(option_id)
                        
                        # Get matching shows for this option from the hashable map
                        option_shows = hashable_field_options_map.get(hashable_key)
                    else:
                        # Use the option shows we generated earlier
                        option_shows = field_options_map.get(option_id)
                    
                    # Skip options with no matching shows
                    if option_shows is None or (isinstance(option_shows, pd.DataFrame) and option_shows.empty):
                        continue
                        
                    # Get display name for the option
                    try:
                        if isinstance(option_id, (int, float)):
                            display_name = self.field_manager.get_name(current_field, int(option_id))
                        else:
                            display_name = option_name
                    except Exception as e:
                        display_name = option_name
                        
                    # Calculate success rate for this option
                    option_rate, option_info = self._calculate_success_rate(option_shows)
                    
                    # Skip options with no valid success rate
                    if option_rate is None:
                        continue
                        
                    # Calculate impact as difference from base rate
                    impact = option_rate - base_rate
                    
                    # Store impact data
                    if current_field not in impact_scores:
                        impact_scores[current_field] = {}
                        
                    # Store impact data with option information
                    impact_scores[current_field][option_id] = {
                        'impact': impact,
                        'success_rate': option_rate,
                        'sample_size': option_info.get('valid_shows', 0),
                        'recommendation_type': recommendation_types[i],
                        'option_name': display_name
                    }
                    
                    # Only log in debug mode
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Calculated impact for {current_field} option {display_name}: {impact:.4f}")
                        if OptimizerConfig.VERBOSE_DEBUG:
                            st.write(f"DEBUG: Base rate: {base_rate:.4f}, Option rate: {option_rate:.4f}")
                            st.write(f"DEBUG: Sample size: {option_info.get('valid_shows', 0)}")
                            st.write(f"DEBUG: Recommendation type: {recommendation_types[i]}")
                        
                        
                    
            # Check if we have any impact scores after processing all fields
            if not impact_scores and fields_to_process:
                # If no impact scores were generated, log a message in debug mode
                if OptimizerConfig.DEBUG_MODE:
                    st.write("No impact scores could be generated for the current criteria.")
                return {}
                
            return impact_scores

        except ValueError as ve:
            return {}
        except Exception as e:
            return {}
            
    def calculate_component_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, confidence_info: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame] = None) -> Dict[str, ComponentScore]:
        """
        Calculate component scores for the given criteria using provided matched shows.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of shows matching the criteria
            confidence_info: Confidence information from matching
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        component_scores: Dict[str, ComponentScore] = {}

        try:
            # First check if we have any matching shows
            if matching_shows is None or matching_shows.empty:
                return {}
                
            # Calculate match count
            match_count = len(matching_shows)
            
            # Using general minimum_sample from OptimizerConfig
            if match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                pass  # Continue with calculation despite small sample
                
            calculators = [
                SuccessScoreCalculator(),
                AudienceScoreCalculator(),
                CriticsScoreCalculator(),
                LongevityScoreCalculator()
            ]
            
            # Calculate scores for each component using the SAME matching_shows DataFrame
            # This prevents redundant matching operations
            for calculator in calculators:
                try:
                    # Pass the same matching_shows to all calculators
                    score_component = calculator.calculate(matching_shows.copy())  # Pass a copy to avoid modification issues
                    if score_component:  # Ensure a component score object was returned
                        component_scores[calculator.component_name] = score_component
                except Exception as e:
                    # Create a placeholder component score with None value
                    component_scores[calculator.component_name] = ComponentScore(
                        component=calculator.component_name,
                        score=None,  # None will be displayed as N/A
                        sample_size=0,
                        confidence='none',
                        details={'status': 'calculation_error', 'error': str(e)}
                    )

            if not component_scores:
                return {}

            # Component scores calculation complete
            return component_scores
            
        except Exception as e:
            return {}




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
        
    def _get_matching_shows(self, criteria: Dict[str, Any], data: pd.DataFrame = None, flexible: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Get shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            data: Optional DataFrame to use instead of integrated data
            flexible: Whether to use flexible matching (fallback to fewer criteria)
            
        Returns:
            Tuple of (matching_shows, confidence_info, match_info)
        """
        if self.matcher is None:
            return pd.DataFrame(), {'level': 'none', 'score': 0.0}, {'match_level': 0}
            
        # Use find_matches_with_fallback to get matches
        matching_shows, confidence_info = self.matcher.find_matches_with_fallback(
            criteria, data, flexible=flexible)
            
        # Extract match level from confidence info
        match_level = confidence_info.get('match_level', 0) if confidence_info else 0
        match_info = {'match_level': match_level}
        
        return matching_shows, confidence_info, match_info
    
    def calculate_network_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: Pre-matched shows DataFrame
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores
        """
        criteria_str = str(criteria) if isinstance(criteria, dict) else criteria
        
        if not hasattr(self, '_network_calculator'):
            self._network_calculator = NetworkScoreCalculator()
            
        # Pass the field_manager to the network calculator if needed
        if hasattr(self, 'field_manager') and not hasattr(self._network_calculator, 'field_manager'):
            self._network_calculator.field_manager = self.field_manager
            
        return self._network_calculator.calculate_network_scores(criteria_str, matching_shows)
