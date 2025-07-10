"""RecommendationEngine: Handles success factor identification and recommendation generation.

This component is responsible for analyzing show data to identify success factors
and generate recommendations for show concept optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set, Union

from .optimizer_config import OptimizerConfig
from .criteria_scorer import ImpactAnalysisResult, ImpactAnalysisSummary
from .optimizer_data_contracts import (
    CriteriaDict, ConfidenceInfo, IntegratedData,
    NetworkMatch, RecommendationItem, FieldValueData, FieldValueSuccessRate,
    update_confidence_info, create_success_rate_data, create_field_value_key, parse_field_value_key
)


@dataclass
class SuccessFactor:
    """A success factor identified from analysis."""
    criteria_type: str  # e.g., "genre", "character_types"
    criteria_value: Any  # ID of the value
    criteria_name: str   # Display name
    impact_score: float  # Impact on success (-1 to 1)
    confidence: str      # none, low, medium, high
    recommendation_type: str  # add, remove, change
    sample_size: int = 0
    matching_titles: List[str] = field(default_factory=list)  # List of show titles matching this criteria

# Note: The Recommendation dataclass has been removed in favor of the RecommendationItem TypedDict
# defined in optimizer_data_contracts.py


class RecommendationEngine:
    """Engine for generating recommendations based on criteria and matching shows."""
    
    # Recommendation type constants
    REC_TYPE_ADD = 'add'
    REC_TYPE_CHANGE = 'change'
    REC_TYPE_REMOVE = 'remove'
    REC_TYPE_NETWORK_KEEP = 'network_keep'
    REC_TYPE_NETWORK_CHANGE = 'network_change'
    
    """Analyzes show data to identify success factors and generate recommendations."""
    

    def _create_success_rate_entry(self, field_name, field_value, success_rate, sample_size):
        """Create a standardized success rate entry for consistent data structure.
        
        DATA CONTRACT: This method ensures consistent data structure between network-specific
        and overall success rates by using the create_success_rate_data function from
        optimizer_data_contracts.
        
        Args:
            field_name: Name of the field (e.g., 'genre_id', 'tone_id')
            field_value: The field value
            success_rate: Success rate as a float
            sample_size: Number of samples used to calculate the rate
            
        Returns:
            A dictionary with standardized structure for success rate data
        """
        return create_success_rate_data(
            field_name=field_name,
            value=field_value,
            rate=success_rate,
            sample_size=sample_size,
            value_name=self._get_criteria_name(field_name, field_value)
        )    
    
    def __init__(self, shows_analyzer, success_analyzer, field_manager, criteria_scorer=None):
        """Initialize the recommendation engine.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for show data
            success_analyzer: SuccessAnalyzer instance for success metrics
            field_manager: FieldManager instance for field mapping and validation
            criteria_scorer: Optional CriteriaScorer instance for criteria impact calculations
        """
        # Validate required dependencies
        if shows_analyzer is None:
            st.error("Required component missing. Please ensure your application is properly configured.")
            raise ValueError("shows_analyzer cannot be None")
        if success_analyzer is None:
            st.error("Required component missing. Please ensure your application is properly configured.")
            raise ValueError("success_analyzer cannot be None")
        if field_manager is None:
            st.error("Required component missing. Please ensure your application is properly configured.")
            raise ValueError("field_manager cannot be None")
            
        self.shows_analyzer = shows_analyzer
        self.success_analyzer = success_analyzer
        self.field_manager = field_manager
        self.criteria_scorer = criteria_scorer or CriteriaScorer(shows_analyzer, success_analyzer, field_manager)
        self.config = OptimizerConfig
        
        # Constants for recommendation types
        self.REC_TYPE_ADD = "add"
        self.REC_TYPE_CHANGE = "change"
        self.REC_TYPE_REMOVE = "remove"
        self.REC_TYPE_NETWORK_KEEP = "network_keep"
        self.REC_TYPE_NETWORK_CHANGE = "network_change"
        
        # Initialize memoization cache for processed combinations
        self._processed_combinations = {}
        
        # Use network_analyzer from criteria_scorer as per the architecture flow
        # NetworkAnalyzer.analyze_network_compatibility (step 3.4) comes before 
        # RecommendationEngine.generate_recommendations (step 3.5)
        self.network_analyzer = self.criteria_scorer.network_analyzer
    
    def calculate_overall_success_rate(self, criteria: CriteriaDict, matching_shows=None) -> Tuple[float, str]:
        """Calculate the overall success rate for the given criteria.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: Optional pre-filtered shows to use (optimization)
            
        Returns:
            Tuple of (success_rate, confidence_level)
        """
        # Handle missing criteria
        if criteria is None:
            criteria = {}
            
        # Check if we have a criteria_scorer
        if self.criteria_scorer is None:
            # Return None with confidence from config
            return None, self.config.CONFIDENCE_LEVELS.get('none', 'none')
            
        try:
            # If we already have matching shows, use them directly
            if matching_shows is not None and not matching_shows.empty:
                # Calculate all scores including success rate
                all_scores = self.criteria_scorer.calculate_scores(criteria, matching_shows)
                success_rate = all_scores.get('success_rate')
                # Use medium confidence when using pre-filtered shows
                return success_rate, 'medium'
            
            # Otherwise, get matching shows using the matcher directly
            matching_shows, confidence_info = self.criteria_scorer.matcher.find_matches_with_fallback(criteria)
            
            # Get match count from confidence_info
            match_count = confidence_info.get('match_count', 0)
            
            # Return None if no matches found
            if matching_shows.empty or match_count == 0:
                return None, 'none'
            
            # Calculate all scores including success rate
            all_scores = self.criteria_scorer.calculate_scores(criteria, matching_shows)
            success_rate = all_scores.get('success_rate')
            
            # Determine confidence level
            confidence = confidence_info.get('confidence', 'none')
            
            return success_rate, confidence
        except Exception as e:
            # Keep a single top-level try-except for critical errors
            st.error(f"Error calculating success rate: {str(e)}")
            # Log the error but don't stop execution
            return None, 'none'
            
    def create_field_value_criteria(self, field_name: str, field_value: Any) -> CriteriaDict:
        """Create a criteria dictionary for a single field value.
        
        DATA CONTRACT: This method must use exact database column names (IDs) to ensure
        consistent key matching between network rates and overall rates.
        
        Args:
            field_name: The field name (e.g., 'genre_id', 'tone_id')
            field_value: The value to filter by
            
        Returns:
            A criteria dictionary with just the specified field and value
        """
        # IMPORTANT: Use exact field name without any mapping to ensure consistent keys
        # between network rates and overall rates
        
        # Create and return the criteria dictionary using the exact field name
        return {field_name: field_value}
    
    def identify_success_factors(self, criteria: CriteriaDict, matching_shows: pd.DataFrame = None, integrated_data: IntegratedData = None, limit: int = 5) -> List[SuccessFactor]:
        """Identify success factors from the given criteria and matching shows.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of shows matching the criteria (optional)
            integrated_data: Dictionary of integrated data frames conforming to IntegratedData (optional)
            limit: Maximum number of success factors to identify per criteria type
            
        Returns:
            List of SuccessFactor objects
        """
        # If matching_shows not provided or empty, get them using the matcher
        # Use len() instead of .empty for better performance
        if matching_shows is None or (hasattr(matching_shows, 'empty') and len(matching_shows) == 0):
            try:
                # Use the matcher directly instead of going through criteria_scorer
                if hasattr(self.criteria_scorer, 'matcher') and self.criteria_scorer.matcher is not None:
                    matching_shows, confidence_info = self.criteria_scorer.matcher.find_matches_with_fallback(criteria)
                    # Use len() instead of .empty for better performance
                    if len(matching_shows) == 0:
                        st.error("No shows match your criteria. Try adjusting your parameters.")
                        return []
                else:
                    st.error("No matcher available. Cannot find matching shows.")
                    return []
            except Exception as e:
                st.error(f"Unable to analyze shows matching your criteria: {str(e)}")
                return []
        
        try:
            # Create a key for the criteria and matching shows combination
            # This will be used to check if we've already calculated impact for this exact combination
            criteria_key = str(sorted([(k, str(v)) for k, v in criteria.items()]))
            
            # Always show debug message for criteria impact calculations
            # Use a more visible format for these debug messages
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG [CRITERIA IMPACT]: Checking criteria impact calculation for criteria hash: {hash(criteria_key)}")
            
            # Check if we've already calculated impact for this criteria combination
            if self._should_process_combination("criteria_impact", criteria_key):
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG [CRITERIA IMPACT]: NEW CALCULATION - First time processing this criteria combination")
                # Calculate impact data using the criteria scorer
                # Pass integrated_data to ensure matcher has access to full dataset
                impact_result = self.criteria_scorer.calculate_criteria_impact(criteria, matching_shows, integrated_data=integrated_data)
            else:
                # We've already seen this criteria combination, skip calculation
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG [CRITERIA IMPACT]: CACHE HIT - Skipping duplicate calculation")
                    
                # For now, we still need to calculate even though we've seen this before
                # In a future optimization, we could store and reuse the results
                impact_result = self.criteria_scorer.calculate_criteria_impact(criteria, matching_shows, integrated_data=integrated_data)
            
            # Check for errors
            if impact_result.error:

                return []
                
            # Get the impact data from the result
            impact_data = impact_result.criteria_impacts
            
        except Exception as e:
            error_msg = f"Error calculating impact data: {str(e)}"
            st.write(error_msg)
            return []
        
        # Convert to SuccessFactor objects
        success_factors = []
        
        # Process each criteria type in the impact data
        for criteria_type, values in impact_data.items():
            # Skip empty values
            if not values:
                continue
                
            # Skip if no values to process (redundant check removed)
            
            processed_count = 0
            
            # Process each value in the impact data
            for value_id, impact_info in values.items():
                # Limit the number of factors per criteria type
                if processed_count >= limit:
                    break
                
                # Extract impact score and sample size with validation
                if not isinstance(impact_info, dict):
                    raise ValueError(f"Expected dict for impact_info, got {type(impact_info).__name__} for {criteria_type}.{value_id}")
                    
                impact = impact_info.get('impact', self.config.DEFAULT_VALUES['impact_score'])
                sample_size = impact_info.get('sample_size', self.config.DEFAULT_VALUES['fallback_sample_size'])
                
                # Use value directly - create_field_value_key will handle hashability
                criteria_value = value_id
                
                # Get the proper display name
                if value_id == 'remove':
                    name = f"Remove {criteria_type}"
                else:
                    name = self._get_criteria_name(criteria_type, value_id)
                
                # Determine confidence level
                confidence = self.config.get_confidence_level(sample_size)
                
                # Get recommendation type from impact data
                recommendation_type = impact_info.get('recommendation_type')
                
                # If no recommendation_type is specified, use criteria_scorer to determine it properly
                if not recommendation_type:
                    # Check if the field is selected in the criteria
                    is_field_selected = criteria_type in criteria
                    is_option_selected = False
                    
                    # Check if this specific option is selected
                    if is_field_selected:
                        field_value = criteria[criteria_type]
                        if isinstance(field_value, list):
                            is_option_selected = value_id in field_value
                        else:
                            is_option_selected = value_id == field_value
                        
                    # Determine recommendation type based on selection status and impact
                    if not is_field_selected and impact > 0:
                        recommendation_type = self.REC_TYPE_ADD
                    elif is_field_selected and is_option_selected and impact < 0:
                        recommendation_type = self.REC_TYPE_REMOVE
                    elif is_field_selected and not is_option_selected and impact > 0:
                        recommendation_type = self.REC_TYPE_CHANGE
                    else:
                        # Skip this factor if no valid recommendation type
                        continue
                
                # Skip if we still don't have a valid recommendation type
                if not recommendation_type:
                    continue
                
                # Get matching titles for this criteria
                matching_titles = []
                try:
                    # Convert hashable value back to original form if needed - optimize by checking type once
                    is_tuple = isinstance(criteria_value, tuple)
                    match_value = list(criteria_value) if is_tuple else criteria_value
                    
                    # Get shows matching just this single criteria
                    single_criteria = {criteria_type: match_value}
                    single_matches, single_confidence = self.criteria_scorer.matcher.find_matches_with_fallback(single_criteria)
                    
                    if not single_matches.empty and 'title' in single_matches.columns:
                        matching_titles = single_matches['title'].tolist()[:100]  # Limit to 100 titles
                except Exception as e:
                    st.write(f"Error finding matches for single criteria {criteria_type}: {str(e)}")
                
                # Create and add the success factor
                try:
                    factor = SuccessFactor(
                        criteria_type=criteria_type,
                        criteria_value=criteria_value,
                        criteria_name=name,
                        impact_score=impact,
                        confidence=confidence,
                        sample_size=sample_size,
                        matching_titles=matching_titles,
                        recommendation_type=recommendation_type
                    )
                    success_factors.append(factor)
                    processed_count += 1
                except Exception as e:
                    st.write(f"Error creating success factor for {criteria_type}: {str(e)}")
        
        return success_factors
            
    def generate_all_recommendations(self, criteria: CriteriaDict, matching_shows: pd.DataFrame = None, 
                                integrated_data: IntegratedData = None, 
                                top_networks: List[NetworkMatch] = None,
                                confidence_info: Optional[ConfidenceInfo] = None) -> List[RecommendationItem]:
        """Generate all recommendations in a single unified list.
        
        This method generates both general and network-specific recommendations in one pass,
        tagging each recommendation appropriately. All recommendations are returned in a single
        list, sorted by impact score.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of shows matching the criteria (optional)
            integrated_data: Dictionary of integrated data frames (optional)
            top_networks: List of NetworkMatch objects for network-specific recommendations
            confidence_info: Optional confidence information dictionary

            
        Returns:
            A single list of recommendation items with appropriate tagging
        """
        # Reset the processed combinations cache for a fresh start
        self._reset_processed_combinations()
        
        # Initialize empty recommendations list
        recommendations = []
        
        try:
            # If matching_shows not provided or empty, get them using the matcher
            if matching_shows is None or matching_shows.empty:
                try:
                    # Use the matcher directly instead of going through criteria_scorer
                    if hasattr(self.criteria_scorer, 'matcher') and self.criteria_scorer.matcher is not None:
                        matching_shows, confidence_info = self.criteria_scorer.matcher.find_matches_with_fallback(criteria)
                        if matching_shows.empty:
                            # No shows match criteria
                            return []
                    else:
                        # No matcher available
                        return []
                except Exception as e:
                    # Display specific error when finding matches
                    st.write(f"Error finding matches: {str(e)}")
                    return []
            
            # Calculate impact data - this is the expensive operation we want to avoid duplicating
            try:
                # For recommendations, we need to analyze all fields that are already in criteria
                # This limits the expensive calculations to only fields that are already selected
                fields_to_analyze = list(criteria.keys())
                
                # Always include 'network' in fields_to_analyze to ensure network-specific recommendations work
                if 'network' not in fields_to_analyze:
                    fields_to_analyze.append('network')
                
                # Analyze impact for selected fields and network field
                impact_result = self.criteria_scorer.calculate_criteria_impact(
                    criteria, 
                    matching_shows, 
                    integrated_data=integrated_data,
                    fields_to_analyze=fields_to_analyze
                )
                
                # Get the impact data from the result
                impact_data = impact_result.criteria_impacts
                
            except Exception as e:
                # Display specific error when calculating impact data
                st.write(f"Error calculating impact data: {str(e)}")
                return []
                
            # Convert to SuccessFactor objects
            success_factors = self.identify_success_factors(criteria, matching_shows, integrated_data)
            
            # Generate general recommendations from success factors
            general_recs = self._recommend_missing_criteria(criteria, success_factors, matching_shows)
            
            # Add general recommendations to our unified list
            for rec in general_recs:
                # Ensure is_network_specific is set to False for general recommendations
                if 'is_network_specific' not in rec:
                    rec['is_network_specific'] = False
                recommendations.append(rec)
            
            # Apply max suggestions limit to general recommendations
            max_suggestions = self.config.SUGGESTIONS.get('max_suggestions', 5)
            if len(recommendations) > max_suggestions:
                recommendations = recommendations[:max_suggestions]
            
            # Generate network-specific recommendations if networks are provided
            if top_networks and len(top_networks) > 0 and matching_shows is not None and not matching_shows.empty:
                # Only proceed if we have the network_id column in matching_shows
                if 'network_id' not in matching_shows.columns:
                    st.write("Cannot generate network-specific recommendations: missing network_id column in matching shows")
                else:
                    # Calculate success rates for each network
                    network_rates = {}
                    overall_rates = {}
                    
                    # Calculate overall success rates for all criteria fields
                    for field_name in criteria.keys():
                        # Use field_manager to get the correct database column name
                        db_field = self.field_manager.map_field_name(field_name)
                        if not db_field:
                            # Fallback to simple mapping if field_manager doesn't have a mapping
                            db_field = field_name if field_name.endswith('_id') else f"{field_name}_id"
                        field_value = criteria[field_name]
                        
                        # Skip non-scalar values (lists, etc.)
                        if isinstance(field_value, (list, np.ndarray)):
                            continue
                            
                        # Create key using exact database column name format
                        key = create_field_value_key(db_field, field_value)
                        
                        # Calculate overall success rate
                        success_threshold = OptimizerConfig.SUCCESS.get('threshold', 0.7)
                        success_count = matching_shows[matching_shows['success_score'] >= success_threshold].shape[0]
                        total_count = matching_shows.shape[0]
                        
                        if total_count > 0:
                            success_rate = success_count / total_count
                            
                            # Store overall rate using consistent format
                            overall_rates[key] = {
                                'rate': success_rate,
                                'sample_size': total_count,
                                'field_name': db_field,
                                'field_value': field_value
                            }
                
                    # Process each network to calculate network-specific success rates
                    for network in top_networks:
                        if not hasattr(network, 'network_id') or network.network_id is None:
                            continue
                            
                        # Filter shows for this network
                        network_shows = matching_shows[matching_shows['network_id'] == network.network_id]
                        
                        if network_shows.empty:
                            continue
                            
                        # Calculate network-specific success rates for each criteria field
                        network_specific_rates = {}
                        
                        # Process each selected field to calculate success rates for all options
                        for field_name in criteria.keys():
                            # Skip non-scalar values and empty fields
                            field_value = criteria[field_name]
                            if isinstance(field_value, (list, np.ndarray)) or field_value is None:
                                continue
                                
                            # Use field_manager to get the correct database column name
                            db_field = self.field_manager.map_field_name(field_name)
                            if not db_field:
                                # Fallback to simple mapping if field_manager doesn't have a mapping
                                db_field = field_name if field_name.endswith('_id') else f"{field_name}_id"
                            
                            # Create key using exact database column name format for the current value
                            current_key = create_field_value_key(db_field, field_value)
                            
                            # Calculate network-specific success rate for current value
                            success_threshold = OptimizerConfig.SUCCESS.get('threshold', 0.7)
                            success_count = network_shows[network_shows['success_score'] >= success_threshold].shape[0]
                            total_count = network_shows.shape[0]
                            
                            if total_count > 0:
                                network_rate = success_count / total_count
                                
                                # Store network-specific rate for current value using consistent format
                                network_specific_rates[current_key] = {
                                    'rate': network_rate,
                                    'sample_size': total_count,
                                    'field_name': db_field,
                                    'field_value': field_value
                                }
                                
                                # Also calculate success rates for alternative options in this field
                                # Get all options for this field
                                field_options = self.field_manager.get_options(field_name)
                                
                                # Track best alternative option
                                best_alt_rate = 0
                                best_alt_value = None
                                best_alt_name = None
                                
                                # Check each alternative option
                                for option in field_options:
                                    # Skip the current value
                                    if option.id == field_value:
                                        continue
                                        
                                    # Create a filter for shows with this option
                                    option_filter = network_shows[db_field] == option.id
                                    option_shows = network_shows[option_filter]
                                    
                                    # Skip if no shows match this option
                                    if option_shows.empty:
                                        continue
                                        
                                    # Calculate success rate for this option
                                    opt_success_count = option_shows[option_shows['success_score'] >= success_threshold].shape[0]
                                    opt_total_count = option_shows.shape[0]
                                    
                                    if opt_total_count >= OptimizerConfig.SUCCESS['min_data_points']:
                                        opt_rate = opt_success_count / opt_total_count
                                        
                                        # Check if this is better than current best alternative
                                        if opt_rate > best_alt_rate:
                                            best_alt_rate = opt_rate
                                            best_alt_value = option.id
                                            best_alt_name = option.name
                                
                                
                                # Store the best alternative in the current value's data
                                if best_alt_value is not None:
                                    network_specific_rates[current_key]['best_alternative'] = {
                                        'value': best_alt_value,
                                        'name': best_alt_name,
                                        'rate': best_alt_rate
                                    }
                
                            # Now compare network-specific rates with overall rates to generate recommendations
                            for key, network_rate_data in network_specific_rates.items():
                                # Get corresponding overall rate
                                overall_rate_data = overall_rates.get(key)
                                
                                if not overall_rate_data:
                                    continue
                                    
                                # Extract rates for comparison
                                network_rate = network_rate_data['rate']
                                overall_rate = overall_rate_data['rate']
                                sample_size = network_rate_data['sample_size']
                                
                                # Calculate difference between network and overall rates
                                difference = network_rate - overall_rate
                                
                                # Check if we have enough data points for reliable comparison
                                has_sufficient_data = sample_size >= OptimizerConfig.SUCCESS['min_data_points']
                                
                                # Get thresholds from config
                                network_diff_threshold = OptimizerConfig.THRESHOLDS['network_difference']
                                significant_diff_threshold = OptimizerConfig.THRESHOLDS['significant_difference']
                                
                                # Determine if the difference is significant enough for a recommendation
                                condition1 = abs(difference) >= significant_diff_threshold  # Large difference
                                condition2 = has_sufficient_data and abs(difference) > network_diff_threshold  # Smaller difference with sufficient data
                                should_generate = condition1 or condition2
                                
                                # Create recommendation if the difference is significant
                                if should_generate:
                                    # Extract field information
                                    field_name = network_rate_data['field_name']
                                    field_value = network_rate_data['field_value']
                                    
                                    # Check if we've already processed this network-field-value combination
                                    if not self._should_process_combination("network_specific", network.network_id, field_name, field_value):
                                        # Skip this combination as we've already processed it
                                        if OptimizerConfig.DEBUG_MODE:
                                            st.write(f"DEBUG: Skipping already processed combination: {network.network_name} - {field_name}: {field_value}")
                                        continue
                                    
                                    # Get display name for the current value
                                    # For database column names ending with _id or _ids, we need to get the display field name
                                    display_field = field_name
                                    
                                    # Use the old logic as it's more reliable for this reverse mapping case
                                    if field_name.endswith('_id') and not field_name.endswith('_ids'):
                                        display_field = field_name[:-3]  # Remove '_id' suffix
                                    elif field_name.endswith('_ids'):
                                        display_field = field_name[:-4]  # Remove '_ids' suffix
                                        
                                    current_name = self._get_criteria_name(display_field, field_value)
                                    
                                    # Calculate impact score
                                    impact_score = max(abs(difference), OptimizerConfig.THRESHOLDS['significant_difference']) * (1 if difference > 0 else -1)
                                    
                                    # Determine recommendation type and create explanation
                                    if difference > 0:
                                        # This is a positive recommendation - keep the current value
                                        rec_type = self.REC_TYPE_NETWORK_KEEP
                                        explanation = f"{network.network_name} shows a {network_rate*100:.1f}% success rate with {network.network_name}: {current_name} compared to the overall average of {overall_rate*100:.1f}%. Keeping this element could increase success probability by {abs(difference)*100:.1f}%."
                                        
                                        # Create recommendation with no suggested alternative
                                        recommendation = {
                                            "recommendation_type": rec_type,
                                            "field": field_name,
                                            "current_value": field_value,
                                            "suggested_value": None,
                                            "suggested_name": f"{network.network_name}: {current_name}",
                                            "impact": impact_score,
                                            "confidence": "medium",
                                            "explanation": explanation,
                                            "is_network_specific": True,
                                            "sample_size": sample_size,
                                            "metadata": {
                                                "network_id": network.network_id,
                                                "network_name": network.network_name,
                                                "network_rate": network_rate,
                                                "overall_rate": overall_rate,
                                                "difference": difference,
                                                "alt_found": False,
                                                "alt_met_threshold": False
                                            }
                                        }
                                    else:
                                        # This is a negative recommendation - suggest changing the current value
                                        rec_type = self.REC_TYPE_NETWORK_CHANGE
                                        
                                        # Initialize variables for alternatives
                                        alt_value = None
                                        alt_name = None
                                        alt_rate = None
                                        
                                        # Initialize tracking flags
                                        has_alternative = False
                                        alt_found = False
                                        alt_met_threshold = False
                                        
                                        # First try to find network-specific alternatives
                                        if 'best_alternative' in network_rate_data:
                                            alt_data = network_rate_data['best_alternative']
                                            alt_value = alt_data.get('value')
                                            alt_name = alt_data.get('name')
                                            alt_rate = alt_data.get('rate')
                                            
                                            # Mark that we found an alternative
                                            alt_found = True
                                            
                                            # Only suggest if the alternative is significantly better
                                            if alt_rate and alt_rate > network_rate + OptimizerConfig.THRESHOLDS['network_difference']:
                                                has_alternative = True
                                                alt_met_threshold = True
                                        
                                        # If no network-specific alternative was found or didn't meet threshold,
                                        # use the best overall option as a fallback
                                        if not has_alternative:
                                            # Get the field options from the field manager
                                            field_options = self.field_manager.get_options(display_field)
                                            
                                            # Find the best option based on overall success rates
                                            best_option = None
                                            best_rate = 0
                                            
                                            # Check each option's overall success rate
                                            for option in field_options:
                                                # Skip the current value
                                                if option.id == field_value:
                                                    continue
                                                
                                                # Look up this option in the overall rates using the same key format
                                                # as used when creating overall_rates (field_name:value_str)
                                                option_key = create_field_value_key(field_name, option.id)
                                                if option_key in overall_rates:
                                                    option_rate = overall_rates[option_key].get('rate', 0)
                                                    
                                                    # Keep track of the best option
                                                    if option_rate > best_rate:
                                                        best_rate = option_rate
                                                        best_option = option
                                            
                                            # If we found a better option and it meets the threshold
                                            if best_option and best_rate > network_rate + OptimizerConfig.THRESHOLDS['network_difference']:
                                                alt_value = best_option.id
                                                alt_name = best_option.name
                                                alt_rate = best_rate
                                                has_alternative = True
                                                alt_found = True
                                                alt_met_threshold = True
                                                
                                                # Add debug to show we're using a fallback
                                                st.write(f"DEBUG: Using fallback alternative for {network.network_name} - {field_name}: {current_name} ({network_rate:.3f}) -> {alt_name} ({alt_rate:.3f})")
                                            else:
                                                # Debug when no fallback is found
                                                st.write(f"DEBUG: No fallback alternative found for {network.network_name} - {field_name}: {current_name} ({network_rate:.3f})")
                                        
                                        # Create explanation based on whether we have a better alternative
                                        if has_alternative:
                                            explanation = f"{network.network_name} shows only {network_rate*100:.1f}% success rate for {current_name}. Consider changing to {alt_name} ({alt_rate*100:.1f}%)."
                                        else:
                                            explanation = f"{network.network_name} shows only {network_rate*100:.1f}% success rate for {current_name} (vs. {overall_rate*100:.1f}% overall)."
                                        
                                        # Create recommendation with suggested alternative if available
                                        recommendation = {
                                            "recommendation_type": rec_type,
                                            "field": field_name,
                                            "current_value": field_value,
                                            "suggested_value": alt_value if has_alternative else None,
                                            "suggested_name": alt_name if has_alternative else f"{network.network_name}: {current_name}",
                                            "impact": impact_score,
                                            "confidence": "medium",
                                            "explanation": explanation,
                                            "is_network_specific": True,
                                            # Add tracking flags to metadata
                                            "sample_size": sample_size,
                                            "metadata": {
                                                "network_id": network.network_id,
                                                "network_name": network.network_name,
                                                "network_rate": network_rate,
                                                "overall_rate": overall_rate,
                                                "difference": difference,
                                                "alternative_value": alt_value if has_alternative else None,
                                                "alternative_name": alt_name if has_alternative else None,
                                                "alternative_rate": alt_rate if has_alternative else None,
                                                "alt_found": alt_found,
                                                "alt_met_threshold": alt_met_threshold
                                            }
                                        }
                                    recommendations.append(recommendation)

            # Sort all recommendations by impact score (descending)            
            recommendations.sort(key=lambda x: abs(x.get('impact', 0)), reverse=True)
            
            # Apply max suggestions limit from config
            max_suggestions = self.config.SUGGESTIONS.get('max_suggestions', 20)
            if len(recommendations) > max_suggestions:
                recommendations = recommendations[:max_suggestions]
            
            # DEBUG: Log recommendation counts
            network_specific = sum(1 for r in recommendations if r.get('is_network_specific', False))
            network_keep = sum(1 for r in recommendations if r.get('is_network_specific', False) and r.get('recommendation_type') == self.REC_TYPE_NETWORK_KEEP)
            network_change = sum(1 for r in recommendations if r.get('is_network_specific', False) and r.get('recommendation_type') == self.REC_TYPE_NETWORK_CHANGE)
            general = len(recommendations) - network_specific
            
            # Count network change recommendations with alternatives and track alternatives found
            network_change_with_alt = 0
            alt_found_count = 0
            alt_threshold_count = 0
            
            # Add a safe debug statement to track alternatives
            st.write(f"DEBUG: Checking if alternatives were found and compared...")
            
            for r in recommendations:
                if r.get('is_network_specific', False) and r.get('recommendation_type') == self.REC_TYPE_NETWORK_CHANGE:
                    has_alt = r.get('suggested_value') is not None
                    if has_alt:
                        network_change_with_alt += 1
                    
                    # Check if metadata contains alternative tracking info
                    if 'metadata' in r:
                        if r['metadata'].get('alt_found', False):
                            alt_found_count += 1
                        if r['metadata'].get('alt_met_threshold', False):
                            alt_threshold_count += 1
            
            st.write(f"DEBUG: Recommendations - Total: {len(recommendations)}, Network: {network_specific} (Keep: {network_keep}, Change: {network_change}, With Alt: {network_change_with_alt}), General: {general}")
            st.write(f"DEBUG: Alternatives - Found: {alt_found_count}, Met threshold: {alt_threshold_count}")

            
            # Return the unified list of tagged recommendations
            return recommendations
            
        except Exception as e:
            # Display specific error but return an empty list
            st.write(f"Error generating recommendations: {str(e)}")
            return []
    
    def _recommend_missing_criteria(self, criteria: CriteriaDict, 
                                   success_factors: List[SuccessFactor],
                                   matching_shows: pd.DataFrame) -> List[RecommendationItem]:
        """Generate recommendations for high-impact criteria that are missing from the concept.
        
        This method processes success factors to create actionable recommendations of types:
        - 'add': For unselected criteria with positive impact
        - 'change': For selected fields but with different options that have positive impact
        - 'remove': For selected criteria with negative impact
        
        Args:
            criteria: Dictionary of criteria key-value pairs from the UI conforming to CriteriaDict
            success_factors: List of SuccessFactor objects with impact scores and recommendation types
            matching_shows: DataFrame of shows matching the current criteria
            
        Returns:
            List of RecommendationItem dictionaries sorted by impact score
        """
        # This method processes success factors to generate actionable recommendations
        # Data flow: success_factors -> filter by impact -> determine recommendation type -> create recommendations
        # The recommendation types follow these rules (as per memory f48dc725):
        # - 'add': For suggesting new unselected fields with positive impact
        # - 'change': For suggesting different values for already selected fields
        # - 'remove': For suggesting removal of selected fields with negative impact
        try:
        
            # Initialize variables for collecting all potential recommendations
            potential_recommendations = []
            min_impact = OptimizerConfig.SUGGESTIONS['minimum_impact']
            
            # First, collect ALL potential recommendations without filtering by recommendation type
            # We'll only filter by the minimum impact threshold to avoid processing negligible impacts
            for factor in success_factors:
                # Skip factors with impact below threshold
                if abs(factor.impact_score) < min_impact:
                    continue
                
                # Get information about the selection status for filtering
                criteria_type = factor.criteria_type
                is_field_selected = criteria_type in criteria
                option_id = factor.criteria_value
                is_option_selected = False
                
                # Check if this specific option is selected
                if is_field_selected:
                    # Handle different criteria field types (array vs. single value)
                    if isinstance(criteria[criteria_type], list):
                        is_option_selected = option_id in criteria[criteria_type]
                    else:
                        is_option_selected = criteria[criteria_type] == option_id
                                
                # Get the recommendation type and impact score
                impact_score = factor.impact_score
                criteria_name = factor.criteria_name
                
                # Use the recommendation type that was already determined in the criteria_scorer
                # This ensures consistency with the recommendation type determination logic
                rec_type = factor.recommendation_type
                
                # Skip factors without a recommendation type
                if rec_type is None:
                    continue
                    
                # Explanation text is handled by the view layer (OptimizerView._generate_explanation_text)
                # We only need to provide the necessary data for the view to generate explanations
                
                # Create recommendation and add to list - ensure full compliance with RecommendationItem contract
                # Tag network-related recommendations as network-specific
                is_network_specific = criteria_type.lower() == 'network'
                
                recommendation = {
                    'recommendation_type': rec_type,
                    'field': criteria_type,
                    'current_value': None,
                    'suggested_value': option_id,
                    'suggested_name': criteria_name,
                    'impact': impact_score,
                    'confidence': factor.confidence,
                    'explanation': factor.explanation if hasattr(factor, 'explanation') and factor.explanation else '',
                    'is_network_specific': is_network_specific
                }
                
                # Add sample size if available
                if hasattr(factor, 'sample_size') and factor.sample_size:
                    recommendation['sample_size'] = factor.sample_size
                    
                potential_recommendations.append(recommendation)
                      
            # Now that we've collected all potential recommendations, sort them by absolute impact (descending)
            potential_recommendations.sort(key=lambda x: abs(x['impact']), reverse=True)
            
            # Use the sorted potential recommendations directly
            # We've already filtered by minimum impact threshold and recommendation type
            recommendations = potential_recommendations
            
            return recommendations
        except Exception as e:
            # Log the error with more context for debugging
            error_message = f"Error generating recommendations for missing criteria: {str(e)}"
            st.error(error_message)
            
            # Return empty list to ensure the UI can still function
            return []
    
    def _identify_limiting_criteria(self, criteria: CriteriaDict, matching_shows: pd.DataFrame, 
                                confidence_info: ConfidenceInfo) -> List[RecommendationItem]:
        """Identify criteria that are limiting match quality and suggest alternatives.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of matching shows
            confidence_info: Dictionary with confidence metrics conforming to ConfidenceInfo
            
        Returns:
            List of RecommendationItem dictionaries with suggestions to improve match quality
        """
        recommendations = []
        
        # Ensure confidence_info conforms to our ConfidenceInfo contract
        # This enforces the contract rather than adding defensive checks
        from .optimizer_data_contracts import update_confidence_info
        confidence_info = update_confidence_info(confidence_info, {})
        
        # Now we can safely extract match_level from confidence_info dictionary
        match_level = confidence_info['match_level']
        
        # Only run this analysis if we're not at match level 1 (perfect match)
        if match_level == 1:
            return []
            
        # For each criterion, try removing it and see if match quality improves
        for criteria_type, criteria_value in criteria.items():
            # Skip non-content criteria or essential criteria that can't be relaxed
            if criteria_type not in OptimizerConfig.CRITERIA_IMPORTANCE:
                continue
                
            importance = OptimizerConfig.CRITERIA_IMPORTANCE.get(criteria_type)
            if importance == 'essential':
                continue
                
            # Create a copy of criteria without this criterion
            test_criteria = {k: v for k, v in criteria.items() if k != criteria_type}
            
            # Get matching shows without this criterion
            # If this fails, the criteria_scorer will return appropriate values or raise an exception
            # that should be handled at a higher level
            test_matches, test_confidence = self.criteria_scorer.matcher.find_matches_with_fallback(
                test_criteria, flexible=True)
                
            # Get the count of test matches
            test_count = len(test_matches) if test_matches is not None else 0
                
            # Ensure test_confidence conforms to our ConfidenceInfo contract
            # This enforces the contract rather than adding defensive checks
            test_confidence = update_confidence_info(test_confidence, {})
            
            # Now we can safely extract match_level from test_confidence dictionary
            test_match_level = test_confidence['match_level']
                
            # If removing this criterion improves match level or significantly increases sample size
            if (test_match_level < match_level or 
                    test_count > len(matching_shows) * 2):  # At least double the sample size
                    
                # Get the name of the criterion
                criteria_name = self._get_criteria_name(criteria_type, criteria_value)
                
                # Calculate the impact score based on the improvement in match quality
                impact_score = 0.2  # Default impact score
                if test_count > 0 and len(matching_shows) > 0:
                    # Impact based on increase in sample size
                    impact_score = min(0.5, (test_count - len(matching_shows)) / len(matching_shows))
                
                # Create explanation text for the recommendation
                explanation_text = f"Removing {criteria_name} would increase the number of matching shows from {len(matching_shows)} to {test_count}, potentially improving match quality."
                
                # Create a RecommendationItem dictionary using the TypedDict contract
                recommendation: RecommendationItem = {
                    'recommendation_type': self.REC_TYPE_REMOVE,
                    'field': criteria_type,  # Renamed from criteria_type to field per TypedDict contract
                    'current_value': criteria_value,
                    'current_name': criteria_name,
                    'suggested_value': None,
                    'suggested_name': '',  # Empty string instead of None for suggested_name
                    'impact': impact_score,  # Renamed from impact_score to impact per TypedDict contract
                    'confidence': "medium",
                    'explanation': explanation_text
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_successful_patterns(self, criteria: CriteriaDict, matching_shows: pd.DataFrame) -> List[RecommendationItem]:
        """Analyze patterns in successful shows and suggest criteria changes.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of matching shows
            
        Returns:
            List of RecommendationItem dictionaries based on successful patterns
        """
        # Skip if we don't have enough data
        if matching_shows is None or matching_shows.empty or len(matching_shows) < self.config.CONFIDENCE.get('minimum_sample', 10):
            return []
            
        # Get success threshold from config
        # Use SUCCESS['threshold'] as the single source of truth for success threshold
        success_threshold = self.config.SUCCESS.get('threshold', 0.6)
            
        # Skip if success_score column is missing
        if 'success_score' not in matching_shows.columns:
            return []
            
        # Get successful shows
        successful_shows = matching_shows[matching_shows['success_score'] >= success_threshold].copy()
        
        # Need at least a few successful shows for analysis
        if len(successful_shows) < 5:
            return []
            
        recommendations = []
        
        # Analyze patterns in successful shows
        for criteria_type in OptimizerConfig.CRITERIA_IMPORTANCE:
            # Skip criteria not present in the dataset
            if criteria_type not in successful_shows.columns:
                continue
                
            # If this criteria is already specified by the user
            if criteria_type in criteria:
                current_value = criteria[criteria_type]
                
                # Use value directly - create_field_value_key will handle hashability when needed
                # No conversion needed here
                
                # Calculate success rate for each value of this criteria
                value_success = {}
                
                # Get a sample of non-null values to check type
                sample_values = successful_shows[criteria_type].dropna().head(5).tolist()
                has_list_values = any(isinstance(val, list) for val in sample_values)
                
                if has_list_values:
                    # For list columns, we need to convert to string for comparison
                    # Get unique string representations
                    unique_str_values = successful_shows[criteria_type].astype(str).unique()
                    
                    for str_val in unique_str_values:
                        if pd.isna(str_val) or str_val == '':
                            continue
                            
                        # Find shows with this string representation
                        shows_with_value = successful_shows[successful_shows[criteria_type].astype(str) == str_val]
                        if shows_with_value.empty:
                            continue
                            
                        # Get the actual value (which might be a list)
                        value = shows_with_value[criteria_type].iloc[0]
                        
                        # Calculate average success
                        avg_success = shows_with_value['success_score'].mean()
                        
                        # Make the value hashable
                        if isinstance(value, list):
                            hashable_value = tuple(value)
                        else:
                            hashable_value = value
                            
                        # Use hashable value as dictionary key
                        value_success[hashable_value] = avg_success
                else:
                    # For non-list columns, use normal unique() method
                    for value in successful_shows[criteria_type].unique():
                        if pd.isna(value) or value == '':
                            continue
                            
                        # Calculate average success score for shows with this value
                        shows_with_value = successful_shows[successful_shows[criteria_type] == value]
                        avg_success = shows_with_value['success_score'].mean()
                        
                        # Make the value hashable if needed
                        if isinstance(value, list):
                            hashable_value = tuple(value)
                        else:
                            hashable_value = value
                            
                        # Use hashable value as dictionary key
                        value_success[hashable_value] = avg_success
                        
                    # Sort values by success score
                    sorted_values = sorted(value_success.items(), key=lambda x: x[1], reverse=True)
                    
                    # If current value is not in top 3 most successful values, suggest alternatives
                    top_values = [v[0] for v in sorted_values[:3]]
                    if current_value not in top_values and len(top_values) > 0:
                        # Get the top value
                        top_value = top_values[0]
                        top_success = value_success[top_value]
                        
                        # Get current success rate
                        current_success = value_success.get(current_value, 0)
                        
                        # Calculate impact score
                        impact_score = (top_success - current_success)
                        
                        # Skip low-impact recommendations
                        min_impact = self.config.SUGGESTIONS.get('minimum_impact', 0.05)
                        if impact_score < min_impact:
                            continue
                            
                        # Get criteria names for reference
                        current_name = self._get_criteria_name(criteria_type, current_value)
                        suggested_name = self._get_criteria_name(criteria_type, top_value)
                        
                        # Determine confidence level based on sample size
                        sample_size = len(shows_with_value)
                        confidence = self.config.get_confidence_level(sample_size)
                        
                        # Create explanation text for the recommendation
                        explanation_text = f"Changing from {current_name} to {suggested_name} could improve success probability by {impact_score*100:.1f}%. Based on analysis of {sample_size} successful shows."
                        
                        # Create a RecommendationItem dictionary using the TypedDict contract
                        recommendation: RecommendationItem = {
                            'recommendation_type': self.REC_TYPE_CHANGE,
                            'field': criteria_type,  # Renamed from criteria_type to field per TypedDict contract
                            'current_value': current_value,
                            'current_name': current_name,
                            'suggested_value': top_value,
                            'suggested_name': suggested_name,
                            'impact': impact_score,  # Renamed from impact_score to impact per TypedDict contract
                            'confidence': confidence,
                            'explanation': explanation_text
                        }
                        recommendations.append(recommendation)
            
            return recommendations
      
    def _should_process_combination(self, combination_type, *args):
        """Check if a combination should be processed or skipped.
        
        Args:
            combination_type: Type of recommendation/processing (e.g., "network_specific")
            *args: Components that make up the unique combination key
            
        Returns:
            Boolean indicating if processing should proceed (True) or skip (False)
        """
        # Initialize set for this combination type if not exists
        if combination_type not in self._processed_combinations:
            self._processed_combinations[combination_type] = set()
            
        # Create a unique key from the args
        key = ":".join(str(arg) for arg in args)
        
        # If already processed, return False to skip
        if key in self._processed_combinations[combination_type]:
            return False
            
        # Mark as processed and return True to proceed
        self._processed_combinations[combination_type].add(key)
        return True
        
    def _reset_processed_combinations(self, combination_type=None):
        """Reset the processed combinations cache.
        
        Args:
            combination_type: Optional specific type to reset, or None for all
        """
        if combination_type:
            if combination_type in self._processed_combinations:
                self._processed_combinations[combination_type] = set()
        else:
            self._processed_combinations = {}
            
    def _get_criteria_name(self, criteria_type, value):
        """Get the reference name for a criteria value.
        
        This method provides a consistent way to get a human-readable reference name
        for criteria values. The OptimizerView will handle the actual display formatting.
        
        Args:
            criteria_type: Type of criteria (e.g., 'genre', 'source_type')
            value: Value of the criteria
            
        Returns:
            Reference name for the criteria value
        """
        if value is None:
            return "None"
        if isinstance(value, (list, np.ndarray)):
            names = [self._get_criteria_name(criteria_type, val) for val in value]
            return ", ".join(names)
        options = self.field_manager.get_options(criteria_type)
        for option in options:
            if option.id == value:
                return option.name
        return str(value)