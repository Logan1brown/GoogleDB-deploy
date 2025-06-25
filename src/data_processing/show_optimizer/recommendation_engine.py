"""RecommendationEngine: Handles success factor identification and recommendation generation.

This component is responsible for analyzing show data to identify success factors
and generate recommendations for show concept optimization.
"""

import pandas as pd
import streamlit as st
import numpy as np
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Set, Union

from .optimizer_config import OptimizerConfig
from .optimizer_data_contracts import (
    CriteriaDict, ConfidenceInfo, IntegratedData,
    NetworkMatch, RecommendationItem, FieldValueData, FieldValueSuccessRate
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

@dataclass
# DEPRECATED: This class is deprecated in favor of the RecommendationItem TypedDict
# defined in optimizer_data_contracts.py
class Recommendation:
    """A recommendation for optimizing a show concept.
    
    DEPRECATED: Use RecommendationItem TypedDict from optimizer_data_contracts.py instead.
    """
    recommendation_type: str  # add, remove, replace, consider, relax, change
    criteria_type: str        # e.g., "genre", "character_types" (renamed to 'field' in TypedDict)
    current_value: Any        # Current value (if any)
    suggested_value: Any      # Suggested value
    suggested_name: str       # Reference name for suggested value
    impact_score: float       # Expected impact on success (-1 to 1) (renamed to 'impact' in TypedDict)
    confidence: str           # none, low, medium, high
    explanation: str          # Raw data for OptimizerView to format
    current_name: str = ""    # Reference name for current value
    metadata: Dict = field(default_factory=dict)  # Additional data for OptimizerView formatting


class RecommendationEngine:
    # Recommendation type constants for standardization across the class
    REC_TYPE_ADD = 'add'
    REC_TYPE_CHANGE = 'change'
    REC_TYPE_REMOVE = 'remove'
    
    # Network-specific recommendation types
    REC_TYPE_NETWORK_KEEP = 'network_keep'
    REC_TYPE_NETWORK_CHANGE = 'network_change'
    """Analyzes show data to identify success factors and generate recommendations."""
    

    
    def _format_key(self, field_name, value_name):
        """Standardize key formatting for consistent matching between network and overall rates.
        
        Args:
            field_name: Field name part of the key
            value_name: Value name part of the key
            
        Returns:
            Tuple of (standard_key, original_key, clean_value_name)
        """
        # Convert value_name to string
        value_name = str(value_name)
        
        # Clean up value_name if it contains "Unknown" with a number in parentheses
        clean_value_name = value_name
        if "Unknown" in value_name and "(" in value_name and ")" in value_name:
            # Extract the value inside parentheses
            start_idx = value_name.find("(") + 1
            end_idx = value_name.find(")")
            if start_idx > 0 and end_idx > start_idx:
                clean_value_name = value_name[start_idx:end_idx].strip()
        
        # Create keys for this field-value combination
        standard_key = f"{field_name}:{clean_value_name}"
        original_key = f"{field_name}:{value_name}"
        
        return standard_key, original_key, clean_value_name
        
    def _make_hashable(self, value):
        """Convert a value to a hashable type for dictionary keys.
        
        Args:
            value: Any value that needs to be made hashable
            
        Returns:
            A hashable version of the value
        """
        if value is None:
            return None
        if isinstance(value, (list, np.ndarray)):
            return tuple(value)
        if isinstance(value, dict):
            return str(value)
        if not isinstance(value, (str, int, float, bool, tuple)) or pd.isna(value):
            return str(value)
        return value
    
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
        self.criteria_scorer = criteria_scorer
        self.config = OptimizerConfig
        
        # Try to get criteria_scorer from success_analyzer if not provided
        if self.criteria_scorer is None and hasattr(success_analyzer, 'criteria_scorer'):
            self.criteria_scorer = success_analyzer.criteria_scorer
            
        if self.criteria_scorer is None:
            st.error("Some recommendation features may be limited due to missing components.")
    
    def calculate_overall_success_rate(self, criteria: CriteriaDict) -> Tuple[float, str]:
        """Calculate the overall success rate for the given criteria.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            
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
            # Get matching shows and count
            matching_shows, match_count, confidence_info = self.criteria_scorer._get_matching_shows(criteria)
            
            # Return None if no matches found
            if matching_shows.empty or match_count == 0:
                return None, 'none'
            
            # Calculate all scores including success rate
            all_scores = self.criteria_scorer.calculate_scores({}, matching_shows)
            success_rate = all_scores.get('success_rate')
            
            # Determine confidence level
            confidence = confidence_info.get('confidence', 'none')
            
            return success_rate, confidence
        except Exception as e:
            # Keep a single top-level try-except for critical errors
            st.error(f"Error calculating success rate: {str(e)}")
            # Log the error but don't stop execution

            return None, 'none'
    
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
        if matching_shows is None or matching_shows.empty:
            try:
                # Use the matcher directly instead of going through criteria_scorer
                if hasattr(self.criteria_scorer, 'matcher') and self.criteria_scorer.matcher is not None:
                    matching_shows, confidence_info = self.criteria_scorer.matcher.find_matches_with_fallback(criteria)
                    if matching_shows.empty:
                        st.error("No shows match your criteria. Try adjusting your parameters.")
                        return []
                else:
                    st.error("No matcher available. Cannot find matching shows.")
                    return []
            except Exception as e:
                st.error(f"Unable to analyze shows matching your criteria: {str(e)}")
                return []
        
        try:
            # Calculate impact data using the criteria scorer
            impact_data = self.criteria_scorer.calculate_criteria_impact(criteria, matching_shows)
            
            # Return empty list if no impact data was found
            if not impact_data or all(len(values) == 0 for field, values in impact_data.items()):
                return []
            
            # Convert to SuccessFactor objects
            success_factors = []
            
            for criteria_type, values in impact_data.items():
                processed_count = 0
                
                # Sort options by absolute impact score to prioritize highest impact options
                sorted_options = sorted(values.items(), key=lambda x: abs(x[1].get('impact', 0)), reverse=True)
                
                for value_id, impact_info in sorted_options:
                    if processed_count >= limit:
                        break
                    
                    # Extract impact score and sample size
                    impact = impact_info.get('impact', self.config.DEFAULT_VALUES['impact_score'])
                    sample_size = impact_info.get('sample_size', self.config.DEFAULT_VALUES['fallback_sample_size'])
                    
                    # Make criteria value hashable
                    criteria_value = self._make_hashable(value_id)
                    
                    # Get the proper display name
                    if value_id == 'remove':
                        name = f"Remove {criteria_type}"
                    else:
                        name = self._get_criteria_name(criteria_type, value_id)
                    
                    # Determine confidence level
                    confidence = self.config.get_confidence_level(sample_size)
                    
                    # Get recommendation type from impact data
                    recommendation_type = impact_info.get('recommendation_type')
                    
                    # If no recommendation_type is specified, determine based on impact
                    if not recommendation_type:
                        recommendation_type = self.REC_TYPE_ADD if impact > 0 else self.REC_TYPE_REMOVE
                    
                    # Get matching titles for this criteria
                    matching_titles = []
                    try:
                        # Convert hashable value back to original form if needed
                        match_value = list(criteria_value) if isinstance(criteria_value, tuple) else criteria_value
                        
                        # Get shows matching just this single criteria
                        single_criteria = {criteria_type: match_value}
                        single_matches, _, _ = self.criteria_scorer._get_matching_shows(single_criteria)
                        
                        if not single_matches.empty and 'title' in single_matches.columns:
                            matching_titles = single_matches['title'].tolist()[:100]  # Limit to 100 titles
                    except Exception as e:
                        st.error(f"Error getting matching titles: {str(e)}")
                    
                    # Create and add the success factor
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
            
            return success_factors
            
        except Exception as e:
            st.error(f"Error identifying success factors: {str(e)}")
            return []
    
    def generate_recommendations(self, criteria: CriteriaDict,
                                success_factors: List[SuccessFactor],
                                top_networks: List[NetworkMatch],
                                matching_shows: pd.DataFrame,
                                confidence_info: ConfidenceInfo,
                                integrated_data: IntegratedData) -> List[RecommendationItem]:
        """Generate recommendations based on criteria analysis.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            success_factors: List of identified success factors
            top_networks: List of NetworkMatch dictionaries
            matching_shows: DataFrame of matching shows
            confidence_info: Dictionary with confidence metrics conforming to ConfidenceInfo
            integrated_data: Dictionary of integrated data frames conforming to IntegratedData
            
        Returns:
            List of RecommendationItem dictionaries with standardized structure
        """
        try:
            # Initialize empty list for recommendations
            recommendations = []
            
            # Analyze missing high-impact criteria
            try:
                missing_criteria_recs = self._recommend_missing_criteria(criteria, success_factors, matching_shows)
                recommendations.extend(missing_criteria_recs)
            except Exception as e:
                st.error(f"Unable to analyze some criteria. Error: {str(e)}")
            
            # Identify limiting criteria that restrict match quality
            if confidence_info and confidence_info.get('match_level', 1) > 1:
                try:
                    limiting_criteria_recs = self._identify_limiting_criteria(criteria, matching_shows, confidence_info)
                    recommendations.extend(limiting_criteria_recs)
                except Exception as e:
                    st.error("Unable to analyze criteria limitations. Some recommendations may be missing.")
            
            # Analyze successful patterns in the matched shows
            if not matching_shows.empty:
                try:
                    pattern_recs = self._analyze_successful_patterns(criteria, matching_shows)
                    recommendations.extend(pattern_recs)
                except Exception as e:
                    st.error("Unable to analyze successful patterns. Some recommendations may be missing.")
                
            # Generate network-specific recommendations
            if top_networks:
                # Limit to top 3 networks for performance
                network_specific_recs = []
                for network in top_networks[:3]:
                    try:
                        network_recs = self.generate_network_specific_recommendations(
                            criteria, network, matching_shows, integrated_data
                        )
                        network_specific_recs.extend(network_recs)
                    except Exception as e:
                        st.error(f"Error generating network recommendations: {str(e)}")
                
                # Add network-specific recommendations to the main list
                recommendations.extend(network_specific_recs)
            
            # Filter valid recommendations and sort by impact score
            valid_recommendations = [rec for rec in recommendations if 'impact' in rec]
            valid_recommendations.sort(key=lambda x: abs(x['impact']), reverse=True)
            
            # Limit to max suggestions
            max_suggestions = self.config.SUGGESTIONS.get('max_suggestions', 5)
            if len(valid_recommendations) > max_suggestions:
                valid_recommendations = valid_recommendations[:max_suggestions]
                
            return valid_recommendations
            
        except Exception as e:
            st.error(f"Unable to generate recommendations based on your criteria: {str(e)}")
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
            List of Recommendation objects sorted by impact score
    """
        # This method processes success factors to generate actionable recommendations
        # Data flow: success_factors -> filter by impact -> determine recommendation type -> create recommendations
        # The recommendation types follow these rules (as per memory f48dc725):
        # - 'add': For suggesting new unselected fields with positive impact
        # - 'change': For suggesting different values for already selected fields
        # - 'remove': For suggesting removal of selected fields with negative impact
        try:
            recommendations = []

            # Process all success factors to create recommendations
            # Get minimum impact threshold from config - this filters out insignificant factors
            # Direct access instead of using .get() with fallback - enforcing the config contract
            min_impact = OptimizerConfig.SUGGESTIONS['minimum_impact']
              
            for factor in success_factors:
                # Skip factors with impact below threshold
                # This is a business rule, not defensive programming
                if abs(factor.impact_score) < min_impact:
                    continue
                
                # Get information about the selection status for filtering
                is_field_selected = factor.criteria_type in criteria
                option_id = factor.criteria_value
                is_option_selected = False
                
                # Check if this specific option is selected
                if is_field_selected:
                    # Handle different criteria field types (array vs. single value)
                    # This explicit type check is necessary as criteria fields can be either:
                    # 1. Lists (e.g., character_types, themes) - check if option is in the list
                    # 2. Single values (e.g., genre, format) - check if option matches exactly
                    # Access the criteria value directly since we've already verified it exists
                    # This removes defensive programming (criteria.get) while maintaining the data contract
                    if isinstance(criteria[factor.criteria_type], list):
                        is_option_selected = option_id in criteria[factor.criteria_type]
                    else:
                        is_option_selected = criteria[factor.criteria_type] == option_id
                
                # Start with the default recommendation type from the success factor
                rec_type = factor.recommendation_type
                
                # Use class-level recommendation type constants for consistency
                # This follows the recommendation from the data flow analysis to standardize types
                
                # Determine recommendation type based on selection status and impact score
                # This ensures consistent recommendation types across the application
                # TODO: Move recommendation type definitions to OptimizerConfig for standardization
                if is_option_selected and factor.impact_score < 0:
                    # Selected option with negative impact should be a 'remove' recommendation
                    rec_type = self.REC_TYPE_REMOVE
                elif not is_field_selected and factor.impact_score > 0:
                    # Unselected field with positive impact should be an 'add' recommendation
                    rec_type = self.REC_TYPE_ADD
                elif is_field_selected and not is_option_selected and factor.impact_score > 0:
                    # Selected field but different option with positive impact should be a 'change' recommendation
                    rec_type = self.REC_TYPE_CHANGE
                elif factor.recommendation_type == self.REC_TYPE_ADD and factor.impact_score < 0:
                    # Don't recommend adding something with negative impact
                    continue
                elif factor.recommendation_type == self.REC_TYPE_REMOVE and not is_field_selected:
                    # Can't remove what's not selected
                    continue
                
                # Update the recommendation type in the factor object for consistency
                factor.recommendation_type = rec_type
                
                # Apply minimum impact thresholds based on recommendation type
                # This ensures recommendations have meaningful impact values for the UI
                # Note: This is business logic that should remain in the recommendation engine
                impact_score = factor.impact_score
                
                # Get threshold values from OptimizerConfig for consistency
                # These thresholds determine the minimum impact required for a recommendation to be visible
                # Direct access enforces the config contract - these values must exist in OptimizerConfig
                min_remove_impact = OptimizerConfig.SUGGESTIONS['min_remove_impact']
                min_other_impact = OptimizerConfig.SUGGESTIONS['min_other_impact']
                
                # Apply minimum thresholds based on recommendation type using constants
                if rec_type == REC_TYPE_REMOVE and abs(impact_score) < min_remove_impact:
                    impact_score = -min_remove_impact
                elif abs(impact_score) < min_other_impact:
                    impact_score = min_other_impact if impact_score > 0 else -min_other_impact
                
                # Recommendation type and impact score finalized
                    
                # Raw data is provided to OptimizerView for formatting
                
                # Create explanation text based on the recommendation type
                explanation_text = ""
                if rec_type == self.REC_TYPE_ADD:
                    explanation_text = f"Adding {factor.criteria_name} could improve success probability by {abs(impact_score)*100:.1f}%."
                elif rec_type == self.REC_TYPE_REMOVE:
                    explanation_text = f"Removing {factor.criteria_name} could improve success probability by {abs(impact_score)*100:.1f}%."
                elif rec_type == self.REC_TYPE_CHANGE:
                    explanation_text = f"Changing to {factor.criteria_name} could improve success probability by {abs(impact_score)*100:.1f}%."
                else:
                    explanation_text = f"Consider {factor.criteria_name} for potential impact of {abs(impact_score)*100:.1f}%."
                
                # Create a RecommendationItem dictionary using the TypedDict contract
                recommendation: RecommendationItem = {
                    'recommendation_type': rec_type,
                    'field': factor.criteria_type,  # Renamed from criteria_type to field per TypedDict contract
                    'current_value': None,
                    'suggested_value': factor.criteria_value,
                    'suggested_name': factor.criteria_name,
                    'impact': impact_score,  # Renamed from impact_score to impact per TypedDict contract
                    'confidence': factor.confidence,
                    'explanation': explanation_text
                }
                
                # Add to recommendations list
                recommendations.append(recommendation)
                 
                # Recommendation processing complete
            return recommendations
        except Exception as e:
            # Log the error with more context for debugging
            error_message = f"Error generating recommendations for missing criteria: {str(e)}"
            st.error(error_message)
            
            # Return empty list to ensure the UI can still function
            return []
    
    def _identify_limiting_criteria(self, criteria: CriteriaDict, matching_shows: pd.DataFrame, 
                                confidence_info: ConfidenceInfo) -> List[Recommendation]:
        """Identify criteria that are limiting match quality and suggest alternatives.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of matching shows
            confidence_info: Dictionary with confidence metrics conforming to ConfidenceInfo
            
        Returns:
            List of Recommendation objects with suggestions to improve match quality
        """
        recommendations = []
        match_level = confidence_info.get('match_level', 1)
        
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
            test_matches, test_count, test_confidence = self.criteria_scorer._get_matching_shows(
                test_criteria, flexible=True)
                
            # If removing this criterion improves match level or significantly increases sample size
            if (test_confidence.get('match_level', match_level) < match_level or 
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
    
    def _analyze_successful_patterns(self, criteria: CriteriaDict, matching_shows: pd.DataFrame) -> List[Recommendation]:
        """Analyze patterns in successful shows and suggest criteria changes.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of matching shows
            
        Returns:
            List of Recommendation objects based on successful patterns
        """
        # Skip if we don't have enough data
        if matching_shows is None or matching_shows.empty or len(matching_shows) < self.config.CONFIDENCE.get('minimum_sample', 10):
            return []
            
        # Get success threshold from config
        success_threshold = self.config.PERFORMANCE.get('success_threshold', 0.7)
            
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
                
                # Make current_value hashable
                current_value = self._make_hashable(current_value)
                
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
                        hashable_value = self._make_hashable(value)
                        value_success[hashable_value] = avg_success
                else:
                    # For non-list columns, use normal unique() method
                    for value in successful_shows[criteria_type].unique():
                        if pd.isna(value) or value == '':
                            continue
                            
                        # Calculate average success score for shows with this value
                        shows_with_value = successful_shows[successful_shows[criteria_type] == value]
                        avg_success = shows_with_value['success_score'].mean()
                        
                        # Make the value hashable
                        hashable_value = self._make_hashable(value)
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

    def generate_network_specific_recommendations(self, criteria: CriteriaDict, 
                                                 network: NetworkMatch,
                                                 matching_shows: pd.DataFrame,
                                                 integrated_data: IntegratedData) -> List[RecommendationItem]:
        """
        Generate network-specific recommendations.        
        Args:
            criteria: Dictionary of criteria values
            network: NetworkMatch object with network information
            matching_shows: DataFrame of shows matching the criteria
            integrated_data: Integrated data for additional context
            
        Returns:
            List of network-specific recommendations
        """
        # Debug logging for network information
        if OptimizerConfig.DEBUG_MODE:
            st.write(f"DEBUG: Processing network {network.network_name} (ID: {network.network_id})")

        
        # Get network-specific success rates
        try:
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Getting network-specific success rates for network {network.network_name}")
                st.write(f"DEBUG: Network object type: {type(network).__name__}")
                st.write(f"DEBUG: Network ID: {network.network_id}")
            
            # Check if network_analyzer has the method we're trying to call
            if OptimizerConfig.DEBUG_MODE and not hasattr(self.network_analyzer, 'get_network_specific_success_rates'):
                st.write("DEBUG: ERROR - network_analyzer does not have get_network_specific_success_rates method")
            
            network_rates = self.network_analyzer.get_network_specific_success_rates(
                matching_shows=matching_shows,
                network_id=network.network_id
            )
        except Exception as e:
            st.error(f"Error getting network-specific success rates: {str(e)}")
            return []
        
        # Calculate overall success rates for comparison with network-specific rates
        overall_rates = {}
        
        if OptimizerConfig.DEBUG_MODE:
            st.write(f"DEBUG: Network rates keys: {list(network_rates.keys())}")
            st.write(f"DEBUG: Criteria keys: {list(criteria.keys())}")
        
        # Process each key in network rates to calculate corresponding overall rates
        for key, network_rate_data in network_rates.items():
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Processing key: {key}")
                st.write(f"DEBUG: Network rate data type: {type(network_rate_data)}")
                if isinstance(network_rate_data, dict):
                    st.write(f"DEBUG: Network rate data keys: {list(network_rate_data.keys())}")
                    if 'success_rate' in network_rate_data:
                        st.write(f"DEBUG: Network rate success_rate: {network_rate_data['success_rate']}")
                    if 'sample_size' in network_rate_data:
                        st.write(f"DEBUG: Network rate sample_size: {network_rate_data['sample_size']}")
                else:
                    st.write(f"DEBUG: Network rate data is not a dict: {type(network_rate_data)}")
                    st.write(f"DEBUG: Network rate data: {network_rate_data}")
                    # This is likely the source of the error
            
            # Extract field name from key using standard format
            field_name = key.split(':', 1)[0] if ':' in key else key
            
            if OptimizerConfig.DEBUG_MODE:
                import streamlit as st
                st.write(f"DEBUG: Extracted field_name: {field_name}")
            
            # Skip if this field is not in our criteria
            if field_name not in criteria:
                if OptimizerConfig.DEBUG_MODE:
                    import streamlit as st
                    st.write(f"DEBUG: Field {field_name} not in criteria, skipping")
                continue
                
            if OptimizerConfig.DEBUG_MODE:
                import streamlit as st
                st.write(f"DEBUG: Creating single_criteria for {field_name}")
                st.write(f"DEBUG: criteria[{field_name}] = {criteria[field_name]}")
            
            # Calculate the overall success rate for this criteria
            single_criteria = {field_name: criteria[field_name]}
            
            # Get matching shows for this single criterion
            if hasattr(self.criteria_scorer, 'matcher') and self.criteria_scorer.matcher is not None:
                single_matches, _ = self.criteria_scorer.matcher.find_matches_with_fallback(single_criteria)
                
                # Calculate all scores including success rate
                all_scores = self.criteria_scorer.calculate_scores(single_criteria, single_matches)
                overall_rate = all_scores.get('success_rate')
                overall_details = all_scores.get('success_info', {})
            
                # Store the overall rate using both key formats for flexible lookup
                overall_rates[key] = overall_rate
                overall_rates[field_name] = overall_rate
        
        recommendations = []
                 
        # First, explicitly determine which fields from network_rates are valid in our criteria
        # This avoids checking field_name in criteria for each iteration
        valid_fields = set(criteria.keys())
        valid_network_rates = {}
        
        if OptimizerConfig.DEBUG_MODE:
            import streamlit as st
            st.write(f"DEBUG: Valid criteria fields: {valid_fields}")
        
        for key, network_rate_data in network_rates.items():
            if OptimizerConfig.DEBUG_MODE:
                import streamlit as st
                st.write(f"DEBUG: Processing network rate key: {key}")
                st.write(f"DEBUG: Network rate data type: {type(network_rate_data)}")
            
            # Extract field name from the key using standard format
            field_name = key.split(':', 1)[0] if ':' in key else key
            
            if OptimizerConfig.DEBUG_MODE:
                import streamlit as st
                st.write(f"DEBUG: Extracted field_name: {field_name}")
            
            # Only process keys that correspond to fields in our criteria
            if field_name in valid_fields:
                if OptimizerConfig.DEBUG_MODE:
                    import streamlit as st
                    st.write(f"DEBUG: Field {field_name} is valid, creating entry in valid_network_rates")
                    st.write(f"DEBUG: criteria[{field_name}] = {criteria[field_name]}")
                
                valid_network_rates[key] = {
                    'field_name': field_name,
                    'network_rate_data': network_rate_data,
                    'current_value': criteria[field_name],
                    'current_name': self._get_criteria_name(field_name, criteria[field_name])
                }
        
        # Now process only the valid network rates
        if OptimizerConfig.DEBUG_MODE:
            import streamlit as st
            st.write(f"DEBUG: Processing {len(valid_network_rates)} valid network rates")
            st.write(f"DEBUG: Valid network rates keys: {list(valid_network_rates.keys())}")
        
        for key, data in valid_network_rates.items():
            if OptimizerConfig.DEBUG_MODE:
                import streamlit as st
                st.write(f"DEBUG: Processing valid network rate key: {key}")
                st.write(f"DEBUG: Data: {data}")
            
            field_name = data['field_name']
            network_rate_data = data['network_rate_data']
            current_value = data['current_value']
            current_name = data['current_name']
            
            if OptimizerConfig.DEBUG_MODE:
                import streamlit as st
                st.write(f"DEBUG: Extracted field_name: {field_name}")
                st.write(f"DEBUG: Network rate data type: {type(network_rate_data)}")
                st.write(f"DEBUG: Current value: {current_value}")
                st.write(f"DEBUG: Current name: {current_name}")
            
            # Get the overall success rate using flexible key lookup
            overall_rate = overall_rates.get(key, overall_rates.get(field_name))
            if OptimizerConfig.DEBUG_MODE:
                import streamlit as st
                st.write(f"DEBUG: Overall rate for {key}/{field_name}: {overall_rate}")
            
            if overall_rate is None:
                if OptimizerConfig.DEBUG_MODE:
                    import streamlit as st
                    st.write(f"DEBUG: No overall rate for {key}/{field_name}, skipping")
                # Skip criteria without overall rates
                continue
            
            if OptimizerConfig.DEBUG_MODE:
                import streamlit as st
                st.write(f"DEBUG: Network rate data keys: {list(network_rate_data.keys()) if isinstance(network_rate_data, dict) else 'Not a dict'}")
                
            # Get network success rate and sample size
            network_rate = network_rate_data['success_rate']
            sample_size = network_rate_data['sample_size']
            
            # Calculate the difference between network and overall rates
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
                
                # Store raw data for OptimizerView to format
                network_rate_value = network_rate
                overall_rate_value = overall_rate
                
                # Create network-specific reference name with direct attribute access
                network_name = network.network_name
                suggested_name = f"{network_name}: {current_name}"
                
                # Calculate impact score with minimum threshold to ensure visibility
                impact_score = max(abs(difference), 0.05) * (1 if difference > 0 else -1)
                
                # Store network data for OptimizerView to use when formatting
                network_data = {
                    "network_id": network.network_id,
                    "network_name": network.network_name,
                    "network_rate": network_rate,
                    "overall_rate": overall_rate,
                    "difference": difference
                }
                
                # Use standardized network recommendation type constants
                network_rec_type = self.REC_TYPE_NETWORK_KEEP if difference > 0 else self.REC_TYPE_NETWORK_CHANGE
                
                # Create explanation text based on the recommendation type
                explanation_text = ""
                if network_rec_type == self.REC_TYPE_NETWORK_KEEP:
                    explanation_text = f"Keep {current_name} for {network_name}. This element performs {abs(difference)*100:.1f}% better on {network_name} than average."
                else:
                    explanation_text = f"Consider changing {current_name} for {network_name}. This element performs {abs(difference)*100:.1f}% worse on {network_name} than average."
                
                # Create a RecommendationItem dictionary using the TypedDict contract
                recommendation: RecommendationItem = {
                    'recommendation_type': network_rec_type,
                    'field': field_name,  # Renamed from criteria_type to field per TypedDict contract
                    'current_value': current_value,
                    'current_name': current_name,
                    'suggested_value': current_value,
                    'suggested_name': suggested_name,
                    'impact': impact_score,  # Renamed from impact_score to impact per TypedDict contract
                    'confidence': network_rate_data.get('confidence', 'medium'),
                    'explanation': explanation_text,
                    'metadata': network_data
                }
                recommendations.append(recommendation)
        
        return recommendations
