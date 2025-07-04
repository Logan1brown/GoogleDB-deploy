"""RecommendationEngine: Handles success factor identification and recommendation generation.

This component is responsible for analyzing show data to identify success factors
and generate recommendations for show concept optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Set, Union

from .optimizer_config import OptimizerConfig
from .criteria_scorer import ImpactAnalysisResult, ImpactAnalysisSummary
from .optimizer_data_contracts import (
    CriteriaDict, ConfidenceInfo, IntegratedData,
    NetworkMatch, RecommendationItem, FieldValueData, FieldValueSuccessRate,
    update_confidence_info
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
    

    
    def _parse_key(self, key):
        """Extract field name and value from a formatted key.
        
        Args:
            key: A key in the format 'field_name:value'
            
        Returns:
            Tuple of (field_name, field_value)
        """
        field_name = key.split(':', 1)[0] if ':' in key else key
        field_value = key.split(':', 1)[1] if ':' in key else None
        return field_name, field_value
        
    def _format_key(self, field_name, value_name):
        """Standardize key formatting for consistent matching.
        
        Args:
            field_name: Field name part of the key
            value_name: Value name part of the key
            
        Returns:
            Formatted key string
        """
        return f"{field_name}:{str(value_name)}"
        
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
        
        # Initialize network_analyzer directly if not available from success_analyzer
        self.network_analyzer = None
        if hasattr(success_analyzer, 'network_analyzer'):
            self.network_analyzer = success_analyzer.network_analyzer
        elif self.criteria_scorer is not None:
            # Import here to avoid circular imports
            from .network_analyzer import NetworkAnalyzer
            self.network_analyzer = NetworkAnalyzer(self.criteria_scorer, self.field_manager)
        
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
            # Get matching shows using the matcher directly
            matching_shows, confidence_info = self.criteria_scorer.matcher.find_matches_with_fallback(criteria)
            
            # Get match count from confidence_info
            match_count = confidence_info.get('match_count', 0)
            
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
            # Pass integrated_data to ensure matcher has access to full dataset
            impact_result = self.criteria_scorer.calculate_criteria_impact(criteria, matching_shows, integrated_data=integrated_data)
            
            # Check for errors
            if impact_result.error:

                return []
                
            # Get the impact data from the result
            impact_data = impact_result.criteria_impacts
            

        except Exception as e:
            error_msg = f"Error calculating impact data: {str(e)}"
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
                    # Convert hashable value back to original form if needed
                    match_value = list(criteria_value) if isinstance(criteria_value, tuple) else criteria_value
                    
                    # Get shows matching just this single criteria
                    single_criteria = {criteria_type: match_value}
                    single_matches, single_confidence = self.criteria_scorer.matcher.find_matches_with_fallback(single_criteria)
                    
                    if not single_matches.empty and 'title' in single_matches.columns:
                        matching_titles = single_matches['title'].tolist()[:100]  # Limit to 100 titles
                except Exception as e:
                    pass
                
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
                    pass
        
        return success_factors
            
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
        recommendations = []
        
        try:
            # Process each success factor to create recommendations
            missing_criteria_recs = []
            
            # Add recommendations from success factors
            for factor in success_factors:
                # Skip factors without a recommendation type
                if not factor.recommendation_type:
                    continue
                    
                # Create the recommendation
                recommendation = {
                    'recommendation_type': factor.recommendation_type,
                    'field': factor.criteria_type,
                    'current_value': criteria.get(factor.criteria_type),
                    'suggested_value': factor.criteria_value,
                    'suggested_name': factor.criteria_name,
                    'impact': factor.impact_score,
                    'confidence': factor.confidence,
                    'sample_size': factor.sample_size,
                    'matching_titles': factor.matching_titles[:5] if factor.matching_titles else []
                }
                
                missing_criteria_recs.append(recommendation)
                
            recommendations.extend(missing_criteria_recs)
                          
        except Exception as e:
            error_msg = f"Error in _recommend_missing_criteria: {str(e)}"
        
        # Process confidence info if available
        if 'confidence_info' in locals() and confidence_info:
            try:
                # update_confidence_info is already imported at the top of the file
                confidence_info = update_confidence_info(confidence_info, {})
                match_level = confidence_info.get('match_level', 1)
                
                # Analyze limiting criteria if match level is not perfect
                if match_level > 1:
                    try:
                        limiting_criteria_recs = self._identify_limiting_criteria(
                            criteria, matching_shows, confidence_info
                        )
                        recommendations.extend(limiting_criteria_recs)
                    except Exception as e:
                        pass
            except Exception as e:
                pass
            
            # Analyze successful patterns in the matched shows if we have data
            if not matching_shows.empty:
                try:
                    pattern_recs = self._analyze_successful_patterns(criteria, matching_shows)
                    recommendations.extend(pattern_recs)

                except Exception as e:
                    error_msg = f"Error in _analyze_successful_patterns: {str(e)}"
            
            # Generate network-specific recommendations if networks are provided
            if top_networks and len(top_networks) > 0:

                
                # Limit to top 3 networks for performance
                for network in top_networks[:3]:
                    try:
                        network_recs = self.generate_network_specific_recommendations(
                            criteria, network, matching_shows, integrated_data, confidence_info
                        )                      
                            
                        # Directly add to network_specific_recommendations
                        # This ensures they're properly categorized as network-specific
                        network_specific_recommendations.extend(network_recs)
                    except Exception as e:
                        error_msg = f"Error generating network recommendations: {str(e)}"
            
    def generate_recommendations(self, criteria: CriteriaDict, matching_shows: pd.DataFrame = None, 
                                integrated_data: IntegratedData = None, 
                                top_networks: List[NetworkMatch] = None,
                                confidence_info: Optional[ConfidenceInfo] = None) -> Dict[str, List[RecommendationItem]]:
        """Generate recommendations based on criteria and matching shows.
        
        This method orchestrates the recommendation generation process, including:
        1. Identifying success factors from criteria and matching shows
        2. Generating general recommendations for missing criteria
        3. Generating network-specific recommendations if networks are provided
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of shows matching the criteria (optional)
            integrated_data: Dictionary of integrated data frames (optional)
            top_networks: List of NetworkMatch objects for network-specific recommendations
            confidence_info: Optional confidence information dictionary
            
        Returns:
            Dictionary with 'general' and 'network_specific' recommendation lists
        """
        # Initialize empty recommendation lists
        recommendations = []
        general_recommendations = []
        network_specific_recommendations = []
        
        try:
            # If matching_shows not provided or empty, get them using the matcher
            if matching_shows is None or matching_shows.empty:
                try:
                    # Use the matcher directly instead of going through criteria_scorer
                    if hasattr(self.criteria_scorer, 'matcher') and self.criteria_scorer.matcher is not None:
                        matching_shows, confidence_info = self.criteria_scorer.matcher.find_matches_with_fallback(criteria)
                        if matching_shows.empty:
                            st.error("No shows match your criteria. Try adjusting your parameters.")
                            return {"general": [], "network_specific": []}
                    else:
                        st.error("No matcher available. Cannot find matching shows.")
                        return {"general": [], "network_specific": []}
                except Exception as e:
                    st.error(f"Unable to analyze shows matching your criteria: {str(e)}")
                    return {"general": [], "network_specific": []}
            
            try:
                # Calculate impact data using the criteria scorer
                # Pass integrated_data to ensure matcher has access to full dataset
                impact_result = self.criteria_scorer.calculate_criteria_impact(criteria, matching_shows, integrated_data=integrated_data)
                
                # Check for errors
                if impact_result.error:
                    return {"general": [], "network_specific": []}
                    
                # Get the impact data from the result
                impact_data = impact_result.criteria_impacts
                

            except Exception as e:
                error_msg = f"Error calculating impact data: {str(e)}"
                return {"general": [], "network_specific": []}
                
            # Convert to SuccessFactor objects
            success_factors = self.identify_success_factors(criteria, matching_shows, integrated_data)
            
            # Generate general recommendations from success factors
            recommendations = self._recommend_missing_criteria(criteria, success_factors, matching_shows)
            
            # Process general recommendations only
            # Network-specific recommendations are already added directly to network_specific_recommendations
            # This ensures proper routing of recommendations to the correct UI tabs:
            # - Network-specific recommendations -> Network Analysis tab
            # - General recommendations -> General Recommendations tab
            for rec in recommendations:
                # Skip recommendations without impact score
                if 'impact' not in rec:
                    continue
                
                # All recommendations in this list are general recommendations
                general_recommendations.append(rec)
            
            # Sort by absolute impact (descending)
            general_recommendations.sort(key=lambda x: abs(x.get('impact', 0)), reverse=True)
            network_specific_recommendations.sort(key=lambda x: abs(x.get('impact', 0)), reverse=True)
            
            # Apply max suggestions limit to general recommendations
            max_suggestions = self.config.SUGGESTIONS.get('max_suggestions', 5)
            if len(general_recommendations) > max_suggestions:
                general_recommendations = general_recommendations[:max_suggestions]
            

                
                # Debug method call removed - was causing AttributeError
            
            return {
                "general": general_recommendations,
                "network_specific": network_specific_recommendations
            }
            
        except Exception as e:
            error_msg = f"Error in generate_recommendations: {str(e)}"
            
            # Always return a dictionary with the expected structure, even on error
            return {
                "general": [],
                "network_specific": []
            }
    
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
                    
                # Create explanation text based on recommendation type
                explanation_templates = {
                    self.REC_TYPE_ADD: "Adding {name} could improve success probability by {impact:.1f}%.",
                    self.REC_TYPE_REMOVE: "Removing {name} could improve success probability by {impact:.1f}%.",
                    self.REC_TYPE_CHANGE: "Changing to {name} could improve success probability by {impact:.1f}%."
                }
                
                template = explanation_templates.get(rec_type, "Consider {name} for potential impact of {impact:.1f}%.")
                explanation_text = template.format(name=criteria_name, impact=abs(impact_score)*100)
                
                # Create recommendation and add to list
                potential_recommendations.append({
                    'recommendation_type': rec_type,
                    'field': criteria_type,
                    'current_value': None,
                    'suggested_value': option_id,
                    'suggested_name': criteria_name,
                    'impact': impact_score,
                    'confidence': factor.confidence,
                    'explanation': explanation_text,
                    'metadata': {}
                })
                    

            
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
                                               integrated_data: IntegratedData,
                                               confidence_info: Optional[ConfidenceInfo] = None) -> List[RecommendationItem]:
        """
        Generate network-specific recommendations.

        Args:
            criteria: Dictionary of criteria values
            network: NetworkMatch object with network information
            matching_shows: DataFrame of shows matching the criteria
            integrated_data: Integrated data for additional context
            confidence_info: Optional confidence information dictionary

        Returns:
            List of network-specific recommendations
        """
        # Ensure confidence_info conforms to our ConfidenceInfo contract
        # This enforces the contract rather than adding defensive checks
        if confidence_info is not None:
            confidence_info = update_confidence_info(confidence_info, {})
            
        # Network object is a NetworkMatch dataclass with attributes like network_id, network_name, etc.
        
        # Check if network_analyzer is available
        if self.network_analyzer is None:
            return []
            
        # Get network-specific success rates
        network_rates = self.network_analyzer.get_network_specific_success_rates(
            matching_shows=matching_shows,
            network_id=network.network_id
        )
        
        if not network_rates:
            return []
        
        # Calculate overall success rates for comparison with network-specific rates
        overall_rates = {}
                
        # Process each key in network rates to calculate corresponding overall rates
        for key, network_rate_data in network_rates.items():
            # Extract field name and value from key
            field_name, field_value = self._parse_key(key)
            
            # Skip if this field is not in our criteria
            if field_name not in criteria:
                continue
            
            # Use the exact field name from the key with no mapping or standardization
            # Calculate the overall success rate for this criteria
            single_criteria = {field_name: criteria[field_name]}
            
            # Skip if matcher is not available
            if not hasattr(self.criteria_scorer, 'matcher') or self.criteria_scorer.matcher is None:
                continue
                
            # Get matching shows for this single criterion
            single_matches, _ = self.criteria_scorer.matcher.find_matches_with_fallback(single_criteria)
            
            # Calculate all scores including success rate
            all_scores = self.criteria_scorer.calculate_scores(single_criteria, single_matches)
            overall_rate = all_scores.get('success_rate')
            
            # Create a consistent data structure that matches network_rate_data
            overall_rates[key] = {
                'success_rate': overall_rate,
                'sample_size': len(single_matches) if single_matches is not None else 0,
                'confidence': 'medium'  # Default confidence level
            }
        
        recommendations = []
                 
        # DATA CONTRACT: Network rates use database column names (IDs) as field names in keys
        # These IDs must match exactly with the criteria keys for proper matching
        # The recommendation engine uses these IDs for all internal processing
        # Human-readable names are derived only for display purposes using _get_criteria_name
        
        # The criteria dictionary must use database column names (IDs) consistently
        valid_fields = set(criteria.keys())
        valid_network_rates = {}
        
        # Process network rates for fields in criteria
        for key, network_rate_data in network_rates.items():
            field_name, _ = self._parse_key(key)
            
            # Only process fields that exist in criteria
            if field_name in valid_fields:
                # Get display name for the current value
                original_field = field_name[:-3] if field_name.endswith('_id') and not field_name.endswith('_ids') else field_name
                current_value = criteria[field_name]
                
                valid_network_rates[key] = {
                    'field_name': field_name,
                    'network_rate_data': network_rate_data,
                    'current_value': current_value,
                    'current_name': self._get_criteria_name(original_field, current_value)
                }
        
        # Now process only the valid network rates
        for key, data in valid_network_rates.items():
            
            field_name = data['field_name']
            network_rate_data = data['network_rate_data']
            current_value = data.get('current_value')
            current_name = data.get('current_name')
            
            # Get the overall success rate using the exact key
            overall_rate_data = overall_rates.get(key)
            
            if overall_rate_data is None:
                # Skip criteria without overall rates
                continue
                
            # Extract the actual success rate value from the overall_rate_data dictionary
            if 'success_rate' not in overall_rate_data or overall_rate_data['success_rate'] is None:
                # Skip criteria without valid success rates
                continue
                
            # Use the explicit success_rate value
            overall_rate = overall_rate_data['success_rate']
                
            network_rate = network_rate_data.get('success_rate')
            sample_size = network_rate_data.get('sample_size', 0)
            
            # Skip if we don't have valid rates to compare
            if network_rate is None or overall_rate is None:
                continue
                
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
                # Calculate impact score with minimum threshold from config
                impact_score = max(abs(difference), OptimizerConfig.THRESHOLDS['significant_difference']) * (1 if difference > 0 else -1)
                
                # Determine recommendation type and create explanation
                if difference > 0:
                    rec_type = self.REC_TYPE_NETWORK_KEEP
                    explanation = f"{network.network_name} shows {network_rate*100:.1f}% success rate for {current_name} (vs. {overall_rate*100:.1f}% overall)."
                else:
                    rec_type = self.REC_TYPE_NETWORK_CHANGE
                    explanation = f"{network.network_name} shows only {network_rate*100:.1f}% success rate for {current_name} (vs. {overall_rate*100:.1f}% overall)."
                
                # Create and append recommendation
                recommendations.append({
                    "recommendation_type": rec_type,
                    "field": field_name,
                    "current_value": current_value,
                    "suggested_value": None,
                    "suggested_name": f"{network.network_name}: {current_name}",
                    "impact": impact_score,
                    "confidence": network_rate_data.get("confidence", "medium"),
                    "explanation": explanation,
                    "metadata": {
                        "network_id": network.network_id,
                        "network_name": network.network_name,
                        "network_rate": network_rate,
                        "overall_rate": overall_rate,
                        "difference": difference
                    }
                })
        
        return recommendations
