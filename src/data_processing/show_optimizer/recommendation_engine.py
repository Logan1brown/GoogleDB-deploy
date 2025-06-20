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
from .score_calculators import NetworkMatch


@dataclass
class SuccessFactor:
    """A success factor identified from analysis."""
    criteria_type: str  # e.g., "genre", "character_types"
    criteria_value: Any  # ID of the value
    criteria_name: str   # Display name
    impact_score: float  # Impact on success (-1 to 1)
    confidence: str      # none, low, medium, high
    sample_size: int = 0
    matching_titles: List[str] = field(default_factory=list)  # List of show titles matching this criteria
    recommendation_type: str = 'add'  # add, remove, change - default to 'add' for backward compatibility


@dataclass
class Recommendation:
    """A recommendation for optimizing a show concept."""
    recommendation_type: str  # add, remove, replace, consider, relax, change
    criteria_type: str        # e.g., "genre", "character_types"
    current_value: Any        # Current value (if any)
    suggested_value: Any      # Suggested value
    suggested_name: str       # Display name for suggested value
    impact_score: float       # Expected impact on success (-1 to 1)
    confidence: str           # none, low, medium, high
    explanation: str          # Human-readable explanation
    current_name: str = ""    # Display name for current value


class RecommendationEngine:
    """Analyzes show data to identify success factors and generate recommendations."""
    

    
    def _format_key(self, field_name, value_name):
        """Standardize key formatting for consistent matching between network and overall rates.
        
        Args:
            field_name: Field name part of the key
            value_name: Value name part of the key
            
        Returns:
            Tuple of (standard_key, original_key, clean_value_name)
        """
        # Convert value_name to string if it's not already
        value_name = str(value_name)
        
        # Clean up value_name if it contains "Unknown" with a number in parentheses
        clean_value_name = value_name
        if isinstance(value_name, str) and "Unknown" in value_name and "(" in value_name and ")" in value_name:
            try:
                # Extract the value inside parentheses
                start_idx = value_name.find("(") + 1
                end_idx = value_name.find(")")
                if start_idx > 0 and end_idx > start_idx:
                    clean_value = value_name[start_idx:end_idx].strip()

                    clean_value_name = clean_value
            except Exception:
                # If extraction fails, keep the original value_name
                pass
        
        # Create keys for this field-value combination
        standard_key = f"{field_name}:{clean_value_name}"
        original_key = f"{field_name}:{value_name}"
        
        return standard_key, original_key, clean_value_name
    
    def __init__(self, shows_analyzer, success_analyzer, field_manager, criteria_scorer=None):
        """Initialize the recommendation engine.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for show data
            success_analyzer: SuccessAnalyzer instance for success metrics
            field_manager: FieldManager instance for field mapping and validation
            criteria_scorer: Optional CriteriaScorer instance for criteria impact calculations
        """
        try:
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
            
            # Ensure we have access to criteria_scorer for network-specific recommendations
            if self.criteria_scorer is None and hasattr(success_analyzer, 'criteria_scorer'):
                self.criteria_scorer = success_analyzer.criteria_scorer
                
            if self.criteria_scorer is None:

                st.error("Some recommendation features may be limited due to missing components.")
        except Exception as e:

            st.error("Error initializing recommendation system. Some features may be unavailable.")
            raise
    
    def calculate_overall_success_rate(self, criteria: Dict[str, Any]) -> Tuple[float, str]:
        """Calculate the overall success rate for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Tuple of (success_rate, confidence)
        """
        try:
            # Handle missing criteria
            if criteria is None:
                criteria = {}
                
            # Check if we have a criteria_scorer
            if self.criteria_scorer is None:
                # Return None with confidence from config
                return None, self.config.CONFIDENCE_LEVELS.get('none', 'none')
                
            # Get matching shows and count
            try:
                matching_shows, match_count, confidence_info = self.criteria_scorer._get_matching_shows(criteria)
                
                if isinstance(matching_shows, pd.DataFrame) and matching_shows.empty or match_count == 0:
                    # Return None instead of 0.0 to indicate no data available
                    return None, 'none'
            except Exception as inner_e:
                # Log the error but don't stop execution

                return None, 'none'
            
            # Calculate success rate
            try:
                success_rate = self.criteria_scorer._calculate_success_rate(matching_shows)
            except Exception as calc_e:
                # If success rate calculation fails, return None but don't stop execution

                return None, 'none'
            
            # Calculate confidence
            try:
                confidence = self.config.get_confidence_level(match_count)
            except Exception as conf_e:
                # If confidence calculation fails, log it and continue with none confidence

                confidence = 'none'
            
            return success_rate, confidence
        except Exception as e:
            # Log the error but don't stop execution

            return None, 'none'
    
    def identify_success_factors(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame = None, integrated_data: Dict[str, pd.DataFrame] = None, limit: int = 5) -> List[SuccessFactor]:
        """Identify success factors from the given criteria and matching shows.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of shows matching the criteria (optional)
            integrated_data: Dictionary of integrated data frames (optional)
            limit: Maximum number of success factors to identify per criteria type
            
        Returns:
            List of SuccessFactor objects
        """
        # Check if criteria is valid
        # Check if matching_shows is valid
        if matching_shows is None or not isinstance(matching_shows, pd.DataFrame) or matching_shows.empty:
            pass  # Skip processing for invalid data
              
        # Process input arguments
        # If matching_shows not provided, get them
        if matching_shows is None or \
           (isinstance(matching_shows, pd.DataFrame) and matching_shows.empty) or \
           (isinstance(matching_shows, dict) and not matching_shows):
            try:
                matching_shows, _, _ = self.criteria_scorer._get_matching_shows(criteria)
                if isinstance(matching_shows, pd.DataFrame) and matching_shows.empty:
                    st.error("No shows match your criteria. Try adjusting your parameters.")
                    return []
            except Exception as inner_e:
                st.error("Unable to analyze shows matching your criteria.")
                return []
                
        try:
            # Calculate criteria impact
            try:
                # Calculate impact data using the criteria scorer
                impact_data = self.criteria_scorer.calculate_criteria_impact(criteria, matching_shows)
                
            
                # Return empty list if no impact data was found
                if not impact_data or all(len(values) == 0 for field, values in impact_data.items()):
                    return []
                
            except Exception as impact_e:

                return []
            # Convert to SuccessFactor objects
            success_factors = []
            
            for criteria_type, values in impact_data.items():
                processed_count = 0
                
                # Sort options by absolute impact score to prioritize highest impact options
                sorted_options = sorted(values.items(), key=lambda x: abs(x[1].get('impact', 0)), reverse=True)
                
                for value_id, impact_data in sorted_options:
                    # Use the original value_id for matching
                    value_id_hashable = value_id
                    if processed_count >= limit:  # Use the limit parameter instead of hardcoded 5
                        break
                        
                    try:
                        if isinstance(impact_data, dict) and 'impact' in impact_data:
                            impact = impact_data['impact']
                            sample_size = impact_data.get('sample_size', None)
                            if sample_size is None:
                                sample_size = self.config.DEFAULT_VALUES['fallback_sample_size']
                        else:
                            impact = self.config.DEFAULT_VALUES['impact_score']
                            sample_size = self.config.DEFAULT_VALUES['fallback_sample_size']
                            # Convert list/array to tuple or string for hashability if needed
                        if isinstance(value_id, (list, np.ndarray)):
                            try:
                                criteria_value = tuple(value_id)
                            except:
                                criteria_value = str(value_id)
                        elif not isinstance(value_id, (str, int, float, bool, tuple)) or pd.isna(value_id):
                            criteria_value = str(value_id)
                        else:
                            criteria_value = value_id
                            
                        # Use field_manager to get the proper option name
                        if value_id == 'remove':
                            name = f"Remove {criteria_type}"
                        else:
                            try:
                                # For numeric option IDs, get the name from field_manager
                                if isinstance(value_id, (int, float)) and not pd.isna(value_id):
                                    name = self.field_manager.get_name(criteria_type, int(value_id))
                                elif isinstance(value_id, (dict, list, np.ndarray)):
                                    name = str(value_id)
                                else:
                                    # Use the name from impact_data if available
                                    name = impact_data.get('option_name', str(value_id))
                            except Exception as e:
                                name = str(value_id)
                        try:
                            if 'sample_size' not in locals() or sample_size is None:
                                sample_size = self.config.DEFAULT_VALUES['fallback_sample_size']
                            confidence = self.config.get_confidence_level(sample_size)
                        except Exception as conf_e:
                            confidence = self.config.DEFAULT_VALUES['confidence']
                        if confidence == 'none' and sample_size > self.config.CONFIDENCE['minimum_sample']:            
                            pass
                            
                        # Get recommendation type from impact data - NEVER override it
                        recommendation_type = impact_data.get('recommendation_type')
                        
                        # If no recommendation_type is specified (which shouldn't happen), fall back to impact-based type
                        if not recommendation_type:  
                            if impact > 0:
                                recommendation_type = 'add'  # Positive impact - recommend adding
                            else:
                                recommendation_type = 'remove'  # Negative impact - recommend removing

                        matching_titles = []
                        try:
                            # Convert tuple back to list for matching if needed
                            if isinstance(criteria_value, tuple):
                                match_value = list(criteria_value)
                            else:
                                match_value = criteria_value
                            
                            single_criteria = {criteria_type: match_value}
                            single_matches, _, _ = self.criteria_scorer._get_matching_shows(single_criteria)
                            
                            if isinstance(single_matches, pd.DataFrame) and not single_matches.empty and 'title' in single_matches.columns:
                                matching_titles = single_matches['title'].tolist()
                                if len(matching_titles) > 100:
                                    matching_titles = matching_titles[:100]
                        except Exception as e:
                            matching_titles = []
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
                            try:
                                hash((criteria_type, criteria_value))
                            except Exception as hash_e:
                                pass
                            success_factors.append(factor)
                        except Exception as factor_e:
                            continue
                    except Exception as e:
                        continue
                    processed_count += 1
            return success_factors
        except Exception as main_e:
            st.error(f"Error identifying success factors: {str(main_e)}")
            return []
            
    def generate_recommendations(self, criteria: Dict[str, Any],
                                success_factors: List[SuccessFactor],
                                top_networks: List[NetworkMatch],
                                matching_shows: pd.DataFrame,
                                confidence_info: Dict[str, Any],
                                integrated_data: Dict[str, pd.DataFrame]) -> List[Recommendation]:
        """Generate recommendations based on criteria analysis.
        
        Args:
            criteria: Dictionary of criteria
            success_factors: List of identified success factors
            top_networks: List of top network matches
            matching_shows: DataFrame of matching shows
            confidence_info: Dictionary with confidence metrics
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            List of Recommendation objects
        """
        # Process recommendations
                
        try:
            # Handle missing inputs gracefully
            if criteria is None:
                criteria = {}
                
            if matching_shows is None:
                matching_shows = pd.DataFrame()
                
            recommendations = []
            
            # Analyze missing high-impact criteria
            try:
                missing_criteria_recs = self._recommend_missing_criteria(criteria, success_factors, matching_shows)
                recommendations.extend(missing_criteria_recs)
            except Exception as e:
                st.error(f"Unable to analyze some criteria. Error: {str(e)}")
                st.error("Results may be incomplete.")
            
            # Identify limiting criteria that restrict match quality
            if confidence_info and confidence_info.get('match_level', 1) > 1:
                try:
                    limiting_criteria_recs = self._identify_limiting_criteria(criteria, matching_shows, confidence_info)
                    recommendations.extend(limiting_criteria_recs)
                except Exception as e:
                    st.error("Unable to analyze criteria limitations. Some recommendations may be missing.")
            
            # Analyze successful patterns in the matched shows
            try:
                if isinstance(matching_shows, pd.DataFrame) and not matching_shows.empty:
                    pattern_recs = self._analyze_successful_patterns(criteria, matching_shows)
                    recommendations.extend(pattern_recs)
            except Exception as e:
                st.error("Unable to analyze successful patterns. Some recommendations may be missing.")
                
            network_specific_recs = []
            if top_networks and len(top_networks) > 0:
                # Limit to top 3 networks for performance
                for network in top_networks[:3]:
                    try:
                        network_recs = self.generate_network_specific_recommendations(
                            criteria, network, matching_shows, integrated_data
                        )
                        network_specific_recs.extend(network_recs)
                    except Exception as e:
                        pass
                            
                # Add network-specific recommendations to the main list
                try:
                            
                    # Ensure network_specific_recs is a list before extending
                    if isinstance(network_specific_recs, list):
                        if network_specific_recs:
                            recommendations.extend(network_specific_recs)
                    else:
                        pass
                except Exception as e:
                    pass
            
            # Ensure recommendations is a list before sorting
            if not isinstance(recommendations, list):

                recommendations = list(recommendations) if recommendations else []
            
            # Sort by impact score (absolute value, as negative impacts are also important)
            try:
                # Ensure all items in recommendations are Recommendation objects
                valid_recommendations = []
                for rec in recommendations:
                    if hasattr(rec, 'impact_score'):
                        valid_recommendations.append(rec)
                    else:
                        pass
                
                recommendations = valid_recommendations
                recommendations.sort(key=lambda x: abs(x.impact_score), reverse=True)
            except Exception as e:

                recommendations = []
            
            # Limit to max suggestions
            max_suggestions = self.config.SUGGESTIONS.get('max_suggestions', 5)
            if len(recommendations) > max_suggestions:
                recommendations = recommendations[:max_suggestions]
                
            return recommendations
            
        except Exception as e:
            st.error("Unable to generate recommendations based on your criteria.")
            return []
    
    def _recommend_missing_criteria(self, criteria: Dict[str, Any], 
                                   success_factors: List[SuccessFactor],
                                   matching_shows: pd.DataFrame) -> List[Recommendation]:
        """Recommend high-impact criteria that are missing from the concept.
        
        Args:
            criteria: Dictionary of criteria
            success_factors: List of success factors with recommendation_type
            matching_shows: DataFrame of shows matching the criteria
            
        Returns:
            List of Recommendation objects
        """
        try:
            recommendations = []

            j# Process all success factors to create recommendations
            min_impact = OptimizerConfig.SUGGESTIONS.get('minimum_impact', 0.01)
              
            for factor in success_factors:
                # Skip factors with impact below threshold
                if abs(factor.impact_score) < min_impact:

                    continue
                
                # Get information about the selection status for filtering
                is_field_selected = factor.criteria_type in criteria
                option_id = getattr(factor, 'criteria_value', None)
                is_option_selected = False
                
                # Check if this specific option is selected
                if is_field_selected and option_id is not None:
                    # For array fields (like character_types), check if the option_id is in the array
                    if isinstance(criteria.get(factor.criteria_type), list):
                        is_option_selected = option_id in criteria[factor.criteria_type]
                    # For single value fields (like genre), check if the option_id matches the value
                    else:
                        is_option_selected = criteria[factor.criteria_type] == option_id
                
                # Determine the correct recommendation type based on selection status and impact
                rec_type = factor.recommendation_type
                skip_recommendation = False
                skip_reason = ""
                
                # Handle recommendation type based on selection status and impact score
                if is_option_selected and factor.impact_score < 0:
                    # Selected option with negative impact should be a 'remove' recommendation
                    rec_type = 'remove'
                    # Option with negative impact - set to remove
                elif not is_field_selected and factor.impact_score > 0:
                    # Unselected field with positive impact should be an 'add' recommendation
                    rec_type = 'add'
                elif is_field_selected and not is_option_selected and factor.impact_score > 0:
                    # Selected field but different option with positive impact should be a 'change' recommendation
                    rec_type = 'change'
                elif factor.recommendation_type == 'add' and factor.impact_score < 0:
                    # Don't recommend adding something with negative impact
                    skip_recommendation = True
                    skip_reason = "unselected field with negative impact"
                elif factor.recommendation_type == 'remove' and not is_field_selected:
                    # Can't remove what's not selected
                    skip_recommendation = True
                    skip_reason = "remove recommendation for unselected field"
                
                # Skip non-actionable recommendations
                if skip_recommendation:
                    continue
                
                # Update the recommendation type in the factor object for consistency
                factor.recommendation_type = rec_type
                
                # Boost impact score for better UI display
                impact_score = factor.impact_score
                
                # Ensure minimum impact thresholds based on recommendation type
                if rec_type == 'remove' and abs(impact_score) < 0.15:
                    # Higher threshold for remove recommendations
                    impact_score = -0.15
                elif abs(impact_score) < 0.05:
                    # Minimum threshold for all other recommendations
                    impact_score = 0.05 if impact_score > 0 else -0.05
                
                # Recommendation type and impact score finalized
                    
                # Format the explanation based on recommendation type
                if rec_type == 'change':
                    # This is a change recommendation (modifying existing field)
                    explanation = f"Changing to '{factor.criteria_name}' could improve success probability by approximately {abs(factor.impact_score)*100:.1f}%."
                elif rec_type == 'add':
                    # This is an add recommendation (new field)
                    if factor.impact_score > 0:
                        explanation = f"Adding '{factor.criteria_name}' could improve success probability by approximately {abs(factor.impact_score)*100:.1f}%."
                    else:
                        explanation = f"Avoiding '{factor.criteria_name}' could improve success probability by approximately {abs(factor.impact_score)*100:.1f}%."
                else:  # rec_type == 'remove'
                    # For remove recommendations
                    explanation = f"Removing '{factor.criteria_name}' could improve success probability by approximately {abs(factor.impact_score)*100:.1f}%."
                
                # Create the recommendation
                recommendation = Recommendation(
                    recommendation_type=rec_type,
                    criteria_type=factor.criteria_type,
                    current_value=None,
                    suggested_value=factor.criteria_value,
                    suggested_name=factor.criteria_name,
                    impact_score=impact_score,  # Use boosted impact score
                    confidence=factor.confidence,
                    explanation=explanation
                )
                
                # Add to recommendations list
                recommendations.append(recommendation)
                

                        
                # Ensure the recommendation has the correct type
                # This is critical for 'remove' recommendations to be properly displayed
                recommendation.recommendation_type = rec_type
                    
                # Check if this field is in the original criteria (selected)
                is_selected = criteria_type in criteria
                expected_type = 'change' if is_selected and impact > 0 else 'add' if impact > 0 else 'remove'

                
                # Recommendation processing complete
            
            return recommendations
        except Exception as e:
            st.error(f"Error recommending missing criteria: {str(e)}")
            return []
    
    def _identify_limiting_criteria(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, 
                                confidence_info: Dict[str, Any]) -> List[Recommendation]:
        """Identify criteria that are limiting match quality and suggest alternatives.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of matching shows
            confidence_info: Dictionary with confidence metrics
            
        Returns:
            List of Recommendation objects with suggestions to improve match quality
        """
        try:
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
                try:
                    test_matches, test_count, test_confidence = self.criteria_scorer._get_matching_shows(
                        test_criteria, flexible=True)
                except Exception as e:
                    continue  # Skip this criterion but continue with others
                    
                # If removing this criterion improves match level or significantly increases sample size
                if (test_confidence.get('match_level', match_level) < match_level or 
                        test_count > len(matching_shows) * 2):  # At least double the sample size
                        
                    # Get the name of the criterion for the explanation
                    criteria_name = self._get_criteria_name(criteria_type, criteria_value)
                        
                    # Get the human-readable field name
                    field_display_name = criteria_type
                    if criteria_type in self.field_manager.FIELD_CONFIGS:
                        field_config = self.field_manager.FIELD_CONFIGS[criteria_type]
                        field_display_name = field_config.name_field.replace('_', ' ').title()
                    
                    # Calculate the impact score based on the improvement in match quality
                    impact_score = 0.2  # Default impact score
                    if test_count > 0 and len(matching_shows) > 0:
                        # Impact based on increase in sample size
                        impact_score = min(0.5, (test_count - len(matching_shows)) / len(matching_shows))
                    
                    recommendation = Recommendation(
                        recommendation_type="remove",
                        criteria_type=criteria_type,
                        current_value=criteria_value,
                        current_name=criteria_name,
                        suggested_value=None,
                        suggested_name=f"Remove {field_display_name}",
                        impact_score=impact_score,
                        confidence="medium",
                        explanation=f"Removing '{criteria_name}' could increase your potential matches from {len(matching_shows)} to {test_count} shows."
                    )
                    recommendations.append(recommendation)
            
            return recommendations
        except Exception as e:
            st.error(f"Error identifying limiting criteria: {str(e)}")
            return []
    
    def _analyze_successful_patterns(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> List[Recommendation]:
        """Analyze patterns in successful shows and suggest criteria changes.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of matching shows
            
        Returns:
            List of Recommendation objects based on successful patterns
        """
        try:
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
                    
                    # Make current_value hashable using the same logic as for other values
                    try:
                        if isinstance(current_value, (list, np.ndarray)):
                            current_value = tuple(current_value)
                        elif isinstance(current_value, dict):
                            current_value = str(current_value)
                        elif not isinstance(current_value, (str, int, float, bool, tuple)) or pd.isna(current_value):
                            current_value = str(current_value)
                    except Exception as e:
                        # Handle conversion error
                        current_value = str(current_value)
                    
                    # Calculate success rate for each value of this criteria
                    value_success = {}
                    
                    # Check if column contains list values
                    try:
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
                                try:
                                    if isinstance(value, list):
                                        hashable_value = tuple(value)
                                    elif isinstance(value, dict):
                                        hashable_value = str(value)
                                    else:
                                        hashable_value = str(value)
                                except Exception:
                                    hashable_value = str(value)
                                    
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
                                try:
                                    if isinstance(value, (list, np.ndarray)):
                                        hashable_value = tuple(value)
                                    elif isinstance(value, dict):
                                        hashable_value = str(value)
                                    elif not isinstance(value, (str, int, float, bool, tuple)) or pd.isna(value):
                                        hashable_value = str(value)
                                    else:
                                        hashable_value = value
                                except Exception:
                                    # If conversion fails, use string representation
                                    hashable_value = str(value)
                                    
                                value_success[hashable_value] = avg_success
                    except Exception as e:
                        continue
                        
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
                            
                        # Get names
                        current_name = self._get_criteria_name(criteria_type, current_value)
                        suggested_name = self._get_criteria_name(criteria_type, top_value)
                        
                        # Determine confidence level based on sample size
                        sample_size = len(successful_shows[successful_shows[criteria_type] == top_value])
                        confidence = self.config.get_confidence_level(sample_size)
                        
                        # Create recommendation
                        recommendation = Recommendation(
                            recommendation_type="change",
                            criteria_type=criteria_type,
                            current_value=current_value,
                            current_name=current_name,
                            suggested_value=top_value,
                            suggested_name=suggested_name,
                            impact_score=impact_score,
                            confidence=confidence,
                            explanation=f"Shows with '{suggested_name}' have a {top_success:.0%} success rate compared to "
                                       f"{current_success:.0%} for shows with '{current_name}'. Consider changing this criterion."
                        )
                        recommendations.append(recommendation)
            
            return recommendations
        except Exception as e:
            # Handle pattern analysis error
            return []
    
    def _get_criteria_name(self, criteria_type, value):
        """Get the display name for a criteria value.
        
        Args:
            criteria_type: Type of criteria (e.g., 'genre', 'source_type')
            value: Value of the criteria
            
        Returns:
            Display name for the criteria value
        """
        try:
            # Handle None values
            if value is None:
                return "None"
            
                
            # Handle list or numpy array values
            if isinstance(value, (list, np.ndarray)):
                try:
                    names = []
                    for val in value:
                        name = self._get_criteria_name(criteria_type, val)
                        names.append(name)
                    return ", ".join(names)
                except:
                    # If iteration fails, use string representation
                    return str(value)
                
            # Handle unhashable types
            if isinstance(value, (dict, list, np.ndarray)):
                return str(value)
                
            # Get options from field manager
            options = self.field_manager.get_options(criteria_type)
            
            # Look for matching option
            for option in options:
                if option.id == value:
                    return option.name
                    
            # Fallback to string representation
            return str(value)
        except Exception as e:
            st.error(f"Error getting criteria name for {criteria_type}: {str(e)}")
            return str(value)
            
    def generate_network_specific_recommendations(self, criteria: Dict[str, Any], 
                                                network: NetworkMatch,
                                                matching_shows: pd.DataFrame,
                                                integrated_data: Dict[str, pd.DataFrame]) -> List[Recommendation]:
        """
        Generate network-specific recommendations.        
        Args:
            criteria: Dictionary of criteria
            network: Target network
            matching_shows: DataFrame of shows matching the criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            List of Recommendation objects specific to the network
        """
        # Skip if we don't have valid criteria or network
        if not criteria or not network:
            return []
            
        # Get network-specific success rates for each criteria using matching_shows
        try:
            network_rates = self.criteria_scorer.network_analyzer.get_network_specific_success_rates(
                matching_shows=matching_shows,
                network_id=network.network_id
            )
        except Exception as e:
            st.error(f"Error getting network-specific success rates: {str(e)}")
            return []
            
        # Skip if network_rates is empty
        if not network_rates:
            return []
        # Calculate overall success rates for comparison with network-specific rates
        overall_rates = {}
        
        # Process each key in network rates to calculate corresponding overall rates
        for key, network_rate_data in network_rates.items():
            # Extract field name from key using standard format
            field_name = key.split(':', 1)[0] if ':' in key else key
            
            # Skip if this field is not in our criteria
            if field_name not in criteria:
                continue
                
            # Calculate the overall success rate for this criteria
            single_criteria = {field_name: criteria[field_name]}
            try:
                overall_rate, overall_details = self.criteria_scorer.calculate_success_rate(
                    single_criteria, integrated_data=integrated_data
                )
                
                # Store the overall rate using both key formats for flexible lookup
                overall_rates[key] = overall_rate
                overall_rates[field_name] = overall_rate
            except Exception:
                # Skip this criteria if calculation fails
                continue
        
        recommendations = []
                 
        # Analyze each criteria to find significant differences between network and overall rates
        for key, network_rate_data in network_rates.items():
            # Extract field name from the key using standard format
            field_name = key.split(':', 1)[0] if ':' in key else key
            
            # Get the overall success rate using flexible key lookup
            overall_rate = overall_rates.get(key, overall_rates.get(field_name))
            if overall_rate is None:
                # Skip criteria without overall rates
                continue
                
            # Get network success rate and sample size
            network_rate = network_rate_data.get('success_rate', 0)
            sample_size = network_rate_data.get('sample_size', 0)
            
            # Calculate the difference between network and overall rates
            difference = network_rate - overall_rate
            
            # Check if we have enough data points for reliable comparison
            has_sufficient_data = sample_size >= OptimizerConfig.SUCCESS['min_data_points']
            
            # Get thresholds from config
            network_diff_threshold = OptimizerConfig.THRESHOLDS.get('network_difference', 0.001)
            significant_diff_threshold = OptimizerConfig.THRESHOLDS['significant_difference']
            
            # Determine if the difference is significant enough for a recommendation
            condition1 = abs(difference) >= significant_diff_threshold  # Large difference
            condition2 = has_sufficient_data and abs(difference) > network_diff_threshold  # Smaller difference with sufficient data
            should_generate = condition1 or condition2
            
            # Create recommendation if the difference is significant
            if should_generate:
                # Get current criteria value and display name
                current_value = criteria.get(field_name)
                current_name = self._get_criteria_name(field_name, current_value)
                
                # Format percentages for explanation
                direction = "higher" if difference > 0 else "lower"
                network_percent = network_rate * 100
                overall_percent = overall_rate * 100
                diff_percent = abs(difference) * 100
                    
                # Create basic explanation text
                explanation = f"Network {network.network_name} has a {direction} success rate for '{field_name}' "
                explanation += f"({network_percent:.1f}% vs {overall_percent:.1f}% overall, {diff_percent:.1f}% difference)."
                
                # Determine recommendation type based on difference direction
                rec_type = "keep" if difference > 0 else "change"
                
                # Add action-oriented text to explanation
                if difference > 0:
                    explanation += f" This criteria performs well on this network."
                else:
                    explanation += f" Consider adjusting this criteria for better results on this network."
                
                # Create network-specific suggested name
                suggested_name = f"{network.network_name}: {current_name}"
                
                # Calculate impact score with minimum threshold to ensure visibility
                impact_score = max(abs(difference), 0.05) * (1 if difference > 0 else -1)
                
                # Create a plain text explanation with clear action steps
                if difference > 0:
                    detailed_explanation = (
                        f"Network Strength: {network.network_name} has a {direction} success rate "
                        f"for '{field_name}' ({network_percent:.1f}% vs {overall_percent:.1f}% overall, "
                        f"{diff_percent:.1f}% difference). "
                        f"Recommendation: Keep this criteria as it performs well on this network."
                    )
                else:
                    detailed_explanation = (
                        f"Network Adjustment: {network.network_name} has a {direction} success rate "
                        f"for '{field_name}' ({network_percent:.1f}% vs {overall_percent:.1f}% overall, "
                        f"{diff_percent:.1f}% difference). "
                        f"Recommendation: Consider adjusting this criteria for better results on this network."
                    )
                
                # Create and add the recommendation object
                recommendations.append(Recommendation(
                    recommendation_type=f"network_{rec_type}",  # Network-specific recommendation type
                    criteria_type=field_name,
                    current_value=current_value,
                    current_name=current_name,
                    suggested_value=current_value,  # Keep same value for network recommendations
                    suggested_name=suggested_name,
                    impact_score=impact_score,
                    confidence=network_rate_data.get('confidence', 'medium'),
                    explanation=detailed_explanation
                ))

        # Return all generated network-specific recommendations
        return recommendations
