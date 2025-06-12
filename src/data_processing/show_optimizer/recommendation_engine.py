"""RecommendationEngine: Handles success factor identification and recommendation generation.

This component is responsible for analyzing show data to identify success factors
and generate recommendations for show concept optimization.
"""

import pandas as pd
import streamlit as st
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from functools import lru_cache

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
                # Debug output removed: shows_analyzer dependency missing
                st.error("Required component missing. Please ensure your application is properly configured.")
                raise ValueError("shows_analyzer cannot be None")
            if success_analyzer is None:
                # Debug output removed: success_analyzer dependency missing
                st.error("Required component missing. Please ensure your application is properly configured.")
                raise ValueError("success_analyzer cannot be None")
            if field_manager is None:
                # Debug output removed: field_manager dependency missing
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
                # Debug output removed: No criteria_scorer available
                st.error("Some recommendation features may be limited due to missing components.")
        except Exception as e:
            # Debug output removed: Error initializing RecommendationEngine
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
                # Debug output removed: Issue retrieving matching shows
                return None, 'none'
            
            # Calculate success rate
            try:
                success_rate = self.criteria_scorer._calculate_success_rate(matching_shows)
            except Exception as calc_e:
                # If success rate calculation fails, return None but don't stop execution
                # Debug output removed: Issue calculating success rate
                return None, 'none'
            
            # Calculate confidence
            try:
                confidence = self.config.get_confidence_level(match_count)
            except Exception as conf_e:
                # If confidence calculation fails, log it and continue with none confidence
                # Debug output removed: Issue calculating confidence level
                confidence = 'none'
            
            return success_rate, confidence
        except Exception as e:
            # Log the error but don't stop execution
            # Debug output removed: Issue in overall success rate calculation
            return None, 'none'
    
    def identify_success_factors(self, criteria: Dict[str, Any], 
                                matching_shows: pd.DataFrame = None,
                                integrated_data: Dict[str, pd.DataFrame] = None,
                                limit: int = 5) -> List[SuccessFactor]:
        """Identify key success factors based on criteria and matching shows.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of matching shows (optional)
            integrated_data: Dictionary of integrated data frames (optional)
            limit: Maximum number of success factors to return
            
        Returns:
            List of SuccessFactor objects
        """
        import traceback
        # Process input arguments
        # If matching_shows not provided, get them
        if matching_shows is None or \
           (isinstance(matching_shows, pd.DataFrame) and matching_shows.empty) or \
           (isinstance(matching_shows, dict) and not matching_shows):
            try:
                matching_shows, _, _ = self.criteria_scorer._get_matching_shows(criteria)
                if isinstance(matching_shows, pd.DataFrame) and matching_shows.empty:
                    # Debug output removed: No matching shows found
                    st.error("No shows match your criteria. Try adjusting your parameters.")
                    return []
            except Exception as inner_e:
                # Error retrieving matching shows
                st.error("Unable to analyze shows matching your criteria.")
                return []
                
        try:
            # Calculate criteria impact
            try:
                impact_data = self.criteria_scorer.calculate_criteria_impact(criteria, matching_shows)
            except Exception as impact_e:
                # Exception in calculate_criteria_impact
                st.error("Critical error during criteria impact calculation.")
                return []
            # Convert to SuccessFactor objects
            success_factors = []
            for criteria_type, values in impact_data.items():
                processed_count = 0
                # Process values directly with proper type handling
                for value_id, impact_data in values.items():
                    # Use the original value_id for matching
                    value_id_hashable = value_id
                    if processed_count >= 5:
                        break
                    try:
                        if isinstance(impact_data, dict) and 'impact' in impact_data:
                            impact = impact_data['impact']
                            sample_size = impact_data.get('sample_size', None)
                            if sample_size is None:
                                sample_size = self.config.DEFAULT_VALUES['fallback_sample_size']
                        else:
                            # Debug output removed: Invalid impact data format
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
                            
                        if isinstance(value_id, (dict, list, np.ndarray)):
                            name = str(value_id)
                        else:
                            options = self.field_manager.get_options(criteria_type)
                            name = str(value_id)
                            for option in options:
                                if option.id == value_id:
                                    name = option.name
                                    break
                        try:
                            if 'sample_size' not in locals() or sample_size is None:
                                sample_size = self.config.DEFAULT_VALUES['fallback_sample_size']
                            confidence = self.config.get_confidence_level(sample_size)
                        except Exception as conf_e:
                            # Debug output removed: Issue determining confidence
                            confidence = self.config.DEFAULT_VALUES['confidence']
                        if confidence == 'none' and sample_size > self.config.CONFIDENCE['minimum_sample']:            
                            pass
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
                            st.error(f"Unable to retrieve matching titles for success factor: {criteria_type}={criteria_value}")
                            matching_titles = []
                        try:
                            factor = SuccessFactor(
                                criteria_type=criteria_type,
                                criteria_value=criteria_value,
                                criteria_name=name,
                                impact_score=impact,
                                confidence=confidence,
                                sample_size=sample_size,
                                matching_titles=matching_titles
                            )
                            try:
                                hash((criteria_type, criteria_value))
                            except Exception as hash_e:
                                # Debug output removed: Unhashable SuccessFactor fields
                                pass
                            success_factors.append(factor)
                        except Exception as factor_e:
                            # Debug output removed: Error creating SuccessFactor
                            # Debug output removed: Impact data
                            continue
                        processed_count += 1
                    except Exception as e:
                        # Debug output removed: Error in inner loop
                        st.error("Unable to create success factor for criteria value")
                        continue
            return success_factors
        except Exception as main_e:
            import traceback
            # Debug output removed: Exception at top level
            # Debug output removed: Input arguments
            st.error("Critical error in identify_success_factors.")
            return []
            
        except Exception as e:
            st.error(f"Error identifying success factors: {str(e)}")
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
        try:
            # Handle missing inputs gracefully
            if criteria is None:
                criteria = {}
                # Log this as a debug message, not an error that stops execution
                st.write("Debug: No criteria provided for recommendation generation, using empty dict")
                
            if matching_shows is None:
                matching_shows = pd.DataFrame()
                # Log this as a debug message, not an error that stops execution
                st.write("Debug: No matching shows provided for recommendation generation, using empty DataFrame")
                
            recommendations = []
            
            # Analyze missing high-impact criteria
            try:
                missing_criteria_recs = self._recommend_missing_criteria(criteria, success_factors, matching_shows)
                recommendations.extend(missing_criteria_recs)
            except Exception as e:
                st.write(f"Debug: Error analyzing missing criteria: {str(e)}")
                st.error("Unable to analyze some criteria. Results may be incomplete.")
            
            # Identify limiting criteria that restrict match quality
            if confidence_info and confidence_info.get('match_level', 1) > 1:
                try:
                    limiting_criteria_recs = self._identify_limiting_criteria(criteria, matching_shows, confidence_info)
                    recommendations.extend(limiting_criteria_recs)
                except Exception as e:
                    st.write(f"Debug: Error identifying limiting criteria: {str(e)}")
                    st.error("Unable to analyze criteria limitations. Some recommendations may be missing.")
            
            # Analyze successful patterns in the matched shows
            try:
                if isinstance(matching_shows, pd.DataFrame) and not matching_shows.empty:
                    pattern_recs = self._analyze_successful_patterns(criteria, matching_shows)
                    recommendations.extend(pattern_recs)
            except Exception as e:
                st.write(f"Debug: Error analyzing successful patterns: {str(e)}")
                st.error("Unable to analyze successful patterns. Some recommendations may be missing.")
            
            # Generate fallback recommendations if needed
            # Only do this if we don't have enough high-quality recommendations already
            if len(recommendations) < self.config.SUGGESTIONS.get('max_suggestions', 5):
                try:
                    fallback_recs = self._generate_fallback_recommendations(criteria, matching_shows, confidence_info)
                    recommendations.extend(fallback_recs)
                except Exception as e:
                    st.write(f"Debug: Error generating fallback recommendations: {str(e)}")
                    st.error("Unable to generate additional recommendations.")
            
            # Sort by impact score (absolute value, as negative impacts are also important)
            recommendations.sort(key=lambda x: abs(x.impact_score), reverse=True)
            
            # Limit to max suggestions
            max_suggestions = self.config.SUGGESTIONS.get('max_suggestions', 5)
            if len(recommendations) > max_suggestions:
                recommendations = recommendations[:max_suggestions]
                
            return recommendations
            
        except Exception as e:
            st.write(f"Debug: Error generating recommendations: {str(e)}")
            st.error("Unable to generate recommendations based on your criteria.")
            return []
    
    def _recommend_missing_criteria(self, criteria: Dict[str, Any], 
                                   success_factors: List[SuccessFactor],
                                   matching_shows: pd.DataFrame) -> List[Recommendation]:
        """Recommend high-impact criteria that are missing from the concept.
        
        Args:
            criteria: Dictionary of criteria
            success_factors: List of success factors
            matching_shows: DataFrame of shows matching the criteria
            
        Returns:
            List of Recommendation objects
        """
        try:
            recommendations = []
            
            # Find high-impact criteria types that are not in the current criteria
            for factor in success_factors:
                if factor.impact_score <= 0:
                    continue  # Skip negative or neutral factors
                    
                if factor.criteria_type not in criteria:
                    # This is a missing criteria type with positive impact
                    recommendation = Recommendation(
                        recommendation_type="add",
                        criteria_type=factor.criteria_type,
                        current_value=None,
                        suggested_value=factor.criteria_value,
                        suggested_name=factor.criteria_name,
                        impact_score=factor.impact_score,
                        confidence=factor.confidence,
                        explanation=f"Adding '{factor.criteria_name}' could improve success probability by approximately {factor.impact_score:.0%}."
                    )
                    recommendations.append(recommendation)
                elif factor.criteria_type in criteria and factor.criteria_value != criteria[factor.criteria_type]:
                    # This criteria type exists but with a different value
                    current_name = str(criteria[factor.criteria_type])
                    options = self.field_manager.get_options(factor.criteria_type)
                    for option in options:
                        if option.id == criteria[factor.criteria_type]:
                            current_name = option.name
                            break
                    
                    # Get the human-readable field name from the field manager
                    field_display_name = factor.criteria_type
                    if factor.criteria_type in self.field_manager.FIELD_CONFIGS:
                        # Use the name from the field config
                        field_config = self.field_manager.FIELD_CONFIGS[factor.criteria_type]
                        field_display_name = field_config.name_field.replace('_', ' ').title()
                            
                    recommendation = Recommendation(
                        recommendation_type="replace",
                        criteria_type=factor.criteria_type,
                        current_value=criteria[factor.criteria_type],
                        current_name=current_name,
                        suggested_value=factor.criteria_value,
                        suggested_name=factor.criteria_name,
                        impact_score=factor.impact_score,
                        confidence=factor.confidence,
                        explanation=f"Replacing '{current_name}' with '{factor.criteria_name}' could improve success probability by approximately {factor.impact_score:.0%}."
                    )
                    recommendations.append(recommendation)
            
            return recommendations
        except Exception as e:
            st.write(f"Debug: Error recommending missing criteria: {str(e)}")
            st.error("Unable to generate criteria recommendations.")
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
                except Exception as inner_e:
                    st.write(f"Debug: Error testing criteria relaxation for {criteria_type}: {str(inner_e)}")
                    continue  # Skip this criterion but continue with others
                    
                # If removing this criterion improves match level or significantly increases sample size
                if (test_confidence.get('match_level', match_level) < match_level or 
                        test_count > len(matching_shows) * 2):  # At least double the sample size
                        
                    # Get the name of the criterion for the explanation
                    if isinstance(criteria_value, (list, np.ndarray)):
                        try:
                            criteria_names = [self._get_criteria_name(criteria_type, v) for v in criteria_value]
                            criteria_name = ", ".join(criteria_names)
                        except:
                            # If iteration fails, use string representation
                            criteria_name = str(criteria_value)
                    else:
                        criteria_name = self._get_criteria_name(criteria_type, criteria_value)
                        
                    # Calculate impact score based on how much the match improves
                    impact_score = 0.1  # Base impact
                    if test_confidence.get('match_level', match_level) < match_level:
                        impact_score += 0.1  # Additional impact for improving match level
                    
                    # Add impact based on sample size increase
                    sample_increase_factor = test_count / max(1, len(matching_shows))
                    if sample_increase_factor > 1:
                        impact_score += min(0.2, (sample_increase_factor - 1) * 0.1)  # Cap at 0.2
                        
                    # Create recommendation
                    recommendation = Recommendation(
                        recommendation_type="relax",
                        criteria_type=criteria_type,
                        current_value=criteria_value,
                        suggested_value=None,  # No specific alternative, just suggesting to relax this criterion
                        suggested_name=criteria_name,
                        impact_score=impact_score,
                        confidence=test_confidence.get('level', 'medium'),
                        explanation=f"Relaxing the '{criteria_name}' requirement would significantly improve match quality "
                                   f"and increase the sample size from {len(matching_shows)} to {test_count} shows."
                    )
                    recommendations.append(recommendation)
            
            return recommendations
        except Exception as e:
            st.write(f"Debug: Error identifying limiting criteria: {str(e)}")
            st.error("Unable to identify criteria limitations.")
            return []
    
    def _analyze_successful_patterns(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> List[Recommendation]:
        """Analyze successful patterns in matching shows and generate recommendations.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of matching shows
            
        Returns:
            List of Recommendation objects based on successful patterns
        """
        try:
            # Handle missing inputs gracefully
            if criteria is None:
                criteria = {}
                
            if matching_shows is None:
                matching_shows = pd.DataFrame()
                
            if isinstance(matching_shows, pd.DataFrame) and matching_shows.empty:
                # Just return empty list, no need for error message
                return []
                
            # Check if we have enough data for meaningful analysis
            try:
                min_sample = self.config.CONFIDENCE.get('minimum_sample')
                if min_sample is None:  # If config value is missing, don't block execution
                    st.write("Debug: Missing minimum_sample configuration")
                elif len(matching_shows) < min_sample:
                    return []
            except Exception as e:
                st.write(f"Debug: Issue accessing configuration: {str(e)}")
                # Continue execution rather than blocking with hardcoded defaults
                
            recommendations = []
            
            # Get successful shows from the matching set
            try:
                # Use PERFORMANCE settings for success threshold
                success_threshold = self.config.PERFORMANCE.get('success_threshold')
                if success_threshold is None:  # If config value is missing, log it but continue
                    st.write("Debug: Missing success_threshold configuration")
                    # Don't use hardcoded default, just continue with empty recommendations
                    return []
                    
                if 'success_score' not in matching_shows.columns:
                    st.write("Debug: Missing success_score column in matching shows data")
                    return []
                    
                successful_shows = matching_shows[matching_shows['success_score'] >= success_threshold].copy()
                
                if len(successful_shows) < 5:  # Need at least a few successful shows for analysis
                    return []
            except Exception as e:
                st.write(f"Debug: Error filtering successful shows: {str(e)}")
                st.error("Unable to analyze successful shows.")
                return []
                
            # Analyze patterns in successful shows
            # 1. Look for common combinations of criteria
            for criteria_type in OptimizerConfig.CRITERIA_IMPORTANCE:
                # Skip criteria not present in the dataset
                if criteria_type not in successful_shows.columns:
                    continue
                    
                # If this criteria is already specified by the user
                if criteria_type in criteria:
                    current_value = criteria[criteria_type]
                    
                    # Make current_value hashable if it's a list
                    if isinstance(current_value, (list, np.ndarray)):
                        try:
                            current_value = tuple(current_value)
                        except:
                            # If conversion fails, use string representation
                            current_value = str(current_value)
                    elif not isinstance(current_value, (str, int, float, bool, tuple)) or pd.isna(current_value):
                        # Handle any other unhashable types
                        current_value = str(current_value)
                    
                    # Check if current value is among the most successful values
                    value_success = {}
                    for value in successful_shows[criteria_type].unique():
                        if pd.isna(value) or value == '':
                            continue
                            
                        # Calculate average success score for shows with this value
                        shows_with_value = successful_shows[successful_shows[criteria_type] == value]
                        avg_success = shows_with_value['success_score'].mean()
                        
                        # Make sure the key is hashable (convert lists/arrays to tuples)
                        if isinstance(value, (list, np.ndarray)):
                            try:
                                hashable_value = tuple(value)
                            except:
                                # If conversion fails, use string representation
                                hashable_value = str(value)
                        elif not isinstance(value, (str, int, float, bool, tuple)) or pd.isna(value):
                            # Handle any other unhashable types
                            hashable_value = str(value)
                        else:
                            hashable_value = value
                            
                        value_success[hashable_value] = avg_success
                        
                    # Sort values by success score
                    sorted_values = sorted(value_success.items(), key=lambda x: x[1], reverse=True)
                    
                    # If current value is not in top 3 most successful values, suggest alternatives
                    top_values = [v[0] for v in sorted_values[:3]]
                    if current_value not in top_values and len(top_values) > 0:
                        # Get the top value
                        top_value = top_values[0]
                        top_success = value_success[top_value]
                        
                        # Calculate impact score based on success difference
                        # Make sure current_value is hashable for dictionary lookup
                        if isinstance(current_value, (list, np.ndarray)):
                            try:
                                hashable_current = tuple(current_value)
                            except:
                                hashable_current = str(current_value)
                        elif not isinstance(current_value, (str, int, float, bool, tuple)) or pd.isna(current_value):
                            # Handle any other unhashable types
                            hashable_current = str(current_value)
                        else:
                            hashable_current = current_value
                            
                        current_success = value_success.get(hashable_current, 0)
                        
                        try:
                            # Use configuration for impact scaling factor
                            impact_score = (top_success - current_success)
                            
                            # Get minimum impact threshold from config
                            min_impact = self.config.SUGGESTIONS.get('minimum_impact')
                            if min_impact is None:
                                st.write("Debug: Missing minimum_impact configuration")
                                continue
                                
                            # Skip low-impact recommendations
                            if impact_score < min_impact:
                                continue
                        except Exception as impact_e:
                            # Log the issue but continue execution
                            st.write(f"Debug: Issue calculating impact score: {str(impact_e)}")
                            continue
                            
                        # Get names
                        current_name = self._get_criteria_name(criteria_type, current_value)
                        suggested_name = self._get_criteria_name(criteria_type, top_value)
                        
                        # Determine confidence level based on sample size
                        try:
                            # Get sample size for this recommendation
                            sample_size = len(shows_with_value)
                            
                            # Determine confidence based on sample size thresholds from config
                            confidence = "medium"  # Default if we can't determine from config
                            
                            # Try to get confidence thresholds from config
                            high_threshold = self.config.CONFIDENCE.get('high_confidence')
                            medium_threshold = self.config.CONFIDENCE.get('medium_confidence')
                            
                            if high_threshold is not None and sample_size >= high_threshold:
                                confidence = "high"
                            elif medium_threshold is not None and sample_size >= medium_threshold:
                                confidence = "medium"
                            else:
                                confidence = "low"
                        except Exception as conf_e:
                            # If there's an issue with confidence calculation, use a safe default
                            st.write(f"Debug: Issue determining confidence: {str(conf_e)}")
                            confidence = "medium"  # Safe default
                        
                        # Create recommendation
                        recommendation = Recommendation(
                            recommendation_type="change",
                            criteria_type=criteria_type,
                            current_value=current_value,
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
            st.write(f"Debug: Error analyzing successful patterns: {str(e)}")
            st.error("Unable to analyze successful patterns.")
            return []
    
    def _generate_fallback_recommendations(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, 
                                        confidence_info: Dict[str, Any]) -> List[Recommendation]:
        """Generate fallback recommendations when data is limited.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of matching shows
            confidence_info: Dictionary with confidence metrics
            
        Returns:
            List of Recommendation objects with fallback suggestions
        """
        try:
            recommendations = []
            
            # Only generate fallback recommendations if we have very few matches
            if isinstance(matching_shows, pd.DataFrame) and not matching_shows.empty and len(matching_shows) >= OptimizerConfig.CONFIDENCE['minimum_sample']:
                return []
                
            # Get the most common successful criteria combinations from the database
            # This is a fallback when we don't have enough matching shows for the specific criteria
            try:
                common_successful_criteria = self.shows_analyzer.get_common_successful_criteria(limit=5)
            except Exception as inner_e:
                st.write(f"Debug: Error retrieving common successful criteria: {str(inner_e)}")
                st.error("Unable to retrieve common successful patterns.")
                return []
            
            # Convert to recommendations
            for criteria_type, values in common_successful_criteria.items():
                # Skip if this criteria type is already in the user's criteria
                if criteria_type in criteria:
                    continue
                    
                # Get the top value for this criteria type
                if not values:
                    continue
                    
                top_value = values[0]['value']
                impact_score = values[0].get('impact', self.config.DEFAULT_VALUES.get('fallback_impact_score', 0.1))
                
                # Get the name for this criteria value
                suggested_name = str(top_value)
                options = self.field_manager.get_options(criteria_type)
                for option in options:
                    if option.id == top_value:
                        suggested_name = option.name
                        break
                
                # Create a fallback recommendation
                recommendation = Recommendation(
                    recommendation_type="consider",  # Use "consider" type for fallbacks
                    criteria_type=criteria_type,
                    current_value=None,
                    suggested_value=top_value,
                    suggested_name=suggested_name,
                    impact_score=impact_score,
                    confidence="low",  # Always low confidence for fallbacks
                    explanation=f"Consider adding '{suggested_name}' as it's commonly found in successful shows. "
                               f"(Note: This is a general recommendation based on limited data for your specific criteria.)"
                )
                recommendations.append(recommendation)
            
            return recommendations
        except Exception as e:
            st.write(f"Debug: Error generating fallback recommendations: {str(e)}")
            st.error("Unable to generate additional recommendations.")
            return []
    
    def _get_criteria_name(self, criteria_type: str, criteria_value: Any) -> str:
        """Get the human-readable name for a criteria value.
        
        Args:
            criteria_type: Type of criteria (e.g., 'genre', 'network')
            criteria_value: Value of the criteria
            
        Returns:
            Human-readable name for the criteria value
        """
        try:
            # Handle None values
            if criteria_value is None:
                return "None"
                
            # Handle list or numpy array values
            if isinstance(criteria_value, (list, np.ndarray)):
                try:
                    names = []
                    for value in criteria_value:
                        name = self._get_criteria_name(criteria_type, value)
                        names.append(name)
                    return ", ".join(names)
                except:
                    # If iteration fails, use string representation
                    return str(criteria_value)
                
            # Handle unhashable types
            if isinstance(criteria_value, (dict, list, np.ndarray)):
                return str(criteria_value)
                
            # Get options from field manager
            options = self.field_manager.get_options(criteria_type)
            
            # Look for matching option
            for option in options:
                if option.id == criteria_value:
                    return option.name
                    
            # Fallback to string representation
            return str(criteria_value)
        except Exception as e:
            st.error(f"Error getting criteria name for {criteria_type}: {str(e)}")
            return str(criteria_value)
            
    def generate_network_specific_recommendations(self, criteria: Dict[str, Any], 
                                               network: NetworkMatch,
                                               matching_shows: pd.DataFrame,
                                               integrated_data: Dict[str, pd.DataFrame]) -> List[Recommendation]:
        """Generate network-specific recommendations.
        
        Args:
            criteria: Dictionary of criteria
            network: Target network
            matching_shows: DataFrame of shows matching the criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            List of Recommendation objects specific to the network
        """
        try:
            recommendations = []
            
            # Get network-specific success rates for each criteria using integrated data
            network_rates = self.criteria_scorer.network_analyzer.get_network_specific_success_rates(criteria, network.network_id, integrated_data)
            
            # Defensive: If network_rates is not a dict, error and return
            if not isinstance(network_rates, dict):
                st.error("NetworkAnalyzer.get_network_specific_success_rates did not return a dict. Cannot generate recommendations.")
                return []
                
            # Process each criteria type in the network rates
            
            # Get overall success rates for each criteria
            overall_rates = {}
            for criteria_type in network_rates.keys():
                if criteria_type not in criteria:
                    continue
                single_criteria = {criteria_type: criteria[criteria_type]}
                overall_rate, _ = self.criteria_scorer.calculate_success_rate(
                    single_criteria, integrated_data=integrated_data
                )
                overall_rates[criteria_type] = overall_rate
            
            # Find criteria where network rate differs significantly from overall rate
            for criteria_type, network_rate_data in network_rates.items():
                if criteria_type not in overall_rates:
                    continue
                if not isinstance(network_rate_data, dict):
                    st.error(f"Network rate data for {criteria_type} is not a dict: {network_rate_data}")
                    continue
                    
                # Check if we have valid data before proceeding
                try:
                    # First check if we have any data at all
                    has_data = network_rate_data.get('has_data', False)
                    
                    # Get the matching_shows data - we know it's always there now
                    matching_shows_data = network_rate_data['matching_shows']
                    
                    # Check if it's empty
                    is_empty = False
                    if matching_shows_data is None:
                        is_empty = True
                    elif isinstance(matching_shows_data, pd.DataFrame):
                        is_empty = matching_shows_data.empty
                    elif isinstance(matching_shows_data, dict) or isinstance(matching_shows_data, list):
                        is_empty = len(matching_shows_data) == 0
                    
                    # Only skip if we have no data at all
                    if is_empty and not has_data:
                        continue
                        
                except Exception as e:
                    # If there's any error in checking the data, skip this criteria
                    continue
                if not network_rate_data.get('has_data') or network_rate_data.get('rate') is None:
                    continue
                network_rate = network_rate_data['rate']
                overall_rate = overall_rates[criteria_type]
                difference = network_rate - overall_rate
                if abs(difference) >= OptimizerConfig.THRESHOLDS['significant_difference']:
                    current_value = criteria[criteria_type]
                    current_name = self._get_criteria_name(criteria_type, current_value)
                    suggested_value = None
                    suggested_name = ""
                    explanation = f"Network {network.display_name} has a significantly different success rate for {criteria_type}."
                    impact_score = difference
                    confidence = network_rate_data.get('confidence', 'unknown')
                    recommendations.append(Recommendation(
                        recommendation_type="network_specific",
                        criteria_type=criteria_type,
                        current_value=current_value,
                        suggested_value=suggested_value,
                        suggested_name=suggested_name,
                        impact_score=impact_score,
                        confidence=confidence,
                        explanation=explanation,
                        current_name=current_name
                    ))
                    # The following block should be inside the previous else branch, not at this indentation level
                    # If you want to suggest alternatives when difference < 0, do so inside an else block
                    # Example fix:
                    # else:
                    #     options = self.field_manager.get_options(criteria_type)
                    #     if options:
                    #         ...

            
            return recommendations
        except Exception as e:
            # Log the error with more specific information but don't display in UI
            # This prevents the three repeated error messages
            if 'empty' in str(e):
                # This is a common error we've fixed, so just log it without showing to user
                print(f"Handled error in network recommendations: {str(e)}")
            else:
                # For other errors, show in UI
                st.error(f"Error generating network-specific recommendations: {str(e)}")
            # Return an empty list to avoid further errors
            return []
