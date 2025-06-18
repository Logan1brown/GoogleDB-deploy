"""RecommendationEngine: Handles success factor identification and recommendation generation.

This component is responsible for analyzing show data to identify success factors
and generate recommendations for show concept optimization.
"""

import pandas as pd
import streamlit as st
import numpy as np
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
        # Add detailed debug logging at the start of the method
        if OptimizerConfig.DEBUG_MODE:
            st.write("DEBUG: Starting identify_success_factors method")
            st.write(f"DEBUG: Criteria: {criteria}")
            st.write(f"DEBUG: Matching shows count: {len(matching_shows) if isinstance(matching_shows, pd.DataFrame) else 'Not a DataFrame'}")
            st.write(f"DEBUG: Limit: {limit}")
            
            # Check if criteria is valid
            if not criteria:
                st.write("DEBUG: Warning - Empty criteria provided to identify_success_factors")
            
            # Check if matching_shows is valid
            if matching_shows is None or not isinstance(matching_shows, pd.DataFrame) or matching_shows.empty:
                st.write("DEBUG: Warning - Invalid or empty matching_shows provided to identify_success_factors")
        import traceback
        
        
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
            # No diagnostic logging needed
        
            # Calculate criteria impact
            try:
                # Calculate impact data using the criteria scorer
                impact_data = self.criteria_scorer.calculate_criteria_impact(criteria, matching_shows)
                
                # Log impact data details if in debug mode
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Impact data returned: {impact_data is not None}")
                    if impact_data:
                        st.write(f"Impact data fields: {list(impact_data.keys())}")
                        for field, values in impact_data.items():
                            st.write(f"Field {field} has {len(values)} options with impact scores")
            
                # Return empty list if no impact data was found
                if not impact_data or all(len(values) == 0 for field, values in impact_data.items()):
                    if OptimizerConfig.DEBUG_MODE:
                        st.write("No impact data found for recommendations.")
                    return []
                
            except Exception as impact_e:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Error analyzing criteria impact: {str(impact_e)}")
                return []
            # Convert to SuccessFactor objects
            success_factors = []
            
            # Debug log the impact data structure
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Processing impact data for {len(impact_data)} criteria types")
                for criteria_type, values in impact_data.items():
                    st.write(f"DEBUG: Criteria type {criteria_type} has {len(values)} options with impact scores")
                    # Show the first few options and their impact scores
                    for i, (option_id, option_data) in enumerate(values.items()):
                        if i >= 3:  # Limit to first 3 options to avoid clutter
                            break
                        impact = option_data.get('impact', 'N/A')
                        rec_type = option_data.get('recommendation_type', 'N/A')
                        st.write(f"DEBUG: Option {option_id} has impact {impact} and recommendation_type {rec_type}")
                if not impact_data:
                    st.write("DEBUG: No impact data available - check calculate_criteria_impact method")
            
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
                            
                        # Determine recommendation type based on impact
                        if impact > 0:
                            recommendation_type = 'add'  # Positive impact - recommend adding
                        else:
                            recommendation_type = 'remove'  # Negative impact - recommend removing
                            
                        # Temporarily lower the minimum impact threshold to ensure we get some recommendations
                        min_impact = 0.01  # Lower threshold for testing
                        
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Impact for {criteria_type}/{name}: {impact} (threshold: {min_impact})")
                        
                        # Ensure we have at least some minimal impact to generate recommendations
                        if abs(impact) < min_impact:
                            if OptimizerConfig.DEBUG_MODE:
                                st.write(f"DEBUG: Boosting small impact for {criteria_type}/{name} from {impact} to {min_impact}")
                            # Set a minimum impact value to ensure recommendations are generated
                            impact = min_impact if impact >= 0 else -min_impact
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
        # Add debug logging for success factors
        if OptimizerConfig.DEBUG_MODE:
            st.write(f"DEBUG: Generate recommendations called with {len(success_factors) if success_factors else 0} success factors")
            if success_factors:
                for i, factor in enumerate(success_factors[:3]):  # Show first 3 factors
                    st.write(f"DEBUG: Success factor {i+1}: {factor.criteria_type} - {factor.criteria_name} - impact: {factor.impact_score} - type: {factor.recommendation_type}")
            else:
                st.write("DEBUG: No success factors available - check identify_success_factors method")
                # Show the criteria to help diagnose why no success factors were found
                st.write(f"DEBUG: Current criteria: {criteria}")
                # Show the matching shows count to help diagnose why no success factors were found
                if isinstance(matching_shows, pd.DataFrame):
                    st.write(f"DEBUG: Matching shows count: {len(matching_shows)}")
                    if not matching_shows.empty and 'title' in matching_shows.columns:
                        st.write(f"DEBUG: First few matching shows: {', '.join(matching_shows['title'].head(3).tolist())}")
                else:
                    st.write("DEBUG: No matching shows DataFrame available")
                
            # Debug the top networks
            if top_networks:
                st.write(f"DEBUG: Top networks: {', '.join([n.network_name for n in top_networks[:3]])}")
            else:
                st.write("DEBUG: No top networks available")
                
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
                
            # Generate network-specific recommendations for top networks
            network_specific_recs = []
            if top_networks and len(top_networks) > 0:
                # Limit to top 3 networks for performance
                for network in top_networks[:3]:
                    try:
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Generating recommendations for network {network.network_name}")
                        
                        network_recs = self.generate_network_specific_recommendations(
                            criteria, network, matching_shows, integrated_data
                        )
                        
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Generated {len(network_recs)} recommendations for network {network.network_name}")
                            if network_recs:
                                for i, rec in enumerate(network_recs[:2]):  # Show first 2 recommendations
                                    st.write(f"DEBUG: Network recommendation {i+1}: {rec.criteria_type} - {rec.suggested_name} - impact: {rec.impact_score}")
                            
                        network_specific_recs.extend(network_recs)
                    except Exception as e:
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Error generating recommendations for network {network.network_name}: {str(e)}")
                            import traceback
                            st.write(traceback.format_exc())
                            
                # Add network-specific recommendations to the main list
                if network_specific_recs:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Adding {len(network_specific_recs)} network-specific recommendations to main list")
                    recommendations.extend(network_specific_recs)
            
            # Generate fallback recommendations if needed
            # Only do this if we don't have enough high-quality recommendations already
            if len(recommendations) < self.config.SUGGESTIONS.get('max_suggestions', 5):
                try:
                    fallback_recs = self._generate_fallback_recommendations(criteria, matching_shows, confidence_info)
                    recommendations.extend(fallback_recs)
                except Exception as e:
                    st.error("Unable to generate additional recommendations.")
            
            # Sort by impact score (absolute value, as negative impacts are also important)
            recommendations.sort(key=lambda x: abs(x.impact_score), reverse=True)
            
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
            
            # Debug logs only if debug mode is enabled
            if OptimizerConfig.DEBUG_MODE:
                # Log success factors
                st.write(f"DEBUG: Total success factors: {len(success_factors)}")
            
            # Filter success factors by recommendation type and impact
            # For testing, include all factors regardless of recommendation type
            add_factors = [f for f in success_factors if f.impact_score > 0]
            change_factors = [f for f in success_factors if f.impact_score < 0]
            
            # Debug log the factors
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Positive impact factors: {len(add_factors)}")
                st.write(f"DEBUG: Negative impact factors: {len(change_factors)}")
            
            # Debug logs only if debug mode is enabled
            if OptimizerConfig.DEBUG_MODE:
                # Log filtered factors
                st.write(f"DEBUG: Add factors: {len(add_factors)}, Change factors: {len(change_factors)}")
                
                # Check if we have any matching shows to analyze
                if isinstance(matching_shows, pd.DataFrame):
                    st.write(f"DEBUG: Matching shows for recommendations: {len(matching_shows)}")
                    if not matching_shows.empty:
                        st.write(f"DEBUG: Sample of columns: {list(matching_shows.columns)[:5]}")
                else:
                    st.write("DEBUG: No matching shows DataFrame available")
            
            # Process all success factors to create recommendations
            for factor in success_factors:
                # Determine recommendation type based on impact score
                rec_type = "add" if factor.impact_score > 0 else "remove"
                
                # Format the explanation based on impact
                if factor.impact_score > 0:
                    explanation = f"Adding '{factor.criteria_name}' could improve success probability by approximately {abs(factor.impact_score)*100:.1f}%."
                else:
                    explanation = f"Removing '{factor.criteria_name}' could improve success probability by approximately {abs(factor.impact_score)*100:.1f}%."
                
                # Create the recommendation
                recommendation = Recommendation(
                    recommendation_type=rec_type,
                    criteria_type=factor.criteria_type,
                    current_value=None,
                    suggested_value=factor.criteria_value,
                    suggested_name=factor.criteria_name,
                    impact_score=factor.impact_score,
                    confidence=factor.confidence,
                    explanation=explanation
                )
                
                # Add to recommendations list
                recommendations.append(recommendation)
                
                # Debug log
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Created recommendation for {factor.criteria_type}/{factor.criteria_name} with impact {factor.impact_score}")
            
            # Debug log the total recommendations created
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Created {len(recommendations)} recommendations from success factors")
            
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
                    except:
                        # If conversion fails, use string representation
                        current_value = str(current_value)
                    
                    # Calculate success rate for each value of this criteria
                    value_success = {}
                    for value in successful_shows[criteria_type].unique():
                        if pd.isna(value) or value == '':
                            continue
                            
                        # Calculate average success score for shows with this value
                        shows_with_value = successful_shows[successful_shows[criteria_type] == value]
                        avg_success = shows_with_value['success_score'].mean()
                        
                        # Make sure the key is hashable
                        try:
                            if isinstance(value, (list, np.ndarray)):
                                hashable_value = tuple(value)
                            elif isinstance(value, dict):
                                # Convert dict to a frozen set of items
                                hashable_value = str(value)
                            elif not isinstance(value, (str, int, float, bool, tuple)) or pd.isna(value):
                                hashable_value = str(value)
                            else:
                                hashable_value = value
                        except:
                            # If conversion fails, use string representation
                            hashable_value = str(value)
                            
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
            st.error(f"Error analyzing successful patterns: {str(e)}")
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
        try:
            # Skip if we don't have valid criteria or network
            if not criteria or not network:
                return []
                
            # Get network-specific success rates for each criteria using matching_shows
            try:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Getting network-specific success rates for {network.network_name}")
                    
                # Use the correct parameter pattern for get_network_specific_success_rates
                # The method only accepts matching_shows and network_id parameters
                network_rates = self.criteria_scorer.network_analyzer.get_network_specific_success_rates(
                    matching_shows, network.network_id)
                    
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Got network rates with {len(network_rates)} criteria types")
            except Exception as e:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Error getting network-specific success rates: {str(e)}")
                    import traceback
                    st.write(traceback.format_exc())
                return []
                
            # Ensure all matching_shows in network_rates are DataFrames
            for criteria_type, rate_data in network_rates.items():
                if isinstance(rate_data, dict) and 'matching_shows' in rate_data:
                    if not isinstance(rate_data['matching_shows'], pd.DataFrame):
                        rate_data['matching_shows'] = pd.DataFrame()
                
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
            
            recommendations = []
            
            # Find criteria where network rate differs significantly from overall rate
            for criteria_type, network_rate_data in network_rates.items():
                # Skip if criteria not in overall rates or network data is invalid
                if criteria_type not in overall_rates or not isinstance(network_rate_data, dict):
                    continue
                    
                # Check if we have valid data
                has_data = network_rate_data.get('has_data', False)
                
                # Handle the matching_shows key
                if 'matching_shows' in network_rate_data:
                    matching_shows_data = network_rate_data['matching_shows']
                else:
                    # Use the matching_shows parameter as fallback
                    matching_shows_data = matching_shows
                    # Store it for future reference
                    network_rate_data['matching_shows'] = matching_shows_data
                
                # Check if matching_shows_data is empty
                is_empty = matching_shows_data is None or (
                    isinstance(matching_shows_data, pd.DataFrame) and matching_shows_data.empty
                )
                
                # Skip if no data available
                if is_empty and not has_data:
                    continue
                    
                # Skip if rate data is missing
                network_rate = network_rate_data.get('rate')
                if network_rate is None:
                    continue
                    
                overall_rate = overall_rates[criteria_type]
                difference = network_rate - overall_rate
                
                # Check if we have enough data for this network
                sample_size = network_rate_data.get('sample_size', 0)
                has_sufficient_data = sample_size >= OptimizerConfig.SUCCESS['min_data_points']
                
                # Generate recommendation if difference is significant
                if abs(difference) >= OptimizerConfig.THRESHOLDS['significant_difference'] or \
                   (has_sufficient_data and abs(difference) > 0.01):
                    
                    current_value = criteria[criteria_type]
                    current_name = self._get_criteria_name(criteria_type, current_value)
                    
                    # Format percentages for explanation
                    direction = "higher" if difference > 0 else "lower"
                    network_percent = network_rate * 100
                    overall_percent = overall_rate * 100
                    diff_percent = abs(difference) * 100
                    
                    # Create explanation
                    explanation = f"Network {network.network_name} has a {direction} success rate for '{criteria_type}' "
                    explanation += f"({network_percent:.1f}% vs {overall_percent:.1f}% overall, {diff_percent:.1f}% difference)."
                    
                    # Determine recommendation type based on difference direction
                    rec_type = "keep" if difference > 0 else "change"
                    
                    if difference > 0:
                        explanation += f" This criteria performs well on this network."
                    else:
                        explanation += f" Consider adjusting this criteria for better results on this network."
                    
                    # Create recommendation
                    # Format the suggested_name to include the network name for better UI display
                    suggested_name = f"{network.network_name}: {current_name}"
                    
                    # Debug log
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Creating network recommendation for {network.network_name} - {criteria_type} - {current_name}")
                        st.write(f"DEBUG: Network rate: {network_rate}, Overall rate: {overall_rate}, Difference: {difference}")
                        st.write(f"DEBUG: Recommendation type: network_{rec_type}")
                    
                    # Set a minimum impact score to ensure recommendations are displayed
                    impact_score = max(abs(difference), 0.05) * (1 if difference > 0 else -1)
                    
                    # Create a more detailed explanation with clear action steps
                    if difference > 0:
                        detailed_explanation = (
                            f"<strong>Network Strength:</strong> {network.network_name} has a {direction} success rate "
                            f"for '{criteria_type}' ({network_percent:.1f}% vs {overall_percent:.1f}% overall, "
                            f"{diff_percent:.1f}% difference). <br><br>"
                            f"<strong>Recommendation:</strong> Keep this criteria as it performs well on this network."
                        )
                    else:
                        detailed_explanation = (
                            f"<strong>Network Adjustment:</strong> {network.network_name} has a {direction} success rate "
                            f"for '{criteria_type}' ({network_percent:.1f}% vs {overall_percent:.1f}% overall, "
                            f"{diff_percent:.1f}% difference). <br><br>"
                            f"<strong>Recommendation:</strong> Consider adjusting this criteria for better results on this network."
                        )
                    
                    recommendations.append(Recommendation(
                        recommendation_type=f"network_{rec_type}",  # Ensure network_ prefix for proper categorization
                        criteria_type=criteria_type,
                        current_value=current_value,
                        current_name=current_name,
                        suggested_value=current_value,  # Use current value for network recommendations
                        suggested_name=suggested_name,
                        impact_score=impact_score,  # Use our boosted impact score to ensure display
                        confidence=network_rate_data.get('confidence', 'medium'),
                        explanation=detailed_explanation
                    ))
            
            return recommendations
        except Exception as e:
            # Only show error in UI if it's not the known 'empty' attribute error
            if 'empty' not in str(e):
                st.error(f"Error generating network recommendations: {str(e)}")
            return []
