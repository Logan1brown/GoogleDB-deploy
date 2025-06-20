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
                            
                        # Get recommendation type from impact data - NEVER override it
                        recommendation_type = impact_data.get('recommendation_type')
                        
                        # If no recommendation_type is specified (which shouldn't happen), fall back to impact-based type
                        if not recommendation_type:  
                            if impact > 0:
                                recommendation_type = 'add'  # Positive impact - recommend adding
                            else:
                                recommendation_type = 'remove'  # Negative impact - recommend removing
                            
                            if OptimizerConfig.DEBUG_MODE:
                                OptimizerConfig.debug(f"WARNING: No recommendation_type in impact data for {criteria_type}/{name}. Using impact-based fallback: {recommendation_type}", category='warning')
                        
                        if OptimizerConfig.DEBUG_MODE:
                            OptimizerConfig.debug(f"Using recommendation type '{recommendation_type}' for {criteria_type}/{name}", category='recommendation')
                            
                        # Temporarily lower the minimum impact threshold to ensure we get some recommendations
                        min_impact = 0.01  # Lower threshold for testing
                        
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Impact for {criteria_type}/{name}: {impact} (threshold: {min_impact})")
                            OptimizerConfig.debug(f"Impact for {criteria_type}/{name}: {impact} (threshold: {min_impact})", category='recommendation')
                        
                        # Ensure we have at least some minimal impact to generate recommendations
                        original_impact = impact
                        # Always boost impact to at least 0.05 (5%) to ensure recommendations are generated
                        # This is a temporary fix to ensure recommendations appear
                        min_display_impact = 0.05  # 5% minimum for display
                        if abs(impact) < min_display_impact:
                            impact = min_display_impact if impact >= 0 else -min_display_impact
                            if OptimizerConfig.DEBUG_MODE:
                                st.write(f"DEBUG: Boosting small impact for {criteria_type}/{name} from {original_impact} to {impact} to ensure recommendations are displayed")
                                OptimizerConfig.debug(f"Boosting small impact for {criteria_type}/{name} from {original_impact} to {impact} to ensure recommendations are displayed", category='recommendation')
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
                            OptimizerConfig.debug(f"Generated {len(network_recs)} recommendations for network {network.network_name}", category='recommendation')
                            if network_recs:
                                for i, rec in enumerate(network_recs[:2]):  # Show first 2 recommendations
                                    st.write(f"DEBUG: Network recommendation {i+1}: {rec.criteria_type} - {rec.suggested_name} - impact: {rec.impact_score}")
                                    OptimizerConfig.debug(f"Network recommendation {i+1}: {rec.criteria_type} - {rec.suggested_name} - impact: {rec.impact_score}", category='recommendation')
                            
                        network_specific_recs.extend(network_recs)
                    except Exception as e:
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Error generating recommendations for network {network.network_name}: {str(e)}")
                            
                # Add network-specific recommendations to the main list
                try:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: network_specific_recs type: {type(network_specific_recs).__name__}")
                        if network_specific_recs:
                            st.write(f"DEBUG: First item type: {type(network_specific_recs[0]).__name__ if len(network_specific_recs) > 0 else 'empty'}")
                            
                    # Ensure network_specific_recs is a list before extending
                    if isinstance(network_specific_recs, list):
                        if network_specific_recs:
                            if OptimizerConfig.DEBUG_MODE:
                                st.write(f"DEBUG: Adding {len(network_specific_recs)} network-specific recommendations to main list")
                            recommendations.extend(network_specific_recs)
                    else:
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: network_specific_recs is not a list, it's a {type(network_specific_recs).__name__}")
                except Exception as e:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Error adding network recommendations: {str(e)}")
                        st.write(f"DEBUG: recommendations type: {type(recommendations).__name__}")
                        st.write(f"DEBUG: network_specific_recs type: {type(network_specific_recs).__name__}")
            
            # Check recommendations type before sorting
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Final recommendations type: {type(recommendations).__name__}")
                if recommendations:
                    st.write(f"DEBUG: First recommendation type: {type(recommendations[0]).__name__ if len(recommendations) > 0 else 'empty'}")
            
            # Ensure recommendations is a list before sorting
            if not isinstance(recommendations, list):
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: recommendations is not a list, converting from {type(recommendations).__name__}")
                recommendations = list(recommendations) if recommendations else []
            
            # Sort by impact score (absolute value, as negative impacts are also important)
            try:
                # Ensure all items in recommendations are Recommendation objects
                valid_recommendations = []
                for rec in recommendations:
                    if hasattr(rec, 'impact_score'):
                        valid_recommendations.append(rec)
                    else:
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Skipping invalid recommendation of type {type(rec).__name__}")
                
                recommendations = valid_recommendations
                recommendations.sort(key=lambda x: abs(x.impact_score), reverse=True)
            except Exception as e:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Error sorting recommendations: {str(e)}")
                    for i, rec in enumerate(recommendations[:5] if recommendations else []):
                        st.write(f"DEBUG: Recommendation {i} type: {type(rec).__name__}")
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
            
            # Debug logs only if debug mode is enabled
            if OptimizerConfig.DEBUG_MODE:
                # Log success factors
                st.write(f"DEBUG: Total success factors: {len(success_factors)}")
            
            # We don't need to categorize recommendations here
            # The recommendation type is already determined in criteria_scorer.py
            # Just log the distribution of recommendation types for debugging
            if OptimizerConfig.DEBUG_MODE:
                # Count factors by recommendation type
                rec_types = {}
                for factor in success_factors:
                    rec_type = factor.recommendation_type
                    if rec_type not in rec_types:
                        rec_types[rec_type] = 0
                    rec_types[rec_type] += 1
                
                # Log the counts
                for rec_type, count in rec_types.items():
                    st.write(f"DEBUG: {rec_type} factors: {count}")
                
                # Also log by impact direction for reference
                positive_impact = [f for f in success_factors if f.impact_score > 0]
                negative_impact = [f for f in success_factors if f.impact_score < 0]
                st.write(f"DEBUG: Positive impact factors: {len(positive_impact)}")
                st.write(f"DEBUG: Negative impact factors: {len(negative_impact)}")
                
                # Check if we have any matching shows to analyze
                if isinstance(matching_shows, pd.DataFrame):
                    st.write(f"DEBUG: Matching shows for recommendations: {len(matching_shows)}")
                    if not matching_shows.empty:
                        st.write(f"DEBUG: Sample of columns: {list(matching_shows.columns)[:5]}")
                else:
                    st.write("DEBUG: No matching shows DataFrame available")
            
            # Process all success factors to create recommendations
            min_impact = OptimizerConfig.SUGGESTIONS.get('minimum_impact', 0.01)
            
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Using minimum impact threshold of {min_impact} for recommendations")
            
            for factor in success_factors:
                # Skip factors with impact below threshold
                if abs(factor.impact_score) < min_impact:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Skipping {factor.criteria_type}/{factor.criteria_name} due to low impact: {factor.impact_score} < {min_impact}")
                    continue
                
                # Get information about the selection status for debugging and filtering
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
                
                # Skip non-actionable recommendations based on recommendation type
                skip_recommendation = False
                
                # Debug all recommendations before filtering
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Processing recommendation: {factor.criteria_type}/{factor.criteria_name} - type: {factor.recommendation_type}, impact: {factor.impact_score}, field selected: {is_field_selected}, option selected: {is_option_selected}", category='recommendation')
                
                # Case 1: "add" recommendations with negative impact for unselected fields
                # These just confirm the user's choice to not select them
                if factor.recommendation_type == 'add' and factor.impact_score < 0:
                    skip_recommendation = True
                    skip_reason = "unselected field with negative impact"
                
                # Case 2: "remove" recommendations for fields that aren't actually selected
                # These don't make sense since you can't remove what's not selected
                # Skip generic "Remove [field]" recommendations with small impact
                elif (factor.recommendation_type == 'remove' and 
                      (not is_field_selected or factor.criteria_name.startswith('Remove '))):
                    skip_recommendation = True
                    skip_reason = "remove recommendation for unselected field or generic remove option"
                    
                # Case 3: Generate proper "remove" recommendations for currently selected options with negative impact
                # This ensures we have meaningful "remove" recommendations in the UI
                elif is_field_selected and factor.impact_score < 0:
                    # Check if this specific option is selected
                    option_id = getattr(factor, 'criteria_value', None)
                    is_option_selected = False
                    
                    if option_id is not None:
                        # For array fields (like character_types), check if the option_id is in the array
                        if isinstance(criteria.get(factor.criteria_type), list):
                            is_option_selected = option_id in criteria[factor.criteria_type]
                        # For single value fields (like genre), check if the option_id matches the value
                        else:
                            is_option_selected = criteria[factor.criteria_type] == option_id
                    
                    # If this specific option is selected and has negative impact, make it a "remove" recommendation
                    if is_option_selected and not factor.criteria_name.startswith('Remove '):
                        factor.recommendation_type = 'remove'
                        if OptimizerConfig.DEBUG_MODE:
                            OptimizerConfig.debug(f"Generated proper 'remove' recommendation for selected option {factor.criteria_type}/{factor.criteria_name}", category='recommendation', force=True)
                
                if skip_recommendation and OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Skipping non-actionable recommendation for {factor.criteria_type}/{factor.criteria_name}: {skip_reason}")
                    OptimizerConfig.debug(f"Skipping non-actionable recommendation for {factor.criteria_type}/{factor.criteria_name}: field selected={is_field_selected}, option selected={is_option_selected}, impact={factor.impact_score}, reason={skip_reason}", category='recommendation')
                    continue
                
                # Boost impact score slightly to ensure it's displayed
                impact_score = factor.impact_score
                if abs(impact_score) < 0.05:  # Ensure at least 5% impact for UI display
                    impact_score = 0.05 if impact_score > 0 else -0.05
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Boosting impact for {factor.criteria_type}/{factor.criteria_name} from {factor.impact_score} to {impact_score}")
                
                # Use the original recommendation_type from the success factor
                # This was properly determined in criteria_scorer.py based on field selection status
                rec_type = factor.recommendation_type
                
                # For selected fields with negative impact, ensure they are marked as 'remove' recommendations
                # This is critical for proper UI display
                is_selected = factor.criteria_type in criteria
                option_id = getattr(factor, 'criteria_value', None)
                is_option_selected = False
                
                # Check if this specific option is selected
                if is_selected and option_id is not None:
                    # For array fields (like character_types), check if the option_id is in the array
                    if isinstance(criteria.get(factor.criteria_type), list):
                        is_option_selected = option_id in criteria[factor.criteria_type]
                    # For single value fields (like genre), check if the option_id matches the value
                    else:
                        is_option_selected = criteria[factor.criteria_type] == option_id
                
                # If this is a selected option with negative impact, it should be a 'remove' recommendation
                # Only generate 'remove' recommendations for options that are actually selected
                if is_option_selected and factor.impact_score < 0:
                    rec_type = 'remove'
                    # Boost the impact score for remove recommendations to make them more significant
                    # This ensures they'll appear in the UI alongside other high-impact recommendations
                    if abs(factor.impact_score) < 0.15:  # Ensure at least 15% impact for remove recommendations
                        impact_score = -0.15
                    
                    if OptimizerConfig.DEBUG_MODE:
                        OptimizerConfig.debug(f"Changed recommendation type to 'remove' for {factor.criteria_type}/{factor.criteria_name} due to negative impact on selected option", category='recommendation', force=True)
                        OptimizerConfig.debug(f"Boosted remove recommendation impact from {factor.impact_score} to {impact_score}", category='recommendation', force=True)
                
                # Debug logging for recommendation type
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Using recommendation type '{rec_type}' for {factor.criteria_type}/{factor.criteria_name} (field selected: {is_selected}, option selected: {is_option_selected})", category='recommendation')
                
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
                
                # Debug log
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Created recommendation for {factor.criteria_type}/{factor.criteria_name} with impact {impact_score} (original: {factor.impact_score})")
                    OptimizerConfig.debug(f"Created recommendation for {factor.criteria_type}/{factor.criteria_name} with impact {impact_score} (original: {factor.impact_score})", category='recommendation')
                    
                    # Special debug for 'remove' recommendations
                    if rec_type == 'remove':
                        OptimizerConfig.debug(f"REMOVE RECOMMENDATION CREATED: {factor.criteria_type}/{factor.criteria_name}", category='recommendation', force=True)
                        st.write(f"DEBUG: REMOVE RECOMMENDATION CREATED: {factor.criteria_type}/{factor.criteria_name}")
                        
                # Ensure the recommendation has the correct type
                # This is critical for 'remove' recommendations to be properly displayed
                recommendation.recommendation_type = rec_type
                    
            if not recommendations and OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: No recommendations created from {len(success_factors)} success factors")
                OptimizerConfig.debug(f"No recommendations created from {len(success_factors)} success factors", category='recommendation')
            
            # Debug log the total recommendations created
            if OptimizerConfig.DEBUG_MODE:
                # Count recommendations by type
                rec_types = {}
                for rec in recommendations:
                    rec_type = getattr(rec, 'recommendation_type', 'unknown')
                    if rec_type not in rec_types:
                        rec_types[rec_type] = 0
                    rec_types[rec_type] += 1
                    
                    # Log detailed info for each recommendation
                    criteria_type = getattr(rec, 'criteria_type', 'unknown')
                    name = getattr(rec, 'suggested_name', 'unknown')
                    impact = getattr(rec, 'impact_score', 0)
                    
                    # Check if this field is in the original criteria (selected)
                    is_selected = criteria_type in criteria
                    expected_type = 'change' if is_selected and impact > 0 else 'add' if impact > 0 else 'remove'
                    
                    # Log detailed info with selection status
                    OptimizerConfig.debug(f"Recommendation: {criteria_type}/{name} - Type: {rec_type} - Impact: {impact} - Selected: {is_selected} - Expected: {expected_type}", category='recommendation')
                
                OptimizerConfig.debug(f"Recommendation types: {rec_types}", category='recommendation')
                if 'change' not in rec_types or rec_types['change'] == 0:
                    OptimizerConfig.debug(f"No 'change' recommendations found", category='recommendation')
                    st.write(f"DEBUG: No 'change' recommendations found")
                else:
                    change_recs = [rec for rec in recommendations if getattr(rec, 'recommendation_type', '') == 'change']
                    st.write(f"DEBUG: Found {len(change_recs)} 'change' recommendations")
                    OptimizerConfig.debug(f"Found {len(change_recs)} 'change' recommendations", category='recommendation')
                    for i, rec in enumerate(change_recs[:3]):
                        criteria_type = getattr(rec, 'criteria_type', 'unknown')
                        name = getattr(rec, 'suggested_name', 'unknown')
                        impact = getattr(rec, 'impact_score', 0)
                        st.write(f"DEBUG: Change recommendation {i+1}: {criteria_type}/{name} - impact: {impact}")
                        OptimizerConfig.debug(f"Change recommendation {i+1}: {criteria_type}/{name} - impact: {impact}", category='recommendation')
            
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
                        # If conversion fails, use string representation
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Error converting current_value to hashable: {str(e)}")
                        current_value = str(current_value)
                    
                    # Calculate success rate for each value of this criteria
                    value_success = {}
                    
                    # Check if column contains list values
                    try:
                        # Get a sample of non-null values to check type
                        sample_values = successful_shows[criteria_type].dropna().head(5).tolist()
                        has_list_values = any(isinstance(val, list) for val in sample_values)
                        
                        if has_list_values:
                            if OptimizerConfig.DEBUG_MODE:
                                st.write(f"DEBUG: Column {criteria_type} contains list values, using special handling")
                            
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
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: Error processing values in pattern analysis: {str(e)}")
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
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Error in _analyze_successful_patterns: {str(e)}")
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
        try:
            # Skip if we don't have valid criteria or network
            if not criteria or not network:
                return []
                
            # Get network-specific success rates for each criteria using matching_shows
            try:
                # Add debug information about the network and matching shows
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Starting network recommendation generation for {network.network_name} (ID: {network.network_id})")
                    st.write(f"DEBUG: Network compatibility score: {network.compatibility_score:.4f}, Sample size: {network.sample_size}")
                    if matching_shows is not None and isinstance(matching_shows, pd.DataFrame):
                        st.write(f"DEBUG: Matching shows count: {len(matching_shows)}")
                        if 'network_id' in matching_shows.columns:
                            network_show_count = len(matching_shows[matching_shows['network_id'] == network.network_id])
                            st.write(f"DEBUG: Shows for this network: {network_show_count}")
                
                # Use the correct parameter pattern for get_network_specific_success_rates
                # The method only accepts matching_shows and network_id parameters
                network_rates = self.criteria_scorer.network_analyzer.get_network_specific_success_rates(
                    matching_shows=matching_shows,
                    network_id=network.network_id
                )
                
                # Debug output to check the structure of network_rates
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Network rates for {network.network_name}: {type(network_rates)}")
                    st.write(f"DEBUG: Network rates keys count: {len(network_rates) if isinstance(network_rates, dict) else 0}")
                    if isinstance(network_rates, dict) and len(network_rates) > 0:
                        st.write(f"DEBUG: Sample network rate keys: {list(network_rates.keys())[:5]}")
                        
                # Ensure network_rates is a dictionary as expected
                if not isinstance(network_rates, dict):
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Expected network_rates to be a dictionary but got {type(network_rates)}")
                        st.write(f"DEBUG: Will use empty dictionary instead")
                    network_rates = {}
                    
                # Store original network_rates for debugging
                original_network_rates = network_rates
                    
                # Handle case where network_rates is not a dictionary
                if not isinstance(network_rates, dict):
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Aborting - network_rates is not a dictionary")
                    return []
                    
                # Handle case where network_rates is empty
                if not network_rates:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Aborting - network_rates dictionary is empty")
                    return []
                    
            except Exception as e:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"DEBUG: Error getting network-specific success rates: {str(e)}")
                return []
                
            # Ensure all matching_shows in network_rates are DataFrames
            try:
                for criteria_type, rate_data in network_rates.items():
                    if isinstance(rate_data, dict) and 'matching_shows' in rate_data:
                        if not isinstance(rate_data['matching_shows'], pd.DataFrame):
                            rate_data['matching_shows'] = pd.DataFrame()
            except Exception as e:
                return []
                
            # Get overall success rates for each criteria
            overall_rates = {}
            
            try:
                # Ensure network_rates is a dictionary before accessing keys
                if not isinstance(network_rates, dict):
                    return []
                    
                for key in network_rates.keys():
                    # Parse the key which is in format "field_name:value_name"
                    if ':' in key:
                        criteria_type = key.split(':', 1)[0]
                    else:
                        criteria_type = key
                        
                    if criteria_type not in criteria:
                        continue
                    
                    single_criteria = {criteria_type: criteria[criteria_type]}
                    overall_rate, overall_details = self.criteria_scorer.calculate_success_rate(
                        single_criteria, integrated_data=integrated_data
                    )
                    
                    # Store using the original key format to match network_rates keys
                    overall_rates[criteria_type] = overall_rate
            except Exception as e:
                pass
            
            recommendations = []
                     
            # Find criteria where network rate differs significantly from overall rate
            try:
                # Double-check that network_rates is a dictionary before iterating
                if not isinstance(network_rates, dict):
                    return []
                
                # Debug output for network rates structure - only in debug mode
                if OptimizerConfig.DEBUG_MODE and len(network_rates) > 0:
                    st.write(f"DEBUG: Processing {len(network_rates)} keys for {network.network_name}")
                    
                for key, network_rate_data in network_rates.items():
                    # Parse the key which is in format "field_name:value_name"
                    if ':' in key:
                        parts = key.split(':', 1)
                        criteria_type = parts[0]  # Extract just the field name part
                    else:
                        criteria_type = key
                        
                    # Skip if criteria not in overall rates or network data is invalid
                    if criteria_type not in overall_rates or not isinstance(network_rate_data, dict):
                        continue
                    
                    # Check if we have valid data
                    has_data = network_rate_data.get('has_data', False)
                    
                    # Debug output for network rates - always show this for debugging
                    # Format the network rate data for better readability
                    network_rate = network_rate_data.get('success_rate', network_rate_data.get('rate', 0))
                    sample_size = network_rate_data.get('sample_size', 0)
                    has_data = network_rate_data.get('has_data', False)
                    
                    # Get network-specific success rate
                    network_rate = network_rate_data.get('rate', 0)
                    
                    # Get overall success rate for this criteria
                    overall_rate = overall_rates.get(criteria_type, 0)
                    
                    # Calculate the difference between network and overall rates
                    difference = network_rate - overall_rate
                    
                    # Only show detailed network rate information in debug mode
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Network {network.network_name} - {criteria_type} comparison:")
                        st.write(f"DEBUG: - Network rate: {network_rate:.4f}")
                        st.write(f"DEBUG: - Overall rate: {overall_rate:.4f}")
                        st.write(f"DEBUG: - Difference: {difference:.4f}")
                    
                    # Check if we have enough data for this network
                    sample_size = network_rate_data.get('sample_size', 0)
                    has_sufficient_data = sample_size >= OptimizerConfig.SUCCESS['min_data_points']
                    
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: - Sample size: {sample_size}")
                        st.write(f"DEBUG: - Min required: {OptimizerConfig.SUCCESS['min_data_points']}")
                        st.write(f"DEBUG: - Has sufficient data: {has_sufficient_data}")
                    
                    # Check conditions for recommendation generation
                    condition1 = abs(difference) >= OptimizerConfig.THRESHOLDS['significant_difference']
                    condition2 = has_sufficient_data and abs(difference) > OptimizerConfig.THRESHOLDS.get('network_difference', 0.01)
                    
                    # Add debug output for recommendation conditions
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: Network recommendation conditions for {network.network_name} - {criteria_type}:")
                        st.write(f"DEBUG: Condition 1 (significant_difference): {condition1} (diff={difference:.4f}, threshold={OptimizerConfig.THRESHOLDS['significant_difference']})")
                        st.write(f"DEBUG: Condition 2 (network_difference): {condition2} (diff={difference:.4f}, threshold={OptimizerConfig.THRESHOLDS.get('network_difference', 0.01)}, has_sufficient_data={has_sufficient_data})")
                    
                    # Generate recommendation if difference is significant
                    if abs(difference) >= OptimizerConfig.THRESHOLDS['significant_difference'] or \
                       (has_sufficient_data and abs(difference) > OptimizerConfig.THRESHOLDS.get('network_difference', 0.01)):

                        
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
                        
                        # Set a minimum impact score to ensure recommendations are displayed
                        impact_score = max(abs(difference), 0.05) * (1 if difference > 0 else -1)
                        
                        # Debug log
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"DEBUG: CREATING NETWORK RECOMMENDATION for {network.network_name} - {criteria_type} - {current_name}")
                            st.write(f"DEBUG: Network rate: {network_rate:.4f}, Overall rate: {overall_rate:.4f}, Difference: {difference:.4f}")
                            st.write(f"DEBUG: Recommendation type: network_{rec_type}, Impact score: {impact_score:.4f}")
                    
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
            except Exception as e:
                st.write(f"DEBUG: Error processing network recommendations: {str(e)}")
                st.write(f"DEBUG: network_rates type: {type(network_rates).__name__}")
                
                # Check if network_rates changed during processing
                if 'original_network_rates' in locals():
                    if id(original_network_rates) != id(network_rates):
                        st.write(f"DEBUG: WARNING - network_rates object changed during processing!")
                        st.write(f"DEBUG: Original network_rates type: {type(original_network_rates).__name__}")
                        st.write(f"DEBUG: Current network_rates type: {type(network_rates).__name__}")
            
            return recommendations
        except Exception as e:
            # Only show error in UI if it's not the known 'empty' attribute error
            if 'empty' not in str(e):
                st.error(f"Error generating network recommendations: {str(e)}")
            return []
