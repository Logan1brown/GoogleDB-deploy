"""SuggestionAnalyzer: Generates optimization recommendations.

This component analyzes successful component combinations, identifies optimal
content elements, and generates prioritized recommendations based on the
patterns identified by CriteriaAnalyzer.

Key concepts:

1. Combination Analysis:
   - Analyze successful combinations of content elements
   - Identify complementary criteria that work well together
   - Detect criteria conflicts that reduce success probability

2. Recommendation Generation:
   - Generate prioritized recommendations for improving show concepts
   - Provide network-specific optimization suggestions
   - Offer alternative approaches when success probability is low

3. Performance:
   - Cache recommendation results
   - Support incremental updates for real-time UI
   - Handle partial criteria sets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
import numpy as np
import logging
import streamlit as st
from functools import lru_cache
from datetime import datetime, timedelta

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .optimizer_config import OptimizerConfig
from .criteria_scorer import CriteriaScorer, NetworkMatch, ComponentScore
from .criteria_analyzer import CriteriaAnalyzer, SuccessFactor

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A recommendation for optimizing a show concept."""
    recommendation_type: str  # add, remove, replace, consider
    criteria_type: str        # e.g., "genre", "character_types"
    current_value: Any        # Current value (if any)
    suggested_value: Any      # Suggested value
    suggested_name: str       # Display name for suggested value
    impact_score: float       # Expected impact on success (-1 to 1)
    confidence: str           # none, low, medium, high
    explanation: str          # Human-readable explanation
    current_name: str = ""    # Display name for current value
    
    
@dataclass
class OptimizationSummary:
    """Summary of optimization recommendations."""
    overall_success_probability: Optional[float]  # Can be None when data is insufficient
    confidence: str
    top_networks: List[NetworkMatch]
    component_scores: Dict[str, ComponentScore]
    recommendations: List[Recommendation]
    success_factors: List[SuccessFactor]
    matching_titles: List[str] = field(default_factory=list)  # Titles of shows matching all criteria
    match_level: int = 1  # Match level used (1-4, where 1 is highest)
    match_quality: float = 1.0  # Quality of the match (0-1)
    confidence_score: float = 1.0  # Confidence score (0-1)
    matching_shows: Any = None  # DataFrame of matching shows (pandas DataFrame)
    match_count: int = 0  # Number of matching shows
    match_counts_by_level: Dict[int, int] = field(default_factory=dict)  # Count of shows by match level
    confidence_info: Dict[str, Any] = field(default_factory=dict)  # Detailed confidence information


class SuggestionAnalyzer:
    """Analyzes successful component combinations and generates recommendations."""
    
    def __init__(self, shows_analyzer: ShowsAnalyzer, success_analyzer: SuccessAnalyzer):
        """Initialize the suggestion analyzer.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for show data
            success_analyzer: SuccessAnalyzer instance for success metrics
        """
        self.shows_analyzer = shows_analyzer
        self.success_analyzer = success_analyzer
        self.criteria_analyzer = CriteriaAnalyzer(shows_analyzer, success_analyzer)
        self.last_analysis = None
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        
    def analyze_show_concept(self, criteria: Dict[str, Any]) -> OptimizationSummary:
        """Analyze a show concept and generate optimization recommendations.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            OptimizationSummary with success probability, recommendations, etc.
        """
        # Start analysis with the given criteria
        try:
            # Get overall success probability
            try:
                logger.info(f"Analyzing show concept with criteria: {criteria}")
                result = self.criteria_analyzer.get_overall_success_rate(criteria)
                logger.info(f"Got result from get_overall_success_rate: {result}")
                
                # Handle the case where result is a tuple with a tuple as first element
                if isinstance(result, tuple) and len(result) == 2:
                    if isinstance(result[0], tuple) and len(result[0]) == 2:
                        # Extract just the success rate from the nested tuple
                        success_probability = result[0][0]
                        confidence_info = result[0][1]
                        confidence = result[1]
                        logger.info(f"Extracted nested tuple: success_probability={success_probability}, confidence={confidence}")
                    else:
                        # Normal case
                        success_probability, confidence = result
                        logger.info(f"Extracted normal tuple: success_probability={success_probability}, confidence={confidence}")
                else:
                    logger.warning(f"Unexpected result format from get_overall_success_rate: {result}")
                    success_probability, confidence = None, 'none'
            except Exception as e:
                logger.warning(f"Could not calculate success probability: {str(e)}")
                logger.error(f"Could not calculate success probability: {e}", exc_info=True)
                success_probability, confidence = None, 'none'
            
            # Get top networks
            try:
                top_networks = self.criteria_analyzer.rank_networks_by_compatibility(criteria, limit=5)
            except Exception as e:
                st.warning(f"Could not calculate network compatibility: {str(e)}")
                top_networks = []
            
            # Get matching shows with flexible matching
            try:
                matching_shows, match_count, confidence_info = self.criteria_analyzer.criteria_scorer._get_matching_shows(criteria, flexible=True)
                logger.info(f"Found {match_count} matching shows with confidence level {confidence_info.get('level', 'unknown')}")
                logger.info(f"Match level: {confidence_info.get('match_level', 'unknown')}")
                
                # Log the first few matching titles for debugging
                if not matching_shows.empty and 'title' in matching_shows.columns:
                    matching_titles = matching_shows['title'].tolist()
                    logger.info(f"First few matching titles: {matching_titles[:5]}")
                else:
                    logger.info("No matching titles found or 'title' column missing")
                    
            except Exception as e:
                logger.warning(f"Could not find matching shows: {str(e)}")
                matching_shows = pd.DataFrame()
                match_count = 0
                confidence_info = {'level': 'none', 'score': 0, 'match_quality': 0, 'sample_size': 0, 'match_level': 4}
            
            # Store the original exact matches even if there are insufficient matches
            original_matches = matching_shows.copy() if not matching_shows.empty else pd.DataFrame()
            original_match_count = match_count
            original_confidence_info = confidence_info.copy() if isinstance(confidence_info, dict) else {}
            
            # If we have some exact matches but not enough for reliable analysis, get additional matches to supplement
            min_sample_size = self.criteria_analyzer.config.CONFIDENCE['minimum_sample']  # Use config value
            need_supplemental = original_match_count > 0 and original_match_count < min_sample_size
            
            # If no matches found or insufficient matches, try fallback with minimal criteria
            if original_matches.empty or need_supplemental:
                # Try with just genre to get supplemental matches
                supplemental_matches = pd.DataFrame()
                supplemental_count = 0
                supplemental_info = {}
                
                try:
                    # First try with just genre
                    if 'genre' in criteria:
                        minimal_criteria = {'genre': criteria['genre']}
                        genre_shows, genre_count, genre_info = self.criteria_analyzer.criteria_scorer._get_matching_shows(minimal_criteria, flexible=True)
                        
                        # Mark these as supplemental matches with appropriate match level
                        # Level 3 means "Core and primary criteria matched" according to OptimizerConfig
                        # For genre-only matches, we should use level 4 ("Only core criteria matched")
                        if not genre_shows.empty and 'match_level' in genre_shows.columns:
                            # Set all to level 4 to indicate they're genre-only matches
                            genre_shows['match_level'] = 4
                        elif not genre_shows.empty:
                            # Add match_level column if it doesn't exist
                            genre_shows['match_level'] = 4
                        
                        supplemental_matches = genre_shows
                        supplemental_count = genre_count
                        supplemental_info = genre_info
                        supplemental_info['match_level'] = 4  # Mark as minimal match (core criteria only)
                except Exception as e:
                    # Silently handle the error - we can't use logger in deployed app
                    pass
                
                # If we have original matches, combine them with supplemental matches
                if not original_matches.empty and not supplemental_matches.empty:
                    # Ensure both DataFrames have the same columns
                    common_columns = list(set(original_matches.columns).intersection(set(supplemental_matches.columns)))
                    
                    # Make sure match_level is included
                    if 'match_level' not in common_columns:
                        if 'match_level' not in original_matches.columns:
                            original_matches['match_level'] = 1  # Mark as exact match
                        if 'match_level' not in supplemental_matches.columns:
                            supplemental_matches['match_level'] = 4  # Mark as minimal match (core criteria only)
                        common_columns.append('match_level')
                    
                    # Filter supplemental matches to remove any that are already in original_matches
                    if 'id' in common_columns:
                        # Filter by ID if available
                        original_ids = set(original_matches['id'].tolist())
                        supplemental_matches = supplemental_matches[~supplemental_matches['id'].isin(original_ids)]
                    elif 'title' in common_columns:
                        # Filter by title if ID not available
                        original_titles = set(original_matches['title'].tolist())
                        supplemental_matches = supplemental_matches[~supplemental_matches['title'].isin(original_titles)]
                    
                    # Combine the matches, with original matches first
                    combined_matches = pd.concat([original_matches[common_columns], supplemental_matches[common_columns]], ignore_index=True)
                    combined_count = len(combined_matches)
                    
                    # Use the combined results
                    matching_shows = combined_matches
                    match_count = combined_count
                    
                    # Update confidence info to reflect the combined matches
                    confidence_info = original_confidence_info.copy()
                    confidence_info['sample_size'] = combined_count
                    
                    # Create a record of match counts by level for the UI
                    match_counts_by_level = {}
                    for level in range(1, 5):
                        level_count = len(matching_shows[matching_shows['match_level'] == level])
                        if level_count > 0:
                            match_counts_by_level[level] = level_count
                    
                    confidence_info['match_counts_by_level'] = match_counts_by_level
                    
                elif not supplemental_matches.empty:
                    # If we had no original matches, just use the supplemental matches
                    matching_shows = supplemental_matches
                    match_count = supplemental_count
                    confidence_info = supplemental_info
                
                # If still empty, create empty DataFrame with necessary columns
                if matching_shows.empty:
                    # No matches found even with fallback criteria
                    pass
                    matching_shows = pd.DataFrame(columns=['title', 'success_score', 'popcornmeter', 'tomatometer', 
                                                         'tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status', 'match_level'])
                    match_count = 0
                    confidence_info = {
                        'match_level': 0,
                        'match_quality': 0.0,
                        'confidence_score': 0.0,
                        'confidence_level': 'none'
                    }
                
                # If still empty, create empty DataFrame with necessary columns
                if matching_shows.empty:
                    matching_shows = pd.DataFrame(columns=['title', 'success_score', 'popcornmeter', 'tomatometer', 
                                                        'tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status'])
                    match_count = 0
                    confidence_info = {
                        'match_level': 0,
                        'match_quality': 0.0,
                        'confidence_score': 0.0,
                        'confidence_level': 'none'
                    }
                
            # Extract confidence information
            match_level = confidence_info.get('match_level', 1)
            match_quality = confidence_info.get('match_quality', 0.0)
            confidence_score = confidence_info.get('confidence_score', 0.0)
            
            # Set confidence based on match count
            if match_count >= 30:
                confidence = 'high'
            elif match_count >= 15:
                confidence = 'medium'
            elif match_count >= 5:
                confidence = 'low'
            else:
                confidence = 'none'
                
            logger.info(f"Setting confidence to {confidence} based on match_count={match_count}")
            
            # Override with confidence from confidence_info if available
            if 'confidence_level' in confidence_info and confidence_info['confidence_level'] != 'none':
                confidence = confidence_info.get('confidence_level')
                logger.info(f"Overriding confidence with value from confidence_info: {confidence}")
                
            # Get matching show titles (up to 100) to include in the summary
            matching_titles = []
            if 'title' in matching_shows.columns:
                matching_titles = matching_shows['title'].tolist()
                logger.info(f"Found {len(matching_titles)} matching titles: {matching_titles[:5]}...")
                # Limit to 100 titles
                if len(matching_titles) > 100:
                    matching_titles = matching_titles[:100]
                
            # Calculate component scores
            try:
                # Get component scores from the criteria analyzer
                component_scores = self.criteria_analyzer.analyze_components(criteria)
            except Exception as e:
                st.warning(f"Could not calculate component scores: {str(e)}")
                # Create placeholder component scores with None values
                component_scores = {
                    'critics': ComponentScore(component="critics", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
                    'audience': ComponentScore(component="audience", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
                    'longevity': ComponentScore(component="longevity", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
                    'completion': ComponentScore(component="completion", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'})
                }
            
            # Handle component scores gracefully
            import streamlit as st
            if not component_scores:
                st.warning("No component scores available for analysis - using N/A values")
                # Create empty component scores dictionary with None values
                component_scores = {}
                
            # Check for missing component scores and add placeholders
            expected_components = ['audience', 'critics', 'longevity']
            missing_components = [comp for comp in expected_components if comp not in component_scores]
            
            if missing_components:
                st.warning(f"Some component scores are missing and will be shown as N/A: {missing_components}")
                # Add placeholder component scores for missing components
                for comp in missing_components:
                    component_scores[comp] = ComponentScore(
                        component=comp,
                        score=None,  # None will be displayed as N/A
                        sample_size=0,
                        confidence='none',
                        details={'status': 'insufficient_data'}
                    )
            
            # Get success factors
            success_factors = self.criteria_analyzer.identify_success_factors(criteria, limit=5)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(criteria, success_factors, top_networks, matching_shows, confidence_info)
            
            # Create and return the optimization summary
            
            # Calculate match counts by level
            match_counts_by_level = {}
            if not matching_shows.empty and 'match_level' in matching_shows.columns:
                for level in range(1, 5):
                    count = len(matching_shows[matching_shows['match_level'] == level])
                    if count > 0:
                        match_counts_by_level[level] = count
            
            summary = OptimizationSummary(
                overall_success_probability=success_probability,
                confidence=confidence,
                top_networks=top_networks,
                component_scores=component_scores,
                recommendations=recommendations,
                success_factors=success_factors,
                matching_titles=matching_titles,  # List of titles
                match_level=confidence_info.get('match_level', 0),
                match_quality=confidence_info.get('match_quality', 0.0),
                confidence_score=confidence_info.get('confidence_score', 0.0),
                matching_shows=matching_shows,  # The actual DataFrame
                match_count=match_count,
                match_counts_by_level=match_counts_by_level,
                confidence_info=confidence_info
            )
            
            return summary
        except Exception as e:
            # Silently handle the error - we can't use logger in deployed app
            pass
            
            # Create placeholder component scores
            component_scores = {
                'critics': ComponentScore(component="critics", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
                'audience': ComponentScore(component="audience", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
                'longevity': ComponentScore(component="longevity", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
                'completion': ComponentScore(component="completion", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'})
            }
            
            # Return a minimal valid summary without using streamlit
            logger.info("Creating fallback OptimizationSummary in exception handler")
            return OptimizationSummary(
                overall_success_probability=None,
                confidence='none',
                top_networks=[],
                component_scores=component_scores,
                success_factors=[],
                recommendations=[],
                matching_titles=[],
                match_level=0,
                match_quality=0.0,
                confidence_score=0.0,
                matching_shows=pd.DataFrame(),
                match_count=0,
                match_counts_by_level={},
                confidence_info={}
            )
    
    def generate_recommendations(self, criteria: Dict[str, Any], 
                               success_factors: List[SuccessFactor],
                               top_networks: List[NetworkMatch],
                               matching_shows: pd.DataFrame,
                               confidence_info: Dict[str, Any]) -> List[Recommendation]:
        """Generate recommendations for optimizing a show concept.
        
        Args:
            criteria: Dictionary of criteria
            success_factors: List of success factors
            top_networks: List of top matching networks
            matching_shows: DataFrame of matching shows
            confidence_info: Dictionary of confidence information
            
        Returns:
            List of Recommendation objects
        """
        recommendations = []
        
        # Analyze missing high-impact criteria
        missing_criteria_recs = self._recommend_missing_criteria(criteria, success_factors)
        recommendations.extend(missing_criteria_recs)
        
        # Analyze network-specific optimizations
        if top_networks:
            network_recs = self._recommend_network_optimizations(criteria, top_networks[0])
            recommendations.extend(network_recs)
        
        # Analyze criteria conflicts
        conflict_recs = self._recommend_conflict_resolutions(criteria)
        recommendations.extend(conflict_recs)
        
        # Identify limiting criteria that restrict match quality
        if confidence_info.get('match_level', 1) > 1:
            limiting_criteria_recs = self._identify_limiting_criteria(criteria, matching_shows, confidence_info)
            recommendations.extend(limiting_criteria_recs)
        
        # Analyze successful patterns in the matched shows
        pattern_recs = self._analyze_successful_patterns(criteria, matching_shows)
        recommendations.extend(pattern_recs)
        
        # Generate fallback recommendations if needed
        # Only do this if we don't have enough high-quality recommendations already
        if len(recommendations) < OptimizerConfig.SUGGESTIONS['max_suggestions']:
            fallback_recs = self._generate_fallback_recommendations(criteria, matching_shows, confidence_info)
            recommendations.extend(fallback_recs)
        
        # Sort by impact score (absolute value, as negative impacts are also important)
        recommendations.sort(key=lambda x: abs(x.impact_score), reverse=True)
        
        # Limit to max suggestions
        max_suggestions = OptimizerConfig.SUGGESTIONS['max_suggestions']
        if len(recommendations) > max_suggestions:
            recommendations = recommendations[:max_suggestions]
            
        return recommendations
    
    def _recommend_missing_criteria(self, criteria: Dict[str, Any], 
                                  success_factors: List[SuccessFactor]) -> List[Recommendation]:
        """Recommend high-impact criteria that are missing from the concept.
        
{{ ... }}
        Args:
            criteria: Dictionary of criteria
            success_factors: List of success factors
            
        Returns:
            List of Recommendation objects
        """
        recommendations = []
        
        # Get field manager from criteria analyzer
        field_manager = self.criteria_analyzer.criteria_scorer.field_manager
        
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
                options = field_manager.get_options(factor.criteria_type)
                for option in options:
                    if option.id == criteria[factor.criteria_type]:
                        current_name = option.name
                        break
                
                # Get the human-readable field name from the field manager
                field_display_name = factor.criteria_type
                if factor.criteria_type in field_manager.FIELD_CONFIGS:
                    # Use the name from the field config
                    field_config = field_manager.FIELD_CONFIGS[factor.criteria_type]
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
    
    def _recommend_network_optimizations(self, criteria: Dict[str, Any], 
                                       network: NetworkMatch) -> List[Recommendation]:
        """Recommend optimizations specific to a target network.
        
        Args:
            criteria: Dictionary of criteria
            network: Target network
            
        Returns:
            List of Recommendation objects
        """
        recommendations = []
        
        # Get field manager from criteria analyzer
        field_manager = self.criteria_analyzer.criteria_scorer.field_manager
        
        # Get network-specific success rates for each criteria
        network_rates = self.criteria_analyzer.get_network_specific_success_rates(
            criteria, network.network_id
        )
        
        # Get overall success rates for each criteria
        overall_rates = {}
        for criteria_type in network_rates.keys():
            # Create a criteria dict with just this one criteria
            single_criteria = {criteria_type: criteria[criteria_type]}
            result = self.criteria_analyzer.get_overall_success_rate(single_criteria)
            
            # Handle the case where result is a tuple with a tuple as first element
            if isinstance(result, tuple) and len(result) == 2:
                if isinstance(result[0], tuple) and len(result[0]) == 2:
                    # Extract just the success rate from the nested tuple
                    rate = result[0][0]
                else:
                    # Normal case
                    rate, _ = result
            else:
                # Fallback
                rate = None
                
            overall_rates[criteria_type] = rate
        
        # Find criteria where network rate differs significantly from overall rate
        for criteria_type, network_rate_data in network_rates.items():
            if criteria_type not in overall_rates:
                continue
                
            # Skip if we don't have enough data for this criteria
            if not network_rate_data['has_data'] or network_rate_data['rate'] is None:
                continue
                
            # Get the actual rate value from the dictionary
            network_rate = network_rate_data['rate']
            overall_rate = overall_rates[criteria_type]
            
            # Calculate the difference
            difference = network_rate - overall_rate
            
            # If the difference is significant, make a recommendation
            if abs(difference) >= OptimizerConfig.THRESHOLDS['significant_difference']:
                if difference > 0:
                    # This criteria works better for this network than overall
                    recommendation = Recommendation(
                        recommendation_type="consider",
                        criteria_type=criteria_type,
                        current_value=criteria[criteria_type],
                        suggested_value=criteria[criteria_type],  # Same value, just emphasizing it
                        suggested_name=self._get_criteria_name(criteria_type, criteria[criteria_type]),
                        impact_score=difference,
                        confidence="medium",  # Network-specific recommendations have medium confidence
                        explanation=f"'{self._get_criteria_name(criteria_type, criteria[criteria_type])}' works particularly well for {network.network_name}, with {difference:.0%} higher success rate than average."
                    )
                    recommendations.append(recommendation)
                else:
                    # This criteria works worse for this network than overall
                    # Look for alternative values that might work better
                    # Use the existing field_manager from criteria_scorer
                    field_manager = self.criteria_analyzer.criteria_scorer.field_manager
                    options = field_manager.get_options(criteria_type)
                    if options:
                        # Suggest a different option
                        alternative = options[0]  # Default to first option
                        for option in options:
                            if option.id != criteria[criteria_type]:
                                alternative = option
                                break
                                
                        recommendation = Recommendation(
                            recommendation_type="replace",
                            criteria_type=criteria_type,
                            current_value=criteria[criteria_type],
                            suggested_value=alternative.id,
                            suggested_name=alternative.name,
                            impact_score=-difference,  # Convert to positive impact
                            confidence="medium",  # Network-specific recommendations have medium confidence
                            explanation=f"'{self._get_criteria_name(criteria_type, criteria[criteria_type])}' performs {-difference:.0%} worse than average for {network.network_name}. Consider alternatives like '{alternative.name}'."
                        )
                        recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_conflict_resolutions(self, criteria: Dict[str, Any]) -> List[Recommendation]:
        """Recommend resolutions for criteria conflicts.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            List of Recommendation objects
        """
        recommendations = []
        
        # This would require analyzing combinations of criteria to identify conflicts
        # For now, we'll implement a simplified version that looks for known conflicts
        
        # Example: If we have both "comedy" and "horror" genres, that might be a conflict
        if 'genre' in criteria and isinstance(criteria['genre'], list) and len(criteria['genre']) > 1:
            # Check for potential genre conflicts from OptimizerConfig
            genre_conflicts = OptimizerConfig.CRITERIA_CONFLICTS.get('genre', {})
            
            genre_ids = criteria['genre']
            for i, genre1 in enumerate(genre_ids):
                for genre2 in genre_ids[i+1:]:
                    conflict_key = f"{genre1}_{genre2}"
                    reverse_key = f"{genre2}_{genre1}"
                    
                    if conflict_key in genre_conflicts or reverse_key in genre_conflicts:
                        # Found a conflict
                        conflict_score = genre_conflicts.get(conflict_key, genre_conflicts.get(reverse_key, -0.2))
                        
                        # Get genre names
                        genre1_name = self._get_criteria_name('genre', genre1)
                        genre2_name = self._get_criteria_name('genre', genre2)
                        
                        # Recommend removing the less important genre
                        # For simplicity, we'll suggest removing the second one
                        recommendation = Recommendation(
                            recommendation_type="remove",
                            criteria_type='genre',
                            current_value=genre2,
                            suggested_value=None,
                            suggested_name=genre2_name,
                            impact_score=-conflict_score,  # Convert to positive impact
                            confidence="medium",
                            explanation=f"The combination of '{genre1_name}' and '{genre2_name}' may reduce success probability by approximately {-conflict_score:.0%}. Consider focusing on one primary genre."
                        )
                        recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_complementary_criteria(self, criteria: Dict[str, Any]) -> List[Recommendation]:
        """Recommend complementary criteria that work well with existing criteria based on data analysis.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            List of Recommendation objects with data-driven complementary criteria
        """
        # Get matching shows with flexible matching
        matching_shows, match_count, confidence_info = self.criteria_analyzer.criteria_scorer._get_matching_shows(
            criteria, flexible=True)
            
        if matching_shows.empty or match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
            return []
            
        recommendations = []
        
        # Get successful shows from the matching set
        success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
        successful_shows = matching_shows[matching_shows['success_score'] >= success_threshold].copy()
        
        if len(successful_shows) < 5:  # Need at least a few successful shows for analysis
            return []
            
        # Analyze criteria that appear frequently in successful shows but aren't in the input criteria
        for criteria_type in OptimizerConfig.CRITERIA_IMPORTANCE:
            # Skip criteria already specified by the user
            if criteria_type in criteria:
                continue
                
            # Skip criteria not present in the dataset
            if criteria_type not in successful_shows.columns:
                continue
                
            # Count occurrences of each value for this criteria type
            value_counts = successful_shows[criteria_type].value_counts(normalize=True)
            
            # Find values that appear in at least 30% of successful shows
            common_values = value_counts[value_counts >= 0.3]
            
            for value, frequency in common_values.items():
                if pd.isna(value) or value == '':
                    continue
                    
                # Calculate impact score based on frequency and criteria importance
                importance_weight = 1.0
                if criteria_type in OptimizerConfig.CRITERIA_IMPORTANCE:
                    importance = OptimizerConfig.CRITERIA_IMPORTANCE[criteria_type]
                    importance_weight = OptimizerConfig.IMPORTANCE_WEIGHTS.get(importance, 1.0)
                    
                impact_score = frequency * importance_weight * 0.1  # Scale to reasonable impact score
                
                # Skip low-impact recommendations
                if impact_score < OptimizerConfig.SUGGESTIONS['minimum_impact']:
                    continue
                    
                # Get the name of the criteria value
                value_name = self._get_criteria_name(criteria_type, value)
                
                # Create the recommendation
                recommendation = Recommendation(
                    recommendation_type="add",
                    criteria_type=criteria_type,
                    current_value=None,
                    suggested_value=value,
                    suggested_name=value_name,
                    impact_score=impact_score,
                    confidence=confidence_info.get('confidence_level', 'medium'),
                    explanation=f"'{value_name}' appears in {frequency:.0%} of successful shows with similar criteria. "
                               f"Adding this could improve success probability by approximately {impact_score:.0%}."
                )
                recommendations.append(recommendation)
        
        return recommendations
    
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
            test_matches, test_count, test_confidence = self.criteria_analyzer.criteria_scorer._get_matching_shows(
                test_criteria, flexible=True)
                
            # If removing this criterion improves match level or significantly increases sample size
            if (test_confidence.get('match_level', match_level) < match_level or 
                    test_count > len(matching_shows) * 2):  # At least double the sample size
                    
                # Get the name of the criterion for the explanation
                if isinstance(criteria_value, list):
                    criteria_names = [self._get_criteria_name(criteria_type, v) for v in criteria_value]
                    criteria_name = ", ".join(criteria_names)
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
                    confidence=test_confidence.get('confidence_level', 'medium'),
                    explanation=f"Relaxing the '{criteria_name}' requirement would significantly improve match quality "
                               f"and increase the sample size from {len(matching_shows)} to {test_count} shows."
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_successful_patterns(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> List[Recommendation]:
        """Analyze successful patterns in matching shows and generate recommendations.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of matching shows
            
        Returns:
            List of Recommendation objects based on successful patterns
        """
        if matching_shows.empty or len(matching_shows) < OptimizerConfig.CONFIDENCE['minimum_sample']:
            return []
            
        recommendations = []
        
        # Get successful shows from the matching set
        success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
        successful_shows = matching_shows[matching_shows['success_score'] >= success_threshold].copy()
        
        if len(successful_shows) < 5:  # Need at least a few successful shows for analysis
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
                
                # Check if current value is among the most successful values
                value_success = {}
                for value in successful_shows[criteria_type].unique():
                    if pd.isna(value) or value == '':
                        continue
                        
                    # Calculate average success score for shows with this value
                    shows_with_value = successful_shows[successful_shows[criteria_type] == value]
                    avg_success = shows_with_value['success_score'].mean()
                    value_success[value] = avg_success
                    
                # Sort values by success score
                sorted_values = sorted(value_success.items(), key=lambda x: x[1], reverse=True)
                
                # If current value is not in top 3 most successful values, suggest alternatives
                top_values = [v[0] for v in sorted_values[:3]]
                if current_value not in top_values and len(top_values) > 0:
                    # Get the top value
                    top_value = top_values[0]
                    top_success = value_success[top_value]
                    
                    # Calculate impact score based on success difference
                    current_success = value_success.get(current_value, 0)
                    impact_score = (top_success - current_success) * 0.5  # Scale to reasonable impact score
                    
                    # Skip low-impact recommendations
                    if impact_score < OptimizerConfig.SUGGESTIONS['minimum_impact']:
                        continue
                        
                    # Get names
                    current_name = self._get_criteria_name(criteria_type, current_value)
                    suggested_name = self._get_criteria_name(criteria_type, top_value)
                    
                    # Create recommendation
                    recommendation = Recommendation(
                        recommendation_type="change",
                        criteria_type=criteria_type,
                        current_value=current_value,
                        suggested_value=top_value,
                        suggested_name=suggested_name,
                        impact_score=impact_score,
                        confidence="high",
                        explanation=f"Shows with '{suggested_name}' have a {top_success:.0%} success rate compared to "
                                   f"{current_success:.0%} for shows with '{current_name}'. Consider changing this criterion."
                    )
                    recommendations.append(recommendation)
        
        # 2. Look for criteria combinations that appear together frequently in successful shows
        # This is a more complex analysis that would require examining co-occurrence patterns
        # For now, we'll implement a simplified version that looks at pairs of criteria
        
        # Focus on criteria types that are important but not yet specified by the user
        important_criteria = [c for c in OptimizerConfig.CRITERIA_IMPORTANCE 
                             if OptimizerConfig.CRITERIA_IMPORTANCE[c] in ['high', 'medium'] 
                             and c not in criteria 
                             and c in successful_shows.columns]
                             
        # For each criteria in the user's input, look for common pairings in successful shows
        for user_criteria_type, user_criteria_value in criteria.items():
            # Skip if not a content criterion or not in the dataset
            if user_criteria_type not in OptimizerConfig.CRITERIA_IMPORTANCE or user_criteria_type not in successful_shows.columns:
                continue
                
            # For each important criteria not yet specified
            for target_criteria_type in important_criteria:
                # Skip self-pairings
                if target_criteria_type == user_criteria_type:
                    continue
                    
                # Get shows that match the user's criteria for this type
                matching_value_shows = successful_shows[successful_shows[user_criteria_type] == user_criteria_value]
                
                if len(matching_value_shows) < 3:  # Need at least a few matching shows
                    continue
                    
                # Count occurrences of each value for the target criteria type
                value_counts = matching_value_shows[target_criteria_type].value_counts(normalize=True)
                
                # Find values that appear in at least 40% of matching shows (higher threshold for pairings)
                common_values = value_counts[value_counts >= 0.4]
                
                for value, frequency in common_values.items():
                    if pd.isna(value) or value == '':
                        continue
                        
                    # Calculate impact score based on frequency and criteria importance
                    importance = OptimizerConfig.CRITERIA_IMPORTANCE[target_criteria_type]
                    importance_weight = OptimizerConfig.IMPORTANCE_WEIGHTS.get(importance, 1.0)
                    impact_score = frequency * importance_weight * 0.15  # Higher impact for strong pairings
                    
                    # Skip low-impact recommendations
                    if impact_score < OptimizerConfig.SUGGESTIONS['minimum_impact']:
                        continue
                        
                    # Get names
                    user_criteria_name = self._get_criteria_name(user_criteria_type, user_criteria_value)
                    target_value_name = self._get_criteria_name(target_criteria_type, value)
                    
                    # Create recommendation
                    recommendation = Recommendation(
                        recommendation_type="add",
                        criteria_type=target_criteria_type,
                        current_value=None,
                        suggested_value=value,
                        suggested_name=target_value_name,
                        impact_score=impact_score,
                        confidence="high",
                        explanation=f"'{target_value_name}' appears in {frequency:.0%} of successful shows with '{user_criteria_name}'. "
                                   f"This combination has a strong success pattern."
                    )
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _get_relaxed_criteria(self, criteria: Dict[str, Any], relaxation_tier: str) -> List[Dict[str, Any]]:
        """Generate sets of relaxed criteria by removing criteria of the specified importance tier.
        
        Args:
            criteria: Dictionary of criteria to relax
            relaxation_tier: Importance tier to relax ('secondary', 'primary', 'core')
            
        Returns:
            List of dictionaries with relaxed criteria sets
        """
        # Find criteria that belong to the specified relaxation tier
        relaxable_criteria = []
        for criteria_type, criteria_value in criteria.items():
            # Skip if criteria type is not in the importance mapping
            if criteria_type not in OptimizerConfig.CRITERIA_IMPORTANCE:
                continue
                
            # Get the importance level of this criterion
            importance = OptimizerConfig.CRITERIA_IMPORTANCE[criteria_type]
            
            # Add to relaxable list if it matches the tier and is not essential
            if importance == relaxation_tier and importance != 'essential':
                relaxable_criteria.append(criteria_type)
        
        # If no relaxable criteria found, return empty list
        if not relaxable_criteria:
            return []
            
        # Limit the number of criteria to relax based on config
        max_to_relax = min(len(relaxable_criteria), 
                          OptimizerConfig.FALLBACK_SYSTEM['relaxation']['max_criteria_to_relax'])
        
        # Generate relaxed criteria sets by removing one criterion at a time
        relaxed_criteria_sets = []
        
        # First try relaxing individual criteria
        for criteria_type in relaxable_criteria:
            # Create a copy without this criterion
            relaxed = {k: v for k, v in criteria.items() if k != criteria_type}
            relaxed_criteria_sets.append({
                'criteria': relaxed,
                'relaxed_type': criteria_type,
                'relaxed_value': criteria[criteria_type],
                'relaxed_name': self._get_criteria_name(criteria_type, criteria[criteria_type])
            })
            
        # If allowed to relax more than one criterion and we have enough criteria,
        # try combinations of two criteria to relax
        if max_to_relax >= 2 and len(relaxable_criteria) >= 2:
            for i, type1 in enumerate(relaxable_criteria):
                for type2 in relaxable_criteria[i+1:]:
                    # Create a copy without these two criteria
                    relaxed = {k: v for k, v in criteria.items() if k != type1 and k != type2}
                    relaxed_criteria_sets.append({
                        'criteria': relaxed,
                        'relaxed_type': f"{type1},{type2}",
                        'relaxed_value': f"{criteria[type1]},{criteria[type2]}",
                        'relaxed_name': f"{self._get_criteria_name(type1, criteria[type1])} and "
                                      f"{self._get_criteria_name(type2, criteria[type2])}"
                    })
        
        return relaxed_criteria_sets
    
    def _calculate_relevance_score(self, original_criteria: Dict[str, Any], matched_show: pd.Series) -> float:
        """Calculate a weighted relevance score for a show based on how well it matches the original criteria.
        
        Args:
            original_criteria: Dictionary of original criteria
            matched_show: Series representing a show from the database
            
        Returns:
            Relevance score between 0 and 1
        """
        total_weight = 0
        weighted_score = 0
        
        # Get the weight multipliers from config
        weight_multipliers = OptimizerConfig.FALLBACK_SYSTEM['relevance']['weight_multipliers']
        
        # For each criterion in the original criteria
        for criteria_type, criteria_value in original_criteria.items():
            # Skip if criteria type is not in the importance mapping
            if criteria_type not in OptimizerConfig.CRITERIA_IMPORTANCE:
                continue
                
            # Get the importance level and base weight for this criterion
            importance = OptimizerConfig.CRITERIA_IMPORTANCE[criteria_type]
            base_weight = OptimizerConfig.IMPORTANCE_WEIGHTS.get(importance, 1.0)
            
            # Apply the weight multiplier based on importance
            weight = base_weight * weight_multipliers.get(importance, 1.0)
            total_weight += weight
            
            # Check if the show matches this criterion
            if criteria_type in matched_show:
                # For list-type fields (e.g., genres, plot_elements)
                if isinstance(matched_show[criteria_type], list) or isinstance(matched_show[criteria_type], np.ndarray):
                    if criteria_value in matched_show[criteria_type]:
                        weighted_score += weight
                # For scalar fields
                elif matched_show[criteria_type] == criteria_value:
                    weighted_score += weight
        
        # Calculate final relevance score
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.0
    
    def _generate_fallback_recommendations(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, 
                                         confidence_info: Dict[str, Any]) -> List[Recommendation]:
        """Generate fallback recommendations when strict matching yields insufficient results.
        
        This method implements a tiered relaxation strategy combined with weighted relevance scoring:
        1. First tries relaxing secondary criteria
        2. If needed, relaxes primary criteria
        3. As a last resort, relaxes core criteria (never relaxes essential criteria)
        
        For each relaxation tier, it evaluates the quality of matches using weighted relevance scoring.
        
        Args:
            criteria: Dictionary of original criteria
            matching_shows: DataFrame of matching shows (may be empty or have few matches)
            confidence_info: Dictionary with confidence metrics
            
        Returns:
            List of Recommendation objects with fallback suggestions
        """
        recommendations = []
        
        # Check if fallback is needed based on match count and confidence
        match_count = len(matching_shows)
        min_matches = OptimizerConfig.FALLBACK_SYSTEM['relaxation']['min_matches_before_fallback']
        min_confidence = OptimizerConfig.FALLBACK_SYSTEM['relaxation']['min_confidence_before_fallback']
        confidence_level = confidence_info.get('confidence_level', 'none')
        
        # Define confidence level hierarchy for comparison
        confidence_levels = ['none', 'very_low', 'low', 'medium', 'high']
        
        # If we have enough matches and confidence is high enough, no fallback needed
        if match_count >= min_matches and confidence_levels.index(confidence_level) >= confidence_levels.index(min_confidence):
            return []
            
        # Get relaxation tiers from config
        relaxation_tiers = OptimizerConfig.FALLBACK_SYSTEM['relaxation']['relaxation_tiers']
        
        # Try each relaxation tier in order
        for relaxation_tier in relaxation_tiers:
            # Get sets of relaxed criteria for this tier
            relaxed_criteria_sets = self._get_relaxed_criteria(criteria, relaxation_tier)
            
            # If no relaxable criteria in this tier, continue to next tier
            if not relaxed_criteria_sets:
                continue
                
            # Try each relaxed criteria set
            for relaxed_set in relaxed_criteria_sets:
                relaxed_criteria = relaxed_set['criteria']
                relaxed_type = relaxed_set['relaxed_type']
                relaxed_name = relaxed_set['relaxed_name']
                
                # Get matching shows with relaxed criteria
                relaxed_matches, relaxed_count, relaxed_confidence = self.criteria_analyzer.criteria_scorer._get_matching_shows(
                    relaxed_criteria, flexible=True)
                    
                # Check if relaxation improved the situation significantly
                if relaxed_count >= match_count * OptimizerConfig.FALLBACK_SYSTEM['relaxation']['min_sample_increase_factor']:
                    # Calculate relevance scores for the new matches
                    relevance_scores = []
                    for _, show in relaxed_matches.iterrows():
                        score = self._calculate_relevance_score(criteria, show)
                        relevance_scores.append(score)
                    
                    # Add relevance scores to the DataFrame
                    relaxed_matches['relevance_score'] = relevance_scores
                    
                    # Filter by minimum relevance score
                    min_relevance = OptimizerConfig.FALLBACK_SYSTEM['relevance']['min_relevance_score']
                    relevant_matches = relaxed_matches[relaxed_matches['relevance_score'] >= min_relevance]
                    
                    # If we have relevant matches, create a recommendation
                    if not relevant_matches.empty:
                        # Calculate average success probability of relevant matches
                        avg_success = relevant_matches['success_score'].mean() if 'success_score' in relevant_matches.columns else 0.5
                        
                        # Calculate impact score based on sample size increase and relevance
                        sample_increase_factor = len(relevant_matches) / max(1, len(matching_shows))
                        avg_relevance = relevant_matches['relevance_score'].mean()
                        impact_score = (sample_increase_factor - 1) * avg_relevance * avg_success
                        
                        # Create recommendation
                        recommendation = Recommendation(
                            recommendation_type="fallback",
                            criteria_type=relaxed_type,
                            current_value=None,
                            suggested_value=None,
                            suggested_name=f"Relax '{relaxed_name}' requirement",
                            impact_score=impact_score,
                            confidence=relaxed_confidence.get('confidence_level', 'low'),
                            explanation=f"Relaxing the '{relaxed_name}' requirement would increase the sample size "
                                       f"from {match_count} to {len(relevant_matches)} shows with an average relevance "
                                       f"score of {avg_relevance:.2f} and success probability of {avg_success:.2f}."
                        )
                        recommendations.append(recommendation)
                        
                        # If we found a good recommendation in this tier, no need to try more tiers
                        if impact_score > OptimizerConfig.SUGGESTIONS['high_impact_threshold']:
                            return recommendations
        
        # Sort by impact score
        recommendations.sort(key=lambda x: x.impact_score, reverse=True)
        
        return recommendations
    
    def _get_criteria_name(self, criteria_type: str, criteria_value: Any) -> str:
        """Get the display name for a criteria value.
        
        Args:
            criteria_type: Type of criteria
            criteria_value: Value of criteria
            
        Returns:
            Display name for the criteria value
        """
        field_manager = self.criteria_analyzer.criteria_scorer.field_manager
        options = field_manager.get_options(criteria_type)
        
        # Find the option with this ID
        name = str(criteria_value)  # Default if not found
        for option in options:
            if option.id == criteria_value:
                name = option.name
                break
                
        return name
