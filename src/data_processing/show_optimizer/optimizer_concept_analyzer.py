"""Concept Analyzer component for Show Optimizer.

This module handles the coordination of the analysis pipeline for show concepts,
integrating results from different analyzers to generate comprehensive optimization summaries.

Key responsibilities:
- Coordinate data flow between components
- Manage sequence of operations (matching → scoring → analysis → recommendations)
- Handle integration of results from different analyzers
- Generate optimization summaries with recommendations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .optimizer_config import OptimizerConfig
from .recommendation_engine import RecommendationEngine
from .models import ComponentScore, NetworkMatch, Recommendation, SuccessFactor, OptimizationSummary
from .optimizer_cache import OptimizerCache
from .optimizer_matcher import Matcher
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer
from .score_calculators import ComponentScore
from .network_analyzer import NetworkMatch
from .suggestion_analyzer import OptimizationSummary
from .recommendation_engine import RecommendationEngine, SuccessFactor, Recommendation


class ConceptAnalyzer:
    """Analyzes show concepts and generates optimization recommendations.
    
    This class coordinates the analysis pipeline for show concepts, including
    matching shows, calculating success probabilities, finding top networks,
    calculating component scores, identifying success factors, and generating
    recommendations.
    """
    
    def __init__(self, shows_analyzer, success_analyzer, matcher, field_manager, criteria_scorer, optimizer_cache=None):
        """Initialize the ConceptAnalyzer.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for show data analysis
            success_analyzer: SuccessAnalyzer instance for success metrics
            matcher: Matcher instance for finding matching shows
            field_manager: FieldManager instance for field mapping
            criteria_scorer: CriteriaScorer instance for scoring components
            optimizer_cache: OptimizerCache instance for caching results (optional)
        """
        self.shows_analyzer = shows_analyzer
        self.success_analyzer = success_analyzer
        self.matcher = matcher
        self.field_manager = field_manager
        self.criteria_scorer = criteria_scorer
        self.config = OptimizerConfig
        self.optimizer_cache = optimizer_cache
        
        # Initialize the recommendation engine
        self.recommendation_engine = RecommendationEngine(
            shows_analyzer=shows_analyzer,
            success_analyzer=success_analyzer,
            field_manager=field_manager,
            criteria_scorer=criteria_scorer
        )
        
        # Initialize matching shows cache
        self._matching_shows_cache = {}
        self._matching_shows_last_update = None
        
    def invalidate_cache(self) -> None:
        """Invalidate the matching shows cache.
        
        This forces a fresh search on the next request.
        """
        self._matching_shows_cache = {}
        self._matching_shows_last_update = None
        st.write("Matching shows cache invalidated")
    
    def display_cache_status(self, show_details: bool = False) -> None:
        """Display cache status information in the Streamlit UI.
        
        Args:
            show_details: Whether to show detailed cache information
        """
        if self.optimizer_cache is None:
            st.write("No OptimizerCache instance available")
            return
            
        # Display OptimizerCache status
        self.optimizer_cache.display_cache_status(show_details)
        
        # Display matching shows cache status
        current_time = datetime.now()
        cache_valid = False
        cache_age = None
        
        if self._matching_shows_last_update is not None:
            cache_age = (current_time - self._matching_shows_last_update).total_seconds()
            cache_valid = cache_age <= self.optimizer_cache.cache_duration
            
        # Create user-friendly status indicator
        cache_status = "✅ Valid" if cache_valid else "⚠️ Expired"
        
        if show_details:
            st.write("**Matching Shows Cache:**", cache_status)
            st.write(f"Cached criteria sets: {len(self._matching_shows_cache)}")
            if self._matching_shows_last_update:
                st.write(f"Last updated: {self._matching_shows_last_update.isoformat()}")
                st.write(f"Age: {round(cache_age) if cache_age else None} seconds")
            else:
                st.write("Not initialized yet")
        else:
            st.write(f"Matching Shows Cache: {cache_status}")
    
    def _create_fallback_summary(self, error_message: str) -> OptimizationSummary:
        """Create a fallback summary when analysis fails.
        
        Args:
            error_message: Error message to include in the summary
            
        Returns:
            Basic OptimizationSummary with error information
        """
        st.error(f"Analysis failed: {error_message}")
        
        # Create placeholder component scores
        component_scores = self._create_placeholder_component_scores()
        
        # Create a minimal summary with error information
        return OptimizationSummary(
            overall_success_probability=None,
            confidence='none',
            top_networks=[],
            component_scores=component_scores,
            recommendations=[],
            success_factors=[],
            matching_titles=[],
            match_level=0,
            match_quality=0.0,
            confidence_score=0.0,
            matching_shows=pd.DataFrame(),
            match_count=0,
            match_counts_by_level={},
            confidence_info={'error': error_message, 'confidence_level': 'none'}
        )
    
    def analyze_concept(self, criteria: Dict[str, Any], force_refresh: bool = False) -> OptimizationSummary:
        """Analyze a show concept and generate optimization recommendations.
        
        This is the main entry point for concept analysis, coordinating the entire
        analysis pipeline from matching to recommendations.
        
        Args:
            criteria: Dictionary of criteria defining the show concept
            force_refresh: Whether to force a refresh of cached data
            
        Returns:
            OptimizationSummary with success probability, recommendations, etc.
        """
        try:
            st.write("Analyzing show concept...")
            
            # Step 1: Find matching shows with fallback
            matching_shows, confidence_info = self._find_matching_shows(criteria, force_refresh=force_refresh)
            
            # Extract match information
            match_count = len(matching_shows) if not matching_shows.empty else 0
            match_level = confidence_info.get('match_level', 0)
            match_quality = confidence_info.get('match_quality', 0.0)
            confidence_score = confidence_info.get('confidence_score', 0.0)
            
            # Calculate match counts by level
            match_counts_by_level = {}
            if not matching_shows.empty and 'match_level' in matching_shows.columns:
                for level in range(1, 5):
                    count = len(matching_shows[matching_shows['match_level'] == level])
                    if count > 0:
                        match_counts_by_level[level] = count
            
            # Step 2: Calculate success probability
            success_probability, confidence = self._calculate_success_probability(criteria, matching_shows)
            
            # Step 3: Find top networks
            top_networks = self._find_top_networks(criteria)
            
            # Step 4: Calculate component scores
            component_scores = self._get_component_scores(criteria)
            
            # Step 5: Identify success factors
            success_factors = self._identify_success_factors(criteria)
            
            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(
                criteria, matching_shows, success_factors, top_networks, confidence_info
            )
            
            # Get matching show titles (up to 100) to include in the summary
            matching_titles = []
            if not matching_shows.empty and 'title' in matching_shows.columns:
                matching_titles = matching_shows['title'].tolist()
                # Limit to 100 titles
                if len(matching_titles) > 100:
                    matching_titles = matching_titles[:100]
            
            # Create and return the optimization summary
            summary = OptimizationSummary(
                overall_success_probability=success_probability,
                confidence=confidence,
                top_networks=top_networks,
                component_scores=component_scores,
                recommendations=recommendations,
                success_factors=success_factors,
                matching_titles=matching_titles,
                match_level=match_level,
                match_quality=match_quality,
                confidence_score=confidence_score,
                matching_shows=matching_shows,
                match_count=match_count,
                match_counts_by_level=match_counts_by_level,
                confidence_info=confidence_info
            )
            
            st.write(f"Analysis complete: Found {match_count} matching shows with confidence level '{confidence}'")
            return summary
            
        except Exception as e:
            st.error(f"Error during concept analysis: {str(e)}")
            import traceback
            # Use st.write for detailed stack trace as it's supplementary information
            st.write(f"Error details: {traceback.format_exc()}")
            return self._create_fallback_summary(f"Unexpected error: {str(e)}")
    
    def _get_criteria_hash(self, criteria: Dict[str, Any]) -> str:
        """Generate a hash for criteria to use as a cache key.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            String hash representing the criteria
        """
        # Create a sorted string representation of the criteria for consistent hashing
        criteria_str = str(sorted([(k, str(v)) for k, v in criteria.items()]))
        return str(hash(criteria_str))
    
    def _is_matching_shows_cache_valid(self, criteria_hash: str, force_refresh: bool = False) -> bool:
        """Check if the matching shows cache is valid for the given criteria.
        
        Args:
            criteria_hash: Hash of the criteria
            force_refresh: If True, ignore cache and return False
            
        Returns:
            True if cache is valid, False otherwise
        """
        # If no optimizer_cache is available, always return False
        if self.optimizer_cache is None:
            return False
            
        try:
            if force_refresh:
                return False
                
            # Check if we have a cache for these criteria
            if (criteria_hash in self._matching_shows_cache and 
                self._matching_shows_last_update is not None):
                
                current_time = datetime.now()
                cache_age = (current_time - self._matching_shows_last_update).total_seconds()
                cache_valid = cache_age <= self.optimizer_cache.cache_duration
                return cache_valid
            
            return False
        except Exception as e:
            st.error(f"Matching shows cache validation error: {str(e)}")
            # If there's an error, assume cache is invalid to force refresh
            return False
    
    def _find_matching_shows(self, criteria: Dict[str, Any], force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Find shows matching the given criteria with fallback strategies.
        
        This method uses caching when available to avoid redundant matching operations.
        
        Args:
            criteria: Dictionary of criteria
            force_refresh: If True, bypass cache and force a new search
            
        Returns:
            Tuple of (matching_shows DataFrame, confidence_info dictionary)
        """
        try:
            # Generate a hash for the criteria to use as cache key
            criteria_hash = self._get_criteria_hash(criteria)
            
            # Check if we have a valid cache for these criteria
            if not force_refresh and self._is_matching_shows_cache_valid(criteria_hash):
                cached_result = self._matching_shows_cache[criteria_hash]
                st.write(f"Using cached matching shows ({len(cached_result[0])} shows)")
                return cached_result
            
            # Get the criteria data from the criteria scorer
            criteria_data = self.criteria_scorer.get_criteria_data()
            
            # Set the criteria data in the matcher
            self.matcher.set_criteria_data(criteria_data)
            
            # Get minimum sample size from config
            min_sample_size = self.config.CONFIDENCE['minimum_sample']
            
            # Find matches with fallback
            st.write("Finding shows matching your criteria...")
            matching_shows, confidence_info = self.matcher.find_matches_with_fallback(
                criteria, criteria_data, min_sample_size
            )
            
            # Log the match results
            match_count = len(matching_shows) if not matching_shows.empty else 0
            st.write(f"Found {match_count} matching shows with confidence level '{confidence_info.get('confidence_level', 'unknown')}'") 
            
            # Cache the results if we have a cache manager
            if self.optimizer_cache is not None:
                self._matching_shows_cache[criteria_hash] = (matching_shows, confidence_info)
                self._matching_shows_last_update = datetime.now()
            
            return matching_shows, confidence_info
            
        except Exception as e:
            st.error(f"Error finding matching shows: {str(e)}")
            # Return empty DataFrame and minimal confidence info
            return pd.DataFrame(), {
                'level': 'none',
                'score': 0,
                'match_quality': 0,
                'sample_size': 0,
                'match_level': 0,
                'confidence_level': 'none'
            }
    
    def _calculate_success_probability(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> Tuple[Optional[float], str]:
        """Calculate the success probability based on matching shows.
        
        Args:
            matching_shows: DataFrame of matching shows
            
        Returns:
            Tuple of (success_probability, confidence_level)
        """
        try:
            st.write("Calculating success probability...")
            
            # If no matching shows, return None
            if matching_shows.empty:
                st.warning("No matching shows found for success probability calculation")
                return None, 'none'
            
            # Use CriteriaScorer to calculate success rate
            success_rate, confidence_info = self.criteria_scorer.calculate_success_rate(
                matching_shows, 
                threshold=self.config.SUCCESS['threshold']
            )
            
            if success_rate is not None:
                confidence_level = confidence_info.get('confidence_level', 'none')
                st.write(f"Success probability: {success_rate:.2f} (confidence: {confidence_level})")
                return success_rate, confidence_level
            
            st.warning("Could not calculate success probability: missing success scores")
            return None, 'none'
            
        except Exception as e:
            st.error(f"Error calculating success probability: {str(e)}")
            return None, 'none'
    
    def _find_top_networks(self, criteria: Dict[str, Any]) -> List[NetworkMatch]:
        """Find top networks compatible with the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            List of NetworkMatch objects sorted by compatibility score
        """
        try:
            st.write("Finding top networks...")
            
            # Get the matching shows that were already found
            matching_shows, confidence_info = self._find_matching_shows(criteria)
            
            # Use CriteriaScorer to analyze network compatibility with the already matched shows
            network_matches = self.criteria_scorer.calculate_network_scores(criteria, matching_shows)
            
            # Take top 5
            top_networks = network_matches[:5] if network_matches else []
            
            st.write(f"Found {len(top_networks)} top networks")
            return top_networks
            
        except Exception as e:
            st.error(f"Error finding top networks: {str(e)}")
            return []
            
    def get_network_specific_recommendations(self, criteria: Dict[str, Any], network: NetworkMatch) -> List[Dict[str, Any]]:
        """Get network-specific recommendations for a given network.
        
        This method coordinates between the NetworkAnalyzer and RecommendationEngine
        to generate network-specific recommendations.
        
        Args:
            criteria: Dictionary of criteria
            network: NetworkMatch object to generate recommendations for
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Access the network_analyzer from criteria_scorer
            network_analyzer = self.criteria_scorer.network_analyzer
            if network_analyzer is None:
                st.write("Warning: NetworkAnalyzer not available. Cannot generate network recommendations.")
                return []
                
            # Use the NetworkAnalyzer to generate recommendations, passing this ConceptAnalyzer
            # so it can access the RecommendationEngine
            return network_analyzer.get_network_recommendations(criteria, network, self)
            
        except Exception as e:
            st.error(f"Error generating network-specific recommendations: {str(e)}")
            return []
    
    def _get_component_scores(self, criteria: Dict[str, Any]) -> Dict[str, ComponentScore]:
        """Calculate component scores for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        try:
            st.write("Analyzing component scores...")
            
            # Get the matching shows that were already found
            matching_shows, confidence_info = self._find_matching_shows(criteria)
            
            # Use CriteriaScorer to calculate component scores with the already matched shows
            component_scores = self.criteria_scorer.calculate_component_scores(
                criteria, matching_shows, confidence_info
            )
            
            if component_scores:
                st.write(f"Analyzed scores for {len(component_scores)} components")
                return component_scores
            else:
                st.warning("No component scores could be calculated")
                return self._create_placeholder_component_scores()
            
        except Exception as e:
            st.error(f"Error calculating component scores: {str(e)}")
            return self._create_placeholder_component_scores()
    
    def _create_placeholder_component_scores(self) -> Dict[str, ComponentScore]:
        """Create placeholder component scores with None values.
        
        Returns:
            Dictionary of placeholder component scores
        """
        return {
            'critics': ComponentScore(component="critics", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
            'audience': ComponentScore(component="audience", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
            'longevity': ComponentScore(component="longevity", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'}),
            'completion': ComponentScore(component="completion", score=None, sample_size=0, confidence='none', details={'status': 'insufficient_data'})
        }
        
    def _get_confidence_level(self, sample_size: int, match_level: int = 1) -> str:
        """Get confidence level based on sample size and match level.
        
        Args:
            sample_size: Number of samples
            match_level: Match level (1-4, where 1 is highest)
            
        Returns:
            Confidence level string from OptimizerConfig
        """
        # Use the OptimizerConfig method for consistent confidence levels
        return self.config.get_confidence_level(sample_size, match_level)
    
    def _identify_success_factors(self, criteria: Dict[str, Any]) -> List[SuccessFactor]:
        """Identify success factors for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            List of SuccessFactor objects
        """
        try:
            # Get matching shows
            matching_shows, _ = self._find_matching_shows(criteria)
            
            # If no matching shows, return empty list
            if matching_shows.empty:
                st.warning("No matching shows found for success factor analysis")
                return []
            
            # Delegate to RecommendationEngine for success factor identification
            return self.recommendation_engine.identify_success_factors(
                criteria=criteria,
                matching_shows=matching_shows,
                limit=5
            )
            
        except Exception as e:
            st.error(f"Error identifying success factors: {str(e)}")
            return []

    def _generate_recommendations(
        self, 
        criteria: Dict[str, Any],
        matching_shows: pd.DataFrame,
        success_factors: List[SuccessFactor],
        top_networks: List[NetworkMatch],
        confidence_info: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations for optimizing the show concept.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of matching shows
            success_factors: List of success factors
            top_networks: List of top networks
            confidence_info: Dictionary of confidence information
            
        Returns:
            List of Recommendation objects
        """
        try:
            st.write("Generating optimization recommendations...")
            
            # Delegate to RecommendationEngine for general recommendation generation
            recommendations = self.recommendation_engine.generate_recommendations(
                criteria=criteria,
                success_factors=success_factors,
                top_networks=top_networks,
                matching_shows=matching_shows,
                confidence_info=confidence_info
            )
            
            # Generate network-specific recommendations for each top network
            if top_networks:
                st.write("Generating network-specific recommendations...")
                for network in top_networks[:3]:  # Limit to top 3 networks
                    try:
                        # Generate network-specific recommendations using the RecommendationEngine directly
                        network_recommendations = self.recommendation_engine.generate_network_specific_recommendations(
                            criteria=criteria,
                            network=network
                        )
                        
                        # Add network-specific recommendations to the list
                        if network_recommendations:
                            recommendations.extend(network_recommendations)
                    except Exception as network_error:
                        st.error(f"Error generating recommendations for network {network.network_name}: {str(network_error)}")
            
            return recommendations
            
        except Exception as e:
            st.warning(f"Could not generate recommendations: {str(e)}")
            return []
