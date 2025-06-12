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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .optimizer_config import OptimizerConfig
from .optimizer_cache import OptimizerCache
from .optimizer_matcher import Matcher
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer
from .score_calculators import ComponentScore, NetworkMatch
from .network_analyzer import NetworkAnalyzer
from .recommendation_engine import RecommendationEngine, SuccessFactor, Recommendation


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
    
    @property
    def network_compatibility(self) -> List[NetworkMatch]:
        """Return the top networks as network compatibility data.
        
        This property ensures backward compatibility with UI code that expects
        a network_compatibility attribute.
        """
        return self.top_networks


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
    
    def _handle_analysis_error(self, error_message: str) -> OptimizationSummary:
        """Handle analysis errors and return a minimal summary.
        
        Args:
            error_message: Error message to display
            
        Returns:
            Minimal OptimizationSummary with error information
        """
        st.error(f"Analysis failed: {error_message}")
        
        # Create a minimal summary with error information
        return OptimizationSummary(
            overall_success_probability=None,
            confidence='none',
            top_networks=[],
            component_scores={},
            recommendations=[],
            success_factors=[],
            matching_titles=[],
            match_level=0,
            match_quality=0.0,
            confidence_score=0.0,
            matching_shows=pd.DataFrame(),
            match_count=0,
            match_counts_by_level={},
            confidence_info={'error': error_message, 'level': 'none'}
        )
    
    def analyze_concept(self, criteria: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame], force_refresh: bool = False) -> OptimizationSummary:
        """Analyze a show concept and generate optimization recommendations.
        
        This is the main entry point for concept analysis, coordinating the entire
        analysis pipeline from matching to recommendations.
        
        Args:
            criteria: Dictionary of criteria defining the show concept
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            force_refresh: Whether to force a refresh of cached data
            
        Returns:
            OptimizationSummary with success probability, recommendations, etc.
        """
        try:
            st.write("Analyzing show concept...")
            
            # Step 1: Find matching shows using integrated data
            matching_shows, confidence_info = self._find_matching_shows(criteria, integrated_data, force_refresh=force_refresh)
            
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
            top_networks = self._find_top_networks(criteria, integrated_data=integrated_data)
            
            # Step 4: Calculate component scores
            component_scores = self._get_component_scores(criteria, matching_shows, integrated_data)
            
            # Step 5: Identify success factors
            success_factors = self._identify_success_factors(criteria, matching_shows, integrated_data)
            
            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(
                criteria, matching_shows, success_factors, top_networks, confidence_info, integrated_data
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
            
            # No backward compatibility or fallbacks - rely solely on top_networks
                
            st.write(f"Analysis complete: Found {match_count} matching shows with confidence level '{confidence}'")
            return summary
            
        except Exception as e:
            st.error(f"Error during concept analysis: {str(e)}")
            import traceback
            # Use st.write for detailed stack trace as it's supplementary information
            st.write(f"Error details: {traceback.format_exc()}")
            return self._handle_analysis_error(f"Unexpected error: {str(e)}")
    
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
    
    def _find_matching_shows(self, criteria: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame], force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Find shows matching the given criteria with fallback strategies.
        
        This method uses caching when available to avoid redundant matching operations.
        
        Args:
            criteria: Dictionary of criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
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
            
            # Get shows data from integrated data
            if 'shows' not in integrated_data or integrated_data['shows'].empty:
                st.error("No shows data available in integrated data")
                return pd.DataFrame(), {'level': 'none', 'error': 'No shows data available'}
                
            shows_data = integrated_data['shows']
            st.write(f"Using integrated data with {len(shows_data)} shows")
            
            # Set the criteria data in the matcher
            self.matcher.set_criteria_data(shows_data)
            
            # Get minimum sample size from config
            min_sample_size = self.config.CONFIDENCE['minimum_sample']
            
            # Find matches with fallback
            st.write("Finding shows matching your criteria...")
            matching_shows, confidence_info = self.matcher.find_matches_with_fallback(
                criteria, shows_data, min_sample_size
            )
            
            # Log the match results
            match_count = len(matching_shows) if not matching_shows.empty else 0
            st.write(f"Found {match_count} matching shows with confidence level '{confidence_info.get('level', 'unknown')}'")
            # Check columns in matching_shows
            
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
                'level': 'none'
            }
    
    def _calculate_success_probability(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> Tuple[Optional[float], str]:
        """Calculate the success probability based on matching shows.
        
        Args:
            matching_shows: DataFrame of matching shows
            
        Returns:
            Tuple of (success_probability, confidence_level)
        """
        try:
            # Calculate success probability
            
            # Always show debug info for this critical calculation
            st.write(f"DEBUG SUCCESS CALC: Received {len(matching_shows)} matching shows for success probability calculation")
            if 'success_score' in matching_shows.columns:
                non_null_count = matching_shows['success_score'].notna().sum()
                st.write(f"DEBUG SUCCESS CALC: Found {non_null_count} shows with non-null success_score values")
                if non_null_count > 0:
                    min_threshold = self.config.SCORE_NORMALIZATION['success_filter_min']
                    above_min = (matching_shows['success_score'] > min_threshold).sum()
                    st.write(f"DEBUG SUCCESS CALC: Found {above_min} shows with success_score > {min_threshold}")
            else:
                st.write("DEBUG SUCCESS CALC: No 'success_score' column found in matching shows")
            
            # If no matching shows, return None
            if matching_shows.empty:
                st.warning("No matching shows found for success probability calculation")
                return None, 'none'
            
            # Use CriteriaScorer to calculate success rate
            st.write("DEBUG SUCCESS CALC: Calling criteria_scorer.calculate_success_rate")
            success_rate, confidence_info = self.criteria_scorer.calculate_success_rate(
                matching_shows, 
                threshold=self.config.PERFORMANCE['success_threshold']
            )
            
            # Always show the result of calculate_success_rate
            st.write(f"DEBUG SUCCESS CALC: calculate_success_rate returned success_rate={success_rate}, confidence_info={confidence_info}")
            
            if success_rate is not None:
                # Get sample size from confidence info or use the number of matching shows
                sample_size = confidence_info.get('sample_size', len(matching_shows))
                st.write(f"DEBUG SUCCESS CALC: Using sample_size={sample_size} for confidence calculation")
                
                # Determine confidence level based on sample size
                if sample_size >= self.config.CONFIDENCE['high_confidence']:
                    confidence_level = 'high'
                elif sample_size >= self.config.CONFIDENCE['medium_confidence']:
                    confidence_level = 'medium'
                elif sample_size >= self.config.CONFIDENCE['low_confidence']:
                    confidence_level = 'low'
                elif sample_size >= self.config.CONFIDENCE['minimum_sample']:
                    confidence_level = 'very_low'
                else:
                    confidence_level = 'none'
                
                # Override with confidence level from confidence_info if available
                if 'confidence' in confidence_info:
                    confidence_level = confidence_info['confidence']
                    st.write(f"DEBUG SUCCESS CALC: Overriding confidence_level with {confidence_level} from confidence_info")
                
                st.write(f"DEBUG SUCCESS CALC: Final success_rate={success_rate}, confidence_level={confidence_level}")
                # Success probability calculated successfully
                return success_rate, confidence_level
            
            st.warning("Could not calculate success probability: missing success scores")
            return None, 'none'
            
        except Exception as e:
            st.error(f"Error calculating success probability: {str(e)}")
            import traceback
            st.write(f"DEBUG SUCCESS CALC ERROR: {traceback.format_exc()}")
            return None, 'none'
    
    def _find_top_networks(self, criteria: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame]) -> List[NetworkMatch]:
        """Find top networks compatible with the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            List of NetworkMatch objects sorted by compatibility score
        """
        try:
            # Validate inputs
            if not criteria:
                if st.session_state.get('debug_mode', False):
                    st.write("Debug: No criteria provided to _find_top_networks")
                return []
                
            if not integrated_data:
                if st.session_state.get('debug_mode', False):
                    st.write("Debug: No integrated data provided to _find_top_networks")
                return []
                
            if st.session_state.get('debug_mode', False):
                st.write("Finding top networks...")
                st.write(f"Debug: Criteria keys: {list(criteria.keys())}")
                st.write(f"Debug: Integrated data keys: {list(integrated_data.keys())}")
            
            # Get the matching shows that were already found
            matching_shows, confidence_info = self._find_matching_shows(criteria, integrated_data=integrated_data)
            
            if matching_shows is None or matching_shows.empty:
                if st.session_state.get('debug_mode', False):
                    st.write("Debug: No matching shows found for network analysis")
                return []
            
            # Use NetworkAnalyzer to rank networks by compatibility
            # The limit is controlled by OptimizerConfig.DEFAULT_NETWORK_LIMIT
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Using cached matching shows ({len(matching_shows)} shows) for network ranking")
                if 'network_id' in matching_shows.columns:
                    network_counts = matching_shows['network_id'].value_counts()
                    st.write(f"Debug: Top 5 networks in matching shows: {network_counts.head()}")
                elif 'network' in matching_shows.columns:
                    network_counts = matching_shows['network'].value_counts()
                    st.write(f"Debug: Top 5 networks in matching shows: {network_counts.head()}")
            
            # Check if network_analyzer is available
            if self.criteria_scorer.network_analyzer is None:
                if st.session_state.get('debug_mode', False):
                    st.write("Debug: NetworkAnalyzer is not available")
                return []
                
            # Call rank_networks_by_compatibility with proper error handling
            network_matches = self.criteria_scorer.network_analyzer.rank_networks_by_compatibility(
                criteria, integrated_data, matching_shows
            )
            
            if network_matches is None:
                if st.session_state.get('debug_mode', False):
                    st.write("Debug: rank_networks_by_compatibility returned None")
                return []
            
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Found {len(network_matches)} top networks")
                if len(network_matches) > 0:
                    st.write(f"Debug: Top network: {network_matches[0].network_name} with compatibility score {network_matches[0].compatibility_score}")
            
            return network_matches
            
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Error finding top networks: {str(e)}")
                import traceback
                st.write(f"Debug: Traceback: {traceback.format_exc()}")
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
    
    def _get_component_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, integrated_data: Dict[str, pd.DataFrame]) -> Dict[str, ComponentScore]:
        """Calculate component scores for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of shows matching the criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        try:
            st.write("Analyzing component scores...")
            
            # Get confidence info from the matching shows
            confidence_info = {'match_level': 1}  # Default to exact match level
            
            # Extract match level from matching_shows if available
            if not matching_shows.empty:
                if 'match_level' in matching_shows.columns:
                    confidence_info['match_level'] = matching_shows['match_level'].max()
                    confidence_info['level'] = 'high' if confidence_info['match_level'] <= 1 else 'medium' if confidence_info['match_level'] <= 2 else 'low'
                # Also include match count for better confidence calculation
                confidence_info['match_count'] = len(matching_shows)
                
                # Add summary of match levels for debugging
                if 'match_level' in matching_shows.columns:
                    level_counts = matching_shows['match_level'].value_counts().to_dict()
                    confidence_info['level_counts'] = level_counts
            
            # Use CriteriaScorer to calculate component scores with the provided matching shows and integrated data
            component_scores = self.criteria_scorer.calculate_component_scores(
                criteria, matching_shows, confidence_info, integrated_data=integrated_data
            )
            
            if component_scores:
                # Component scores analyzed successfully
                return component_scores
            else:
                st.warning("No component scores could be calculated")
                return {}
            
        except Exception as e:
            st.error(f"Error calculating component scores: {str(e)}")
            return {}
    
    
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
    
    def _identify_success_factors(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, integrated_data: Dict[str, pd.DataFrame]) -> List[SuccessFactor]:
        """Identify success factors for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of shows matching the criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            List of SuccessFactor objects
        """
        try:
            # If no matching shows, return empty list
            if matching_shows.empty:
                st.warning("No matching shows found for success factor analysis")
                return []
            
            # Delegate to RecommendationEngine for success factor identification
            return self.recommendation_engine.identify_success_factors(
                criteria=criteria,
                matching_shows=matching_shows,
                integrated_data=integrated_data,
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
        confidence_info: Dict[str, Any],
        integrated_data: Dict[str, pd.DataFrame]
    ) -> List[Recommendation]:
        """Generate recommendations for optimizing the show concept.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of matching shows
            success_factors: List of success factors
            top_networks: List of top networks
            confidence_info: Dictionary of confidence information
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
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
                confidence_info=confidence_info,
                integrated_data=integrated_data
            )
            
            # Generate network-specific recommendations for each top network
            if top_networks:
                st.write("Generating network-specific recommendations...")
                
                # First check if matching_shows is valid to avoid multiple errors
                if matching_shows is None or (isinstance(matching_shows, pd.DataFrame) and matching_shows.empty):
                    st.warning("No matching shows available for network-specific recommendations")
                else:
                    for network in top_networks[:3]:  # Limit to top 3 networks
                        try:
                            # Generate network-specific recommendations using the RecommendationEngine directly
                            network_recommendations = self.recommendation_engine.generate_network_specific_recommendations(
                                criteria=criteria,
                                network=network,
                                matching_shows=matching_shows,
                                integrated_data=integrated_data
                            )
                            
                            # Add network-specific recommendations to the list
                            if network_recommendations:
                                recommendations.extend(network_recommendations)
                        except Exception as network_error:
                            # Use a more specific error message that includes the network name
                            st.error(f"Error generating recommendations for network {network.network_name}: {str(network_error)}")
            
            return recommendations
            
        except Exception as e:
            st.warning(f"Could not generate recommendations: {str(e)}")
            return []
