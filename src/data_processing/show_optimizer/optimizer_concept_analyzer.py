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
from typing import Dict, List, Optional, Tuple, Any, Union, TypedDict, NotRequired
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from .optimizer_config import OptimizerConfig

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .optimizer_matcher import Matcher
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer

# Define explicit data contracts using TypedDict
class CriteriaDict(TypedDict):
    """Explicit contract for criteria dictionary used throughout the optimizer.
    
    This defines the expected structure of criteria dictionaries passed between components.
    All fields are optional since criteria can be partially specified.
    """
    genre: NotRequired[List[str]]
    network_id: NotRequired[Union[int, List[int]]]
    studio_id: NotRequired[Union[int, List[int]]]
    source_type_id: NotRequired[Union[int, List[int]]]
    character_type_ids: NotRequired[List[int]]
    plot_element_ids: NotRequired[List[int]]
    theme_element_ids: NotRequired[List[int]]
    tone_ids: NotRequired[List[int]]
    time_setting_ids: NotRequired[List[int]]
    location_setting_ids: NotRequired[List[int]]
    episode_count: NotRequired[int]
    order_type_id: NotRequired[int]
    team_ids: NotRequired[List[int]]

class ConfidenceInfo(TypedDict):
    """Explicit contract for confidence information dictionary.
    
    This defines the expected structure of confidence info dictionaries used to
    track match quality and confidence levels throughout the analysis pipeline.
    """
    level: str  # 'high', 'medium', 'low', 'very_low', or 'none'
    match_level: NotRequired[float]  # Average match level (1=exact, 2=close, 3=partial)
    match_count: NotRequired[int]  # Number of matching shows
    sample_size: NotRequired[int]  # Sample size used for calculations
    max_match_level: NotRequired[float]  # Maximum match level in the sample
    min_match_level: NotRequired[float]  # Minimum match level in the sample
    mean_match_level: NotRequired[float]  # Mean match level in the sample
    level_counts: NotRequired[Dict[int, int]]  # Counts of each match level
    error: NotRequired[str]  # Error message if applicable

class IntegratedData(TypedDict):
    """Explicit contract for integrated data dictionary.
    
    This defines the expected structure of the integrated data dictionary
    that contains all the dataframes used throughout the analysis pipeline.
    """
    shows: pd.DataFrame  # Main shows dataframe
    networks: NotRequired[pd.DataFrame]  # Networks dataframe
    studios: NotRequired[pd.DataFrame]  # Studios dataframe
    genres: NotRequired[pd.DataFrame]  # Genres dataframe
    success_metrics: NotRequired[pd.DataFrame]  # Success metrics dataframe
    team: NotRequired[pd.DataFrame]  # Team dataframe
from .score_calculators import ComponentScore, NetworkMatch
from .network_analyzer import NetworkAnalyzer
from .recommendation_engine import RecommendationEngine, SuccessFactor, Recommendation


def validate_criteria(criteria: Dict[str, Any]) -> CriteriaDict:
    """Validate and standardize criteria dictionary.
    
    Args:
        criteria: Raw criteria dictionary from user input or other sources
        
    Returns:
        Standardized CriteriaDict with validated fields
    """
    # Create a new dictionary that will match our CriteriaDict structure
    validated: Dict[str, Any] = {}
    
    # Validate and standardize each field if present
    if 'genre' in criteria:
        # Ensure genre is always a list
        if isinstance(criteria['genre'], str):
            validated['genre'] = [criteria['genre']]
        else:
            validated['genre'] = criteria['genre']
    
    # Handle network_id which can be single value or list
    if 'network_id' in criteria:
        if isinstance(criteria['network_id'], list):
            validated['network_id'] = criteria['network_id']
        else:
            validated['network_id'] = criteria['network_id']
    
    # Handle studio_id which can be single value or list
    if 'studio_id' in criteria:
        if isinstance(criteria['studio_id'], list):
            validated['studio_id'] = criteria['studio_id']
        else:
            validated['studio_id'] = criteria['studio_id']
    
    # Handle source_type_id which can be single value or list
    if 'source_type_id' in criteria:
        if isinstance(criteria['source_type_id'], list):
            validated['source_type_id'] = criteria['source_type_id']
        else:
            validated['source_type_id'] = criteria['source_type_id']
    
    # Ensure all *_ids fields are lists
    for field in ['character_type_ids', 'plot_element_ids', 'theme_element_ids', 
                 'tone_ids', 'time_setting_ids', 'location_setting_ids', 'team_ids']:
        if field in criteria:
            if not isinstance(criteria[field], list):
                validated[field] = [criteria[field]]
            else:
                validated[field] = criteria[field]
    
    # Simple scalar fields
    for field in ['episode_count', 'order_type_id']:
        if field in criteria:
            validated[field] = criteria[field]
    
    # Copy any other fields that might be used by extensions
    for key, value in criteria.items():
        if key not in validated:
            validated[key] = value
    
    return validated


def create_default_confidence_info() -> ConfidenceInfo:
    """Create a default confidence info dictionary with standard values.
    
    Returns:
        Default ConfidenceInfo dictionary
    """
    return {
        'level': 'none',
        'match_level': 0.0,
        'match_count': 0,
        'sample_size': 0
    }


def update_confidence_info(base_info: Dict[str, Any], updates: Dict[str, Any]) -> ConfidenceInfo:
    """Update confidence info with new values while maintaining the contract.
    
    Args:
        base_info: Existing confidence info dictionary
        updates: New values to update
        
    Returns:
        Updated ConfidenceInfo dictionary
    """
    # Start with default values if base_info is empty
    if not base_info:
        result = create_default_confidence_info()
    else:
        # Create a copy to avoid modifying the original
        result = dict(base_info)
    
    # Update with new values
    for key, value in updates.items():
        result[key] = value
    
    # Ensure the required 'level' field is present
    if 'level' not in result:
        result['level'] = 'none'
    
    return result


def validate_integrated_data(data: Dict[str, pd.DataFrame]) -> IntegratedData:
    """Validate integrated data dictionary and ensure it has required fields.
    
    Args:
        data: Dictionary of dataframes from various sources
        
    Returns:
        Validated IntegratedData dictionary
        
    Raises:
        ValueError: If required 'shows' dataframe is missing or empty
    """
    if 'shows' not in data or data['shows'].empty:
        raise ValueError("Integrated data must contain a non-empty 'shows' dataframe")
    
    # Create a new dictionary that will match our IntegratedData structure
    validated: Dict[str, pd.DataFrame] = {'shows': data['shows']}
    
    # Copy optional dataframes if present
    for field in ['networks', 'studios', 'genres', 'success_metrics', 'team']:
        if field in data and not data[field].empty:
            validated[field] = data[field]
    
    return validated


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
        
    # Private attribute to store formatted data
    _formatted_data_dict: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def formatted_data(self) -> Dict[str, Any]:
        """Format data for UI display.
        
        Returns a dictionary with formatted networks and recommendations data
        ready for display in the UI.
        """
        # If _formatted_data_dict has been set, return it
        if self._formatted_data_dict:
            return self._formatted_data_dict
            
        # Otherwise generate formatted data
        formatted = {
            'networks': [],
            'recommendations': {
                'general': [],
                'network_specific': []
            }
        }
        
        # Format networks data
        if self.top_networks:
            for network in self.top_networks:
                formatted['networks'].append({
                    'network_id': network.network_id,
                    'network_name': network.network_name,
                    'compatibility_score': network.compatibility_score,
                    'success_probability': network.success_probability,
                    'sample_size': network.sample_size,
                    'confidence': network.confidence
                })
        
        # Format recommendations data
        if self.recommendations:
            for rec in self.recommendations:
                rec_dict = {
                    'recommendation_type': rec.recommendation_type,
                    'criteria_type': rec.criteria_type,
                    'current_value': rec.current_value,
                    'current_name': rec.current_name,
                    'suggested_value': rec.suggested_value,
                    'suggested_name': rec.suggested_name,
                    'impact_score': rec.impact_score,
                    'confidence': rec.confidence,
                    'description': rec.explanation
                }
                
                # Categorize recommendations
                if rec.recommendation_type.startswith('network_'):
                    formatted['recommendations']['network_specific'].append(rec_dict)
                else:
                    formatted['recommendations']['general'].append(rec_dict)
        
        # Store the formatted data for future access
        self._formatted_data_dict = formatted
        return formatted
        
    @formatted_data.setter
    def formatted_data(self, value: Dict[str, Any]):
        """Set formatted data.
        
        Args:
            value: Dictionary with formatted data
        """
        self._formatted_data_dict = value


class ConceptAnalyzer:
    """Analyzes show concepts and generates optimization recommendations.
    
    This class coordinates the analysis pipeline for show concepts, including
    matching shows, calculating success probabilities, finding top networks,
    calculating component scores, identifying success factors, and generating
    recommendations.
    """
    
    def __init__(self, shows_analyzer, success_analyzer, field_manager, criteria_scorer):
        """Initialize the ConceptAnalyzer.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for show data analysis
            success_analyzer: SuccessAnalyzer instance for success metrics
            field_manager: FieldManager instance for field mapping
            criteria_scorer: CriteriaScorer instance for scoring components

        """
        self.shows_analyzer = shows_analyzer
        self.success_analyzer = success_analyzer
        self.field_manager = field_manager
        self.criteria_scorer = criteria_scorer
        self.config = OptimizerConfig

        
        # Initialize the recommendation engine
        self.recommendation_engine = RecommendationEngine(
            shows_analyzer=shows_analyzer,
            success_analyzer=success_analyzer,
            field_manager=field_manager,
            criteria_scorer=criteria_scorer
        )
        

    
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
    
    def analyze_concept(self, criteria: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame]) -> OptimizationSummary:
        """Analyze a show concept and generate optimization recommendations.
        
        This is the main entry point for concept analysis, coordinating the entire
        analysis pipeline from matching to recommendations.
        
        Args:
            criteria: Dictionary of criteria defining the show concept
            integrated_data: Dictionary of integrated data frames from ShowOptimizer

            
        Returns:
            OptimizationSummary with success probability, recommendations, etc.
        """
        try:
            
            # Step 1: Find matching shows using integrated data
            matching_shows, confidence_info = self._find_matching_shows(criteria, integrated_data)
            
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
            
            # Step 3: Find top networks - pass the existing matching_shows to avoid redundant matching
            top_networks = self._find_top_networks(criteria, integrated_data=integrated_data, matching_shows=matching_shows)
            
            # Step 4: Calculate component scores
            component_scores = self._get_component_scores(criteria, matching_shows, integrated_data)
            
            # Step 5: Identify success factors
            success_factors = self._identify_success_factors(criteria, matching_shows, integrated_data)
            
            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(
                criteria, matching_shows, success_factors, top_networks, confidence_info, integrated_data
            )
            
            # Get matching show titles (up to MAX_RESULTS) to include in the summary
            matching_titles = []
            if not matching_shows.empty and 'title' in matching_shows.columns:
                matching_titles = matching_shows['title'].tolist()
                # Limit to MAX_RESULTS titles
                if len(matching_titles) > self.config.MAX_RESULTS:
                    matching_titles = matching_titles[:self.config.MAX_RESULTS]
            
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
            
            # Explicitly access formatted_data to ensure it's populated before returning
            # This ensures the UI will have access to properly formatted recommendations
            formatted = summary.formatted_data
            
            # Log summary creation through centralized debug method
            # Debug removed for clarity
            return summary
            
        except Exception as e:
            st.error(f"Error during concept analysis: {str(e)}")

            return self._handle_analysis_error(f"Unexpected error: {str(e)}")
    

    
    def _find_matching_shows(self, criteria: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, ConfidenceInfo]:
        """Find shows matching the given criteria with fallback strategies.
        
        Args:
            criteria: Dictionary of criteria (will be validated to CriteriaDict)
            integrated_data: Dictionary of integrated data frames (will be validated to IntegratedData)
            
        Returns:
            Tuple of (matching_shows DataFrame, confidence_info dictionary conforming to ConfidenceInfo)
        """
        try:
            # Validate inputs using our helper methods
            validated_criteria = validate_criteria(criteria)
            validated_data = validate_integrated_data(integrated_data)
            
            shows_data = validated_data['shows']
            
            # Get minimum sample size from config
            min_sample_size = self.config.CONFIDENCE['minimum_sample']
            
            # Find matches with fallback using the criteria_scorer's matcher
            if hasattr(self.criteria_scorer, 'matcher') and self.criteria_scorer.matcher is not None:
                # Set the criteria data in the matcher
                self.criteria_scorer.matcher.set_criteria_data(shows_data)
                
                # Use the matcher from criteria_scorer
                matching_shows, confidence_info = self.criteria_scorer.matcher.find_matches_with_fallback(
                    validated_criteria, shows_data, min_sample_size
                )
                
                # Ensure confidence_info follows our contract
                confidence_info = update_confidence_info(confidence_info, {})
                
                # Log the match results
                match_count = len(matching_shows) if not matching_shows.empty else 0
                
                # Update match count in confidence info
                confidence_info = update_confidence_info(confidence_info, {'match_count': match_count})
                
                # Log match results through config debug
                self.config.debug(
                    f"Found {match_count} matching shows with confidence level: {confidence_info['level']}", 
                    category='matching'
                )
                
                return matching_shows, confidence_info
            else:
                # Create an error confidence info using our helper
                error_info: ConfidenceInfo = create_default_confidence_info()
                error_info['error'] = 'No matcher available'
                error_info['level'] = 'none'  # Explicitly set confidence level
                
                st.error("No matcher available in CriteriaScorer. Cannot find matching shows.")
                return pd.DataFrame(), error_info
                
        except ValueError as e:
            # Handle validation errors
            error_info: ConfidenceInfo = create_default_confidence_info()
            error_info['error'] = str(e)
            error_info['level'] = 'none'  # Explicitly set confidence level
            
            st.error(f"Error finding matching shows: {str(e)}")
            return pd.DataFrame(), error_info
        
    def _calculate_success_probability(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> Tuple[Optional[float], str]:
        """Calculate the success probability based on matching shows.
        
        Args:
            criteria: Dictionary of criteria (should be validated with validate_criteria)
            matching_shows: DataFrame of matching shows from _find_matching_shows
            
        Returns:
            Tuple of (success_probability, confidence_level)
            where confidence_level is one of: 'high', 'medium', 'low', 'very_low', 'none'
        """
        try:
            # If no matching shows, return None with appropriate logging
            if matching_shows.empty:
                self.config.debug("No matching shows found for success probability calculation", category='success')
                return None, 'none'
            
            # Use CriteriaScorer to calculate all scores including success rate
            # We use an empty criteria dict since we're just calculating success rate for matching shows
            all_scores = self.criteria_scorer.calculate_scores(
                {}, 
                matching_shows
            )
            
            # Extract success rate and info from the comprehensive scores
            success_rate = all_scores.get('success_rate')
            confidence_info = all_scores.get('success_info', {})
            
            if success_rate is not None:
                # Get sample size from confidence info or use the number of matching shows
                sample_size = confidence_info.get('sample_size', len(matching_shows))
                
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
                
                # Success probability calculated successfully
                return success_rate, confidence_level
            
            self.config.debug("Could not calculate success probability: missing success scores", category='success')
            return None, 'none'
            
        except Exception as e:
            st.error(f"Error calculating success probability: {str(e)}")
            return None, 'none'
    
    def _find_top_networks(self, criteria: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame], matching_shows: pd.DataFrame = None) -> List[NetworkMatch]:
        """Find top networks compatible with the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            matching_shows: Optional DataFrame of shows already matched to criteria
            
        Returns:
            List of NetworkMatch objects sorted by compatibility score
        """
        try:
            # Validate inputs
            if not criteria:
                return []
                
            if not integrated_data:
                return []
                          
            # Use the matching shows that were passed in, or get them if not provided
            if matching_shows is None or matching_shows.empty:
                # Only call _find_matching_shows if we don't already have matching shows
                matching_shows, confidence_info = self._find_matching_shows(criteria, integrated_data=integrated_data)
                
                if matching_shows is None or matching_shows.empty:
                    pass
                    return []
            
            # Use NetworkAnalyzer to rank networks by compatibility
            # The limit is controlled by OptimizerConfig.DEFAULT_NETWORK_LIMIT
            if 'network_id' in matching_shows.columns:
                network_counts = matching_shows['network_id'].value_counts()
            elif 'network' in matching_shows.columns:
                network_counts = matching_shows['network'].value_counts()
            
            # Check if network_analyzer is available
            if self.criteria_scorer.network_analyzer is None:
                pass
                return []
                
            # Call rank_networks_by_compatibility with proper error handling
            # The method only accepts matching_shows and an optional limit parameter
            network_matches = self.criteria_scorer.network_analyzer.rank_networks_by_compatibility(
                matching_shows
            )
            
            if network_matches is None:
                return []
                
            return network_matches
            
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
                st.warning("NetworkAnalyzer not available. Cannot generate network recommendations.")
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
            
            # Get confidence info from the matching shows
            confidence_info = {'match_level': 1}  # Default to exact match level
            
            # Extract match level statistics from matching_shows if available
            if not matching_shows.empty:
                if 'match_level' in matching_shows.columns:
                    # Store the distribution of match levels but don't override the original match_level
                    # that was set by the matcher
                    confidence_info['max_match_level'] = matching_shows['match_level'].max()
                    confidence_info['min_match_level'] = matching_shows['match_level'].min()
                    confidence_info['mean_match_level'] = matching_shows['match_level'].mean()
                    
                    # Only set the confidence level if it's not already set
                    if 'level' not in confidence_info:
                        match_level = confidence_info.get('match_level', confidence_info['max_match_level'])
                        confidence_info['level'] = 'high' if match_level <= 1 else 'medium' if match_level <= 2 else 'low'
                # Also include match count for better confidence calculation
                confidence_info['match_count'] = len(matching_shows)
                
                # Add summary of match levels for debugging
                if 'match_level' in matching_shows.columns:
                    level_counts = matching_shows['match_level'].value_counts().to_dict()
                    confidence_info['level_counts'] = level_counts
            
            # Use CriteriaScorer to calculate all scores with the provided matching shows and integrated data
            all_scores = self.criteria_scorer.calculate_scores(
                criteria, matching_shows, integrated_data=integrated_data
            )
            
            if all_scores and 'component_scores' in all_scores:
                # Component scores analyzed successfully
                return all_scores['component_scores']
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
            
            # Delegate to RecommendationEngine for general recommendation generation
            recommendations = self.recommendation_engine.generate_recommendations(                criteria=criteria,
                success_factors=success_factors,
                top_networks=top_networks,
                matching_shows=matching_shows,
                confidence_info=confidence_info,
                integrated_data=integrated_data
            )
            
            # Generate network-specific recommendations for each top network
            if top_networks:
                OptimizerConfig.debug(f"Processing network recommendations for top networks")
                
                # First check if matching_shows is valid to avoid multiple errors
                if matching_shows is None or (isinstance(matching_shows, pd.DataFrame) and matching_shows.empty):
                    st.warning("No matching shows available for network-specific recommendations")
                else:
                    # Log network recommendation processing through config debug
                    factor_count = len(success_factors) if success_factors else 0
                    show_count = len(matching_shows) if isinstance(matching_shows, pd.DataFrame) else 0
                    network_count = min(3, len(top_networks))
                    
                    self.config.debug(f"Processing recommendations for top {network_count} networks with {factor_count} success factors and {show_count} matching shows", category='recommendation')
                    
                    for network in top_networks[:3]:  # Limit to top 3 networks
                        try:
                            # Get network-specific shows for analysis
                            network_shows = matching_shows[matching_shows['network_id'] == network.network_id] if isinstance(matching_shows, pd.DataFrame) else None
                            show_count = len(network_shows) if network_shows is not None else 0
                            self.config.debug(f"Analyzing network {network.network_name} with {show_count} matching shows", category='network')
                            
                            # Generate network-specific recommendations using the RecommendationEngine directly
                            network_recommendations = self.recommendation_engine.generate_network_specific_recommendations(
                                criteria=criteria,
                                network=network,
                                matching_shows=matching_shows,
                                integrated_data=integrated_data
                            )
                            
                            # Process network recommendations
                            if network_recommendations:
                                self.config.debug(f"Generated {len(network_recommendations)} recommendations for network {network.network_name}", category='recommendation')
                                recommendations.extend(network_recommendations)
                            else:
                                self.config.debug(f"No recommendations generated for network {network.network_name}", category='recommendation')
                        except Exception as network_error:
                            # Use a more specific error message that includes the network name
                            st.error(f"Error generating recommendations for network {network.network_name}: {str(network_error)}")
                            import traceback
                            self.config.debug(f"Network recommendation error: {traceback.format_exc()}", category='recommendation', force=True)
            
            return recommendations
            
        except Exception as e:
            st.warning(f"Could not generate recommendations: {str(e)}")
            return []
