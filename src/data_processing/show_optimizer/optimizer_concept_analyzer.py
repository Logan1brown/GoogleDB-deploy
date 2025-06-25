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
import traceback
from datetime import datetime
from .optimizer_config import OptimizerConfig

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .optimizer_matcher import Matcher
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer
from .optimizer_data_contracts import CriteriaDict, ConfidenceInfo, IntegratedData, NetworkMatch, validate_criteria, validate_integrated_data, create_default_confidence_info, update_confidence_info

# Data contracts are now imported from optimizer_data_contracts.py
from .score_calculators import ComponentScore
from .network_analyzer import NetworkAnalyzer
from .recommendation_engine import RecommendationEngine, SuccessFactor, Recommendation


# Helper functions are now imported from optimizer_data_contracts.py


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
    confidence_info: ConfidenceInfo = field(default_factory=dict)  # Detailed confidence information
    
    @property
    def network_compatibility(self) -> List[NetworkMatch]:
        """Return the top networks as network compatibility data.
        
        This property ensures backward compatibility with UI code that expects
        a network_compatibility attribute.
        """
        return self.top_networks
        
    # Private attribute to store formatted data
    _formatted_data_dict: Dict[str, Union[float, str, List[Dict[str, Any]]]] = field(default_factory=dict)
    
    @property
    def formatted_data(self) -> Dict[str, Union[float, str, List[Dict[str, Any]]]]:
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
            },
            'component_scores': self.component_scores if self.component_scores else {}
        }
        
        # Format networks data using direct attribute access
        if self.top_networks:
            for network in self.top_networks:
                network_data = {
                    'network_id': network.network_id,
                    'network_name': network.network_name,
                    'compatibility_score': network.compatibility_score,
                    'success_probability': network.success_probability,
                    'sample_size': network.sample_size,
                    'confidence': network.confidence
                }
                formatted['networks'].append(network_data)
        
        # Format recommendations data - handle both dictionary and object access
        if self.recommendations:
            for rec in self.recommendations:
                # Check if rec is a dictionary or an object
                if isinstance(rec, dict):
                    # Dictionary-style access for TypedDict
                    rec_type = rec.get('recommendation_type', '')
                    rec_dict = {
                        'recommendation_type': rec_type,
                        'criteria_type': rec.get('field', ''),
                        'current_value': rec.get('current_value', None),
                        'current_name': rec.get('current_name', ''),
                        'suggested_value': rec.get('suggested_value', None),
                        'suggested_name': rec.get('suggested_name', ''),
                        'impact_score': rec.get('impact', 0.0),
                        'confidence': rec.get('confidence', 'none'),
                        'description': rec.get('explanation', 'No explanation available')
                    }
                    
                    # Categorize recommendations
                    if rec_type.startswith('network_'):
                        formatted['recommendations']['network_specific'].append(rec_dict)
                    else:
                        formatted['recommendations']['general'].append(rec_dict)
                else:
                    # For object-style access (e.g., NetworkMatch)
                    # Skip objects that don't have the necessary attributes
                    if not hasattr(rec, 'recommendation_type'):
                        continue
                        
                    rec_type = rec.recommendation_type
                    rec_dict = {
                        'recommendation_type': rec_type,
                        'criteria_type': getattr(rec, 'field', ''),
                        'current_value': getattr(rec, 'current_value', None),
                        'current_name': getattr(rec, 'current_name', ''),
                        'suggested_value': getattr(rec, 'suggested_value', None),
                        'suggested_name': getattr(rec, 'suggested_name', ''),
                        'impact_score': getattr(rec, 'impact', 0.0),
                        'confidence': getattr(rec, 'confidence', 'none'),
                        'description': getattr(rec, 'explanation', 'No explanation available')
                    }
                    
                    # Categorize recommendations
                    if rec_type.startswith('network_'):
                        formatted['recommendations']['network_specific'].append(rec_dict)
                    else:
                        formatted['recommendations']['general'].append(rec_dict)
        
        # Store the formatted data for future access
        self._formatted_data_dict = formatted
        return formatted
        
    @formatted_data.setter
    def formatted_data(self, value: Dict[str, Union[float, str, List[Dict[str, Any]]]]):
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
        
        # Create default component scores to prevent UI errors
        default_component_scores = {
            'audience': {'score': 0.0, 'confidence': 'none', 'sample_size': 0},
            'concept': {'score': 0.0, 'confidence': 'none', 'sample_size': 0},
            'execution': {'score': 0.0, 'confidence': 'none', 'sample_size': 0},
            'marketability': {'score': 0.0, 'confidence': 'none', 'sample_size': 0},
            'overall': {'score': 0.0, 'confidence': 'none', 'sample_size': 0}
        }
        
        # Create a minimal summary with error information
        return OptimizationSummary(
            overall_success_probability=None,
            confidence='none',
            top_networks=[],
            component_scores=default_component_scores,
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
    
    def analyze_concept(self, criteria: CriteriaDict, integrated_data: IntegratedData) -> OptimizationSummary:
        """Analyze a show concept and generate optimization recommendations.
        
        This is the main entry point for concept analysis, coordinating the entire
        analysis pipeline from matching to recommendations.
        
        Args:
            criteria: Dictionary of criteria defining the show concept conforming to CriteriaDict
            integrated_data: Dictionary of integrated data frames from ShowOptimizer conforming to IntegratedData
            
        Returns:
            OptimizationSummary with success probability, recommendations, etc.
        """
        try:
            if OptimizerConfig.DEBUG_MODE:
                st.write("DEBUG: Starting analyze_concept method")
            
            # Step 1: Find matching shows using integrated data
            if OptimizerConfig.DEBUG_MODE:
                st.write("DEBUG: Finding matching shows")
            matching_shows, confidence_info = self._find_matching_shows(criteria, integrated_data)
            
            # Store matching_shows for later use in get_network_specific_recommendations
            self._last_matching_shows = matching_shows
            
            # Extract match information
            match_count = len(matching_shows) if not matching_shows.empty else 0
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Found {match_count} matching shows")
                st.write(f"DEBUG: Confidence info: {confidence_info}")
                st.write(f"DEBUG: Matching shows columns: {matching_shows.columns.tolist() if not matching_shows.empty else []}")

            
            # Safely extract match_level with detailed error tracing
            try:
                match_level_value = confidence_info.get('match_level')
                OptimizerConfig.debug(f"match_level_value type: {type(match_level_value)}, value: {match_level_value}", category='analyzer', force=True)
                
                if match_level_value is None:
                    match_level = 0
                    OptimizerConfig.debug(f"match_level is None, defaulting to 0", category='analyzer', force=True)
                elif isinstance(match_level_value, (int, float)):
                    match_level = match_level_value
                    OptimizerConfig.debug(f"match_level is numeric: {match_level}", category='analyzer', force=True)
                elif hasattr(match_level_value, 'match_level'):
                    match_level = match_level_value.match_level
                    OptimizerConfig.debug(f"match_level extracted from object: {match_level}", category='analyzer', force=True)
                else:
                    # If we get here, we have an object without match_level attribute
                    OptimizerConfig.debug(f"match_level_value has unexpected type: {type(match_level_value)}", category='analyzer', force=True)
                    match_level = 0
            except Exception as e:
                OptimizerConfig.debug(f"Error extracting match_level: {str(e)}\nTraceback: {traceback.format_exc()}", category='analyzer', force=True)
                match_level = 0
                    
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
            if OptimizerConfig.DEBUG_MODE:
                st.write("DEBUG: Calculating success probability")
            success_probability, confidence = self._calculate_success_probability(criteria, matching_shows)
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Success probability: {success_probability}, confidence: {confidence}")
            
            # Step 3: Find top networks - pass the existing matching_shows to avoid redundant matching
            if OptimizerConfig.DEBUG_MODE:
                st.write("DEBUG: Finding top networks")
            top_networks = self._find_top_networks(criteria, integrated_data=integrated_data, matching_shows=matching_shows)
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Found {len(top_networks)} top networks")
                for i, network in enumerate(top_networks):
                    st.write(f"DEBUG: Network {i+1}: {network.network_name} (ID: {network.network_id})")
                    st.write(f"DEBUG: Network {i+1} type: {type(network).__name__}")
            
            # Step 4: Calculate component scores
            if OptimizerConfig.DEBUG_MODE:
                st.write("DEBUG: Calculating component scores")
            component_scores = self._get_component_scores(criteria, matching_shows, integrated_data)
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Component scores: {component_scores}")
            
            # Step 5: Identify success factors
            if OptimizerConfig.DEBUG_MODE:
                st.write("DEBUG: Identifying success factors")
            success_factors = self._identify_success_factors(criteria, matching_shows, integrated_data)
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG: Found {len(success_factors)} success factors")
            
            # Step 6: Generate recommendations
            if OptimizerConfig.DEBUG_MODE:
                st.write("DEBUG: Generating recommendations")
            try:
                recommendations = self._generate_recommendations(
                    criteria, matching_shows, success_factors, top_networks, confidence_info, integrated_data
                )
            except Exception as rec_error:
                import traceback
                error_traceback = traceback.format_exc()
                st.write("### Detailed Error in Recommendation Generation")
                st.error(f"Error: {str(rec_error)}")
                st.code(error_traceback)
                
                # Log the error for debugging
                OptimizerConfig.debug(f"RECOMMENDATION ERROR: {str(rec_error)}", category='error', force=True)
                OptimizerConfig.debug(f"TRACEBACK: {error_traceback}", category='error', force=True)
                
                # Continue with empty recommendations
                recommendations = []
            
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
            error_msg = f"Error in analyze_concept: {str(e)}"
            trace = traceback.format_exc()
            OptimizerConfig.debug(f"{error_msg}\n{trace}", category='analyzer', force=True)
            st.error(error_msg)

            return self._handle_analysis_error(f"Analysis error: {str(e)}")
    

    
    def _find_matching_shows(self, criteria: CriteriaDict, integrated_data: IntegratedData) -> Tuple[pd.DataFrame, ConfidenceInfo]:
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
        
    def _calculate_success_probability(self, criteria: CriteriaDict, matching_shows: pd.DataFrame) -> Tuple[Optional[float], str]:
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
    
    def _find_top_networks(self, criteria: CriteriaDict, integrated_data: IntegratedData, matching_shows: pd.DataFrame = None) -> List[NetworkMatch]:
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
            
    def get_network_specific_recommendations(self, criteria: CriteriaDict, network: NetworkMatch) -> List[Dict[str, Any]]:
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
            network_analyzer = self.criteria_scorer.network_analyzer
            if network_analyzer is None:
                st.warning("NetworkAnalyzer not available. Cannot generate network recommendations.")
                return []
            
            # Get matching shows for the criteria from the most recent analysis
            # This is stored during analyze_concept method execution
            matching_shows = None
            if hasattr(self, '_last_matching_shows') and self._last_matching_shows is not None:
                matching_shows = self._last_matching_shows
            else:
                # If we don't have matching shows cached, we can't generate recommendations
                st.warning("No matching shows available for network recommendations.")
                return []
                
            # Use the NetworkAnalyzer to generate recommendations with the correct parameter order:
            # matching_shows, network, concept_analyzer
            return network_analyzer.get_network_recommendations(matching_shows, network, self)
            
        except Exception as e:
            st.error(f"Error generating network-specific recommendations: {str(e)}")
            return []
    
    def _get_component_scores(self, criteria: CriteriaDict, matching_shows: pd.DataFrame, integrated_data: IntegratedData) -> Dict[str, ComponentScore]:
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
    
    def _identify_success_factors(self, criteria: CriteriaDict, matching_shows: pd.DataFrame, integrated_data: IntegratedData) -> List[SuccessFactor]:
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

    def _generate_recommendations(self, criteria, matching_shows, success_factors, top_networks, confidence_info, integrated_data):
        """Generate recommendations based on criteria, matching shows, and analysis results.
        
        Args:
            criteria: Dictionary of criteria for the show concept
            matching_shows: DataFrame of shows matching the criteria
            success_factors: List of identified success factors
            top_networks: List of top compatible networks as NetworkMatch objects
            confidence_info: Dictionary with confidence metrics
            integrated_data: Dictionary of integrated data frames
            
        Returns:
            List of Recommendation objects
        """
        try:
            # Store matching_shows for later use in get_network_specific_recommendations
            self._last_matching_shows = matching_shows
            
            # If no matching shows, return empty list
            if matching_shows is None or matching_shows.empty:
                st.warning("No matching shows found for recommendation generation")
                return []
                
            # Generate general recommendations
            general_recommendations = self.recommendation_engine.generate_recommendations(
                criteria=criteria,
                matching_shows=matching_shows,
                success_factors=success_factors,
                top_networks=[],  # Empty list for general recommendations
                confidence_info=confidence_info,
                integrated_data=integrated_data
            )
            
            # Generate network-specific recommendations for top networks
            network_recommendations = []
            
            # Limit to top 3 networks to avoid overwhelming the user
            for network in top_networks[:3]:
                network_id = network.network_id
                network_name = network.network_name
                network_shows = matching_shows[matching_shows['network_id'] == network_id]
                show_count = len(network_shows)
                self.config.debug(f"Analyzing network {network_name} with {show_count} matching shows", category='network')
                network_recommendations.extend(
                    self.recommendation_engine.generate_network_specific_recommendations(
                        criteria=criteria,
                        network=network,
                        matching_shows=matching_shows,
                        integrated_data=integrated_data
                    )
                )
            
            # Combine and return all recommendations
            return general_recommendations + network_recommendations
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return []

