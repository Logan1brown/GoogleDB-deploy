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
from .optimizer_data_contracts import CriteriaDict, ConfidenceInfo, IntegratedData, NetworkMatch, validate_criteria, validate_integrated_data, update_confidence_info

# Data contracts are now imported from optimizer_data_contracts.py
from .score_calculators import ComponentScore
from .network_analyzer import NetworkAnalyzer
from .recommendation_engine import RecommendationEngine, SuccessFactor
from .optimizer_data_contracts import RecommendationItem


# Helper functions are now imported from optimizer_data_contracts.py


@dataclass
class OptimizationSummary:
    """Summary of optimization recommendations."""
    overall_success_probability: Optional[float]  # Can be None when data is insufficient
    confidence: str
    top_networks: List[NetworkMatch]
    component_scores: Dict[str, ComponentScore]
    recommendations: Dict[str, List[Dict[str, Any]]]  # Dictionary with 'general' and 'network_specific' recommendation lists
    success_factors: List[SuccessFactor]
    matching_titles: List[str] = field(default_factory=list)  # Titles of shows matching all criteria
    match_level: int = 1  # Match level used (1-4, where 1 is highest)
    match_quality: float = 1.0  # Quality of the match (0-1)
    confidence_score: float = 1.0  # Confidence score (0-1)
    matching_shows: Any = None  # DataFrame of matching shows (pandas DataFrame)
    match_count: int = 0  # Number of matching shows
    match_counts_by_level: Dict[int, int] = field(default_factory=dict)  # Count of shows by match level
    confidence_info: ConfidenceInfo = field(default_factory=dict)  # Detailed confidence information
    
    def __post_init__(self):
        """Post-initialization validation and logging."""
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"OptimizationSummary created with success probability: {self.overall_success_probability}", category='components')
            
            if self.component_scores:
                OptimizerConfig.debug(f"Component scores: {list(self.component_scores.keys())}", category='components')
                if 'success' in self.component_scores:
                    success_score = self.component_scores['success']
                    OptimizerConfig.debug(f"Success component score: {success_score.score}, confidence: {success_score.confidence}", category='components')
    
    @property
    def network_compatibility(self) -> List[NetworkMatch]:
        """Return the top networks as network compatibility data.
        
        This property ensures backward compatibility with UI code that expects
        a network_compatibility attribute.
        """
        return self.top_networks
        
    # Private attribute to store formatted data
    _formatted_data_dict: Dict[str, Union[float, str, List[Dict[str, Any]]]] = None
    
    def __post_init__(self):
        """Initialize instance variables after dataclass initialization."""
        # Initialize _formatted_data_dict as an instance variable
        self._formatted_data_dict = {}
    
    def _format_component_scores(self) -> Dict[str, Dict[str, Union[float, int, str]]]:
        """Format component scores for UI display.
        
        Returns:
            Dictionary with formatted component scores
        """
        # Component scores should always be present, but return empty dict if not
        # This maintains the contract with the UI layer
        if not self.component_scores:
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("No component scores available to format", category='components')
            return {}
            
        # Debug log the component scores before formatting
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Formatting {len(self.component_scores)} component scores: {list(self.component_scores.keys())}", category='components')
            
        # Format each ComponentScore object into a dictionary
        formatted_scores = {}
        for component_name, component_score in self.component_scores.items():
            # Create a clean dictionary with the required fields
            score_dict = {
                'score': float(component_score.score) if component_score.score is not None else None,
                'sample_size': component_score.sample_size,
                'confidence': component_score.confidence
            }
            
            # Add error information if present in details
            if component_score.details and 'error' in component_score.details:
                score_dict['error'] = component_score.details['error']
                
            # Debug log each component score
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Component {component_name}: score={score_dict['score']}, sample_size={score_dict['sample_size']}, confidence={score_dict['confidence']}", category='components')
                
            formatted_scores[component_name] = score_dict
            
        return formatted_scores
    
    def _format_success_probability(self) -> Dict[str, str]:
        """Format success probability for UI display.
        
        Returns:
            Dictionary with formatted success probability data
        """
        # Check for error in confidence_info
        if isinstance(self.confidence_info, dict) and 'error' in self.confidence_info:
            return {
                'display': 'Error',
                'subtitle': f"Analysis failed: {self.confidence_info['error']}"
            }
            
        # Format success probability for display
        if not hasattr(self, 'overall_success_probability') or self.overall_success_probability is None:
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Success probability not available in OptimizationSummary", category='components')
            return {
                'display': 'N/A',
                'subtitle': 'Success probability not available'
            }
            
        if not self.confidence_info:
            return {
                'display': f"{self.overall_success_probability:.0%}",
                'subtitle': 'Confidence data not available'
            }
            
        # Format the success probability - confidence_info is always a dictionary (ConfidenceInfo TypedDict)
        confidence = self.confidence_info.get('level', 'none')
            
        return {
            'display': f"{self.overall_success_probability:.0%}",
            'subtitle': f"Confidence: {confidence.capitalize()}"
        }
    
    @property
    def formatted_data(self) -> Dict[str, Union[float, str, List[Dict[str, Any]]]]:
        """Format data for UI display.
        
        Returns a dictionary with formatted networks and recommendations data
        ready for display in the UI.
        """
        # No debug statements needed here - we've fixed the confidence_info handling
            
        # If _formatted_data_dict has been set, return it
        if self._formatted_data_dict:
            # Return cached formatted data
            return self._formatted_data_dict
            
        # Otherwise generate formatted data
        # Generate new formatted data
        component_scores = self._format_component_scores()
        
        formatted = {
            'networks': [],
            'recommendations': {
                'general': [],
                'network_specific': []
            },
            'component_scores': component_scores,
            'success_factors': [],  # Initialize empty list for success factors
            'success_probability': self._format_success_probability(),
            'confidence_info': self.confidence_info  # Include confidence_info for error handling
        }
        
        # Component scores are now explicitly included
        
        # Format networks and recommendations data
        
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
                # Format recommendations data - now using the new dictionary structure
        if self.recommendations:
            # Process general recommendations
            general_recs = self.recommendations.get('general', [])
            for rec in general_recs:
                # Ensure rec is a dictionary
                if not isinstance(rec, dict):
                    # Skip non-dictionary recommendations
                    continue
                    
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
                
                # Add to general recommendations
                formatted['recommendations']['general'].append(rec_dict)
            
            # Process network-specific recommendations
            network_recs = self.recommendations.get('network_specific', [])
            for rec in network_recs:
                # Ensure rec is a dictionary
                if not isinstance(rec, dict):
                    # Skip non-dictionary recommendations
                    continue
                    
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
                
                # Add to network-specific recommendations
                formatted['recommendations']['network_specific'].append(rec_dict)
        
        # Cache and return the formatted data
            
        # Store the formatted data for future access
        self._formatted_data_dict = formatted
        
        # Verify that component_scores is in the dictionary before returning
        if 'component_scores' not in formatted:
            st.write("ERROR: component_scores missing from formatted_data!")
            formatted['component_scores'] = component_scores
            
        # Debug log the formatted data structure
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Formatted data structure: {formatted.keys()}", category='format')
            if 'component_scores' in formatted:
                OptimizerConfig.debug(f"Component scores keys: {formatted['component_scores'].keys()}", category='format')
            if 'success_probability' in formatted:
                OptimizerConfig.debug(f"Success probability: {formatted['success_probability']}", category='format')
            
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
        
        # Create a minimal summary with error information
        # Include placeholder component scores with error information for better UI display
        component_scores = {
            'success': ComponentScore(
                component='success',
                score=None,
                sample_size=0,
                confidence='none',
                details={'error': error_message}
            ),
            'audience': ComponentScore(
                component='audience',
                score=None,
                sample_size=0,
                confidence='none',
                details={'error': error_message}
            ),
            'critics': ComponentScore(
                component='critics',
                score=None,
                sample_size=0,
                confidence='none',
                details={'error': error_message}
            ),
            'longevity': ComponentScore(
                component='longevity',
                score=None,
                sample_size=0,
                confidence='none',
                details={'error': error_message}
            )
        }
        
        return OptimizationSummary(
            overall_success_probability=None,
            confidence='none',
            top_networks=[],
            component_scores=component_scores,  # Include component scores with error info
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
            # VERY OBVIOUS UI CHANGE TO CONFIRM CODE UPDATES ARE WORKING
            # Version indicator is now in the sidebar
            

            
            # Step 1: Find matching shows using integrated data

            matching_shows, confidence_info = self._find_matching_shows(criteria, integrated_data)
            
            # Store matching_shows for later use in get_network_specific_recommendations
            self._last_matching_shows = matching_shows
            
            # Extract match information
            match_count = len(matching_shows) if not matching_shows.empty else 0
            
            # Ensure confidence_info conforms to our ConfidenceInfo contract
            # This is better than adding defensive checks - we enforce the contract
            confidence_info = update_confidence_info(confidence_info, {})
            
            # Now we can safely extract values from confidence_info
            # The contract guarantees these keys exist
            match_level = confidence_info['match_level']
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
                OptimizerConfig.debug("Calculating success probability", category='analysis')
            success_probability, confidence = self._calculate_success_probability(criteria, matching_shows)
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Success probability: {success_probability}, confidence: {confidence}", category='analysis')
            
            # Step 3: Find top networks - pass the existing matching_shows and confidence_info to avoid redundant matching
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Finding top networks", category='analysis')
            top_networks = self._find_top_networks(criteria, integrated_data=integrated_data, matching_shows=matching_shows, confidence_info=confidence_info)
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Found {len(top_networks)} top networks", category='analysis')
                for i, network in enumerate(top_networks):
                    OptimizerConfig.debug(f"Network {i+1}: {network.network_name} (ID: {network.network_id})", category='networks')
                    OptimizerConfig.debug(f"Network {i+1} type: {type(network).__name__}", category='networks')
            
            # Step 4: Calculate component scores
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Calculating component scores", category='analysis')
            component_scores = self._get_component_scores(criteria, matching_shows, integrated_data)
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Component scores: {component_scores}", category='analysis')
            
            # Step 5: Identify success factors
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Identifying success factors", category='analysis')
            success_factors = self._identify_success_factors(criteria, matching_shows, integrated_data)
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Found {len(success_factors)} success factors", category='analysis')
            
            # Step 6: Generate recommendations
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Generating recommendations", category='analysis')
            recommendations = self._generate_recommendations(
                criteria, matching_shows, success_factors, top_networks, confidence_info, integrated_data
            )
            if OptimizerConfig.DEBUG_MODE:
                # Handle the new dictionary structure for recommendations
                general_count = len(recommendations.get('general', []))
                network_specific_count = len(recommendations.get('network_specific', []))
                total_count = general_count + network_specific_count
                OptimizerConfig.debug(f"Generated {total_count} recommendations ({general_count} general, {network_specific_count} network-specific)", category='analysis')
                # Safely get recommendation types, handling both dict and object access
                if recommendations and total_count > 0:
                    # Enforce RecommendationItem TypedDict contract - recommendations should always be dictionaries
                    # No need to debug recommendation types
                    pass
            
            # Get matching show titles (up to MAX_RESULTS) to include in the summary
            matching_titles = []
            if not matching_shows.empty and 'title' in matching_shows.columns:
                matching_titles = matching_shows['title'].tolist()
                # Limit to MAX_RESULTS titles
                if len(matching_titles) > self.config.MAX_RESULTS:
                    matching_titles = matching_titles[:self.config.MAX_RESULTS]
            
            # Create the optimization summary with confidence info
                
            # Prepare to create the optimization summary
            
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
            
            # Ensure component_scores is included in the formatted_data
            if 'component_scores' not in formatted:
                # Add component scores directly to ensure they're available to the UI
                formatted['component_scores'] = summary._format_component_scores()
                # Update the cached formatted_data
                summary.formatted_data = formatted
            
            # Return the summary with populated formatted_data
            
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
                # Create an error confidence info
                error_info: ConfidenceInfo = update_confidence_info({}, {
                    'error': 'No matcher available',
                    'level': 'none',
                    'match_level': 1  # Explicitly set match_level to avoid attribute errors
                })
                
                st.error("No matcher available in CriteriaScorer. Cannot find matching shows.")
                return pd.DataFrame(), error_info
                
        except ValueError as e:
            # Handle validation errors
            error_info: ConfidenceInfo = update_confidence_info({}, {
                'error': str(e),
                'level': 'none',
                'match_level': 1  # Explicitly set match_level to avoid attribute errors
            })
            
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
    
    def _find_top_networks(self, criteria: CriteriaDict, integrated_data: IntegratedData, matching_shows: pd.DataFrame = None, confidence_info: Optional[Dict[str, Any]] = None) -> List[NetworkMatch]:
        """Find top networks compatible with the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            matching_shows: Optional DataFrame of shows already matched to criteria
            confidence_info: Optional dictionary with match confidence information
            
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
                matching_shows, new_confidence_info = self._find_matching_shows(criteria, integrated_data=integrated_data)
                
                if matching_shows is None or matching_shows.empty:
                    return []
                    
                # Use the confidence_info from _find_matching_shows if none was provided
                if confidence_info is None:
                    confidence_info = new_confidence_info
            
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
                
            # Ensure confidence_info conforms to our ConfidenceInfo contract
            # This enforces the contract rather than adding defensive checks
            confidence_info = update_confidence_info(confidence_info, {})
            
            # Call rank_networks_by_compatibility with proper error handling
            # Now we can safely pass confidence_info to ensure match_level is properly accessed
            network_matches = self.criteria_scorer.network_analyzer.rank_networks_by_compatibility(
                matching_shows,
                confidence_info
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
            
            # Initialize confidence_info without hardcoded defaults
            confidence_info = {}
            
            # Extract match level statistics from matching_shows if available
            if not matching_shows.empty:
                if 'match_level' in matching_shows.columns:
                    # Store the distribution of match levels
                    confidence_info['max_match_level'] = matching_shows['match_level'].max()
                    confidence_info['min_match_level'] = matching_shows['match_level'].min()
                    confidence_info['mean_match_level'] = matching_shows['match_level'].mean()
                    
                    # Set match_level based on max_match_level (best match)
                    confidence_info['match_level'] = int(confidence_info['max_match_level'])
                    
                    # Only set the confidence level if it's not already set
                    if 'level' not in confidence_info:
                        match_level = confidence_info['match_level']
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
            List of RecommendationItem dictionaries
        """
        try:
            # Store matching_shows for later use in get_network_specific_recommendations
            self._last_matching_shows = matching_shows
            
            # If no matching shows, return empty list
            if matching_shows is None or matching_shows.empty:
                st.warning("No matching shows found for recommendation generation")
                return []
                
            # Ensure confidence_info conforms to our ConfidenceInfo contract
            # This enforces the contract rather than adding defensive checks
            confidence_info = update_confidence_info(confidence_info, {})
            
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
            for i, network in enumerate(top_networks[:3]):
                # Add detailed debugging for NetworkMatch objects
                if self.config.DEBUG_MODE:
                    st.write(f"DEBUG: Network object type: {type(network).__name__}")

                    st.write(f"DEBUG: Network ID: {network.network_id}")
                    st.write(f"DEBUG: Network Name: {network.network_name}")
                
                network_id = network.network_id
                network_name = network.network_name
                network_shows = matching_shows[matching_shows['network_id'] == network_id]
                show_count = len(network_shows)
                self.config.debug(f"Analyzing network {network_name} with {show_count} matching shows", category='network')
                
                try:
                    # Wrap the call in a try-except to catch any errors specific to this network
                    # Ensure confidence_info is passed to avoid attribute errors
                    network_specific_recs = self.recommendation_engine.generate_network_specific_recommendations(
                        criteria=criteria,
                        network=network,
                        matching_shows=matching_shows,
                        integrated_data=integrated_data,
                        confidence_info=confidence_info
                    )
                    network_recommendations.extend(network_specific_recs)
                    
                except Exception as net_error:
                    # Log the error but continue processing other networks
                    if self.config.DEBUG_MODE:
                        st.write(f"DEBUG: Error generating recommendations for {network_name}: {str(net_error)}")
                    self.config.debug(f"Error generating recommendations for {network_name}: {str(net_error)}", 
                                     category='error', force=True)
            
            # Return recommendations in the expected structure for the UI
            # Instead of a flat list, return a dictionary with 'general' and 'network_specific' keys
            return {
                'general': general_recommendations,
                'network_specific': network_recommendations
            }
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            # Always return the expected dictionary structure, even on error
            return {
                'general': [],
                'network_specific': []
            }

