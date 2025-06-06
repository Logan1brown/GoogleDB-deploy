"""NetworkAnalyzer: Specialized analyzer for network compatibility and recommendations.

Handles all network-related analysis in the Show Optimizer, including:
1. Network Compatibility Analysis:
   - Calculate network compatibility scores
   - Rank networks based on criteria
   - Group networks into tiers based on compatibility

2. Network Recommendations:
   - Generate network-specific recommendations
   - Analyze network-specific success patterns
   - Identify optimal network features

3. Network Categorization:
   - Classify networks into tiers
   - Analyze network categories and types
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache
from datetime import datetime, timedelta

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .field_manager import FieldManager
from .optimizer_config import OptimizerConfig
from .score_calculators import NetworkMatch, NetworkScoreCalculator
from .criteria_scorer import CriteriaScorer
from .optimizer_matcher import Matcher


@dataclass
class NetworkTier:
    """A tier of networks with similar compatibility."""
    tier_name: str  # e.g., "Excellent Match", "Good Match", "Fair Match"
    networks: List[NetworkMatch]
    min_score: float
    max_score: float


class NetworkAnalyzer:
    """Specialized analyzer for network compatibility and recommendations."""
    
    def __init__(self, criteria_scorer: CriteriaScorer, field_manager: FieldManager = None):
        """Initialize the network analyzer.
        
        Args:
            criteria_scorer: CriteriaScorer instance for scoring calculations
            field_manager: Optional FieldManager instance for field mapping
        """
        self.criteria_scorer = criteria_scorer
        self.field_manager = field_manager or criteria_scorer.field_manager
        self.network_score_calculator = NetworkScoreCalculator(criteria_scorer)
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        self._last_network_analysis = None
        self._network_analysis_cache = {}
        
    def rank_networks_by_compatibility(self, criteria: Dict[str, Any], 
                                     limit: int = 10) -> List[NetworkMatch]:
        """Rank networks by compatibility with the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            limit: Maximum number of networks to return
            
        Returns:
            List of NetworkMatch objects sorted by compatibility score
        """
        try:
            # Get network matches from NetworkScoreCalculator
            network_matches = self.network_score_calculator.calculate_network_scores(criteria)
            
            # Sort by compatibility score (descending)
            network_matches.sort(key=lambda x: x.compatibility_score if x.compatibility_score is not None else -1, reverse=True)
            
            # Return top networks
            return network_matches[:limit]
        except Exception as e:
            st.error(f"Error ranking networks: {str(e)}")
            return []
    
    def get_network_tiers(self, criteria: Dict[str, Any], 
                        min_confidence: str = 'low') -> Dict[str, NetworkTier]:
        """Group networks into tiers based on compatibility with criteria.
        
        Args:
            criteria: Dictionary of criteria
            min_confidence: Minimum confidence level to include (none, low, medium, high)
            
        Returns:
            Dictionary mapping tier names to NetworkTier objects
        """
        try:
            # Get network matches from NetworkScoreCalculator
            network_matches = self.network_score_calculator.calculate_network_scores(criteria)
            
            # Filter by confidence
            confidence_levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
            min_confidence_level = confidence_levels.get(min_confidence.lower(), 0)
            
            # More efficient list comprehension instead of loop
            filtered_matches = [
                match for match in network_matches
                if confidence_levels.get(match.confidence.lower(), 0) >= min_confidence_level
            ]
            
            # Sort by compatibility score (descending)
            filtered_matches.sort(key=lambda x: x.compatibility_score if x.compatibility_score is not None else -1, reverse=True)
            
            # Group into tiers
            tiers = {}
            
            # Define tier thresholds from OptimizerConfig
            tier_thresholds = OptimizerConfig.NETWORK_TIERS
            
            # Excellent matches (highest tier)
            excellent_threshold = tier_thresholds.get('excellent', 0.85)
            excellent_matches = [m for m in filtered_matches if m.compatibility_score >= excellent_threshold]
            if excellent_matches:
                tiers['excellent'] = NetworkTier(
                    tier_name="Excellent Match",
                    networks=excellent_matches,
                    min_score=excellent_threshold,
                    max_score=1.0
                )
            
            # Good matches (second tier)
            good_threshold = tier_thresholds.get('good', 0.7)
            good_matches = [m for m in filtered_matches if good_threshold <= m.compatibility_score < excellent_threshold]
            if good_matches:
                tiers['good'] = NetworkTier(
                    tier_name="Good Match",
                    networks=good_matches,
                    min_score=good_threshold,
                    max_score=excellent_threshold
                )
            
            # Fair matches (third tier)
            fair_threshold = tier_thresholds.get('fair', 0.5)
            fair_matches = [m for m in filtered_matches if fair_threshold <= m.compatibility_score < good_threshold]
            if fair_matches:
                tiers['fair'] = NetworkTier(
                    tier_name="Fair Match",
                    networks=fair_matches,
                    min_score=fair_threshold,
                    max_score=good_threshold
                )
            
            # Poor matches (lowest tier)
            poor_matches = [m for m in filtered_matches if m.compatibility_score < fair_threshold]
            if poor_matches:
                tiers['poor'] = NetworkTier(
                    tier_name="Poor Match",
                    networks=poor_matches,
                    min_score=0.0,
                    max_score=fair_threshold
                )
            
            return tiers
        except Exception as e:
            st.error(f"Error grouping networks into tiers: {str(e)}")
            # Return empty tiers dictionary on error
            return {}
    
    def get_network_specific_success_rates(self, criteria: Dict[str, Any], 
                                         network_id: int) -> Dict[str, Any]:
        """Get network-specific success rates for each criteria.
        
        Args:
            criteria: Dictionary of criteria
            network_id: ID of the network to analyze
            
        Returns:
            Dictionary mapping criteria names to success rate information
        """
        try:
            # Fetch criteria data
            data = self.criteria_scorer.fetch_criteria_data()
            if data.empty:
                return {}
            
            # Filter to this network
            network_data = data[data['network_id'] == network_id]
            if network_data.empty:
                return {}
            
            # Calculate success rates for each criteria
            success_rates = {}
            
            for field_name, value in criteria.items():
                if field_name not in self.field_manager.FIELD_CONFIGS:
                    continue
                    
                config = self.field_manager.FIELD_CONFIGS[field_name]
                
                # Filter shows with this criteria
                if config.is_array:
                    # Use the array field mapping from field_manager
                    array_field_mapping = self.field_manager.get_array_field_mapping()
                    
                    # Get the correct column name for this field
                    if field_name in array_field_mapping:
                        field_id = array_field_mapping[field_name]
                    else:
                        st.error(f"ERROR: Field '{field_name}' not found in array field mapping")
                        success_rates[field_name] = {
                            'rate': None,
                            'sample_size': 0,
                            'has_data': False,
                            'matching_titles': []
                        }
                        continue
                    
                    # Check if the column exists in the DataFrame
                    if field_id not in network_data.columns:
                        st.error(f"ERROR: Column '{field_id}' for field '{field_name}' not found in data")
                        success_rates[field_name] = {
                            'rate': None,
                            'sample_size': 0,
                            'has_data': False,
                            'matching_titles': []
                        }
                        continue
                    
                    # Check data format (list or scalar)
                    sample = network_data[field_id].iloc[0] if not network_data.empty else None
                    
                    if isinstance(value, list):
                        # Multiple values
                        value_set = set(value)
                        if isinstance(sample, list):
                            # If column contains lists, use intersection
                            matching_shows = network_data[network_data[field_id].apply(
                                lambda x: isinstance(x, list) and bool(value_set.intersection(x))
                            )]
                        else:
                            # If column doesn't contain lists, use isin
                            matching_shows = network_data[network_data[field_id].isin(value)]
                    else:
                        # Single value
                        if isinstance(sample, list):
                            # If column contains lists, check if value is in list
                            matching_shows = network_data[network_data[field_id].apply(
                                lambda x: isinstance(x, list) and value in x
                            )]
                        else:
                            # If column doesn't contain lists, use equality
                            matching_shows = network_data[network_data[field_id] == value]
                else:
                    # Scalar field - use field_manager to map field name
                    field_id = self.field_manager.map_field_name(field_name, network_data.columns)
                    if isinstance(value, list):
                        matching_shows = network_data[network_data[field_id].isin(value)]
                    else:
                        matching_shows = network_data[network_data[field_id] == value]
                
                # Calculate success rate and sample size
                if not matching_shows.empty:
                    # Calculate success rate using threshold from OptimizerConfig
                    success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
                    success_count = matching_shows[matching_shows['success_score'] >= success_threshold].shape[0]
                    total_count = matching_shows.shape[0]
                    
                    # Get matching show titles (up to MAX_RESULTS)
                    matching_titles = []
                    if 'title' in matching_shows.columns:
                        matching_titles = matching_shows['title'].tolist()
                        # Limit to MAX_RESULTS titles
                        if len(matching_titles) > OptimizerConfig.MAX_RESULTS:
                            matching_titles = matching_titles[:OptimizerConfig.MAX_RESULTS]
                    
                    if total_count > 0:
                        success_rate = success_count / total_count
                        success_rates[field_name] = {
                            'rate': success_rate,
                            'sample_size': total_count,
                            'has_data': True,
                            'matching_titles': matching_titles
                        }
                    else:
                        success_rates[field_name] = {
                            'rate': None,
                            'sample_size': 0,
                            'has_data': False,
                            'matching_titles': []
                        }
                else:
                    # No matching shows
                    success_rates[field_name] = {
                        'rate': None,
                        'sample_size': 0,
                        'has_data': False,
                        'matching_titles': []
                    }
            
            return success_rates
        except Exception as e:
            st.error(f"Error calculating network-specific success rates: {str(e)}")
            return {}
    
    def get_network_recommendations(self, criteria: Dict[str, Any], 
                                  network: NetworkMatch) -> List[Dict[str, Any]]:
        """Generate network-specific recommendations.
        
        Args:
            criteria: Dictionary of criteria
            network: Target network
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            recommendations = []
            
            # Get network-specific success rates for each criteria
            network_rates = self.get_network_specific_success_rates(criteria, network.network_id)
            
            # Get overall success rates for each criteria
            overall_rates = {}
            for criteria_type in network_rates.keys():
                # Create a criteria dict with just this one criteria
                single_criteria = {criteria_type: criteria[criteria_type]}
                overall_rate, _ = self.criteria_scorer._calculate_success_rate_with_confidence(
                    single_criteria, min_sample_size=OptimizerConfig.CONFIDENCE['minimum_sample']
                )
                overall_rates[criteria_type] = overall_rate
            
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
                        recommendations.append({
                            'recommendation_type': 'consider',
                            'criteria_type': criteria_type,
                            'current_value': criteria[criteria_type],
                            'suggested_value': criteria[criteria_type],  # Same value, just emphasizing it
                            'impact_score': difference,
                            'confidence': 'medium',  # Network-specific recommendations have medium confidence
                            'explanation': f"'{self._get_criteria_name(criteria_type, criteria[criteria_type])}' works particularly well for {network.network_name}, with {difference:.0%} higher success rate than average."
                        })
                    else:
                        # This criteria works worse for this network than overall
                        # Look for alternative values that might work better
                        options = self.field_manager.get_options(criteria_type)
                        if options:
                            # Suggest a different option
                            alternative = options[0]  # Default to first option
                            for option in options:
                                if option.id != criteria[criteria_type]:
                                    alternative = option
                                    break
                                    
                            recommendations.append({
                                'recommendation_type': 'replace',
                                'criteria_type': criteria_type,
                                'current_value': criteria[criteria_type],
                                'suggested_value': alternative.id,
                                'suggested_name': alternative.name,
                                'impact_score': -difference,  # Convert to positive impact
                                'confidence': 'medium',  # Network-specific recommendations have medium confidence
                                'explanation': f"'{self._get_criteria_name(criteria_type, criteria[criteria_type])}' performs {-difference:.0%} worse than average for {network.network_name}. Consider alternatives like '{alternative.name}'."
                            })
            
            return recommendations
        except Exception as e:
            st.error(f"Error generating network recommendations: {str(e)}")
            return []
    
    def calculate_network_compatibility(self, criteria: Dict[str, Any], network_id: int) -> Dict[str, Any]:
        """Calculate compatibility score between criteria and a specific network.
        
        Args:
            criteria: Dictionary of criteria
            network_id: ID of the network to analyze
            
        Returns:
            Dictionary with compatibility score and details
        """
        try:
            # Create network-specific criteria
            network_criteria = criteria.copy()
            network_criteria['network'] = network_id
            
            # Get all network matches
            network_matches = self.network_score_calculator.calculate_network_scores(criteria)
            
            # Find the match for this network
            for match in network_matches:
                if match.network_id == network_id:
                    return {
                        'compatibility_score': match.compatibility_score,
                        'success_probability': match.success_probability,
                        'sample_size': match.sample_size,
                        'confidence': match.confidence,
                        'details': match.details
                    }
            
            # Network not found
            return {
                'compatibility_score': None,
                'success_probability': None,
                'sample_size': 0,
                'confidence': 'none',
                'details': {}
            }
        except Exception as e:
            st.error(f"Error calculating network compatibility: {str(e)}")
            return {
                'compatibility_score': None,
                'success_probability': None,
                'sample_size': 0,
                'confidence': 'none',
                'details': {'error': str(e)}
            }
    
    def _get_criteria_name(self, criteria_type: str, criteria_value: Any) -> str:
        """Get the display name for a criteria value.
        
        Args:
            criteria_type: Type of criteria (e.g., 'genre', 'format')
            criteria_value: Value of the criteria (ID)
            
        Returns:
            Display name for the criteria value
        """
        try:
            # Get options for this criteria type
            options = self.field_manager.get_options(criteria_type)
            
            # Find the option with matching ID
            for option in options:
                if option.id == criteria_value:
                    return option.name
            
            # If not found, return the raw value
            return str(criteria_value)
        except Exception as e:
            st.error(f"Error getting criteria name: {str(e)}")
            return str(criteria_value)
    
    def _calculate_success_rate_with_confidence(self, criteria: Dict[str, Any], 
                                              min_sample_size: int = 10) -> Tuple[float, str]:
        """Calculate success rate with confidence level.
        
        Args:
            criteria: Dictionary of criteria
            min_sample_size: Minimum sample size for confidence
            
        Returns:
            Tuple of (success_rate, confidence_level)
        """
        try:
            # Use the criteria scorer to get matching shows
            matching_shows, count, confidence_info = self.criteria_scorer._matching_calculator.get_matching_shows(criteria)
            
            if matching_shows.empty or count < min_sample_size:
                return None, 'none'
            
            # Calculate success rate
            success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
            success_count = matching_shows[matching_shows['success_score'] >= success_threshold].shape[0]
            success_rate = success_count / count if count > 0 else None
            
            # Determine confidence level
            confidence = confidence_info.get('level', OptimizerConfig.get_confidence_level(count, confidence_info.get('match_level', 1)))
            
            return success_rate, confidence
        except Exception as e:
            st.error(f"Error calculating success rate: {str(e)}")
            return None, 'none'
