"""NetworkAnalyzer: Specialized analyzer for network compatibility.

Handles network-related analysis in the Show Optimizer, focusing on:
1. Network Compatibility Analysis:
   - Calculate network compatibility scores directly from matching shows
   - Rank networks based on compatibility with show criteria
   - Group networks into tiers based on compatibility thresholds

2. Network-Specific Success Analysis:
   - Calculate success rates for specific networks
   - Analyze network preferences for different criteria
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import streamlit as st

from .optimizer_config import OptimizerConfig
from .optimizer_data_contracts import ConfidenceInfo, update_confidence_info
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer
from .optimizer_data_contracts import (
    CriteriaDict, ConfidenceInfo, IntegratedData, 
    FieldValueSuccessRate, RecommendationItem,
    create_field_value_key, create_success_rate_data
)
# Import NetworkMatch from optimizer_data_contracts instead of score_calculators
# to avoid circular imports
from .optimizer_data_contracts import NetworkMatch


class NetworkAnalyzer:
    """Specialized analyzer for network compatibility using a single data stream approach."""
    
    def __init__(self, criteria_scorer: CriteriaScorer, field_manager: FieldManager = None):
        """Initialize the network analyzer.
        
        Args:
            criteria_scorer: CriteriaScorer instance for score calculations
            field_manager: Optional FieldManager instance for field mapping
        """
        self.criteria_scorer = criteria_scorer
        self.field_manager = field_manager or criteria_scorer.field_manager
        
    def rank_networks_by_compatibility(self, matching_shows: pd.DataFrame, confidence_info: Optional[ConfidenceInfo] = None, limit: Optional[int] = None) -> List[NetworkMatch]:
        """Rank networks by compatibility using only the matching shows DataFrame.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level column
            confidence_info: Dictionary with confidence metrics conforming to ConfidenceInfo
            limit: Maximum number of networks to return
            
        Returns:
            List of NetworkMatch objects sorted by compatibility score
        """
        # Ensure confidence_info conforms to our ConfidenceInfo contract
        # This enforces the contract rather than adding defensive checks
        if confidence_info is not None:
            confidence_info = update_confidence_info(confidence_info, {})
        try:
            # Validate inputs
            if matching_shows is None or matching_shows.empty:
                return []
            
            # Check if network_id column exists
            if 'network_id' not in matching_shows.columns:
                return []
            
            # Calculate network compatibility scores directly from matching shows
            network_scores = {}
            
            # Group by network_id and calculate compatibility scores
            for network_id, network_shows in matching_shows.groupby('network_id'):
                # Skip if no shows for this network
                if network_shows.empty:
                    continue
                    
                # Get network name
                network_name = "Unknown Network"
                if self.field_manager:
                    try:
                        network_name = self.field_manager.get_name('network', network_id) or "Unknown Network"
                    except Exception as e:
                        # Keep the default name
                        pass
                
                # Calculate compatibility score based on match levels in the network shows DataFrame
                # Lower match_level is better (1 is exact match)
                if 'match_level' in network_shows.columns:
                    avg_match_level = network_shows['match_level'].mean()
                    # Convert to score (1 is best, lower match_level = higher score)
                    compatibility_score = max(0.1, 1.0 - ((avg_match_level - 1) * 0.2))
                else:
                    # Default if no match_level column
                    compatibility_score = 0.5
                    
                # Calculate success probability if success_score column exists
                success_probability = None
                sample_size = len(network_shows)
                confidence = "none"
                
                if 'success_score' in network_shows.columns:
                    # Calculate average success score
                    success_probability = network_shows['success_score'].mean()
                    
                    # Determine confidence based on sample size and match level using OptimizerConfig
                    avg_match_level = network_shows['match_level'].mean() if 'match_level' in network_shows.columns else 1
                    confidence = OptimizerConfig.get_confidence_level(sample_size, int(avg_match_level))
                
                # Create NetworkMatch using dataclass constructor
                network_match = NetworkMatch(
                    network_id=network_id,
                    network_name=network_name,  # Using network_name field from dataclass
                    compatibility_score=compatibility_score,
                    success_probability=success_probability,
                    sample_size=sample_size,
                    confidence=confidence,
                    details={}  # Empty details dictionary
                )
                
                network_scores[network_id] = network_match
            
            # Convert to list and sort by compatibility score (descending)
            network_matches = list(network_scores.values())
            network_matches.sort(key=lambda x: x.compatibility_score if x.compatibility_score is not None else -1, reverse=True)
            
            # Use config for default limit if not specified
            if limit is None:
                limit = OptimizerConfig.NETWORK.get('default_limit', 10)
                
            # Return top networks
            return network_matches[:limit]
        except Exception as e:
            return []
            
    def group_networks_into_tiers(self, network_matches: List[NetworkMatch]) -> Dict[str, List[NetworkMatch]]:
        """Group networks into tiers based on compatibility score.
        
        Args:
            network_matches: List of NetworkMatch objects sorted by compatibility score
            
        Returns:
            Dictionary of tier_name -> list of NetworkMatch objects
        """
        try:
            # Validate inputs
            if not network_matches:
                return {}
                
            # Initialize tiers dictionary with empty lists for each tier
            tiers = {tier_name: [] for tier_name in OptimizerConfig.NETWORK_TIERS.keys()}
            # Add 'poor' tier if not in config
            if 'poor' not in tiers:
                tiers['poor'] = []
            
            # Get sorted tier thresholds for comparison (highest to lowest)
            sorted_tiers = sorted(
                [(name, threshold) for name, threshold in OptimizerConfig.NETWORK_TIERS.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Assign networks to tiers
            for network in network_matches:
                score = network.compatibility_score
                assigned = False
                
                # Check each tier threshold in order
                for tier_name, threshold in sorted_tiers:
                    if score >= threshold:
                        tiers[tier_name].append(network)
                        assigned = True
                        break
                        
                # If not assigned to any tier, put in 'poor' tier
                if not assigned:
                    tiers['poor'].append(network)
            
            return tiers
            
        except Exception as e:
            return {}
    
    def get_network_tiers(self, matching_shows: pd.DataFrame, min_confidence: str = 'low') -> Dict[str, List[NetworkMatch]]:
        """Group networks into tiers based on compatibility using matching shows.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level column
            min_confidence: Minimum confidence level to include (none, low, medium, high)
            
        Returns:
            Dictionary mapping tier names to lists of NetworkMatch objects
        """
        try:
            # Validate inputs - require matching_shows to be provided
            if matching_shows is None or matching_shows.empty:
                st.error("Error: No matching shows provided for network tier analysis")
                return {}
            
            # Validate inputs for both calling patterns
            if matching_shows is None or matching_shows.empty:
                return {}
            
            # Get network matches using the simplified approach
            # Pass confidence_info if available, otherwise None
            confidence_info = kwargs.get('confidence_info', None)
            network_matches = self.rank_networks_by_compatibility(matching_shows, confidence_info)
            
            # Filter by confidence using OptimizerConfig
            # Map confidence levels to numeric values for comparison
            confidence_order = {
                'none': 0,
                'minimal': 1,
                'very_low': 2,
                'low': 3,
                'medium': 4,
                'high': 5
            }
            min_confidence_level = confidence_order.get(min_confidence.lower(), 0)
            
            filtered_matches = []
            for network in network_matches:
                network_confidence = confidence_order.get(network.confidence, 0)
                if network_confidence >= min_confidence_level:
                    filtered_matches.append(network)
            
            # Use the simplified group_networks_into_tiers method
            return self.group_networks_into_tiers(filtered_matches)
        except Exception as e:
            st.error(f"Error getting network tiers: {str(e)}")
            return {}
    
    # The get_network_specific_success_rates method has been removed
    # Network-specific success rates are now calculated directly in the recommendation engine
    # This ensures consistent key formats and eliminates potential key mismatches
    
    # The get_network_recommendations method has been removed as part of unifying the recommendation generation.
    # All recommendations (both general and network-specific) are now generated through the unified generate_all_recommendations
    # method in the RecommendationEngine and tagged with is_network_specific=True/False for proper categorization.
    # 
    # Network-specific recommendations are now generated alongside general recommendations in a single pass,
    # eliminating redundant calculations and ensuring consistent data contracts.
    
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
            # Error in _get_criteria_name
            return str(criteria_value)

    def _calculate_success_rate_with_confidence(self, matching_shows: pd.DataFrame, 
                                                min_sample_size: int = 10) -> Tuple[float, str]:
        """Calculate success rate with confidence level.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level and success_score columns
            min_sample_size: Minimum sample size for confidence
            
        Returns:
            Tuple of (success_rate, confidence_level)
        """
        try:
            # Validate input
            if matching_shows is None or not isinstance(matching_shows, pd.DataFrame) or matching_shows.empty:
                return None, 'none'
            
            # Get sample size
            count = len(matching_shows)
            if count < min_sample_size:
                return None, 'none'
            
            # Calculate success rate
            # Use SUCCESS['threshold'] as the single source of truth for success threshold
            success_threshold = OptimizerConfig.SUCCESS['threshold']
            
            # Calculate success rate if success_score column exists
            if 'success_score' in matching_shows.columns:
                success_count = matching_shows[matching_shows['success_score'] >= success_threshold].shape[0]
                success_rate = success_count / count if count > 0 else None
            else:
                success_rate = None
            
            # Calculate average match level for confidence
            avg_match_level = 1
            if 'match_level' in matching_shows.columns:
                # Ensure we're only calculating mean for rows where match_level is a number
                valid_match_levels = []
                for ml in matching_shows['match_level']:
                    if isinstance(ml, (int, float)) and not pd.isna(ml):
                        valid_match_levels.append(ml)
                
                if valid_match_levels:
                    avg_match_level = sum(valid_match_levels) / len(valid_match_levels)
                else:
                    avg_match_level = 1
            
            # Determine confidence level using OptimizerConfig
            confidence = OptimizerConfig.get_confidence_level(count, int(avg_match_level))
            
            return success_rate, confidence
        except Exception as e:
            # Error calculating success rate
            return None, 'none'
