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
    create_field_value_key, create_success_rate
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
    
    def get_network_tiers(self, matching_shows: pd.DataFrame = None, min_confidence: str = 'low', criteria: CriteriaDict = None, integrated_data: IntegratedData = None) -> Dict[str, List[NetworkMatch]]:
        """Group networks into tiers based on compatibility using matching shows.
        
        This method supports two calling patterns:
        1. With matching_shows directly (legacy pattern)
        2. With criteria and integrated_data to find matching shows first (new pattern)
        
        Args:
            matching_shows: Optional DataFrame of shows matching the criteria with match_level column
            min_confidence: Minimum confidence level to include (none, low, medium, high)
            criteria: Optional criteria dictionary to find matching shows if not provided
            integrated_data: Optional integrated data needed to find matching shows
            
        Returns:
            Dictionary mapping tier names to lists of NetworkMatch objects
        """
        try:
            # Handle the case where criteria and integrated_data are provided instead of matching_shows
            if matching_shows is None and criteria is not None and integrated_data is not None:
                # Use criteria_scorer to find matching shows
                if self.criteria_scorer and hasattr(self.criteria_scorer, 'matcher') and self.criteria_scorer.matcher:
                    matching_shows, _ = self.criteria_scorer.matcher.find_matches_with_fallback(criteria, integrated_data)
                    
                    if matching_shows is None or matching_shows.empty:
                        return {}
                else:
                    # No matcher available
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
    
    def get_network_specific_success_rates(self, matching_shows: pd.DataFrame, network_id: int) -> Dict[str, FieldValueSuccessRate]:
        """Get success rates for specific criteria for a given network using matching shows.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level and success_score columns
            network_id: ID of the network to analyze
            
        Returns:
            Dictionary mapping field-value keys to FieldValueSuccessRate dictionaries
        """
        try:
            # Initialize success rates dictionary
            success_rates = {}
            
            # Validate inputs
            if matching_shows is None or matching_shows.empty:
                return {}
                
            # Filter to this network
            if 'network_id' not in matching_shows.columns:
                return {}
                
            network_shows = matching_shows[matching_shows['network_id'] == network_id]
            
            if network_shows.empty:
                return {}
                
            # Check if success_score column exists
            if 'success_score' not in network_shows.columns:
                return {}
            
            # Get success threshold from config
            success_threshold = OptimizerConfig.SUCCESS['threshold']
            
            # Define standard columns to exclude
            standard_columns = ['show_id', 'title', 'success_score', 'match_level', 'network_id', 'network_name']
            
            # Only process ID columns for consistency with the rest of the system
            # This follows the CriteriaDict contract which uses IDs for all fields
            id_columns = [col for col in network_shows.columns 
                         if (col.endswith('_id') or col.endswith('_ids')) 
                         and col not in standard_columns]
            
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Network {network_id} analysis: Found {len(id_columns)} ID columns to analyze", category='recommendation')
            
            # Process ID columns for network-specific success rates
            
            # Process each valid criteria column (ID columns)
            for column in id_columns:
                # Get unique values for this column
                try:
                    # Skip columns with all null values
                    if network_shows[column].isna().all():

                        continue
                        
                    # Get unique non-null values
                    unique_values = network_shows[column].dropna().unique()
                    
                    # Skip if no unique values
                    if len(unique_values) == 0:

                        continue
                        

                        
                    # For each unique value, calculate success rate
                    for value in unique_values:
                        # Skip null values
                        if pd.isna(value):
                            continue
                            
                        # Handle array fields differently
                        is_array_field = False
                        try:
                            if isinstance(value, (list, tuple)):
                                is_array_field = True
                                # For list-type fields, find shows where the column contains this value
                                value_shows = network_shows[network_shows[column].apply(
                                    lambda x: isinstance(x, list) and all(item in x for item in value) if isinstance(x, list) else False
                                )]
                            elif isinstance(value, str) and column.endswith('_names') and '[' in value:
                                is_array_field = True
                                # Skip string representations of arrays as they need special parsing
                                continue
                            else:
                                # For scalar fields, use direct comparison
                                value_shows = network_shows[network_shows[column] == value]
                        except Exception as e:
                            continue
                        
                        # Skip if no shows
                        if value_shows.empty:
                            continue
                            
                        # Calculate success rate if we have data
                        success_count = value_shows[value_shows['success_score'] >= success_threshold].shape[0]
                        total_count = value_shows.shape[0]
                        
                        if total_count > 0:
                            success_rate = success_count / total_count
                            
                            # Use the exact database column name (ID) for field_name
                            field_name = column
                            
                            # Get value name for display
                            value_name = str(value)
                            
                            # Clean up value_name if it contains "Unknown" with a number in parentheses
                            clean_value_name = value_name
                            if isinstance(value_name, str) and "Unknown" in value_name and "(" in value_name and ")" in value_name:
                                try:
                                    # Extract the value inside parentheses
                                    start_idx = value_name.find("(") + 1
                                    end_idx = value_name.find(")")
                                    if start_idx > 0 and end_idx > start_idx:
                                        clean_value = value_name[start_idx:end_idx].strip()
                                        clean_value_name = clean_value
                                except Exception:
                                    # Keep original value_name if extraction fails
                                    pass
                            
                            # Get matching show titles (up to MAX_RESULTS)
                            matching_titles = []
                            if 'title' in value_shows.columns:
                                matching_titles = value_shows['title'].tolist()
                                if len(matching_titles) > OptimizerConfig.MAX_RESULTS:
                                    matching_titles = matching_titles[:OptimizerConfig.MAX_RESULTS]
                            
                            # Use the field manager to create standardized success rate data
                            # This ensures consistent handling of field values, especially list-type fields
                            success_rate_data = self.field_manager.create_success_rate_data(
                                field_name=field_name,
                                value=value,
                                rate=success_rate,
                                sample_size=total_count,
                                matching_shows=matching_titles
                            )
                            
                            # Add cleaned value name if different from the generated one
                            if clean_value_name != success_rate_data['value_name']:
                                success_rate_data['original_value_name'] = success_rate_data['value_name']
                                success_rate_data['value_name'] = clean_value_name
                            
                            # Create a key using the original field name (which is already an ID column)
                            try:
                                # Import the key creation function if needed
                                from ..utils.optimizer_utils import create_field_value_key
                                
                                # Create a key using the original field name without standardization
                                key = create_field_value_key(field_name, value)
                                

                                
                                # Add success rate data to the dictionary
                                success_rates[key] = success_rate_data
                            except Exception as e:
                                # Error adding success rate data
                                pass
                except Exception as e:
                    # Error processing column
                    continue
            
            return success_rates
        except Exception as e:
            return {}
    
    def get_network_recommendations(self, matching_shows: pd.DataFrame, 
                                     network: NetworkMatch, 
                                     concept_analyzer=None) -> List[RecommendationItem]:
        """Generate network-specific recommendations using the RecommendationEngine from ConceptAnalyzer.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level column
            network: Target network with compatibility score and other metrics
            concept_analyzer: ConceptAnalyzer instance that contains the RecommendationEngine
            
        Returns:
            List of RecommendationItem dictionaries with standardized structure
        """
        try:
            # Debug logging to verify network object type and attributes
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"DEBUG [network_analyzer.py:get_network_recommendations]: Network object type: {type(network).__name__}")

            
            # Get the RecommendationEngine from the ConceptAnalyzer
            recommendation_engine = concept_analyzer.recommendation_engine
                
            # Get network ID using direct attribute access
            network_id = network.network_id
                
            # Filter matching_shows to this network if needed
            network_shows = matching_shows
            if 'network_id' in matching_shows.columns:
                network_shows = matching_shows[matching_shows['network_id'] == network_id]
                
                if network_shows.empty:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"DEBUG: No shows found for network ID {network_id}")
                    return []
            
            # Extract criteria from matching_shows if possible
            # This is more reliable than passing potentially stale criteria
            criteria: CriteriaDict = {}
            if hasattr(matching_shows, 'criteria') and isinstance(matching_shows.criteria, dict):
                # Use the criteria exactly as provided - no conversion needed
                # The system should consistently use database column names (IDs) throughout
                criteria = matching_shows.criteria
                

            
            # Get integrated data from concept_analyzer if available
            integrated_data = None
            if hasattr(concept_analyzer, 'integrated_data'):
                integrated_data = concept_analyzer.integrated_data
            
            # Call the RecommendationEngine with the network-specific shows
            # This will generate recommendations specific to this network

            
            # Call the recommendation engine with the network object
            # Create a default confidence_info dictionary to ensure contract compliance
            confidence_info = update_confidence_info({}, {})
            return recommendation_engine.generate_network_specific_recommendations(
                criteria=criteria,
                network=network,
                matching_shows=network_shows,
                integrated_data=integrated_data,
                confidence_info=confidence_info
            )
        except Exception as e:
            st.error(f"Error generating network recommendations: {str(e)}")
            return []
    
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
