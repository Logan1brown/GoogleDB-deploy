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
    
    
@dataclass
class OptimizationSummary:
    """Summary of optimization recommendations."""
    overall_success_probability: float
    confidence: str
    top_networks: List[NetworkMatch]
    component_scores: Dict[str, ComponentScore]
    recommendations: List[Recommendation]
    success_factors: List[SuccessFactor]


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
        import streamlit as st
        
        try:
            # Get overall success probability
            st.write("DEBUG - Getting overall success rate")
            success_probability, confidence = self.criteria_analyzer.get_overall_success_rate(criteria)
            st.write(f"DEBUG - Success probability: {success_probability}, Confidence: {confidence}")
            
            # Get top networks
            st.write("DEBUG - Ranking networks by compatibility")
            top_networks = self.criteria_analyzer.rank_networks_by_compatibility(criteria, limit=5)
            st.write(f"DEBUG - Top networks count: {len(top_networks)}")
            
            # Get component scores
            st.write("DEBUG - Analyzing components")
            component_scores = self.criteria_analyzer.analyze_components(criteria)
            st.write(f"DEBUG - Component scores: {component_scores}")
            
            # Verify component scores are valid
            if not component_scores:
                st.error("DEBUG ERROR: Empty component scores returned from analyze_components")
                raise ValueError("No component scores available for analysis")
                
            # Check that all expected component scores are present
            expected_components = ['audience', 'critics', 'longevity']
            missing_components = [comp for comp in expected_components if comp not in component_scores]
            if missing_components:
                st.error(f"DEBUG ERROR: Missing component scores: {missing_components}")
                raise ValueError(f"Missing required component scores: {missing_components}")
            
            # Get success factors
            st.write("DEBUG - Identifying success factors")
            success_factors = self.criteria_analyzer.identify_success_factors(criteria, limit=5)
            st.write(f"DEBUG - Success factors count: {len(success_factors)}")
            
            # Generate recommendations
            st.write("DEBUG - Generating recommendations")
            recommendations = self.generate_recommendations(criteria, success_factors, top_networks)
            st.write(f"DEBUG - Recommendations count: {len(recommendations)}")
            
            # Create summary
            st.write("DEBUG - Creating OptimizationSummary")
            summary = OptimizationSummary(
                overall_success_probability=success_probability,
                confidence=confidence,
                top_networks=top_networks,
                component_scores=component_scores,
                recommendations=recommendations,
                success_factors=success_factors
            )
            st.write(f"DEBUG - Summary created successfully: {type(summary)}")
            return summary
                
        except Exception as e:
            st.write(f"DEBUG - Error in analyze_show_concept: {str(e)}")
            import traceback
            st.write(f"DEBUG - Traceback: {traceback.format_exc()}")
            return None
    
    def generate_recommendations(self, criteria: Dict[str, Any], 
                               success_factors: List[SuccessFactor],
                               top_networks: List[NetworkMatch]) -> List[Recommendation]:
        """Generate recommendations for optimizing a show concept.
        
        Args:
            criteria: Dictionary of criteria
            success_factors: List of success factors
            top_networks: List of top matching networks
            
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
        
        # Analyze complementary criteria
        complementary_recs = self._recommend_complementary_criteria(criteria)
        recommendations.extend(complementary_recs)
        
        # Sort by impact score (absolute value, as negative impacts are also important)
        recommendations.sort(key=lambda x: abs(x.impact_score), reverse=True)
        
        return recommendations
    
    def _recommend_missing_criteria(self, criteria: Dict[str, Any], 
                                  success_factors: List[SuccessFactor]) -> List[Recommendation]:
        """Recommend high-impact criteria that are missing from the concept.
        
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
                        
                recommendation = Recommendation(
                    recommendation_type="replace",
                    criteria_type=factor.criteria_type,
                    current_value=criteria[factor.criteria_type],
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
            rate, _ = self.criteria_analyzer.get_overall_success_rate(single_criteria)
            overall_rates[criteria_type] = rate
        
        # Find criteria where network rate differs significantly from overall rate
        for criteria_type, network_rate in network_rates.items():
            if criteria_type not in overall_rates:
                continue
                
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
        """Recommend complementary criteria that work well with existing criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            List of Recommendation objects
        """
        recommendations = []
        
        # This would require analyzing combinations of criteria to identify complementary pairs
        # For now, we'll implement a simplified version that looks for known complementary pairs
        
        # Example: If we have "sci-fi" genre, recommend "futuristic" setting
        if 'genre' in criteria:
            genre_ids = criteria['genre'] if isinstance(criteria['genre'], list) else [criteria['genre']]
            
            for genre_id in genre_ids:
                # Check for complementary criteria from OptimizerConfig
                complementary = OptimizerConfig.COMPLEMENTARY_CRITERIA.get('genre', {}).get(str(genre_id), [])
                
                for comp in complementary:
                    comp_type = comp['criteria_type']
                    comp_value = comp['criteria_value']
                    comp_impact = comp['impact_score']
                    
                    # Skip if this criteria is already included
                    if comp_type in criteria:
                        current_value = criteria[comp_type]
                        if isinstance(current_value, list) and comp_value in current_value:
                            continue
                        if current_value == comp_value:
                            continue
                    
                    # Get names
                    genre_name = self._get_criteria_name('genre', genre_id)
                    comp_name = self._get_criteria_name(comp_type, comp_value)
                    
                    recommendation = Recommendation(
                        recommendation_type="add",
                        criteria_type=comp_type,
                        current_value=criteria.get(comp_type),
                        suggested_value=comp_value,
                        suggested_name=comp_name,
                        impact_score=comp_impact,
                        confidence="medium",
                        explanation=f"'{comp_name}' works well with '{genre_name}' and could improve success probability by approximately {comp_impact:.0%}."
                    )
                    recommendations.append(recommendation)
        
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
