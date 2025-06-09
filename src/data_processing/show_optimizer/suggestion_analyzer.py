"""SuggestionAnalyzer: Formats and presents optimization results.

This component focuses on formatting and presenting results to the UI.
It takes analysis results from other components and prepares them for display.

Key responsibilities:

1. Result Formatting:
   - Format recommendations for UI display
   - Generate user-friendly explanations
   - Prepare data structures for UI components

2. Result Presentation:
   - Format criteria names and values for display
   - Format success probability and confidence information
   - Format component scores and network matches

3. UI Integration:
   - Support UI-specific formatting needs
   - Provide consistent presentation of results
   - Format complex data for easy understanding
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import streamlit as st

from .optimizer_config import OptimizerConfig
from .criteria_scorer import ComponentScore
from .network_analyzer import NetworkMatch
from .recommendation_engine import SuccessFactor, Recommendation
from .field_manager import FieldManager
    
    
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
    """Formats and presents optimization results for UI display."""
    
    def __init__(self, criteria_scorer):
        """Initialize the suggestion analyzer.
        
        Args:
            criteria_scorer: CriteriaScorer instance for scoring criteria
        """
        self.field_manager = criteria_scorer.field_manager
        self.criteria_scorer = criteria_scorer
        self.config = OptimizerConfig
        
    def format_optimization_summary(self, summary: OptimizationSummary) -> Dict[str, Any]:
        """Format an OptimizationSummary for UI display.
        
        Args:
            summary: OptimizationSummary to format
            
        Returns:
            Dictionary with formatted data for UI display
        """
        try:
            # Format success probability for display
            formatted_success = self._format_success_probability(summary.overall_success_probability, summary.confidence)
            
            # Format recommendations for display
            formatted_recommendations = self._format_recommendations(summary.recommendations)
            
            # Format network matches for display
            formatted_networks = self._format_network_matches(summary.top_networks)
            
            # Format component scores for display
            formatted_scores = self._format_component_scores(summary.component_scores)
            
            # Format match quality information
            match_quality_info = self._format_match_quality(
                summary.match_level, 
                summary.match_count,
                summary.match_counts_by_level,
                summary.confidence_score
            )
            
            # Return formatted data
            return {
                "success_probability": formatted_success,
                "recommendations": formatted_recommendations,
                "networks": formatted_networks,
                "component_scores": formatted_scores,
                "match_quality": match_quality_info,
                "matching_titles": summary.matching_titles[:10] if summary.matching_titles else []
            }
            
        except Exception as e:
            st.write(f"Debug: Error formatting optimization summary: {str(e)}")
            st.error("Unable to format optimization results. Please try again or contact support.")
            return {"error": "Formatting error", "details": str(e)}
            
    def _format_success_probability(self, probability: Optional[float], confidence: str) -> Dict[str, Any]:
        """Format success probability for display.
        
        Args:
            probability: Success probability value (0-1) or None
            confidence: Confidence level string
            
        Returns:
            Dictionary with formatted success probability information
        """
        try:
            # Format the probability as a percentage with appropriate precision
            if probability is None:
                formatted_value = "Unknown"
                percentage = None
                color = self.config.UI_COLORS['neutral']
            else:
                percentage = probability * 100
                if percentage < 1 and percentage > 0:
                    formatted_value = "<1%"
                elif percentage > 99 and percentage < 100:
                    formatted_value = ">99%"
                else:
                    formatted_value = f"{int(percentage)}%"
                
                # Determine color based on probability using config thresholds
                if percentage >= (self.config.THRESHOLDS['strong_compatibility'] * 100):
                    color = self.config.UI_COLORS['success']
                elif percentage >= (self.config.THRESHOLDS['minimum_compatibility'] * 100):
                    color = self.config.UI_COLORS['warning']
                else:
                    color = self.config.UI_COLORS['danger']
            
            # Format confidence level using config
            confidence_display = self.config.CONFIDENCE_DISPLAY.get(confidence, "Unknown confidence")
            
            return {
                "value": probability,
                "formatted": formatted_value,
                "percentage": percentage,
                "confidence": confidence,
                "confidence_display": confidence_display,
                "color": color
            }
        except Exception as e:
            st.write(f"Debug: Error formatting success probability: {str(e)}")
            st.error("Unable to format success probability data.")
            return {"formatted": "Error", "confidence_display": "Error", "color": self.config.UI_COLORS['neutral']}
            
    def _format_recommendations(self, recommendations: List[Recommendation]) -> List[Dict[str, Any]]:
        """Format recommendations for display.
        
        Args:
            recommendations: List of Recommendation objects
            
        Returns:
            List of dictionaries with formatted recommendation information
        """
        try:
            formatted_recommendations = []
            
            # Use recommendation types from config
            recommendation_types = self.config.RECOMMENDATION_TYPES
            
            for rec in recommendations:
                # Format impact score as a percentage with sign
                impact_percentage = abs(rec.impact_score * 100)
                if impact_percentage < 1 and impact_percentage > 0:
                    impact_formatted = "<1%"
                elif impact_percentage > 99:
                    impact_formatted = ">99%"
                else:
                    impact_formatted = f"{int(impact_percentage)}%"
                
                if rec.impact_score > 0:
                    impact_display = f"+{impact_formatted}"
                    impact_color = self.config.UI_COLORS['success']
                elif rec.impact_score < 0:
                    impact_display = f"-{impact_formatted}"
                    impact_color = self.config.UI_COLORS['danger']
                else:
                    impact_display = "0%"
                    impact_color = self.config.UI_COLORS['neutral']
                
                # Get recommendation type display name
                rec_type_display = recommendation_types.get(rec.rec_type, "Other Recommendation")
                
                # Format confidence level using config
                confidence_display = self.config.CONFIDENCE_DISPLAY.get(rec.confidence, "Unknown confidence")
                
                # Create formatted recommendation
                formatted_rec = {
                    "type": rec.rec_type,
                    "type_display": rec_type_display,
                    "explanation": rec.explanation,
                    "impact_score": rec.impact_score,
                    "impact_display": impact_display,
                    "impact_color": impact_color,
                    "confidence": rec.confidence,
                    "confidence_display": confidence_display,
                    "sample_size": rec.sample_size,
                    "matching_titles": rec.matching_titles[:5] if rec.matching_titles else []
                }
                
                formatted_recommendations.append(formatted_rec)
            
            return formatted_recommendations
            
        except Exception as e:
            st.write(f"Debug: Error formatting recommendations: {str(e)}")
            st.error("Unable to format recommendation data.")
            return []
            
    def _format_network_matches(self, networks: List[NetworkMatch]) -> List[Dict[str, Any]]:
        """Format network matches for display.
        
        Args:
            networks: List of NetworkMatch objects
            
        Returns:
            List of dictionaries with formatted network information
        """
        try:
            formatted_networks = []
            
            for network in networks:
                # Format compatibility score as a percentage
                compatibility_percentage = network.compatibility_score * 100
                if compatibility_percentage < 1 and compatibility_percentage > 0:
                    compatibility_formatted = "<1%"
                elif compatibility_percentage > 99:
                    compatibility_formatted = ">99%"
                else:
                    compatibility_formatted = f"{int(compatibility_percentage)}%"
                
                # Determine color based on compatibility using config thresholds
                if compatibility_percentage >= (self.config.THRESHOLDS['strong_compatibility'] * 100):
                    color = self.config.UI_COLORS['success']
                elif compatibility_percentage >= (self.config.THRESHOLDS['minimum_compatibility'] * 100):
                    color = self.config.UI_COLORS['warning']
                else:
                    color = self.config.UI_COLORS['danger']
                
                # Get network name
                network_name = network.name if hasattr(network, 'name') else "Unknown Network"
                
                # Create formatted network
                formatted_network = {
                    "id": network.network_id,
                    "name": network_name,
                    "compatibility_score": network.compatibility_score,
                    "compatibility_formatted": compatibility_formatted,
                    "color": color,
                    "tier": network.tier if hasattr(network, 'tier') else None,
                    "success_rate": network.success_rate if hasattr(network, 'success_rate') else None
                }
                
                formatted_networks.append(formatted_network)
            
            return formatted_networks
            
        except Exception as e:
            st.write(f"Debug: Error formatting network matches: {str(e)}")
            st.error("Unable to format network match data.")
            return []
                
    def _format_component_scores(self, component_scores: Dict[str, ComponentScore]) -> Dict[str, Dict[str, Any]]:
        """Format component scores for display.
        
        Args:
            component_scores: Dictionary of component scores
            
        Returns:
            Dictionary with formatted component score information
        """
        try:
            formatted_scores = {}
            
            for component_name, score in component_scores.items():
                # Format the score as a percentage
                score_percentage = score.score * 100
                if score_percentage < 1 and score_percentage > 0:
                    score_formatted = "<1%"
                elif score_percentage > 99:
                    score_formatted = ">99%"
                else:
                    score_formatted = f"{int(score_percentage)}%"
                
                # Determine color based on score using config thresholds
                if score_percentage >= (self.config.THRESHOLDS['strong_compatibility'] * 100):
                    color = self.config.UI_COLORS['success']
                elif score_percentage >= (self.config.THRESHOLDS['minimum_compatibility'] * 100):
                    color = self.config.UI_COLORS['warning']
                else:
                    color = self.config.UI_COLORS['danger']
                
                # Get display name for component
                display_name = self.field_manager.get_display_name(component_name) if hasattr(self, 'field_manager') else component_name
                
                # Create formatted component score
                formatted_scores[component_name] = {
                    "name": component_name,
                    "display_name": display_name,
                    "score": score.score,
                    "formatted": score_formatted,
                    "color": color,
                    "weight": score.weight,
                    "sample_size": score.sample_size
                }
            
            return formatted_scores
            
        except Exception as e:
            st.write(f"Debug: Error formatting component scores: {str(e)}")
            st.error("Unable to format component score data.")
            return {}
    def _format_match_quality(self, match_level: int, match_count: int, match_counts_by_level: Dict[int, int], confidence_score: float) -> Dict[str, Any]:
        """Format match quality information for display.
        
        Args:
            match_level: Match level used (1-4, where 1 is highest)
            match_count: Number of matching shows
            match_counts_by_level: Count of shows by match level
            confidence_score: Confidence score (0-1)
            
        Returns:
            Dictionary with formatted match quality information
        """
        try:
            # Format match level description using config
            match_level_display = self.config.MATCH_LEVEL_DESCRIPTIONS.get(match_level, "Unknown match level")
            
            # Format confidence score as a percentage
            confidence_percentage = confidence_score * 100
            if confidence_percentage < 1 and confidence_percentage > 0:
                confidence_formatted = "<1%"
            elif confidence_percentage > 99:
                confidence_formatted = ">99%"
            else:
                confidence_formatted = f"{int(confidence_percentage)}%"
            
            # Determine color based on confidence using config thresholds
            if confidence_percentage >= (self.config.THRESHOLDS['strong_compatibility'] * 100):
                color = self.config.UI_COLORS['success']
            elif confidence_percentage >= (self.config.THRESHOLDS['minimum_compatibility'] * 100):
                color = self.config.UI_COLORS['warning']
            else:
                color = self.config.UI_COLORS['danger']
            
            # Format match counts by level
            formatted_counts = {}
            for level, count in match_counts_by_level.items():
                level_desc = self.config.MATCH_LEVEL_DESCRIPTIONS.get(level, f"Level {level}")
                formatted_counts[str(level)] = {
                    "level": level,
                    "description": level_desc,
                    "count": count
                }
            
            return {
                "match_level": match_level,
                "match_level_display": match_level_display,
                "match_count": match_count,
                "confidence_score": confidence_score,
                "confidence_formatted": confidence_formatted,
                "color": color,
                "match_counts_by_level": formatted_counts
            }
            
        except Exception as e:
            st.write(f"Debug: Error formatting match quality: {str(e)}")
            st.error("Unable to format match quality data.")
            return {"match_level_display": "Error", "confidence_formatted": "Error", "color": self.config.UI_COLORS['neutral']}
                        
    def format_criteria_for_display(self, criteria: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Format criteria for display in the UI.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary with formatted criteria information
        """
        try:
            formatted_criteria = {}
            
            for field_name, value in criteria.items():
                # Get display name for field
                display_name = self.field_manager.get_display_name(field_name) if hasattr(self, 'field_manager') else field_name
                
                # Format value based on field type
                formatted_value = self._format_field_value(field_name, value)
                
                # Create formatted field
                formatted_criteria[field_name] = {
                    "name": field_name,
                    "display_name": display_name,
                    "value": value,
                    "formatted_value": formatted_value
                }
            
            return formatted_criteria
            
        except Exception as e:
            st.write(f"Debug: Error formatting criteria: {str(e)}")
            st.error("Unable to format criteria data.")
            return {}
    def _format_field_value(self, field_name: str, value: Any) -> str:
        """Format a field value for display based on field type.
        
        Args:
            field_name: Name of the field
            value: Value to format
            
        Returns:
            Formatted string representation of the value
        """
        try:
            # Handle None values
            if value is None:
                return ""
                
            # Handle list values
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
                
            # Handle boolean values
            if isinstance(value, bool):
                return "Yes" if value else "No"
                
            # Handle numeric values
            if isinstance(value, (int, float)):
                # Format percentages
                if field_name.endswith('_percent') or field_name.endswith('_rate'):
                    return f"{value * 100:.1f}%"
                return str(value)
                
            # Default: return as string
            return str(value)
        except Exception as e:
            st.write(f"Debug: Error formatting field value for {field_name}: {str(e)}")
            return str(value) if value is not None else ""  # Safe fallback
        
    def _get_criteria_name(self, criteria_type: str, criteria_value: Any) -> str:
        """Get the display name for a criteria value.
        
        Args:
            criteria_type: Type of criteria
            criteria_value: Value of criteria
            
        Returns:
            Display name for the criteria value
        """
        try:
            # Handle None values
            if criteria_value is None:
                return "None"
                
            # Handle list values
            if isinstance(criteria_value, list):
                names = [self._get_criteria_name(criteria_type, v) for v in criteria_value]
                return ", ".join(names)
                
            # Get options from field manager
            options = self.field_manager.get_options(criteria_type)
            
            # Find the option with this ID
            name = str(criteria_value)  # Default if not found
            for option in options:
                if option.id == criteria_value:
                    name = option.name
                    break
                    
            return name
        except Exception as e:
            st.write(f"Debug: Error getting criteria name for {criteria_type}: {str(e)}")
            return str(criteria_value)  # Fallback to string representation
