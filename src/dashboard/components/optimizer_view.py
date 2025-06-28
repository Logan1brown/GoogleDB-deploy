"""OptimizerView: Bridge between ShowOptimizer and UI.

This component serves as the bridge between the ShowOptimizer analysis engine
and the UI layer, handling initialization, state management, and result formatting.

Key responsibilities:

1. Initialization:
   - Initialize ShowOptimizer and related components
   - Cache field options and display options in state
   - Manage initialization status

2. Analysis Execution:
   - Run analysis with normalized criteria
   - Handle errors and provide user feedback
   - Store results in state

3. Result Formatting:
   - Format recommendations for UI display
   - Generate user-friendly explanations
   - Format success probability and confidence information
   - Format component scores and network matches
   - Prepare data structures for UI components
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import streamlit as st

from src.data_processing.show_optimizer.optimizer_config import OptimizerConfig
from src.data_processing.show_optimizer.criteria_scorer import ComponentScore
from src.data_processing.show_optimizer.recommendation_engine import SuccessFactor
from src.data_processing.show_optimizer.optimizer_data_contracts import RecommendationItem
from src.data_processing.show_optimizer.field_manager import FieldManager
from src.data_processing.show_optimizer.optimizer_concept_analyzer import OptimizationSummary
from src.data_processing.show_optimizer.optimizer_data_contracts import (
    CriteriaDict, ConfidenceInfo, IntegratedData, 
    NetworkMatch, RecommendationItem, FieldValueData, FieldValueSuccessRate
)


class OptimizerView:
    """View component for formatting ShowOptimizer results for UI display.
    
    This component is responsible for transforming raw optimization data into UI-ready formats.
    It centralizes all UI formatting logic, ensuring a clear separation between business logic
    and presentation. The OptimizerView maintains a consistent data contract with UI components
    by providing standardized formatted data structures.
    
    Key responsibilities:
    1. Format raw optimization results into UI-ready data structures
    2. Generate human-readable explanations for recommendations
    3. Format success probability and confidence information
    4. Format component scores and network matches
    5. Prepare data structures for UI components with consistent interfaces
    
    Data contracts:
    - Input: Raw data objects from ShowOptimizer (Recommendation, NetworkMatch, etc.)
    - Output: Formatted dictionaries with consistent keys and display-ready values
    """
    
    def __init__(self, show_optimizer):
        """Initialize the optimizer view.
        
        Args:
            show_optimizer: ShowOptimizer instance that this view will format results for.
        """
        self.optimizer = show_optimizer
        self.field_manager = show_optimizer.field_manager
        self.criteria_scorer = show_optimizer.criteria_scorer
        self.config = OptimizerConfig
        
    def update_references(self):
        """Update references to field_manager and criteria_scorer from the optimizer.
        
        This method should be called if the optimizer is updated after this view is created.
        """
        self.field_manager = self.optimizer.field_manager
        self.criteria_scorer = self.optimizer.criteria_scorer
    

    # The run_analysis method has been removed as it's now handled by ShowOptimizer directly
    # OptimizerView now focuses solely on formatting and rendering
        
    def format_optimization_summary(self, summary: OptimizationSummary) -> OptimizationSummary:
        """Format the optimization summary for display in the UI.
        
        This method creates a comprehensive UI-ready data structure attached to summary.formatted_data.
        All UI components should use this structure for rendering, which provides a consistent
        data contract between the business logic and UI rendering.
        
        DATA CONTRACT:
        The formatted_data dictionary contains the following keys and structures:
        
        - 'networks': List of dictionaries with network match data
          - 'Network': str - Display name of the network
          - 'Compatibility': str - Formatted compatibility score (e.g., '85%')
          - 'Success Probability': str - Formatted success probability (e.g., '72.5%')
          - 'Sample Size': int - Number of shows in the sample
          - 'Confidence': str - Display text for confidence level
          - '_compatibility_raw': float - Raw compatibility score for sorting
          - '_success_prob_raw': float - Raw success probability for sorting
        
        - 'component_scores': Dict mapping component names to score dictionaries
          - Each score dict contains: 'score', 'confidence', 'sample_size'
        
        - 'match_quality': Dict with match quality information
          - 'match_level': int - The match level (1-4)
          - 'match_level_name': str - Human-readable match level name
          - 'match_count': int - Number of matching shows
          - 'match_counts_by_level': Dict - Counts of shows by match level
          - 'confidence_score': float - Confidence score for the match quality
        
        - 'success_factors': List of dictionaries with success factor data
          - 'Type': str - Display name of the criteria type
          - 'Name': str - Display name of the criteria value
          - 'Impact': float - Raw impact score
          - 'ImpactDisplay': str - Formatted impact score
          - 'Confidence': str - Display text for confidence level
          - 'Sample': int - Sample size
          - 'SampleDisplay': str - Formatted sample size text
          - '_impact_raw': float - Raw impact score for sorting
          - '_confidence_level': int - Numeric confidence level for sorting
          - '_matching_titles': List[str] - List of matching show titles if available
        
        - 'recommendations': Dict containing grouped recommendations
          - 'grouped': Dict mapping recommendation types to lists of recommendation dictionaries
          - 'network_specific': List of network-specific recommendation dictionaries
          - 'all': List of all recommendation dictionaries
          - Each recommendation dict contains formatted display values and metadata
        
        - 'matching_shows': DataFrame with matching shows data
          - Contains original columns plus 'match_level_desc' for display
        
        - 'success_probability': Dict with success probability information
          - 'value': str - Formatted probability value (e.g., '75%')
          - 'confidence': str - Display text for confidence level
          - 'confidence_level': str - Raw confidence level for logic
        
        Args:
            summary: The optimization summary to format (raw data from ShowOptimizer)
            
        Returns:
            The formatted optimization summary with attached formatted_data dictionary
            that follows the data contract described above
        """
        # Create a new formatted_data attribute as a dictionary
        # Important: Use a different attribute name to avoid recursive reference
        setattr(summary, '_formatted_data_dict', {})
        
        # Format network matches if available
        if hasattr(summary, 'top_networks') and summary.top_networks:
            summary._formatted_data_dict['networks'] = self._format_network_matches(summary.top_networks)
            
        # Format component scores directly in the view layer
        if hasattr(summary, 'component_scores'):
            # Format component scores for UI display - expect all scores to be ComponentScore objects
            component_scores = {}
            for component, score in summary.component_scores.items():
                # Format the ComponentScore object for UI display
                component_scores[component] = {
                    'score': float(score.score) if score.score is not None else None,
                    'sample_size': score.sample_size,
                    'confidence': score.confidence,
                    'label': component.replace('_', ' ').title(),
                    'description': self._get_component_description(component)
                }
            
            # Debug log the component scores
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Formatted component scores: {component_scores.keys()}", category='format')
                
            summary._formatted_data_dict['component_scores'] = component_scores
            
        # Format match quality if available
        if hasattr(summary, 'match_level') and hasattr(summary, 'match_count'):
            # Create a proper match quality dictionary with all required fields
            match_quality_data = {
                'match_level': summary.match_level,
                'match_count': summary.match_count,
                'match_counts_by_level': summary.match_counts_by_level,
                'confidence_score': summary.confidence_score,
                'match_quality_score': summary.match_quality  # The float value
            }
            summary._formatted_data_dict['match_quality'] = match_quality_data
            
        # Format success factors if available
        if hasattr(summary, 'success_factors') and summary.success_factors:
            summary._formatted_data_dict['success_factors'] = self._format_success_factors(summary.success_factors)
            
        # Format recommendations if available
        if hasattr(summary, 'recommendations') and summary.recommendations:
            summary._formatted_data_dict['recommendations'] = self._format_recommendations(summary.recommendations)
            
        # Format matching shows if available
        if hasattr(summary, 'matching_shows') and not summary.matching_shows.empty:
            # Format the matching shows for UI display
            formatted_shows = self._format_matching_shows(summary.matching_shows)
            
            # Store the formatted shows in the summary
            summary._formatted_data_dict['matching_shows'] = formatted_shows
            
            # Replace the original DataFrame with the formatted one
            summary.matching_shows = formatted_shows
        
        # Format success probability - expect it to be present
        # The attribute is named overall_success_probability in the OptimizationSummary class
        probability = summary.overall_success_probability
        confidence = summary.confidence
        
        # Debug log the success probability
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Formatting success probability: {probability:.4f}, confidence: {confidence}", category='format')
            
        # The key in the formatted data dictionary is 'success_probability'
        summary._formatted_data_dict['success_probability'] = self._format_success_probability(
            probability, confidence
        )
        
        # Set the formatted_data property to the dictionary
        summary.formatted_data = summary._formatted_data_dict
        
        return summary
    
    def _get_component_description(self, component_name: str) -> str:
        """
        Get the description for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Description of the component
        """
        # Define descriptions for each component
        descriptions = {
            'success': 'Overall probability of success based on historical data',
            'audience': 'Audience appeal and engagement potential',
            'critics': 'Projected critical reception and reviews',
            'longevity': 'Expected show lifespan and sustainability'
        }
        
        # Return the description or empty string if not found
        return descriptions.get(component_name, '')
    
    def _format_success_probability(self, probability: Optional[float], confidence: str) -> Dict[str, str]:
        """
        Format success probability for UI display.
        
        Args:
            probability: Success probability (0-1) or None
            confidence: Confidence level (none, low, medium, high)
            
        Returns:
            Dictionary with formatted success probability data including display, subtitle, etc.
        """
        # Format probability as percentage
        if probability is not None:
            # Get confidence display text from config - expect it to be a valid key
            confidence_display = self.config.CONFIDENCE_DISPLAY[confidence]
            
            return {
                "display": f"{probability:.0%}",
                "subtitle": f"Confidence: {confidence_display}"
            }
        else:
            return {
                "display": "N/A",
                "subtitle": "Success probability not available"
            }
        
    def _format_recommendations(self, recommendations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Union[List[Dict[str, Union[str, float, int, bool]]], Dict[str, List[Dict[str, Union[str, float, int, bool]]]]]]:
        """
        Format recommendations for display in the UI.
        
        This method formats and groups recommendations by their type (add, change, remove, network_keep, 
        network_change). The recommendation types are determined in criteria_scorer.py and preserved 
        throughout the pipeline. This method does not modify the recommendation types but only formats 
        them for UI display.
        
        Recommendation types follow these rules:
        - 'add': For suggesting new unselected fields with positive impact
        - 'change': For suggesting different values for already selected fields
        - 'remove': For suggesting removal of selected fields with negative impact
        - 'network_keep': For suggesting to keep elements that work well with a specific network
        - 'network_change': For suggesting to change elements that don't work well with a specific network
        
        Args:
            recommendations: Dictionary with 'general' and 'network_specific' recommendation lists
            
        Returns:
            Dictionary with formatted recommendations grouped by type:
            - 'grouped': Dict mapping recommendation types to lists of formatted recommendations
            - 'network_specific': List of network-specific recommendations
            - 'all': List of all formatted recommendations
            
            Each recommendation contains fields like field, option, impact, explanation, etc.
        """
        # Initialize with standard recommendation types and their display headers
        config = OptimizerConfig()
        grouped = {
            'add': {'items': [], 'header': "Add to Your Concept"},
            'change': {'items': [], 'header': "Consider Changing"},
            'remove': {'items': [], 'header': "Consider Removing"},
            'network_specific': {'items': [], 'header': "Network Recommendations"}
        }
        
        # Add any additional types from config
        for rec_type in config.RECOMMENDATION_TYPES.keys():
            if rec_type not in grouped:
                # Get display header from config or create a default one
                header = config.RECOMMENDATION_TYPES.get(rec_type, f"{rec_type.replace('_', ' ').title()} Recommendations")
                grouped[rec_type] = {'items': [], 'header': header}
            
        # Extract general and network-specific recommendations from the dictionary structure
        general_recs = recommendations.get('general', [])
        network_specific_recs = recommendations.get('network_specific', [])
        
        # Track formatted network-specific recommendations
        network_specific_formatted = []
        
        # Debug log the recommendations count
        if OptimizerConfig.DEBUG_MODE:
            general_count = len(general_recs)
            network_count = len(network_specific_recs)
            total_count = general_count + network_count
            OptimizerConfig.debug(f"Formatting {total_count} recommendations ({general_count} general, {network_count} network-specific)", category='recommendation', force=True)
            
            if general_recs:
                # Count recommendations by type - use dictionary-style access
                type_counts = {}
                for rec in general_recs:
                    rec_type = rec['recommendation_type']
                    if rec_type not in type_counts:
                        type_counts[rec_type] = 0
                    type_counts[rec_type] += 1
                
                OptimizerConfig.debug(f"General recommendation types: {type_counts}", category='recommendation', force=True)
                
                # Special debug for 'remove' recommendations - use dictionary-style access
                remove_recs = [rec for rec in general_recs if rec['recommendation_type'] == 'remove']
                if remove_recs:
                    OptimizerConfig.debug(f"Found {len(remove_recs)} 'remove' recommendations before formatting", category='recommendation', force=True)
                    # Inspect the first few remove recommendations in detail
                    for i, rec in enumerate(remove_recs[:3]):
                        OptimizerConfig.debug(f"Remove recommendation {i+1}:", category='recommendation', force=True)
                        OptimizerConfig.debug(f"  - Type: {rec['recommendation_type']}", category='recommendation', force=True)
                        OptimizerConfig.debug(f"  - Field: {rec.get('field', rec.get('criteria_type', 'unknown'))}", category='recommendation', force=True)
                        OptimizerConfig.debug(f"  - Impact: {rec.get('impact_score', rec.get('impact', 0.0))}", category='recommendation', force=True)
                    for rec in remove_recs:
                        OptimizerConfig.debug(f"Remove rec: {rec['criteria_type']}/{rec['suggested_name']}", category='recommendation', force=True)
                else:
                    OptimizerConfig.debug("No 'remove' recommendations found in original recommendations", category='recommendation', force=True)
                
                # Show the first few general recommendations in detail
                for i, rec in enumerate(general_recs[:3]):
                    OptimizerConfig.debug(f"General recommendation {i+1} details:", category='recommendation', force=True)
                    OptimizerConfig.debug(f"  - Type: {rec['recommendation_type']}", category='recommendation')
                    OptimizerConfig.debug(f"  - Field: {rec.get('field', 'unknown')}", category='recommendation')
                    OptimizerConfig.debug(f"  - Suggested Name: {rec.get('suggested_name', 'unknown')}", category='recommendation')
                    OptimizerConfig.debug(f"  - Impact: {rec.get('impact', 0.0)}", category='recommendation')
                    OptimizerConfig.debug(f"  - Explanation: {rec.get('explanation', rec.get('description', 'No explanation'))}", category='recommendation')
                    # Debug all attributes
                    # Debug removed for clarity
                
        # If no recommendations, add debug output
        if not recommendations:
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("No recommendations to format", category='recommendation', force=True)
            # Return empty structure to avoid errors
            return {
                "grouped": grouped,
                "network_specific": [],
                "all": []
            }
            
        formatted_recommendations = []
        
        # Process general recommendations
        for rec in general_recs:
            # Use recommendation_type with dictionary-style access
            rec_type = rec['recommendation_type']
                
            # Format impact percentage for display
            # Always use 'impact' as the standard field name from the recommendation engine
            impact_score = rec.get('impact', 0.0)
            impact_percent = abs(impact_score * 100)
            impact_direction = "Increase" if impact_score > 0 else "Decrease"
            
            # Special debug for 'remove' recommendations
            if rec_type == 'remove':
                field_name = rec.get('field', 'unknown')  # Always use 'field' as the standard field name
                suggested_name = rec.get('suggested_name', 'unknown')
                OptimizerConfig.debug(f"REMOVE RECOMMENDATION FOUND IN OPTIMIZER_VIEW: {field_name}/{suggested_name}", category='recommendation', force=True)
            
            # Create recommendation title without impact percentage
            if rec_type.startswith('network_'):
                # For network recommendations
                clean_rec_type = rec_type.replace('network_', '')
                # Extract network name from suggested_name
                network_name = rec.get('suggested_name', '')
                if ':' in network_name:
                    network_name = network_name.split(':', 1)[0].strip()
                    
                field_name = rec.get('field', '')  # Always use 'field' as the standard field name
                title = f"{network_name} - {clean_rec_type.capitalize()} {field_name}"
            else:
                # Format the title to include only the field and suggested name
                # Always use 'field' as the standard field name from the recommendation engine
                field_name = rec.get('field', '').replace('_', ' ').title()
                suggested_name = rec.get('suggested_name', '')
                
                # Create a clean title without the impact information
                title = f"{field_name}: {suggested_name}"
                
            # Generate explanation text based on recommendation type and data
            # This is now the responsibility of the OptimizerView since the RecommendationEngine no longer provides formatted text
            explanation = self._generate_explanation_text(rec)
            
            # Make sure the explanation ends with a period for consistency
            if explanation and not explanation.strip().endswith('.'):
                explanation = f"{explanation}."
            
            # Create a formatted recommendation object with all necessary fields
            formatted_rec = {
                "title": title,
                "explanation": explanation,
                "impact_percent": impact_percent,
                "impact_direction": impact_direction,
                "recommendation_type": rec_type,  # Preserve the original recommendation_type
                "category": rec_type,  # Keep category for backward compatibility
                "criteria_type": rec.get('field', ''),  # Add criteria_type mapped from field
                "impact": impact_score,
                "field": rec.get('field', ''),  # Always use 'field' as the standard field name
                "suggested_name": rec.get('suggested_name', ''),
                # Dictionary-style access for values
                "current_value": rec.get('current_value', None),
                "suggested_value": rec.get('suggested_value', None),
                
                # Raw data for sorting and filtering
                "_impact_raw": impact_score,
                "_confidence_level": self._get_confidence_level(rec.get('confidence', 'medium')),
                "_rec_type": rec_type
            }
            
            # Keep debug output for 'remove' recommendations as they're specifically related to recommendations
            if rec_type == 'remove' and OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Created formatted 'remove' recommendation: {title}", category='recommendation', force=True)
            
            # Add to formatted recommendations list
            formatted_recommendations.append(formatted_rec)
            
            # Add to appropriate group based on recommendation type
            if rec_type.startswith('network_'):
                # Handle network-specific recommendations from general recommendations
                # This should be rare since we now separate them at the source
                network_specific_formatted.append(formatted_rec)
                
                # Ensure network-specific recommendations are properly grouped
                if 'network_specific' not in grouped:
                    grouped['network_specific'] = {'items': [], 'header': "Network Recommendations"}
                grouped['network_specific']['items'].append(formatted_rec)
                
                # Keep debug output for network recommendations as they're specifically related to recommendations
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Added network recommendation from general recs to 'network_specific' group: {formatted_rec['title']}", category='recommendation')
                
            # Ensure the group exists
            if rec_type not in grouped:
                # Create a display header for this group
                header = rec_type.replace('_', ' ').title()
                if rec_type.startswith('network_'):
                    network_name = rec_type.replace('network_', '').replace('_', ' ').title()
                    header = f"{network_name} Network Recommendations"
                grouped[rec_type] = {'items': [], 'header': header}
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Created missing group '{rec_type}' in grouped dictionary", category='recommendation', force=True)
                
            # Add the recommendation to its group
            grouped[rec_type]['items'].append(formatted_rec)
            
            # Debug output
            if OptimizerConfig.DEBUG_MODE:
                # Debug removed for clarity
                pass
                
            # Special debug for 'remove' recommendations
            if rec_type == 'remove' and OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"VERIFY: Added 'remove' recommendation to grouped dictionary", category='recommendation', force=True)
                OptimizerConfig.debug(f"  - Group 'remove' now has {len(grouped['remove'])} recommendations", category='recommendation', force=True)
        
        # Now process the network-specific recommendations that are provided separately
        if network_specific_recs:
            # Add focused debugging to identify if any recommendations are strings
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Processing {len(network_specific_recs)} network-specific recommendations", category='recommendation', force=True)
                # Check each recommendation's type
                for i, rec in enumerate(network_specific_recs):
                    OptimizerConfig.debug(f"Network recommendation {i} type: {type(rec)}", category='recommendation', force=True)
                    if isinstance(rec, str):
                        OptimizerConfig.debug(f"FOUND STRING RECOMMENDATION: {rec}", category='recommendation', force=True)
                    elif isinstance(rec, dict):
                        OptimizerConfig.debug(f"Dictionary recommendation keys: {list(rec.keys())}", category='recommendation', force=True)
                    else:
                        OptimizerConfig.debug(f"Unexpected recommendation type: {type(rec)}", category='recommendation', force=True)
            
            for rec in network_specific_recs:
                # Use recommendation_type with dictionary-style access
                rec_type = rec['recommendation_type']
                
                # Format impact percentage for display
                # Always use 'impact' as the standard field name from the recommendation engine
                impact_score = rec['impact']
                impact_percent = abs(impact_score * 100)
                impact_direction = "Increase" if impact_score > 0 else "Decrease"
                
                # Special debug for 'remove' recommendations
                if rec_type == 'remove' and OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"REMOVE RECOMMENDATION FOUND IN OPTIMIZER_VIEW: {rec['field']}/{rec['suggested_name']}", category='recommendation', force=True)
                
                # Create recommendation title without impact percentage
                if rec_type.startswith('network_'):
                    # For network recommendations
                    clean_rec_type = rec_type.replace('network_', '')
                    # Extract network name from suggested_name
                    network_name = rec['suggested_name']
                    if ':' in network_name:
                        network_name = network_name.split(':', 1)[0].strip()
                        
                    # Always use 'field' as the standard field name
                    field_name = rec['field']
                    title = f"{network_name} - {clean_rec_type.capitalize()} {field_name}"
                else:
                    # Format the title to include only the field and suggested name
                    # Always use 'field' as the standard field name
                    field_name = rec['field'].replace('_', ' ').title()
                    suggested_name = rec['suggested_name']
                    
                    # Create a clean title without the impact information
                    title = f"{field_name}: {suggested_name}"
                    
                # Generate explanation text based on recommendation type and data
                explanation = self._generate_explanation_text(rec)
                
                # Make sure the explanation ends with a period for consistency
                if explanation and not explanation.strip().endswith('.'):
                    explanation = f"{explanation}."
                
                # Create formatted recommendation dictionary with dictionary-style access
                formatted_rec = {
                    # Display values
                    "title": title,
                    "description": explanation,
                    "importance": rec['confidence'],
                    "category": rec_type,  # This is the key field for grouping
                    "impact": impact_score,
                    "field": rec['field'],  # Always use 'field' as the standard field name
                    "suggested_name": rec['suggested_name'],
                    # Dictionary-style access for values
                    "current_value": rec['current_value'],
                    "suggested_value": rec['suggested_value'],
                    
                    # Raw data for sorting and filtering
                    "_impact_raw": impact_score,
                    "_confidence_level": self._get_confidence_level(rec['confidence']),
                    "_rec_type": rec_type
                }
                
                # Add to formatted recommendations list
                formatted_recommendations.append(formatted_rec)
                
                # Add to network_specific group and tracking list
                grouped['network_specific']['items'].append(formatted_rec)
                network_specific_formatted.append(formatted_rec)
                
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Added network-specific recommendation: {formatted_rec['title']}", category='recommendation')
        
        # Sort each group by impact score (descending)
        for rec_type in grouped:
            if 'items' in grouped[rec_type]:
                grouped[rec_type]['items'].sort(key=lambda x: abs(x["_impact_raw"]), reverse=True)
        
        # Debug log the formatted recommendations structure
        if OptimizerConfig.DEBUG_MODE:
            non_empty_groups = [k for k, v in grouped.items() if v.get('items', [])]
            OptimizerConfig.debug(f"Formatted recommendations structure", category='recommendation')
            OptimizerConfig.debug(f"Non-empty groups: {non_empty_groups}", category='recommendation')
            OptimizerConfig.debug(f"Network-specific recommendations: {len(network_specific_formatted)}", category='recommendation')
            
            # Log counts for each recommendation type
            for rec_type in config.RECOMMENDATION_TYPES.keys():
                count = len(grouped.get(rec_type, {}).get('items', []))
                OptimizerConfig.debug(f"Group '{rec_type}' has {count} recommendations", category='recommendation')
                if count > 0:
                    OptimizerConfig.debug(f"First item in group '{rec_type}': {grouped[rec_type]['items'][0]['title']}", category='recommendation')
                
        # Sort recommendations within each group by impact (absolute value)
        for rec_type, group_data in grouped.items():
            if group_data.get('items', []):
                group_data['items'].sort(key=lambda x: abs(x.get('_impact_raw', 0)), reverse=True)
        
        # Final debug output - simplified to only essential recommendation information
        if OptimizerConfig.DEBUG_MODE:
            # Log counts of different recommendation types
            remove_recs = grouped.get('remove', {}).get('items', [])
            change_recs = grouped.get('change', {}).get('items', [])
            add_recs = grouped.get('add', {}).get('items', [])
            network_recs = len(network_specific_formatted)
            
            OptimizerConfig.debug(f"Recommendation counts: Add={len(add_recs)}, Change={len(change_recs)}, Remove={len(remove_recs)}, Network={network_recs}", category='recommendation')
            
            # Only log detailed info for remove recommendations as they're critical
            if remove_recs:
                for i, rec in enumerate(remove_recs[:3]):
                    OptimizerConfig.debug(f"  Remove rec {i+1}: {rec['title']} - Impact: {rec['impact']}", category='recommendation')
            else:
                OptimizerConfig.debug("No 'remove' recommendations found", category='recommendation')
        
        # Make sure 'remove' recommendations are included in the grouped dictionary
        # This is a critical check to ensure we don't lose any remove recommendations
        remove_recs_formatted = [rec for rec in formatted_recommendations if rec['category'] == 'remove']
        if remove_recs_formatted and 'remove' not in grouped:
            grouped['remove'] = {'items': [], 'header': "Consider Removing"}
            
        # Add any remove recommendations that might have been missed
        for rec in remove_recs_formatted:
            if rec not in grouped.get('remove', {}).get('items', []):
                if 'remove' not in grouped:
                    grouped['remove'] = {'items': [], 'header': "Consider Removing"}
                grouped['remove']['items'].append(rec)
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Added missing remove recommendation: {rec['title']}", category='recommendation')
        
        # Check if there are recommendations but no grouped recommendations with items
        all_formatted_recs = [rec for group_data in grouped.values() for rec in group_data.get('items', [])]        
        if formatted_recommendations and (not all_formatted_recs or all(len(group_data.get('items', [])) == 0 for group_data in grouped.values())):
            # Create default group with all recommendations
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("No recommendations found in any group, creating default group", category='recommendation', force=True)
            grouped = {"add": {"items": formatted_recommendations, "header": "Add to Your Concept"}}
            
        # Return the formatted recommendations
        # Ensure network-specific recommendations aren't duplicated in the 'all' list
        # Since they're already included in grouped['network_specific']
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Preparing final recommendations dictionary", category='recommendation', force=True)
            OptimizerConfig.debug(f"Network-specific count: {len(network_specific_formatted)}", category='recommendation', force=True)
            OptimizerConfig.debug(f"Grouped keys: {list(grouped.keys())}", category='recommendation', force=True)
            for k, group_data in grouped.items():
                OptimizerConfig.debug(f"Group '{k}' has {len(group_data.get('items', []))} items with header '{group_data.get('header', '')}'")
            
        # Return the structure expected by render_recommendations
        # This is a direct structure with 'general' and 'network_specific' keys
        # We need to exclude the 'network_specific' group from the general recommendations
        general_flattened = []
        
        # Only include recommendations from 'add', 'change', and 'remove' groups
        for group_name, group_data in grouped.items():
            if group_name in ['add', 'change', 'remove']:
                for rec in group_data.get('items', []):
                    general_flattened.append(rec)
        
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Flattened general recommendations count: {len(general_flattened)}", category='recommendation', force=True)
            for i, rec in enumerate(general_flattened[:3]):
                OptimizerConfig.debug(f"Flattened general rec {i+1} type: {rec.get('recommendation_type', 'MISSING')}", category='recommendation', force=True)
        
        return {
            "general": general_flattened,
            "network_specific": network_specific_formatted
        }
    
    def _format_network_matches(self, network_matches: List[NetworkMatch]) -> List[Dict[str, Any]]:
        """Format network matches for display.
        
        Args:
            network_matches: List of NetworkMatch dataclass objects
            
        Returns:
            List of formatted network match dictionaries for UI display
        """
        formatted = []
        
        if not network_matches:
            return []
        
        for match in network_matches:
            # Use attribute-style access for NetworkMatch objects
            network_id = match.network_id
            
            # Convert network_id to integer if needed
            if isinstance(network_id, (str, float)):
                network_id = int(float(network_id))
                
            # Get name from field manager - use the network_name attribute if available
            network_name = match.network_name if hasattr(match, 'network_name') and match.network_name else self.field_manager.get_name('network', network_id)
            if not network_name:
                network_name = "Unknown Network"
            
            # Format compatibility score as percentage
            compatibility_value = float(match.compatibility_score)
            # Check if value is already a percentage (>1.0)
            if compatibility_value > 1.0:
                compatibility_value = compatibility_value / 100.0  # Convert to decimal
            compatibility_display = f"{compatibility_value*100:.1f}%"
            compatibility_raw = compatibility_value
            
            # Format success probability as percentage
            success_value = float(match.success_probability) if match.success_probability is not None else 0.0
            # Check if value is already a percentage (>1.0)
            if success_value > 1.0:
                success_value = success_value / 100.0  # Convert to decimal
            success_display = f"{success_value*100:.1f}%"
            success_raw = success_value
                
            # Get confidence display text from config
            confidence = match.confidence or "unknown"
            sample_size = match.sample_size if hasattr(match, 'sample_size') else 0
            confidence_display = self.config.CONFIDENCE_DISPLAY.get(confidence, confidence.capitalize())
            
            # Format the network match with all data needed for UI display
            formatted.append({
                # Display values
                'Network': network_name,
                'Compatibility': compatibility_display,
                'Success Probability': success_display,
                'Sample Size': sample_size,
                'Confidence': confidence_display,
                
                # Raw values for sorting (not displayed)
                '_compatibility_raw': compatibility_raw,
                '_success_prob_raw': success_raw
            })
            
        # Sort the formatted data by compatibility (descending), then by success probability
        formatted.sort(key=lambda x: (x['_compatibility_raw'] or 0, x['_success_prob_raw'] or 0), reverse=True)
        
        return formatted
    
    # _format_component_scores method removed - now using pre-formatted data from OptimizationSummary
    # This ensures consistent formatting and avoids duplicate code
        
    def _get_match_level_description(self, level: int) -> str:
        """Get a human-readable description for a match level.
        
        Args:
            level: Match level (1-4)
            
        Returns:
            Human-readable description of the match level
        """
        # Use the config's match level descriptions
        if level in self.config.MATCH_LEVELS:
            return self.config.MATCH_LEVELS[level]['name']
        
        # Fallback for unexpected levels
        return f"Match Level {level}"
        
    def _generate_explanation_text(self, recommendation) -> str:
        """Generate human-readable explanation text for a recommendation.
        
        This method creates formatted explanation text based on the recommendation type and metadata.
        Following the separation of concerns pattern, the RecommendationEngine provides raw data in the
        recommendation object's metadata field, and this method formats it into user-friendly explanations.
        
        The explanation text is generated based on the recommendation_type:
        - 'add': Explains the impact of adding a new element
        - 'change': Explains the impact of changing from current to suggested value
        - 'remove': Explains the impact of removing an element
        - 'network_keep': Explains why keeping an element is good for a specific network
        - 'network_change': Explains why changing an element could be beneficial for a network
        
        Args:
            recommendation: RecommendationItem dictionary with recommendation_type, impact, 
                          field, suggested_value, suggested_name, and other required keys
            
        Returns:
            Formatted explanation text for display in the UI
        """
        # Ensure we're only using dictionary-style access on dictionaries
        # RecommendationItem objects should always be dictionaries (TypedDict)
        if not isinstance(recommendation, dict):
            # If we somehow got a non-dictionary, return a generic explanation
            return "No detailed explanation available for this recommendation."
            
        # Extract common fields - dictionary-style access for TypedDict
        rec_type = recommendation['recommendation_type']
        # Always use 'impact' as the standard field name from the recommendation engine
        impact_score = recommendation.get('impact', 0.0)
        # Always use 'field' as the standard field name from the recommendation engine
        field_name = recommendation.get('field', '').replace('_', ' ').title()
        suggested_name = recommendation.get('suggested_name', '')
        current_name = recommendation.get('current_name', '')
        metadata = recommendation.get('metadata', {})
        
        # Format impact percentage for display
        impact_percent = abs(impact_score * 100)
        impact_direction = "increase" if impact_score > 0 else "decrease"
        
        # Generate explanation based on recommendation type
        if rec_type == 'add':
            return f"Adding {suggested_name} could {impact_direction} success probability by {impact_percent:.1f}%."
        
        elif rec_type == 'change':
            return f"Changing from {current_name} to {suggested_name} could {impact_direction} success probability by {impact_percent:.1f}%."
        
        elif rec_type == 'remove':
            return f"Removing {suggested_name} could {impact_direction} success probability by {impact_percent:.1f}%."
        
        elif rec_type == 'network_keep':
            # Use metadata for network-specific explanations
            network_name = metadata.get('network_name', 'this network')
            network_rate = metadata.get('network_rate', 0) * 100
            overall_rate = metadata.get('overall_rate', 0) * 100
            difference = metadata.get('difference', 0) * 100
            
            return f"{network_name} shows a {network_rate:.1f}% success rate with {suggested_name} compared to the overall average of {overall_rate:.1f}%. Keeping this element could {impact_direction} success probability by {impact_percent:.1f}%."
        
        elif rec_type == 'network_change':
            # Use metadata for network-specific explanations
            network_name = metadata.get('network_name', 'this network')
            network_rate = metadata.get('network_rate', 0) * 100
            overall_rate = metadata.get('overall_rate', 0) * 100
            difference = metadata.get('difference', 0) * 100
            
            return f"{network_name} shows only a {network_rate:.1f}% success rate with {suggested_name} compared to the overall average of {overall_rate:.1f}%. Consider changing this element to {impact_direction} success probability by {impact_percent:.1f}%."
        
        # Default explanation if no specific format is defined
        return f"This recommendation could {impact_direction} success probability by {impact_percent:.1f}%."
    

        
    def _get_confidence_level(self, confidence: str) -> int:
        """Convert confidence string to numeric level for sorting.
        
        Args:
            confidence: Confidence string (none, low, medium, high)
            
        Returns:
            Numeric confidence level (0-3)
        """
        if confidence == 'high':
            return 3
        elif confidence == 'medium':
            return 2
        elif confidence == 'low':
            return 1
        else:
            return 0
        
    def _format_success_factors(self, success_factors: List[SuccessFactor]) -> List[Dict[str, Union[str, float, int]]]:
        """Format success factors for display.
        
        Args:
            success_factors: List of success factors
            
        Returns:
            List of formatted success factors with fields like name, impact, description, etc.
        """
        if not success_factors:
            return []
            
        formatted = []
        
        for factor in success_factors:
            # Get proper display name for criteria type using field_manager
            criteria_type = factor.criteria_type
            criteria_type_display = criteria_type.replace("_", " ").title()
            
            # Format the success factor with all data needed for UI display
            formatted.append({
                # Display values
                "Type": criteria_type_display,
                "Name": factor.criteria_name,
                "Impact": factor.impact_score,
                "ImpactDisplay": f"{factor.impact_score:.2f}",
                "Confidence": factor.confidence.capitalize(),
                "Sample": factor.sample_size,
                "SampleDisplay": f"Sample: {factor.sample_size}",
                
                # Raw data for charts and sorting
                "_impact_raw": factor.impact_score,
                "_confidence_level": self._get_confidence_level(factor.confidence),
                "_matching_titles": factor.matching_titles
            })
        
        # Sort by absolute impact (descending)
        formatted.sort(key=lambda x: abs(x["_impact_raw"]), reverse=True)
        
        return formatted
    
    # _format_match_quality method has been removed and replaced with direct dictionary creation
    # in the format_optimization_summary method to maintain explicit data contracts
    
    def _format_matching_shows(self, matching_shows: pd.DataFrame) -> pd.DataFrame:
        """Format matching shows for display in the UI.
        
        This method ensures that the DataFrame is properly formatted for UI consumption.
        The match_level column should already exist from the matching process.
        
        Args:
            matching_shows: DataFrame of matching shows
            
        Returns:
            Formatted DataFrame of matching shows
        """
        # Create a copy to avoid modifying the original
        formatted_shows = matching_shows.copy()
        
        # Ensure match_level is an integer type for proper sorting
        if 'match_level' in formatted_shows.columns:
            formatted_shows['match_level'] = formatted_shows['match_level'].astype(int)
        
        # Only generate match_level_desc if it doesn't exist but match_level does
        if 'match_level_desc' not in formatted_shows.columns and 'match_level' in formatted_shows.columns:
            # Generate descriptions based on match_level
            formatted_shows['match_level_desc'] = formatted_shows['match_level'].apply(
                lambda x: self._get_match_level_description(x)
            )
        
        # Debug output to help diagnose match_level issues
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"After formatting, matching_shows columns: {formatted_shows.columns.tolist()}", category='recommendation')
            if 'match_level' in formatted_shows.columns:
                OptimizerConfig.debug(f"After formatting, match_level values: {formatted_shows['match_level'].value_counts().to_dict()}", category='recommendation')
            else:
                OptimizerConfig.debug("WARNING - match_level column is missing from matching shows!", category='recommendation', force=True)
        
        return formatted_shows
        
    def format_criteria_display(self, criteria: CriteriaDict) -> Dict[str, str]:
        """Format criteria for display.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            
        Returns:
            Dictionary mapping criteria types to display names
        """
        formatted = {}
        
        for criteria_type, criteria_value in criteria.items():
            # Skip None values
            if criteria_value is None:
                continue
                
            # Get display name
            display_name = self._get_criteria_name(criteria_type, criteria_value)
            formatted[criteria_type] = display_name
            
        return formatted
    
    def _get_criteria_name(self, criteria_type: str, criteria_value: Any) -> str:
        """Get the display name for a criteria value.
        
        Args:
            criteria_type: The type of criteria (e.g., 'genre', 'network')
            criteria_value: The value of the criteria (usually an ID)
            
        Returns:
            The display name for the criteria value
        """
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
