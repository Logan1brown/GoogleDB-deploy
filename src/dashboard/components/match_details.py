"""Manager for match details display logic."""

import streamlit as st
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FieldMatch:
    """Represents a match for a single field."""
    name1: str  # Source show value
    name2: str  # Target/selected value 
    selected: bool  # Whether criteria was selected
    match: bool  # Whether values match
    score: float  # Points awarded for this match
    max_score: float  # Maximum possible points

@dataclass
class ArrayFieldMatch(FieldMatch):
    """Represents a match for a field with multiple values."""
    values1: List[str]  # Source show values
    values2: List[str]  # Target/selected values
    matches: List[str]  # Values that match between source and target

class MatchDetailsManager:
    """Manages transformation of comp data into UI-friendly match details.
    
    This class is responsible for:
    1. Converting IDs to display names
    2. Adding UI state information (selected/matched)
    3. Organizing data for display
    
    It does NOT:
    1. Calculate scores (that's CompAnalyzer's job)
    2. Transform CompAnalyzer's data structure
    3. Add any business logic
    """
    
    def __init__(self, comp_analyzer):
        # Import here to prevent early initialization
        from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
        self.comp_analyzer = comp_analyzer
        
        # Get field configs from CompAnalyzer
        self.field_configs = comp_analyzer.field_manager.FIELD_CONFIGS
        
        # Map field names to match data fields
        self.id_field_map = {
            'genre': 'genre_id',
            'source_type': 'source_type_id',
            'character_types': 'character_type_ids',
            'plot_elements': 'plot_element_ids',
            'thematic_elements': 'thematic_element_ids',
            'tone': 'tone_id',
            'time_setting': 'time_setting_id',
            'location_setting': 'location_setting_id',
            'network': 'network_id',
            'studio': 'studios',
            'team': 'team_member_ids',
            'episodes': 'episode_count',
            'order_type': 'order_type_id'
        }
        
        # Map fields to their name fields
        self.name_field_map = {
            'genre': 'genre_name',
            'source_type': 'source_type_name', 
            'character_types': 'character_type_names',
            'plot_elements': 'plot_element_names',
            'thematic_elements': 'thematic_element_names',
            'tone': 'tone_name',
            'time_setting': 'time_setting_name',
            'location_setting': 'location_setting_name',
            'network': 'network_name',
            'studio': 'studio_names',
            'team': 'team_member_names',
            'episodes': 'episode_count',
            'order_type': 'order_type_name'
        }
        
    def _get_component_score(self, match: Dict, field: str) -> float:
        """Get a component's score from the comp_score dictionary.
        
        Args:
            match: Match data containing comp_score
            field: Field to get score for
            
        Returns:
            Score for the field, or 0 if not found
        """
        if not match or 'comp_score' not in match:
            return 0
            
        components = match['comp_score'].get('components', {})
        
        # Get score directly from components
        return float(components.get(field, 0))
        
    def get_field_name(self, field: str, id: Optional[int], match: Optional[Dict] = None, default: str = 'Unknown') -> str:
        """Get display name for a field value.
        
        Args:
            field: Field type (e.g. 'genre', 'source_type')
            id: ID of the field value
            match: Optional match data containing name fields. If not provided, will use field_manager
            default: Default value if name not found
            
        Returns:
            Display name for the field value
        """
        if id is None:
            return default
            
        # Map field to its corresponding name field in the match data
        name_field_map = {
            'genre': 'genre_name',
            'source_type': 'source_type_name',
            'tone': 'tone_name',
            'time_setting': 'time_setting_name',
            'location_setting': 'location_setting_name',
            'network': 'network_name',
            'order_type': 'order_type_name'
        }
        
        # If match data is provided and has the name field, use that
        if match is not None:
            # First check the direct name field map
            if field in name_field_map:
                name = match.get(name_field_map[field])
                if name is not None:
                    return name
            
            # Then check array fields
            array_name_map = {
                'character_types': 'character_type_names',
                'plot_elements': 'plot_element_names',
                'thematic_elements': 'thematic_element_names',
                'studios': 'studio_names',
                'team_members': 'team_member_names'
            }
            
            if field in array_name_map:
                names = match.get(array_name_map[field], [])
                # Try to find the name at the same index as the ID in the IDs array
                ids = match.get(f'{field}_ids', [])
                try:
                    idx = ids.index(id)
                    if idx < len(names):
                        return names[idx]
                except (ValueError, IndexError):
                    pass
                    
        # If we didn't find the name in match data, use field_manager
        try:
            return self.comp_analyzer.get_field_display_name(field, id) or default
        except:
            return default
            
    def get_field_names(self, field: str, ids: List[int], match: Optional[Dict] = None, default: str = 'Unknown') -> List[str]:
        """Get display names for field values.
        
        Args:
            field: Field type (e.g. 'genre', 'source_type')
            ids: List of field value IDs
            match: Optional match data containing name fields
            default: Default value if name not found
            
        Returns:
            List of display names for the field values
        """
        return [self.get_field_name(field, id, match, default) for id in ids]
        
    def create_match_details(self, match: Dict, criteria: Dict) -> Dict:
        """Add UI display information to CompAnalyzer's match data.
        
        This method takes the raw match data and criteria, and adds:
        1. Display names for IDs
        2. Selected state information
        3. Match state information
        
        It does NOT modify any scoring - that comes directly from CompAnalyzer.
        """
        if not match:
            return {}
            
        # Get comp_score if it exists, otherwise use empty dict
        comp_score = match.get('comp_score', {})
        components = comp_score.get('components', {})
        
        # Add UI info to each component while preserving CompAnalyzer's structure
        details = {}
        
        # Content section
        content_components = {}
        
        # Genre
        genre = components.get('genre', {})
        genre_id = match.get('genre_id')
        target_genre_id = criteria.get('genre_id')
        content_components['genre'] = {
            **genre,  # Keep CompAnalyzer's score data
            'display': FieldMatch(
                name1=self.get_field_name('genre', genre_id, match),
                name2=self.get_field_name('genre', target_genre_id),
                selected=target_genre_id is not None,
                match=genre_id == target_genre_id if genre_id and target_genre_id else False,
                score=genre.get('score', 0),
                max_score=genre.get('max', 0)
            )
        }
        
        # Source Type
        source = components.get('source_type', {})
        source_id = match.get('source_type_id')
        target_source_id = criteria.get('source_type_id')
        content_components['source_type'] = {
            **source,
            'display': FieldMatch(
                name1=self.get_field_name('source_type', source_id, match),
                name2=self.get_field_name('source_type', target_source_id),
                selected=target_source_id is not None,
                match=source_id == target_source_id if source_id and target_source_id else False,
                score=source.get('score', 0),
                max_score=source.get('max', 0)
            )
        }
        
        # Array fields (character_types, plot_elements, thematic_elements)
        for field in ['character_types', 'plot_elements', 'thematic_elements']:
            field_data = components.get(field, {})
            values = match.get(self.id_field_map[field], [])
            selected = criteria.get(self.id_field_map[field], [])
            content_components[field] = {
                **field_data,
                'display': ArrayFieldMatch(
                    values1=self.get_field_names(field, values, match),
                    values2=self.get_field_names(field, selected),
                    matches=self.get_field_names(field, field_data.get('matches', [])),
                    selected=bool(selected),
                    match=field_data.get('match', False),
                    score=field_data.get('score', 0),
                    max_score=field_data.get('max', 0)
                )
            }
            
        # Single fields (tone, time_setting, location_setting)
        for field in ['tone', 'time_setting', 'location_setting']:
            field_data = components.get(field, {})
            value_id = match.get(self.id_field_map[field])
            target_id = criteria.get(self.id_field_map[field])
            content_components[field] = {
                **field_data,
                'display': FieldMatch(
                    name1=self.get_field_name(field, value_id, match),
                    name2=self.get_field_name(field, target_id),
                    selected=target_id is not None,
                    match=value_id == target_id if value_id and target_id else False,
                    score=field_data.get('score', 0),
                    max_score=field_data.get('max', 0)
                )
            }
            
        details['content'] = {
            'score': comp_score.get('content', {}).get('score', 0),
            'max_score': comp_score.get('content', {}).get('max', 0),
            'components': content_components
        }
        
        # Production section
        production = comp_score.get('production', {})
        production_components = {}
        
        # Network
        network = components.get('network', {})
        network_id = match.get('network_id')
        target_network_id = criteria.get('network_id')
        production_components['network'] = {
            **network,
            'display': FieldMatch(
                name1=self.get_field_name('network', network_id, match),
                name2=self.get_field_name('network', target_network_id),
                selected=target_network_id is not None,
                match=network_id == target_network_id if network_id and target_network_id else False,
                score=network.get('score', 0),
                max_score=network.get('max', 0)
            )
        }
        
        # Array fields (studio, team)
        for field in ['studio', 'team']:
            field_data = components.get(field, {})
            values = match.get(self.id_field_map[field], [])
            selected = criteria.get(self.id_field_map[field], [])
            production_components[field] = {
                **field_data,
                'display': ArrayFieldMatch(
                    values1=self.get_field_names(field, values, match),
                    values2=self.get_field_names(field, selected),
                    matches=self.get_field_names(field, field_data.get('matches', [])),
                    selected=bool(selected),
                    match=field_data.get('match', False),
                    score=field_data.get('score', 0),
                    max_score=field_data.get('max', 0)
                )
            }
        
        # Add production section to details
        details['production'] = {
            'score': production.get('score', 0),
            'max_score': production.get('max', 0),
            'components': production_components
        }

        # Format section
        format_section = comp_score.get('format', {})
        format_components = {}
        
        # Episodes
        episodes = components.get('episodes', {})
        episode_count = match.get('episode_count')
        target_episode_count = criteria.get('episode_count')
        format_components['episodes'] = {
            **episodes,
            'display': FieldMatch(
                name1=str(episode_count) if episode_count is not None else 'Unknown',
                name2=str(target_episode_count) if target_episode_count is not None else 'Unknown',
                selected=target_episode_count is not None,
                match=episodes.get('match', False),
                score=episodes.get('score', 0),
                max_score=episodes.get('max', 0)
            )
        }
        
        # Order Type
        order = components.get('order_type', {})
        order_type_id = match.get('order_type_id')
        target_order_type_id = criteria.get('order_type_id')
        format_components['order_type'] = {
            **order,
            'display': FieldMatch(
                name1=self.get_field_name('order_type', order_type_id, match),
                name2=self.get_field_name('order_type', target_order_type_id),
                selected=target_order_type_id is not None,
                match=order_type_id == target_order_type_id if order_type_id and target_order_type_id else False,
                score=order.get('score', 0),
                max_score=order.get('max', 0)
            )
        }
        
        details['format'] = {
            'score': format_section.get('score', 0),
            'max_score': format_section.get('max', 0),
            'components': format_components
        }
        
        # Get total scores from comp_score
        details['total'] = {
            'score': comp_score.get('total', 0),
            'max_score': comp_score.get('max', 0)
        }
        
        return details
        

        

