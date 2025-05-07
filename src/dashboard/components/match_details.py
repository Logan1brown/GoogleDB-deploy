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
            'subgenres': 'subgenres',  # Direct field name from data
            'source_type': 'source_type_id',
            'character_types': 'character_type_ids',
            'plot_elements': 'plot_element_ids',
            'thematic_elements': 'thematic_element_ids',
            'tone': 'tone_id',
            'time_setting': 'time_setting_id',
            'location_setting': 'location_setting_id',
            'network': 'network_id',
            'studios': 'studios',  # Direct field name from data
            'team_members': 'team_member_ids',
            'episodes': 'episode_count',
            'order_type': 'order_type_id'
        }
        
        # Map fields to their name fields
        self.name_field_map = {
            'genre': 'genre_name',
            'subgenres': 'subgenre_names',
            'source_type': 'source_type_name', 
            'character_types': 'character_type_names',
            'plot_elements': 'plot_element_names',
            'thematic_elements': 'thematic_element_names',
            'tone': 'tone_name',
            'time_setting': 'time_setting_name',
            'location_setting': 'location_setting_name',
            'network': 'network_name',
            'studios': 'studio_names',  # Match field config name
            'team_members': 'team_member_names',  # Match field config name
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
        
        # Get score from comp_score using CompAnalyzer's method
        return self.comp_analyzer.get_field_score(field, match)
        
    def get_field_name(self, field: str, id: Optional[int], match: Optional[Dict] = None, default: str = 'None') -> str:
        """Get display name for a field value.
        
        Args:
            field: Field type (e.g. 'genre', 'source_type')
            id: ID of the field value
            match: Match data containing name fields. If None, uses field manager.
            default: Value to return for null fields
            
        Returns:
            Display name for the field value, or default for null fields
        """
        # Handle null field values
        if id is None:
            return default
            
        # If no match data, use field manager (for criteria display)
        if match is None:
            return self.comp_analyzer.get_field_display_name(field, id) or default
            
        name_field = self.name_field_map.get(field)
        if not name_field:
            raise ValueError(f"No name field mapping for {field}")
            
        config = self.field_configs.get(field)
        if not config:
            raise ValueError(f"No field config for {field}")
            
        # Handle non-array fields
        if not config.is_array:
            if field == 'genre':
                st.write(f"Debug - Match data for genre (id={id}):\nname_field={name_field}\nmatch={match}")
            name = match.get(name_field)
            if name is None:
                # If name is missing in match data, try field manager
                return self.comp_analyzer.get_field_display_name(field, id) or default
            return name
            
        # Handle array fields
        names = match.get(name_field)
        ids = match.get(self.id_field_map.get(field))
        
        # Arrays should exist if field is array type
        if names is None or ids is None:
            raise ValueError(f"Array data missing for {field}")
            
        try:
            idx = ids.index(id)
            if idx >= len(names):
                raise ValueError(f"Name missing for {field} ID {id} at index {idx}")
            return names[idx]
        except ValueError:
            raise ValueError(f"ID {id} not found in {field} list")
            
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
        if not ids:
            return []
            
        # Check if this is an array field
        config = self.field_configs.get(field)
        is_array = config and config.is_array if config else False
            
        # First try to get names from match data if provided
        if match is not None:
            name_field = self.name_field_map.get(field)
            if name_field:
                if is_array:
                    # For array fields, match data should already have the names
                    names = match.get(name_field, [])
                    if names and len(names) == len(ids):
                        return names
                else:
                    # For single fields, get the specific name
                    name = match.get(name_field)
                    if name is not None:
                        return [name]
        
        # If no match data or names not found, use field manager
        try:
            return [self.comp_analyzer.get_field_display_name(field, id) or default for id in ids]
        except:
            return [default for _ in ids]
        
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
        
        # Get scoring config from CompAnalyzer
        scoring = self.comp_analyzer.score_engine.SCORING
        content_scoring = scoring['content']['components']
        
        # Genre (combines base + overlap)
        genre_score = components.get('genre_base', 0) + components.get('genre_overlap', 0)
        genre_max = content_scoring['genre']['base'] + content_scoring['genre']['overlap']
        genre_id = match.get('genre_id')
        target_genre_id = criteria.get('genre_id')
        content_components['genre'] = {
            'display': FieldMatch(
                name1=self.get_field_name('genre', genre_id, match),
                name2=self.get_field_name('genre', target_genre_id),
                selected=target_genre_id is not None,
                match=genre_id == target_genre_id if genre_id and target_genre_id else False,
                score=genre_score,
                max_score=genre_max
            )
        }
        
        # Subgenres (part of genre scoring)
        subgenre_ids = match.get('subgenres', [])
        target_subgenre_ids = criteria.get('subgenres', [])
        content_components['subgenres'] = {
            'display': ArrayFieldMatch(
                name1='',  # Not used for array fields
                name2='',  # Not used for array fields
                values1=self.get_field_names('subgenres', subgenre_ids, match),
                values2=self.get_field_names('subgenres', target_subgenre_ids),
                matches=[v for v in subgenre_ids if v in target_subgenre_ids],
                selected=bool(target_subgenre_ids),
                match=bool(set(subgenre_ids) & set(target_subgenre_ids)),
                score=components.get('genre_overlap', 0),  # Already included in genre_score
                max_score=content_scoring['genre']['overlap']
            )
        }
        
        # Source Type
        source_score = components.get('source_type', 0)
        source_max = content_scoring['source_type']['match']
        source_type_id = match.get('source_type_id')
        target_source_type_id = criteria.get('source_type_id')
        content_components['source_type'] = {
            'display': FieldMatch(
                name1=self.get_field_name('source_type', source_type_id, match),
                name2=self.get_field_name('source_type', target_source_type_id),
                selected=target_source_type_id is not None,
                match=source_type_id == target_source_type_id if source_type_id and target_source_type_id else False,
                score=source_score,
                max_score=source_max
            )
        }
        
        # Array fields (character_types, plot_elements, thematic_elements)
        for field in ['character_types', 'plot_elements', 'thematic_elements']:
            field_score = components.get(field, 0)
            field_max = content_scoring[field]['first'] + content_scoring[field]['second']
            values = match.get(self.id_field_map[field], [])
            selected = criteria.get(self.id_field_map[field], [])
            matches = [v for v in values if v in selected]
            
            content_components[field] = {
                'display': ArrayFieldMatch(
                    name1='',  # Not used for array fields
                    name2='',  # Not used for array fields
                    values1=self.get_field_names(field, values, match),
                    values2=self.get_field_names(field, selected),
                    matches=self.get_field_names(field, matches),
                    selected=bool(selected),
                    match=bool(matches),
                    score=field_score,
                    max_score=field_max
                )
            }
        
        # Single-value fields
        # Note: We store the score under the UI field name (location_setting)
        # even though it comes from 'location' in comp_score
        field_map = {
            'tone': 'tone',
            'time_setting': 'time_setting',
            'location_setting': 'location'
        }
        
        for field, score_field in field_map.items():
            field_score = components.get(score_field, 0)
            field_max = content_scoring[field]['match']
            value_id = match.get(self.id_field_map[field])
            target_id = criteria.get(self.id_field_map[field])
            
            # Store under UI field name for consistency
            content_components[field] = {
                'display': FieldMatch(
                    name1=self.get_field_name(field, value_id, match),
                    name2=self.get_field_name(field, target_id),
                    selected=target_id is not None,
                    match=value_id == target_id if value_id and target_id else False,
                    score=field_score,
                    max_score=field_max
                )
            }
            
        details['content'] = {
            'score': comp_score.get('content_score', 0),
            'max_score': scoring['content']['total'],
            'components': content_components
        }
        
        # Production section
        production_scoring = scoring['production']['components']
        production_components = {}
        
        # Network (matches exactly in comp_score)
        network_score = components.get('network', 0)
        network_max = production_scoring['network']['match']
        network_id = match.get('network_id')
        target_network_id = criteria.get('network_id')
        production_components['network'] = {
            'display': FieldMatch(
                name1=self.get_field_name('network', network_id, match),
                name2=self.get_field_name('network', target_network_id),
                selected=target_network_id is not None,
                match=network_id == target_network_id if network_id and target_network_id else False,
                score=network_score,
                max_score=network_max
            )
        }
        
        # Array fields use plural names (studios, team_members) in the data
        # but singular names (studio, team) in scoring to distinguish between:
        # - Data fields (plural): collections of IDs (e.g. studios[])
        # - Scoring fields (singular): individual match scores (e.g. studio.primary)
        
        # Add subgenres to content section
        subgenre_score = components.get('genre_overlap', 0)  # Overlap score from matching subgenres
        subgenre_max = content_scoring['genre']['overlap']  # Max overlap score
        subgenre_ids = match.get('subgenres', [])
        target_subgenre_ids = criteria.get('subgenres', [])
        matches = [v for v in subgenre_ids if v in target_subgenre_ids]
        content_components['subgenres'] = {
            'display': ArrayFieldMatch(
                name1='',  # Not used for array fields
                name2='',  # Not used for array fields
                values1=self.get_field_names('subgenre', subgenre_ids, match),
                values2=self.get_field_names('subgenre', target_subgenre_ids),
                matches=self.get_field_names('subgenre', matches),
                selected=bool(target_subgenre_ids),
                match=bool(matches),
                score=subgenre_score,
                max_score=subgenre_max
            )
        }
        
        # Map plural field names to singular for production components
        field_map = {
            'studios': 'studio',
            'team_members': 'team'
        }
        
        for field_plural, field_singular in field_map.items():
            # Get score using singular name
            field_score = components.get(field_singular, 0)
            
            # Get max score from scoring config
            field_max = (
                production_scoring[field_singular]['primary' if field_plural == 'studios' else 'first'] +
                production_scoring[field_singular]['max_additional']
            )
            
            # Get values and selected from match data using full field name
            values = match.get(self.id_field_map[field_plural], [])
            selected = criteria.get(self.id_field_map[field_plural], [])
            matches = [v for v in values if v in selected]
            
            production_components[field_singular] = {
                'display': ArrayFieldMatch(
                    name1='',  # Not used for array fields
                    name2='',  # Not used for array fields
                    values1=self.get_field_names(field_plural, values, match),
                    values2=self.get_field_names(field_plural, selected),
                    matches=self.get_field_names(field_plural, matches),
                    selected=bool(selected),
                    match=bool(matches),
                    score=field_score,
                    max_score=field_max
                )
            }
        
        # Add production section to details
        details['production'] = {
            'score': comp_score.get('production_score', 0),
            'max_score': scoring['production']['total'],
            'components': production_components
        }
        
        # Format match section
        format_scoring = scoring['format']['components']
        format_components = {}
        
        # Episodes (matches exactly in comp_score)
        episode_score = components.get('episodes', 0)
        episode_max = format_scoring['episodes']['within_2']
        episode_count = match.get('episode_count')
        target_episode_count = criteria.get('episode_count')
        format_components['episodes'] = {
            'display': FieldMatch(
                name1=str(episode_count) if episode_count is not None else 'Unknown',
                name2=str(target_episode_count) if target_episode_count is not None else 'Unknown',
                selected=target_episode_count is not None,
                match=abs(episode_count - target_episode_count) <= 2 if episode_count is not None and target_episode_count is not None else False,
                score=episode_score,
                max_score=episode_max
            )
        }
        
        # Order Type (matches exactly in comp_score)
        order_score = components.get('order_type', 0)
        order_max = format_scoring['order_type']['match']
        order_type_id = match.get('order_type_id')
        target_order_type_id = criteria.get('order_type_id')
        format_components['order_type'] = {
            'display': FieldMatch(
                name1=self.get_field_name('order_type', order_type_id, match),
                name2=self.get_field_name('order_type', target_order_type_id),
                selected=target_order_type_id is not None,
                match=order_type_id == target_order_type_id if order_type_id and target_order_type_id else False,
                score=order_score,
                max_score=order_max
            )
        }
        
        details['format'] = {
            'score': comp_score.get('format_score', 0),
            'max_score': scoring['format']['total'],
            'components': format_components
        }
        
        # Get total scores from comp_score
        details['total'] = {
            'score': comp_score.get('total', 0),
            'max_score': sum([
                scoring['content']['total'],
                scoring['production']['total'],
                scoring['format']['total']
            ])
        }
        
        return details
