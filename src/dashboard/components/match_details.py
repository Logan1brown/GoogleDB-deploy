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
    """Manages transformation of comp data into UI-friendly match details."""
    
    def __init__(self, comp_analyzer):
        # Import here to prevent early initialization
        from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
        self.comp_analyzer = comp_analyzer
        self.scoring = comp_analyzer.score_engine.SCORING
        
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
            
        # Handle genre specially since it's split into base and overlap
        if field == 'genre':
            base = match['comp_score']['components'].get('genre_base', 0)
            overlap = match['comp_score']['components'].get('genre_overlap', 0)
            return float(base) + float(overlap)
            
        # Look for the field directly in components
        components = match['comp_score']['components']
        if field in components:
            return float(components[field])
            
        return 0
        
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
                
            return default
            
    def get_field_names(self, field: str, ids: List[int], default: str = 'Unknown') -> List[str]:
        """Get display names for field values."""
        return [self.get_field_name(field, id, default) for id in ids]
        
    def create_match_details(self, match: Dict, criteria: Dict) -> Dict:
        """Transform match and criteria into UI-ready match details."""
        # Store match data for field name lookups
        self.match = match
        details = {}
        
        # Content match details
        details['genre'] = self._process_genre_match(match, criteria)
        # Process subgenres with overlap scoring - any match is worth the overlap points
        subgenre_scoring = {
            'first': self.scoring['content']['components']['genre']['overlap'],  # Any match gets overlap points
            'second': 0  # No additional points for more matches
        }
        details['subgenres'] = self._process_array_field_match(
            'genre', match.get('subgenres', []), criteria.get('subgenres', []),
            subgenre_scoring
        )
        details['source_type'] = self._process_single_field_match(
            'source_type', match.get('source_type_id'), criteria.get('source_type_id'),
            self.scoring['content']['components']['source_type']['match'],
            match
        )
        details['character_types'] = self._process_array_field_match(
            'character_types', match.get('character_type_ids', []), criteria.get('character_type_ids', []),
            self.scoring['content']['components']['character_types'],
            match
        )
        details['plot_elements'] = self._process_array_field_match(
            'plot_elements', match.get('plot_element_ids', []), criteria.get('plot_element_ids', []),
            self.scoring['content']['components']['plot_elements'],
            match
        )
        details['thematic_elements'] = self._process_array_field_match(
            'thematic_elements', match.get('thematic_element_ids', []), criteria.get('thematic_element_ids', []),
            self.scoring['content']['components']['thematic_elements'],
            match
        )
        details['tone'] = self._process_single_component(
            'tone', match, criteria
        )
        
        # Production match details
        details['network'] = self._process_single_field_match(
            'network', match.get('network_id'), criteria.get('network_id'),
            self.scoring['production']['components']['network']['match'],
            match
        )
        details['studio'] = self._process_production_field_match(
            'studios', match.get('studios', []), criteria.get('studio_ids', []),
            self.scoring['production']['components']['studio'],
            match=match
        )
        # Get team member matches
        source_team = match.get('team_member_ids', [])
        target_team = criteria.get('team_member_ids', [])
        
        # For team members, we need to check all IDs for each name
        matching_names = set()
        matching_ids = set()
        source_names = match.get('team_member_names', [])
        target_names = criteria.get('team_member_names', [])
        
        # For each source ID, get all possible target IDs with the same name
        for source_id, source_name in zip(source_team, source_names):
            for target_id, target_name in zip(target_team, target_names):
                if source_name == target_name:
                    matching_names.add(target_name)  # Use exact name from criteria
                    matching_ids.add(target_id)
        
        details['team'] = ArrayFieldMatch(
            name1='Multiple' if source_team else 'None',
            name2='Multiple' if target_team else 'None',
            selected=bool(target_team),
            match=bool(matching_ids),
            score=self._get_component_score(match, 'team'),
            max_score=self.scoring['production']['components']['team']['first'] + 
                      (self.scoring['production']['components']['team'].get('additional', 0) 
                       if len(target_team) > 1 else 0),
            values1=match.get('team_member_names', []),  # Show's team names
            values2=criteria.get('team_member_names', []),  # Selected names
            # Use matching_names which contains matched names from criteria
            matches=list(matching_names)
        )
        
        # Map internal field names to UI display names
        details['source'] = self._process_single_component('source_type', match, criteria)
        details['time'] = self._process_single_component('time_setting', match, criteria)
        details['location'] = self._process_single_component('location_setting', match, criteria)
        
        # Format match details
        format_details = self._process_format_match(
            match.get('episode_count'), criteria.get('episode_count'),
            match.get('order_type_id'), criteria.get('order_type_id'),
            match
        )
        details['episodes'] = format_details['episodes']
        details['order'] = format_details['order_type']
        
        # Calculate section scores by summing their components
        # Map internal field names to UI display names
        # Only process fields that are selected in criteria
        content_components = {}
        if criteria.get('genre_ids'):
            content_components['genre'] = {
                'score': self._get_component_score(match, 'genre'),
                'match_details': self._process_genre_match(match, criteria)
            }
        if criteria.get('source_type_id'):
            content_components['source_type'] = self._process_single_component('source_type', match, criteria)
        if criteria.get('character_type_ids'):
            content_components['character_types'] = self._process_array_component('character_types', match, criteria)
        if criteria.get('plot_element_ids'):
            content_components['plot_elements'] = self._process_array_component('plot_elements', match, criteria)
        if criteria.get('thematic_element_ids'):
            content_components['thematic_elements'] = self._process_array_component('thematic_elements', match, criteria)
        if criteria.get('tone_id'):
            content_components['tone'] = self._process_single_component('tone', match, criteria)
        if criteria.get('time_setting_id'):
            content_components['time_setting'] = self._process_single_component('time_setting', match, criteria)
        if criteria.get('location_setting_id'):
            content_components['location_setting'] = self._process_single_component('location_setting', match, criteria)
            
        production_components = {}
        if criteria.get('network_id'):
            production_components['network'] = self._process_single_component('network', match, criteria)
        if criteria.get('studio_ids'):
            production_components['studio'] = self._process_array_component('studio', match, criteria)
        if criteria.get('team_member_ids'):
            production_components['team'] = self._process_array_component('team', match, criteria)
            
        format_components = {}
        if criteria.get('episode_count'):
            format_components['episodes'] = self._process_single_component('episodes', match, criteria)
        if criteria.get('order_type_id'):
            format_components['order_type'] = self._process_single_component('order_type', match, criteria)
        
        details['content'] = {
            'score': sum(c['score'] for c in content_components.values()),
            'max': self.scoring['content']['total'],
            'components': content_components
        }
        
        details['production'] = {
            'score': sum(c['score'] for c in production_components.values()),
            'max': self.scoring['production']['total'],
            'components': production_components
        }
        
        details['format'] = {
            'score': sum(c['score'] for c in format_components.values()),
            'max': self.scoring['format']['total'],
            'components': format_components
        }
        
        return details
        
    def _process_single_component(self, field: str, match: Dict, criteria: Dict) -> Dict:
        """Process single-value component matches (source_type, tone).
        
        Args:
            field: Field name (e.g. 'source_type')
            match: Match data
            criteria: Criteria data
            
        Returns:
            Dict with score and match details
        """
        # Map field names to their ID fields in the data
        id_field_map = {
            'source_type': 'source_type_id',
            'tone': 'tone_id',
            'time_setting': 'time_setting_id',
            'location_setting': 'location_setting_id',
            'network': 'network_id',
            'episodes': 'tmdb_total_episodes',
            'order_type': 'order_type_id'
        }
        
        id_field = id_field_map[field]
        # Get scoring section and config based on field type
        if field in self.scoring['content']['components']:
            section = 'content'
            max_score = self.scoring[section]['components'][field]['match']
        elif field in self.scoring['production']['components']:
            section = 'production'
            max_score = self.scoring[section]['components'][field]['match']
        elif field in self.scoring['format']['components']:
            section = 'format'
            if field == 'episodes':
                max_score = self.scoring[section]['components'][field]['within_2']
            else:
                max_score = self.scoring[section]['components'][field]['match']
        else:
            raise ValueError(f'Field {field} not found in scoring configuration')
            
        return {
            'score': self._get_component_score(match, field),
            'match_details': self._process_single_field_match(
                field,
                match.get(id_field),
                criteria.get(id_field),
                max_score,
                match
            )
        }

    def _process_array_component(self, field: str, match: Dict, criteria: Dict) -> Dict:
        """Process array-based component matches (character_types, plot_elements, thematic_elements).
        
        Args:
            field: Field name (e.g. 'character_types')
            match: Match data
            criteria: Criteria data
            
        Returns:
            Dict with score and match details
        """
        # Map field names to their ID fields in the data
        id_field_map = {
            'character_types': 'character_type_ids',
            'plot_elements': 'plot_element_ids',
            'thematic_elements': 'thematic_element_ids',
            'studio': 'studio_ids',
            'team': 'team_member_ids'
        }
        
        # Get scoring section and config based on field type
        if field in self.scoring['content']['components']:
            section = 'content'
        elif field in self.scoring['production']['components']:
            section = 'production'
        else:
            raise ValueError(f'Field {field} not found in scoring configuration')
            
        id_field = id_field_map[field]
        return {
            'score': self._get_component_score(match, field),
            'match_details': self._process_array_field_match(
                field,
                match.get(id_field, []),
                criteria.get(id_field, []),
                self.scoring[section]['components'][field],
                match
            )
        }
        
    def _process_genre_match(self, match: Dict, criteria: Dict) -> FieldMatch:
        """Process genre and subgenre matches."""
        genre_id = match.get('genre_id')
        selected_id = criteria.get('genre_id')
        genre_match = genre_id == selected_id
        
        # Use the score from comp_analyzer instead of recalculating
        score = self._get_component_score(match, 'genre')
        max_score = self.scoring['content']['components']['genre']['base']  # Only use base score for primary genre
            
        return FieldMatch(
            name1=self.get_field_name('genre', genre_id),
            name2=self.get_field_name('genre', selected_id),
            selected=selected_id is not None,
            match=genre_match,
            score=score,
            max_score=max_score
        )
        
    def _process_single_field_match(self, field: str, value_id: Optional[int], 
                                  selected_id: Optional[int], max_score: float, 
                                  match: Optional[Dict] = None) -> FieldMatch:
        """Process match for a single-value field."""
        # Get score from comp_analyzer
        score = self._get_component_score(match, field)
        
        is_match = value_id == selected_id
            
        return FieldMatch(
            name1=self.get_field_name(field, value_id),
            name2=self.get_field_name(field, selected_id),
            selected=selected_id is not None,
            match=is_match,
            score=score,
            max_score=max_score
        )
        
    def _process_array_field_match(self, field: str, values: List[int], 
                                 selected: List[int], scoring: Dict,
                                 match: Optional[Dict] = None) -> ArrayFieldMatch:
        """Process match for a multi-value field.
        
        Handles both content scoring (first/second) and production scoring (primary/additional).
        
        Args:
            field: Field name
            values: List of field values
            selected: List of selected values
            scoring: Scoring config for the field
            match: Optional match data containing name fields
            
        Returns:
            ArrayFieldMatch with match details
        """
        """Process match for a multi-value field."""
        value_names = self.get_field_names(field, values)
        selected_names = self.get_field_names(field, selected)
        matches = [n for n in value_names if n in selected_names]
        
        # Calculate max score based on scoring format
        if 'first' in scoring and 'second' in scoring:
            # Content scoring (character_types, plot_elements, etc)
            max_score = scoring['first'] + scoring['second']
        elif 'primary' in scoring and 'max_additional' in scoring:
            # Production scoring (studio)
            max_score = scoring['primary'] + scoring['max_additional']
        elif 'first' in scoring and 'max_additional' in scoring:
            # Team scoring
            max_score = scoring['first'] + scoring['max_additional']
        else:
            raise ValueError(f'Unknown scoring format for field: {scoring}')
        
        return ArrayFieldMatch(
            name1='Multiple' if value_names else 'None',
            name2='Multiple' if selected_names else 'None',
            selected=bool(selected),
            match=bool(matches),
            score=self._get_component_score(match, field),
            max_score=max_score,
            values1=value_names,
            values2=selected_names,
            matches=matches
        )
        

        
    def _process_production_field_match(self, field: str, values: List[int],
                                      selected: List[int], scoring: Dict, match: Dict = None) -> ArrayFieldMatch:
        """Process match for production fields (studio) with special scoring."""
        # For studios, use pre-fetched names from API response
        if field == 'studios' and match:
            value_names = match.get('studio_names', [])
            # For selected studios, still need to look up names
            selected_names = self.get_field_names(field, selected)
        else:
            value_names = self.get_field_names(field, values)
            selected_names = self.get_field_names(field, selected)
            
        value_set = set(values)
        selected_set = set(selected)
        matches = value_set & selected_set
        
        # Get score from comp_analyzer
        score = self._get_component_score(match, field)
                
        # Calculate max score based on field type
        if field == 'team_members':
            max_score = scoring.get('first', 0)
            if len(selected) > 1:
                max_score += scoring.get('additional', 0)
        else:
            max_score = scoring.get('primary', 0)
            max_score += scoring.get('max_additional', 0)
                
        return ArrayFieldMatch(
            name1='Multiple' if value_names else 'None',
            name2='Multiple' if selected_names else 'None',
            selected=bool(selected),
            match=bool(matches),
            score=score,
            max_score=max_score,
            values1=value_names,
            values2=selected_names,
            # For studios, show all values for better display
            matches=value_names if field == 'studios'
                   else self.get_field_names(field, list(matches))
        )
        
    def _process_setting_match(self, time_id: Optional[int], location_id: Optional[int],
                             selected_time_id: Optional[int], 
                             selected_location_id: Optional[int]) -> Dict[str, FieldMatch]:
        """Process time and location setting matches."""
        time_match = self._process_single_field_match(
            'time_setting', time_id, selected_time_id,
            self.scoring['content']['components']['time_setting']['match']
        )
        
        location_match = self._process_single_field_match(
            'location_setting', location_id, selected_location_id,
            self.scoring['content']['components']['location_setting']['match']
        )
        
        return {
            'time_setting': time_match,
            'location_setting': location_match,
            'total_score': time_match.score + location_match.score,
            'max_score': time_match.max_score + location_match.max_score
        }
        
    def _process_format_match(self, episodes: Optional[int], selected_episodes: Optional[int],
                            order_type_id: Optional[int], 
                            selected_order_type_id: Optional[int],
                            match: Optional[Dict] = None) -> Dict[str, FieldMatch]:
        """Process episode count and order type matches."""
        episode_score = self._get_component_score(match, 'episodes')
        diff = abs(episodes - selected_episodes) if episodes is not None and selected_episodes is not None else None
                
        episode_match = FieldMatch(
            name1=str(episodes) if episodes is not None else 'Unknown',
            name2=str(selected_episodes) if selected_episodes is not None else 'Unknown',
            selected=selected_episodes is not None,
            match=diff is not None and diff <= 2,  # Consider a match if within 2 episodes
            score=episode_score,
            max_score=self.scoring['format']['components']['episodes']['within_2']
        )
        
        order_match = self._process_single_field_match(
            'order_type', order_type_id, selected_order_type_id,
            self.scoring['format']['components']['order_type']['match'],
            match
        )
        
        return {
            'episodes': episode_match,
            'order_type': order_match,
            'total_score': episode_match.score + order_match.score,
            'max_score': episode_match.max_score + order_match.max_score
        }
