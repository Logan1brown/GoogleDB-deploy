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
        """Create match details for display in the UI and scoring.
        Creates a properly structured dict for the UI from raw match data.
        """
        if not match or 'comp_score' not in match:
            return {}
            
        comp_score = match['comp_score']
        components = comp_score.get('components', {})
        
        # Create the full details structure
        details = {
            'content': {
                'score': comp_score.get('content', 0),
                'max': 82,  # From SCORING_CONFIG
                'components': {
                    'genre': {
                        'score': components.get('genre_base', 0) + components.get('genre_overlap', 0),
                        'max': 23  # 15 base + 8 overlap
                    },
                    'source_type': {
                        'score': components.get('source_type', 0),
                        'max': 10
                    },
                    'character_types': {
                        'score': components.get('character_types', 0),
                        'max': 12  # 9 first + 3 second
                    },
                    'plot_elements': {
                        'score': components.get('plot_elements', 0),
                        'max': 10  # 7.5 first + 2.5 second
                    },
                    'thematic_elements': {
                        'score': components.get('thematic_elements', 0),
                        'max': 10  # 7.5 first + 2.5 second
                    },
                    'tone': {
                        'score': components.get('tone', 0),
                        'max': 9
                    },
                    'time_setting': {
                        'score': components.get('time_setting', 0),
                        'max': 4
                    },
                    'location_setting': {
                        'score': components.get('location', 0),
                        'max': 4
                    }
                },
                'breakdown': {
                    'genre': self._process_genre_match(match, criteria),
                    'source': self._process_single_field('source_type', match, criteria),
                    'characters': self._process_array_field('character_types', match, criteria),
                    'plot': self._process_array_field('plot_elements', match, criteria),
                    'themes': self._process_array_field('thematic_elements', match, criteria),
                    'tone': self._process_single_field('tone', match, criteria),
                    'time_setting': self._process_single_field('time_setting', match, criteria),
                    'location_setting': self._process_single_field('location_setting', match, criteria)
                }
            },
            'production': {
                'score': comp_score.get('production', 0),
                'max': 13,
                'components': {
                    'network': {
                        'score': components.get('network', 0),
                        'max': 5
                    },
                    'studio': {
                        'score': components.get('studio', 0),
                        'max': 4  # 2 primary + 2 max additional
                    },
                    'team': {
                        'score': components.get('team', 0),
                        'max': 4  # 2 first + 2 max additional
                    }
                },
                'breakdown': {
                    'network': self._process_single_field('network', match, criteria),
                    'studio': self._process_array_field('studio', match, criteria),
                    'team': self._process_array_field('team', match, criteria)
                }
            },
            'format': {
                'score': comp_score.get('format', 0),
                'max': 5,
                'components': {
                    'episodes': {
                        'score': components.get('episodes', 0),
                        'max': 3
                    },
                    'order_type': {
                        'score': components.get('order_type', 0),
                        'max': 2
                    }
                },
                'breakdown': {
                    'episodes': self._process_single_field('episodes', match, criteria),
                    'order_type': self._process_single_field('order_type', match, criteria)
                }
            }
        }
        
        return details

    def _process_single_field(self, field: str, match: Dict, criteria: Dict) -> FieldMatch:
        """Process a single field and return a FieldMatch object for UI display."""
        id_field = self.id_field_map[field]
        
        value_id = match.get(id_field)
        target_id = criteria.get(id_field)
        
        # Get names using field_manager
        value_name = self.get_field_name(field, value_id, match)
        target_name = self.get_field_name(field, target_id)
        
        # Get score and max from comp_score
        comp_score = match.get('comp_score', {})
        components = comp_score.get('components', {})
        score = components.get(field, 0)
        
        # Get max score based on field
        max_scores = {
            'source_type': 10,
            'tone': 9,
            'time_setting': 4,
            'location_setting': 4,
            'network': 5,
            'episodes': 3,
            'order_type': 2
        }
        
        return FieldMatch(
            name1=value_name or 'Unknown',
            name2=target_name or 'Unknown',
            selected=bool(target_id),
            match=value_id == target_id if value_id and target_id else False,
            score=score,
            max_score=max_scores.get(field, 10)  # Default to 10 if not found
        )

    def _process_array_field(self, field: str, match: Dict, criteria: Dict) -> ArrayFieldMatch:
        """Process an array field and return an ArrayFieldMatch object for UI display."""
        id_field = self.id_field_map[field]
        
        values = match.get(id_field, [])
        selected = criteria.get(id_field, [])
        
        # Get names using field_manager
        value_names = self.get_field_names(field, values, match) if values else []
        selected_names = self.get_field_names(field, selected) if selected else []
        matches = list(set(value_names) & set(selected_names))
        
        # Get score from comp_score
        comp_score = match.get('comp_score', {})
        components = comp_score.get('components', {})
        score = components.get(field, 0)
        
        # Get max score based on field
        max_scores = {
            'character_types': 12,  # 9 first + 3 second
            'plot_elements': 10,    # 7.5 first + 2.5 second
            'thematic_elements': 10,# 7.5 first + 2.5 second
            'studio': 4,           # 2 primary + 2 additional
            'team': 4              # 2 first + 2 additional
        }
        
        return ArrayFieldMatch(
            name1='Multiple' if values else 'Unknown',
            name2='Multiple' if selected else 'Unknown',
            selected=bool(selected),
            match=bool(matches),
            score=score,
            max_score=max_scores.get(field, 10),  # Default to 10 if not found
            values1=value_names,
            values2=selected_names,
            matches=matches
        )
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
        
        # Get IDs and check for match
        value_id = match.get(id_field_map[field])
        target_id = criteria.get(id_field_map[field])
        
        # Get display names
        value_name = self.get_field_name(field, value_id, match)
        target_name = self.get_field_name(field, target_id, match)
        
        # Return match details using CompAnalyzer scores
        return {
            'score': match.get(f'{field}_score', 0),
            'max_score': match.get(f'{field}_max_score', 0),
            'match_details': {
                'name1': value_name,
                'name2': target_name,
                'selected': bool(target_id),
                'match': value_id == target_id if value_id and target_id else False
            }
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
        
        id_field = id_field_map[field]
        values = match.get(id_field, [])
        selected = criteria.get(id_field, [])
        
        return {
            'score': match.get(f'{field}_score', 0),
            'max_score': match.get(f'{field}_max_score', 0),
            'match_details': self._process_array_field_match(
                field,
                values,
                selected,
                {'first': 0, 'second': 0},  # Dummy scoring since we get scores from CompAnalyzer
                match
            )
        }
        
    def _process_genre_match(self, match: Dict, criteria: Dict) -> Dict:
        """Process genre and subgenre matches."""
        genre_id = match.get('genre_id')
        selected_id = criteria.get('genre_id')
        genre_match = genre_id == selected_id
        
        return {
            'name1': self.get_field_name('genre', genre_id),
            'name2': self.get_field_name('genre', selected_id),
            'selected': selected_id is not None,
            'match': genre_match
        }
        
    def _process_single_field_match(self, field: str, value_id: Optional[int], 
                                   selected_id: Optional[int], max_score: float, 
                                   match: Optional[Dict] = None) -> Dict:
        """Process match for a single-value field."""
        is_match = value_id == selected_id
        
        # Get score from CompAnalyzer
        score = self.comp_analyzer.get_field_score(field, match) if match else 0
        max_score = self.comp_analyzer.get_field_max_score(field)
        
        return {
            'name1': self.get_field_name(field, value_id),
            'name2': self.get_field_name(field, selected_id),
            'selected': selected_id is not None,
            'match': is_match,
            'score': score,
            'max': max_score
        }
        
    def _empty_match_details(self) -> Dict:
        """Return empty match details for unselected fields."""
        return {
            'score': 0,
            'match_details': FieldMatch(
                name1='Not Selected',
                name2='Not Selected',
                selected=False,
                match=False,
                score=0,
                max_score=0
            )
        }

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
