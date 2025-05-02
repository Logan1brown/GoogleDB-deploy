"""Manager for match details display logic."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer

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
    
    def __init__(self, comp_analyzer: CompAnalyzer):
        self.comp_analyzer = comp_analyzer
        self.scoring = comp_analyzer.score_engine.SCORING
        
    def get_field_name(self, field: str, id: Optional[int], default: str = 'Unknown') -> str:
        """Get display name for a field value."""
        if id is None:
            return default
        try:
            return self.comp_analyzer.get_field_display_name(field, id)
        except:
            return default
            
    def get_field_names(self, field: str, ids: List[int], default: str = 'Unknown') -> List[str]:
        """Get display names for field values."""
        return [self.get_field_name(field, id, default) for id in ids]
        
    def create_match_details(self, match: Dict, criteria: Dict) -> Dict:
        """Transform match and criteria into UI-ready match details."""
        details = {}
        
        # Content match details
        details['genre'] = self._process_genre_match(match, criteria)
        details['source'] = self._process_single_field_match(
            'source_type', match.get('source_type_id'), criteria.get('source_type_id'),
            self.scoring['content']['components']['source_type']['match']
        )
        details['characters'] = self._process_array_field_match(
            'character_types', match.get('character_type_ids', []), criteria.get('character_type_ids', []),
            self.scoring['content']['components']['character_types']
        )
        details['plot'] = self._process_array_field_match(
            'plot_elements', match.get('plot_element_ids', []), criteria.get('plot_element_ids', []),
            self.scoring['content']['components']['plot_elements']
        )
        details['themes'] = self._process_array_field_match(
            'thematic_elements', match.get('thematic_element_ids', []), criteria.get('thematic_element_ids', []),
            self.scoring['content']['components']['theme_elements']
        )
        details['tone'] = self._process_single_field_match(
            'tone', match.get('tone_id'), criteria.get('tone_id'),
            self.scoring['content']['components']['tone']['match']
        )
        
        # Production match details
        details['studio'] = self._process_production_field_match(
            'studios', match.get('studios', []), criteria.get('studio_ids', []),
            self.scoring['production']['components']['studio']
        )
        details['team'] = self._process_production_field_match(
            'team_members', match.get('team_member_ids', []), criteria.get('team_ids', []),
            self.scoring['production']['components']['team']
        )
        
        # Setting match details
        details['setting'] = self._process_setting_match(
            match.get('time_setting_id'), match.get('location_setting_id'),
            criteria.get('time_setting_id'), criteria.get('location_setting_id')
        )
        
        # Format match details
        details['format'] = self._process_format_match(
            match.get('episode_count'), criteria.get('episode_count'),
            match.get('order_type_id'), criteria.get('order_type_id')
        )
        
        return details
        
    def _process_genre_match(self, match: Dict, criteria: Dict) -> FieldMatch:
        """Process genre and subgenre matches."""
        genre_id = match.get('genre_id')
        selected_id = criteria.get('genre_id')
        genre_match = genre_id == selected_id
        
        source_subgenres = set(match.get('subgenres', []))
        target_subgenres = set(criteria.get('subgenres', []))
        has_subgenre_match = bool(source_subgenres and target_subgenres and 
                                source_subgenres & target_subgenres)
        
        score = 0
        if genre_match:
            score += self.scoring['content']['components']['genre']['base']
        if has_subgenre_match:
            score += self.scoring['content']['components']['genre']['overlap']
            
        return FieldMatch(
            name1=self.get_field_name('genre', genre_id),
            name2=self.get_field_name('genre', selected_id),
            selected=selected_id is not None,
            match=genre_match,
            score=score,
            max_score=sum(self.scoring['content']['components']['genre'].values())
        )
        
    def _process_single_field_match(self, field: str, value_id: Optional[int], 
                                  selected_id: Optional[int], max_score: float) -> FieldMatch:
        """Process match for a single-value field."""
        return FieldMatch(
            name1=self.get_field_name(field, value_id),
            name2=self.get_field_name(field, selected_id),
            selected=selected_id is not None,
            match=value_id == selected_id,
            score=max_score if value_id == selected_id else 0,
            max_score=max_score
        )
        
    def _process_array_field_match(self, field: str, values: List[int], 
                                 selected: List[int], scoring: Dict) -> ArrayFieldMatch:
        """Process match for a multi-value field."""
        value_names = self.get_field_names(field, values)
        selected_names = self.get_field_names(field, selected)
        
        value_set = set(values)
        selected_set = set(selected)
        matches = value_set & selected_set
        
        score = 0
        if matches:
            score += scoring['first']
            if len(matches) > 1:
                score += scoring['second']
                
        return ArrayFieldMatch(
            name1='Multiple' if value_names else 'None',
            name2='Multiple' if selected_names else 'None',
            selected=bool(selected),
            match=bool(matches),
            score=score,
            max_score=scoring['first'] + scoring['second'],
            values1=value_names,
            values2=selected_names,
            matches=self.get_field_names(field, list(matches))
        )
        
    def _process_production_field_match(self, field: str, values: List[int],
                                      selected: List[int], scoring: Dict) -> ArrayFieldMatch:
        """Process match for production fields (studio, team) with special scoring."""
        value_names = self.get_field_names(field, values)
        selected_names = self.get_field_names(field, selected)
        
        value_set = set(values)
        selected_set = set(selected)
        matches = value_set & selected_set
        
        score = 0
        if matches:
            score += scoring['primary']
            additional_matches = len(matches) - 1
            if additional_matches > 0:
                additional_score = min(
                    additional_matches * scoring['additional'],
                    scoring.get('max_additional', float('inf'))
                )
                score += additional_score
                
        return ArrayFieldMatch(
            name1='Multiple' if value_names else 'None',
            name2='Multiple' if selected_names else 'None',
            selected=bool(selected),
            match=bool(matches),
            score=score,
            max_score=scoring['primary'] + scoring.get('max_additional', 0),
            values1=value_names,
            values2=selected_names,
            matches=self.get_field_names(field, list(matches))
        )
        
    def _process_setting_match(self, time_id: Optional[int], location_id: Optional[int],
                             selected_time_id: Optional[int], 
                             selected_location_id: Optional[int]) -> Dict[str, FieldMatch]:
        """Process time and location setting matches."""
        time_match = self._process_single_field_match(
            'time_setting', time_id, selected_time_id,
            self.scoring['content']['components']['setting']['time']
        )
        
        location_match = self._process_single_field_match(
            'location_setting', location_id, selected_location_id,
            self.scoring['content']['components']['setting']['location']
        )
        
        return {
            'time': time_match,
            'location': location_match,
            'total_score': time_match.score + location_match.score,
            'max_score': time_match.max_score + location_match.max_score
        }
        
    def _process_format_match(self, episodes: Optional[int], selected_episodes: Optional[int],
                            order_type_id: Optional[int], 
                            selected_order_type_id: Optional[int]) -> Dict[str, FieldMatch]:
        """Process episode count and order type matches."""
        episode_score = 0
        if episodes is not None and selected_episodes is not None:
            diff = abs(episodes - selected_episodes)
            if diff <= 2:
                episode_score = self.scoring['format']['components']['episodes']['within_2']
            elif diff <= 4:
                episode_score = self.scoring['format']['components']['episodes']['within_4']
            elif diff <= 6:
                episode_score = self.scoring['format']['components']['episodes']['within_6']
                
        episode_match = FieldMatch(
            name1=str(episodes) if episodes is not None else 'Unknown',
            name2=str(selected_episodes) if selected_episodes is not None else 'Unknown',
            selected=selected_episodes is not None,
            match=episode_score > 0,
            score=episode_score,
            max_score=self.scoring['format']['components']['episodes']['within_2']
        )
        
        order_match = self._process_single_field_match(
            'order_type', order_type_id, selected_order_type_id,
            self.scoring['format']['components']['order_type']['match']
        )
        
        return {
            'episodes': episode_match,
            'order_type': order_match,
            'total_score': episode_match.score + order_match.score,
            'max_score': episode_match.max_score + order_match.max_score
        }
