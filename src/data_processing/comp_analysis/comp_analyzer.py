"""CompAnalyzer: Source of truth for show comparison scoring.

This component defines the scoring weights and rules that other components
(like CompBuilder) must reference. It uses ShowsAnalyzer as its data provider
to maintain consistency with our established component pattern.

Key Features:
- Scoring system for show similarity
- Field option management
- Show comparison and matching
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
from datetime import datetime
import streamlit as st
import logging

from src.config.supabase_client import get_client
from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class FieldConfig:
    """Configuration for a field in the comparison system."""
    table_name: str  # Reference table name
    id_field: str   # Field containing the ID
    name_field: str  # Field containing display name
    is_array: bool = False  # Whether field contains multiple values
    
@dataclass
class FieldOption:
    """A single option for a field with its ID and display name."""
    id: int
    name: str
    
class FieldManager:
    """Manages field options and mappings."""
    
    # Field configuration mapping
    FIELD_CONFIGS = {
        'character_types': FieldConfig('character_type_types', 'id', 'name', True),  # show_character_type_list
        'genre': FieldConfig('genre_list', 'id', 'genre'),  # Primary genre
        'subgenres': FieldConfig('genre_list', 'id', 'genre', True),  # shows.subgenres[]
        'source_type': FieldConfig('source_types', 'id', 'type'),  # Not array - shows.source_type_id
        'plot_elements': FieldConfig('plot_element_types', 'id', 'name', True),  # show_plot_element_list
        'thematic_elements': FieldConfig('thematic_element_types', 'id', 'name', True),  # show_thematic_element_list
        'tone': FieldConfig('tone_types', 'id', 'name'),  # show_description_analysis.tone_id
        'time_setting': FieldConfig('time_setting_types', 'id', 'name'),  # show_description_analysis.time_setting_id
        'location_setting': FieldConfig('location_setting_types', 'id', 'name'),  # show_description_analysis.location_setting_id
        'network': FieldConfig('network_list', 'id', 'network'),  # shows.network_id
        'studios': FieldConfig('studio_list', 'id', 'studio', True),  # shows.studios[]
        'team_members': FieldConfig('api_show_comp_data', 'team_member_ids', 'team_member_names', True),  # api_show_comp_data
        'order_type': FieldConfig('order_types', 'id', 'type')  # shows.order_type_id
    }
    
    def __init__(self, reference_data: Dict[str, pd.DataFrame]):
        self.reference_data = reference_data
        self.options: Dict[str, List[FieldOption]] = {}
        self._load_options()
        
    def _load_options(self):
        """Load all field options from reference data."""
        for field_name, config in self.FIELD_CONFIGS.items():
            # For subgenres, use the subgenres list
            if field_name == 'subgenres':
                df = self.reference_data['subgenres']
            else:
                df = self.reference_data[field_name]
                
            # Special handling for team members from api_show_comp_data
            if field_name == 'team_members':
                # Use dictionary to maintain unique entries by ID
                unique_members = {}
                    
                # Debug: Print first few rows
                logger.info("First few rows of team member data:")
                logger.info(df.head().to_dict())
                
                # Collect unique team members
                for _, row in df.iterrows():
                    if isinstance(row['team_member_ids'], list) and isinstance(row['team_member_names'], list):
                        # Debug: Print arrays for first row
                        if len(unique_members) == 0:
                            logger.info(f"First row IDs: {row['team_member_ids']}")
                            logger.info(f"First row names: {row['team_member_names']}")
                        
                        for id, name in zip(row['team_member_ids'], row['team_member_names']):
                            # Debug: Print each ID/name pair
                            if name == 'Adam Bernstein':
                                logger.info(f"Found Adam Bernstein with ID {id}")
                            # Only add if we haven't seen this ID before
                            if name not in unique_members:
                                unique_members[name] = id
                                
                # Convert dictionary to list of options
                options = [FieldOption(id=id, name=name) for name, id in unique_members.items()]
                self.options[field_name] = sorted(options, key=lambda x: x.name)
            else:
                clean_members = {}
                for _, row in df.iterrows():
                    if pd.notna(row[config.id_field]) and pd.notna(row[config.name_field]):
                        id = int(row[config.id_field])
                        name = str(row[config.name_field])
                        clean_name = self._normalize_name(name)
                        if clean_name:
                            if clean_name not in clean_members:
                                clean_members[clean_name] = id
                            elif isinstance(clean_members[clean_name], int):
                                # Convert to list if we have a duplicate
                                clean_members[clean_name] = [clean_members[clean_name], id]
                            else:
                                # Add to existing list
                                clean_members[clean_name].append(id)
                        
                # Create options with first ID for display but store all IDs
                options = []
                for name, id_or_ids in clean_members.items():
                    if isinstance(id_or_ids, list):
                        opt = FieldOption(id=id_or_ids[0], name=name)
                        opt.all_ids = id_or_ids
                    else:
                        opt = FieldOption(id=id_or_ids, name=name)
                        opt.all_ids = [id_or_ids]
                    options.append(opt)
                
                self.options[field_name] = sorted(options, key=lambda x: x.name)
                pass  # Options loaded
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a team member name for consistent matching.
        
        Handles variations like:
        - David E. Kelley vs David E Kelley
        - John Smith Jr. vs John Smith Jr
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name for matching
        """
        # Remove extra spaces and periods from middle initials
        return ' '.join(part.strip(' .') for part in name.split())
    
    def _get_team_member_options(self, df: pd.DataFrame) -> List[FieldOption]:
        """Get deduplicated team member options with grouped IDs.
        
        For team members, we want to show unique names in the dropdown
        but preserve all IDs for matching. This helper groups by name
        and collects all IDs for each unique name.
        
        Args:
            df: DataFrame containing team member data
            
        Returns:
            List of FieldOption with unique names and grouped IDs
        """
        # Add normalized names for grouping
        df = df.copy()
        df['normalized_name'] = df['name'].apply(self._normalize_name)
        
        # Group by normalized name to collect all IDs for each unique name
        grouped = df.groupby('normalized_name').agg({
            'id': list,
            'name': 'first'  # Keep first name variant for display
        }).reset_index()
        
        # Create options with first ID (for display) but store all IDs
        options = []
        for _, row in grouped.iterrows():
            # Use first ID for the option (needed for UI)
            opt = FieldOption(id=row['id'][0], name=row['name'])
            # Store all IDs as additional attribute
            opt.all_ids = row['id']
            options.append(opt)
            
        return sorted(options, key=lambda x: x.name)

    def get_options(self, field_name: str) -> List[FieldOption]:
        """Get all options for a field."""
        return self.options.get(field_name, [])
        
    def get_display_options(self, field_name: str) -> List[Tuple[int, str]]:
        """Get (id, name) tuples for display."""
        return [(opt.id, opt.name) for opt in self.get_options(field_name)]
        
    def get_name(self, field_name: str, id: int) -> str:
        """Get display name for an ID."""
        for opt in self.get_options(field_name):
            if opt.id == id:
                return opt.name
        return f'Unknown ({id})'

@dataclass
class CompScore:
    """Score breakdown for a comparable show match (100 points total)."""
    
    # Content Match (82 points)
    genre_base: float = field(default=0)      # Direct genre match (9 points)
    genre_overlap: float = field(default=0)    # Subgenre overlap (8 points)
    source_type: float = field(default=0)      # Source type match (10 points)
    character_types: float = field(default=0)  # Character type overlap (14 points)
    plot_elements: float = field(default=0)    # Plot element overlap (12 points)
    theme_elements: float = field(default=0)   # Theme overlap (13 points)
    tone: float = field(default=0)            # Tone match (9 points)
    time_setting: float = field(default=0)    # Time period match (4 points)
    location: float = field(default=0)        # Location match (3 points)
    
    # Production Match (13 points)
    network: float = field(default=0)         # Network match (5 points)
    studio: float = field(default=0)          # Studio overlap (3 points)
    team: float = field(default=0)            # Team overlap (5 points)
    
    # Format Match (5 points)
    episodes: float = field(default=0)        # Episode count similarity (4 points)
    order_type: float = field(default=0)      # Order type match (1 point)
    
    def __post_init__(self):
        """Validate all scores are non-negative."""
        for field_name, field_value in self.__dict__.items():
            if field_value < 0:
                raise ValueError(f'Score cannot be negative: {field_name} = {field_value}')
                
    def total(self) -> float:
        """Calculate total score across all components."""
        return self.content_score() + self.production_score() + self.format_score()
    
    def content_score(self) -> float:
        """Calculate content match score (82 points)."""
        return float(sum([
            self.genre_base,
            self.genre_overlap,
            self.source_type,
            self.character_types,
            self.plot_elements,
            self.theme_elements,
            self.tone,
            self.time_setting,
            self.location
        ]))
    
    def production_score(self) -> float:
        """Calculate production match score (13 points)."""
        score = float(self.network + self.studio + self.team)
        logger.debug(f"Production score breakdown: network={self.network}, studio={self.studio}, team={self.team}, total={score}")
        return score
    
    def format_score(self) -> float:
        """Calculate format match score (5 points)."""
        return float(sum([
            self.episodes,
            self.order_type
        ]))

    def to_display_dict(self) -> Dict[str, float]:
        """Convert scores to a display-friendly dictionary."""
        return {
            'total': float(self.total()),
            'content': float(self.content_score()),
            'production': float(self.production_score()),
            'format': float(self.format_score())
        }
        
    def get_match_details(self) -> Dict[str, Dict[str, Any]]:
        """Get match details for display."""
        return {
            'content': {
                'score': self.content_score(),
                'max': 82,
                'components': {
                    'genre': self.genre_base + self.genre_overlap,
                    'source_type': self.source_type,
                    'character_types': self.character_types,
                    'plot_elements': self.plot_elements,
                    'theme_elements': self.theme_elements,
                    'tone': self.tone,
                    'setting': self.time_setting + self.location
                }
            },
            'production': {
                'score': self.production_score(),
                'max': 13,
                'components': {
                    'network': self.network,
                    'studio': self.studio,
                    'team': self.team
                }
            },
            'format': {
                'score': self.format_score(),
                'max': 5,
                'components': {
                    'episodes': self.episodes,
                    'order_type': self.order_type
                }
            }
        }

class ScoreEngine:
    """Handles all scoring calculations."""
    
    def __init__(self, field_manager: Optional[FieldManager] = None):
        """Initialize the score engine.
        
        Args:
            field_manager: FieldManager instance for field option lookups
        """
        self.field_manager = field_manager
    
    # Scoring configuration
    SCORING = {
        'content': {
            'total': 82,
            'components': {
                'genre': {'base': 9, 'overlap': 8},
                'source_type': {'match': 10},
                'character_types': {'first': 10.5, 'second': 3.5},
                'plot_elements': {'first': 9, 'second': 3},
                'theme_elements': {'first': 9.75, 'second': 3.25},
                'tone': {'match': 9},
                'setting': {'time': 4, 'location': 3}
            }
        },
        'production': {
            'total': 13,
            'components': {
                'network': {'match': 5},
                'studio': {'primary': 2, 'additional': 0.5, 'max_additional': 2},
                'team': {'first': 3, 'additional': 1, 'max_additional': 2}
            }
        },
        'format': {
            'total': 5,
            'components': {
                'episodes': {'within_2': 4, 'within_4': 3, 'within_6': 2},
                'order_type': {'match': 1}
            }
        }
    }
    
    def calculate_score(self, source: pd.Series, target: pd.Series) -> CompScore:
        """Calculate comparison score between two shows."""
        score = CompScore()
        
        # Content scoring
        # Only apply genre scoring if genre is specified in criteria
        if source.get('genre_id') is not None:
            if source.get('genre_id') == target.get('genre_id'):
                score.genre_base = self.SCORING['content']['components']['genre']['base']
                
            source_subgenres = set(source.get('subgenres') or [])
            target_subgenres = set(target.get('subgenres') or [])
            if source_subgenres and target_subgenres:
                overlap = len(source_subgenres & target_subgenres)
                if overlap > 0:
                    score.genre_overlap = self.SCORING['content']['components']['genre']['overlap']
                
        # Source type
        if source.get('source_type_id') == target.get('source_type_id'):
            score.source_type = self.SCORING['content']['components']['source_type']['match']
            
        # Array field scoring
        score.character_types = self._calculate_array_match(
            source.get('character_type_ids') or [],
            target.get('character_type_ids') or [],
            self.SCORING['content']['components']['character_types']['first'],
            self.SCORING['content']['components']['character_types']['second'],
            'character_types'
        )
        
        score.plot_elements = self._calculate_array_match(
            source.get('plot_element_ids') or [],
            target.get('plot_element_ids') or [],
            self.SCORING['content']['components']['plot_elements']['first'],
            self.SCORING['content']['components']['plot_elements']['second'],
            'plot_elements'
        )
        
        score.theme_elements = self._calculate_array_match(
            source.get('thematic_element_ids') or [],
            target.get('thematic_element_ids') or [],
            self.SCORING['content']['components']['theme_elements']['first'],
            self.SCORING['content']['components']['theme_elements']['second'],
            'thematic_elements'
        )
        
        # Direct matches
        if source.get('tone_id') == target.get('tone_id'):
            score.tone = self.SCORING['content']['components']['tone']['match']
            
        if source.get('time_setting_id') == target.get('time_setting_id'):
            score.time_setting = self.SCORING['content']['components']['setting']['time']
            
        if source.get('location_setting_id') == target.get('location_setting_id'):
            score.location = self.SCORING['content']['components']['setting']['location']
            
        if source.get('network_id') == target.get('network_id'):
            score.network = self.SCORING['production']['components']['network']['match']
            
        # Studio matching
        source_studios = set(source.get('studios') or [])
        target_studios = set(target.get('studios') or [])
        if source_studios and target_studios:
            matches = len(source_studios & target_studios)
            if matches > 0:
                score.studio = self.SCORING['production']['components']['studio']['primary']
                additional = min(
                    matches - 1,
                    self.SCORING['production']['components']['studio']['max_additional']
                )
                if additional > 0:
                    score.studio += additional * self.SCORING['production']['components']['studio']['additional']
                    
        # Team matching
        source_team = source.get('team_member_ids') or []
        target_team = target.get('team_member_ids') or []
        if source_team and target_team:
            score.team = self._calculate_array_match(
                source_team, target_team,
                self.SCORING['production']['components']['team']['first'],
                self.SCORING['production']['components']['team']['additional'],
                'team_members'
            )
                    
        # Episode scoring
        if pd.notna(source['episode_count']) and pd.notna(target['episode_count']):
            diff = abs(source['episode_count'] - target['episode_count'])
            if diff <= 2:
                score.episodes = self.SCORING['format']['components']['episodes']['within_2']
            elif diff <= 4:
                score.episodes = self.SCORING['format']['components']['episodes']['within_4']
            elif diff <= 6:
                score.episodes = self.SCORING['format']['components']['episodes']['within_6']
                
        # Order type
        if source['order_type_id'] == target['order_type_id']:
            score.order_type = self.SCORING['format']['components']['order_type']['match']
            
        return score
    
    def _calculate_array_match(self, source_arr: List, target_arr: List, first_points: float, second_points: float, field_name: str = None) -> float:
        """Calculate score for array field matches.
        
        For team members, we need to check if any of the source IDs match any of the target IDs,
        since we've grouped IDs by name in the UI.
        
        Args:
            source_arr: List of IDs from source show
            target_arr: List of IDs from target show
            first_points: Points for first match
            second_points: Points for additional matches
            field_name: Name of the field being matched (used to identify team members)
        """
        if not source_arr or not target_arr:
            return 0
            
        # For team members, we need to count unique names for scoring
        if field_name == 'team_members':
            # Get all team member options with their grouped IDs
            team_options = self.field_manager.get_options('team_members')
            
            # Create a map of ID -> name that includes all IDs for each name
            id_to_name = {}
            for opt in team_options:
                # Each ID for this name maps to the same name
                # Handle case where all_ids might be None
                if hasattr(opt, 'all_ids') and opt.all_ids:
                    for team_id in opt.all_ids:
                        id_to_name[team_id] = opt.name
                else:
                    # Fallback to single ID if all_ids not available
                    id_to_name[opt.id] = opt.name
                
            # Get unique names for source and target using the lookup map
            source_names = {id_to_name[id] for id in (source_arr or []) if id in id_to_name}
            target_names = {id_to_name[id] for id in (target_arr or []) if id in id_to_name}
                    
            # Count matches by unique names
            matches = source_names & target_names
            
            # Calculate points based on matches
            if len(matches) > 0:
                # First match gets first_points
                score = first_points
                # Additional matches get second_points up to max_additional
                additional_matches = len(matches) - 1
                if additional_matches > 0:
                    score += min(additional_matches * second_points, scoring.get('max_additional', float('inf')))
                return score
            return 0
        else:
            # Normal array matching for other fields
            # Handle None values in arrays
            source_set = set(x for x in source_arr if x is not None)
            target_set = set(x for x in target_arr if x is not None)
            matches = source_set & target_set
            
            if not matches:
                return 0
                
            # First match is worth more
            score = first_points
            
            # Additional matches get lower points
            if len(matches) > 1:
                score += (len(matches) - 1) * second_points
                
            return score
    
    def calculate_episode_score(self, source_eps: int, target_eps: int) -> float:
        """Calculate episode count similarity score.
        
        Args:
            source_eps: Episode count of source show
            target_eps: Episode count of target show
            
        Returns:
            Score based on episode count difference
        """
        if pd.isna(source_eps) or pd.isna(target_eps):
            return 0
            
        diff = abs(source_eps - target_eps)
        if diff <= 2:
            return self.SCORING['format']['components']['episodes']['within_2']
        elif diff <= 4:
            return self.SCORING['format']['components']['episodes']['within_4']
        elif diff <= 6:
            return self.SCORING['format']['components']['episodes']['within_6']
        return 0


class CompAnalyzer:
    """Main analyzer for show comparisons."""

    def __init__(self, shows_analyzer: Optional[ShowsAnalyzer] = None):
        self.shows_analyzer = shows_analyzer or ShowsAnalyzer()
        self.success_analyzer = SuccessAnalyzer()
        self.comp_data = None
        self.reference_data = None
        self.field_manager = None
        self.score_engine = None  # Initialize after field_manager is set
        
    def initialize(self, force: bool = False):
        """Initialize or refresh the analyzer data."""
        if force or self.comp_data is None:
            self.comp_data, self.reference_data = self.shows_analyzer.fetch_comp_data()
            
            # Convert numeric fields for success calculation
            numeric_fields = ['tmdb_seasons', 'tmdb_avg_eps']
            for field in numeric_fields:
                try:
                    self.comp_data[field] = pd.to_numeric(self.comp_data[field], errors='coerce')
                except KeyError:
                    logger.warning(f"{field} not found in comp_data")
            
            # Initialize success analyzer with the comp data
            self.success_analyzer.initialize_data(self.comp_data)
            
        # Initialize field manager
        self.field_manager = FieldManager(self.reference_data)
        # Initialize score engine with field manager
        self.score_engine = ScoreEngine(self.field_manager)
    
    def get_field_options(self, force: bool = False) -> Dict[str, List[Tuple[int, str]]]:
        """Get all unique values for dropdown fields.
        
        Args:
            force: If True, bypass cache and fetch fresh data
            
        Returns:
            Dictionary mapping field names to lists of (id, name) tuples
        """
        try:
            self.initialize(force)
            return {field: self.field_manager.get_display_options(field)
                    for field in FieldManager.FIELD_CONFIGS.keys()}
                    
        except Exception as e:
            st.error(f"Error in get_field_options: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def get_field_display_name(self, field_name: str, id: int) -> str:
        """Get display name for a field value.
        
        Args:
            field_name: Name of the field to get name for
            id: ID to get name for
            
        Returns:
            Display name for the ID
        """
        try:
            self.initialize()
            return self.field_manager.get_name(field_name, id)
        except Exception as e:
            logger.error(f"Error in get_field_display_name: {e}")
            logger.error(f"Field: {field_name}, ID: {id}")
            raise
        
    def get_field_display_options(self, field_name: str) -> List[Tuple[int, str]]:
        """Get display options (id, name) for a field.
        
        Args:
            field_name: Name of the field to get display options for
            
        Returns:
            List of (id, name) tuples sorted by name
        """
        try:
            self.initialize()
            return self.field_manager.get_display_options(field_name)
        except Exception as e:
            logger.error(f"Error in get_field_display_options: {e}")
            logger.error(f"Field: {field_name}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
    def find_by_criteria(self, criteria: Dict) -> List[Tuple[int, CompScore]]:
        """Find shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria to match against
            
        Returns:
            List of tuples containing (show_id, CompScore) for matching shows,
            sorted by total score descending.
        """
        self.initialize()
            
        # Create a dummy show with the criteria
        # Map field names to match database schema
        mapped_criteria = {}
        field_mapping = {
            'studios': 'studio_ids',  # Map DB field to UI field
            'character_type_ids': 'character_type_ids',
            'plot_element_ids': 'plot_element_ids',
            'thematic_element_ids': 'thematic_element_ids',
            'team_member_ids': 'team_member_ids',  # For matching/scoring
            'team_member_names': 'team_member_names',  # For display
            'episode_count': 'episode_count'  # First season episode count
        }
        
        for key, value in criteria.items():
            # Use mapped name if it exists, otherwise use original
            mapped_key = field_mapping.get(key, key)
            mapped_criteria[mapped_key] = value
            
        source = pd.Series(mapped_criteria)
        
        # Score each show
        results = []
        for _, target in self.comp_data.iterrows():
            score = self.score_engine.calculate_score(source, target)
            # Include results with any matching criteria
            if score.total() > 0 or score.character_types > 0 or score.plot_elements > 0 or score.theme_elements > 0:
                # Calculate success score and ensure it's a float
                success_score = float(self.success_analyzer.calculate_success(target))

                # Store debug info in result dictionary
                tmdb_debug = {
                    'id': target['id'],
                    'title': target['title'],
                    'tmdb_status': target.get('tmdb_status'),
                    'tmdb_seasons': target.get('tmdb_seasons'),
                    'tmdb_avg_eps': target.get('tmdb_avg_eps'),
                    'success_score': success_score
                }

                # Include all fields needed for match details
                result = {
                    'id': target['id'],
                    'title': target['title'],
                    'description': target.get('description', ''),  # Add description
                    'success_score': success_score,  # Use calculated success score
                    'comp_score': score,
                    'debug': tmdb_debug,  # Add debug info
                    # Content fields
                    'genre_id': target.get('genre_id'),
                    'subgenres': target.get('subgenres', []),  # Include subgenres
                    'source_type_id': target.get('source_type_id'),
                    'character_type_ids': target.get('character_type_ids', []),
                    'plot_element_ids': target.get('plot_element_ids', []),
                    'thematic_element_ids': target.get('thematic_element_ids', []),
                    'tone_id': target.get('tone_id'),
                    # Setting
                    'time_setting_id': target.get('time_setting_id'),
                    'location_setting_id': target.get('location_setting_id'),
                    # Production
                    'network_id': target.get('network_id'),
                    'team_member_ids': target.get('team_member_ids', []),
                    'team_member_names': target.get('team_member_names', []),
                    'studios': target.get('studios', []),
                    'studio_names': target.get('studio_names', []),  # For display
                    # Format
                    'episode_count': target.get('episode_count'),
                    'order_type_id': target.get('order_type_id')
                }
                results.append(result)
                
        # Sort by total score descending
        return sorted(results, key=lambda x: x['comp_score'].total(), reverse=True)
        
    def get_similar_shows(self, show_id: int, limit: int = 10) -> List[Tuple[int, CompScore]]:
        """Get similar shows for the given show ID.
        
        Args:
            show_id: ID of the show to find similar shows for
            limit: Maximum number of similar shows to return
            
        Returns:
            List of tuples containing (show_id, CompScore) for similar shows,
            sorted by total score descending.
        """
        self.initialize()
            
        # Get source show
        source = self.comp_data[self.comp_data['id'] == show_id].iloc[0]
        
        # Score each show
        scores = []
        for _, target in self.comp_data.iterrows():
            if target['id'] == show_id:
                continue
                
            score = self.score_engine.calculate_score(source, target)
            if score.total() > 0:
                scores.append((target['id'], score))
                
        # Sort by total score descending and limit results
        return sorted(scores, key=lambda x: x[1].total(), reverse=True)[:limit]
        
    def get_comparison_details(self, source_id: int, target_id: int) -> Dict:
        """Get detailed comparison between two shows.
        
        Args:
            source_id: ID of the source show
            target_id: ID of the target show
            
        Returns:
            Dictionary containing comparison details including scores and field values
        """
        self.initialize()
        
        # Get shows
        source = self.comp_data[self.comp_data['id'] == source_id].iloc[0]
        target = self.comp_data[self.comp_data['id'] == target_id].iloc[0]
        
        # Calculate score
        score = self.score_engine.calculate_score(source, target)
        
        # Build response
        details = {
            'scores': score.to_display_dict(),
            'match_details': score.get_match_details(),
            'source': {
                'id': source_id,
                'title': source['title'],
                'fields': {}
            },
            'target': {
                'id': target_id,
                'title': target['title'],
                'fields': {}
            }
        }
        
        # Add field details
        for field in FieldManager.FIELD_CONFIGS.keys():
            # Special handling for team members - use names directly
            if field == 'team_members':
                source_names = source.get('team_member_names', [])
                target_names = target.get('team_member_names', [])
            else:
                # Get field values
                source_val = source[f'{field}_id'] if f'{field}_id' in source else source.get(field, [])
                target_val = target[f'{field}_id'] if f'{field}_id' in target else target.get(field, [])
                
                # Get display names
                if isinstance(source_val, list):
                    source_names = [self.field_manager.get_name(field, id) for id in source_val]
                    target_names = [self.field_manager.get_name(field, id) for id in target_val]
                else:
                    source_names = [self.field_manager.get_name(field, source_val)] if pd.notna(source_val) else []
                    target_names = [self.field_manager.get_name(field, target_val)] if pd.notna(target_val) else []
            
            # Add to details
            details['source']['fields'][field] = source_names
            details['target']['fields'][field] = target_names
        
        return details
    
    def _calculate_score(self, source: pd.Series, target: pd.Series) -> CompScore:
        """Calculate comparison score between two shows.
        
        Args:
            source: Source show data from comp_data
            target: Target show data from comp_data
            
        Returns:
            CompScore object containing score breakdown
        """
        try:
            return self.score_engine.calculate_score(source, target)
        except Exception as e:
            logger.error(f"Error in _calculate_score: {str(e)}")
            logger.error(f"Source columns: {source.index}")
            logger.error(f"Target columns: {target.index}")
            import traceback
            logger.error(traceback.format_exc())
            raise
