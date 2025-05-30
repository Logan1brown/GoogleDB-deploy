"""CompAnalyzer: Source of truth for show comparison scoring.

This component defines the scoring weights and rules that other components
(like CompBuilder) must reference. It uses ShowsAnalyzer as its data provider
to maintain consistency with our established component pattern.

Key concepts:

1. Scoring:
   - Content match (82 points): genre, source type, character types, plot elements,
     theme elements, tone, time/location setting
   - Production match (13 points): network, studios, team
   - Format match (5 points): episodes, order type

2. Field configuration:
   - Each field has a table_name, id_field, and name_field
   - Array fields (e.g. studios[]) vs single fields (e.g. network)
   - Special handling for team members (grouped by name)

3. Data flow:
   - CompAnalyzer gets data from ShowsAnalyzer
   - ShowsAnalyzer gets data from api_show_comp_data view
   - View joins all necessary tables for efficient querying
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import logging

# Test if streamlit writes are working


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
    """A single option for a field with its ID and display name.
    
    For team members, all_ids contains all IDs that map to this name.
    For other fields, all_ids is None.
    """
    id: int
    name: str
    all_ids: Optional[List[int]] = None
    
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
        'team_members': FieldConfig('api_show_comp_data', 'team_member_ids', 'team_member_names', True),  # Use view for processed team data
        'order_type': FieldConfig('order_types', 'id', 'type')  # shows.order_type_id
    }
    
    def __init__(self, reference_data: Dict[str, pd.DataFrame]):
        self.reference_data = reference_data
        self.options: Dict[str, List[FieldOption]] = {}
        self._load_options()
        
    def _load_options(self):
        """Load all field options from reference data."""
        for field_name, config in self.FIELD_CONFIGS.items():
            # Special handling for different data sources
            if field_name == 'subgenres':
                df = self.reference_data['subgenres']
            elif field_name == 'team_members':
                logger.info(f"Loading team members from reference data")
                df = self.reference_data[field_name]  # Use view for processed team data
                logger.info(f"Team members df columns: {list(df.columns)}")
            else:
                df = self.reference_data[field_name]
                
            # Special handling for team members from api_show_comp_data
            if field_name == 'team_members':
                # Use dictionary to maintain unique entries by ID
                unique_members = {}
                
                # Process each team member
                for _, row in df.iterrows():
                    team_member_ids = row.get('team_member_ids', [])
                    team_member_names = row.get('team_member_names', [])
                    
                    # Skip if either array is empty or not a list
                    if not isinstance(team_member_ids, list) or not isinstance(team_member_names, list) \
                        or len(team_member_ids) == 0 or len(team_member_names) == 0:
                        continue
                        
                    # Process each team member
                    for id, name in zip(team_member_ids, team_member_names):
                        # Convert name to string and check if empty
                        name_str = str(name).strip()
                        if len(name_str) > 0:
                            if name_str not in unique_members:
                                unique_members[name_str] = []
                            unique_members[name_str].append(int(id))
                
                # Convert dictionary to list of options
                options = []
                for name, ids in unique_members.items():
                    # Use first ID as primary and store all IDs
                    options.append(FieldOption(id=ids[0], name=name, all_ids=ids))
                self.options[field_name] = sorted(options, key=lambda x: x.name)
            else:
                clean_members = {}
                for _, row in df.iterrows():
                    if pd.notna(row[config.id_field]) and pd.notna(row[config.name_field]):
                        id = int(row[config.id_field])
                        name = str(row[config.name_field])
                        clean_name = self._normalize_name(name)
                        if isinstance(clean_name, str) and len(clean_name) > 0:
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
                
                self.options[field_name] = sorted(options, key=lambda x: str(x.name) if x.name is not None else '')
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
    """Score breakdown for a comparable show match (100 points total).
    
    Note on field naming:
    - Array fields in the data use plural names (studios[], team_members[])
    - Score fields here use singular names (studio, team) since they represent
      individual match scores (e.g. studio.primary, team.first)
    """
    
    # Content Match
    genre_base: float = field(default=0)      # Direct genre match
    genre_overlap: float = field(default=0)    # Subgenre overlap
    source_type: float = field(default=0)      # Source type match
    character_types: float = field(default=0)  # Character type overlap
    plot_elements: float = field(default=0)    # Plot element overlap
    thematic_elements: float = field(default=0)   # Theme overlap
    tone: float = field(default=0)            # Tone match
    time_setting: float = field(default=0)    # Time period match
    location: float = field(default=0)        # Location match
    
    # Production Match
    network: float = field(default=0)         # Network match
    studio: float = field(default=0)          # Studio overlap
    team: float = field(default=0)            # Team overlap
    
    # Format Match
    episodes: float = field(default=0)        # Episode count similarity
    order_type: float = field(default=0)      # Order type match
    
    # Reference to scoring config
    _scoring: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate all scores are non-negative and convert to float."""
        for field_name, field_value in self.__dict__.items():
            if field_name.startswith('_'):
                continue
            try:
                # Convert to float, handle None/empty values
                if field_value is None or field_value == '':
                    field_value = 0.0
                else:
                    field_value = float(field_value)
                if field_value < 0:
                    field_value = 0.0
                # Update the field with converted value
                setattr(self, field_name, field_value)
            except (TypeError, ValueError):
                # If conversion fails, set to 0
                setattr(self, field_name, 0.0)
                
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
            self.thematic_elements,
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
        
    def get_match_details(self) -> Dict[str, Dict[str, Any]]:
        """Get match details for display."""
        # Convert all scores to float to ensure they're serializable
        # Create a deep copy of the scoring config to avoid modifying the original
        scoring = dict(self._scoring)
        
        # Convert all nested values to standard Python types
        content = {
            'score': float(self.content_score()),
            'max': float(scoring['content']['total']),
            'components': {
                'genre_base': {
                    'score': float(self.genre_base),
                    'max': float(scoring['content']['components']['genre']['base'])
                },
                'genre_overlap': {
                    'score': float(self.genre_overlap),
                    'max': float(scoring['content']['components']['genre']['overlap'])
                },
                'source_type': {
                    'score': float(self.source_type),
                    'max': float(scoring['content']['components']['source_type']['match'])
                },
                'character_types': {
                    'score': float(self.character_types),
                    'max': float(scoring['content']['components']['character_types']['first'] + 
                          scoring['content']['components']['character_types']['second'])
                },
                'plot_elements': {
                    'score': float(self.plot_elements),
                    'max': float(scoring['content']['components']['plot_elements']['first'] + 
                          scoring['content']['components']['plot_elements']['second'])
                },
                'thematic_elements': {
                    'score': float(self.thematic_elements),
                    'max': float(scoring['content']['components']['thematic_elements']['first'] + 
                          scoring['content']['components']['thematic_elements']['second'])
                },
                'tone': {
                    'score': float(self.tone),
                    'max': float(scoring['content']['components']['tone']['match'])
                },
                'time_setting': {
                    'score': float(self.time_setting),
                    'max': float(scoring['content']['components']['time_setting']['match'])
                },
                'location_setting': {
                    'score': float(self.location),
                    'max': float(scoring['content']['components']['location_setting']['match'])
                }
            }
        }
        
        production = {
            'score': float(self.production_score()),
            'max': float(scoring['production']['total']),
            'components': {
                'network': {
                    'score': float(self.network),
                    'max': float(scoring['production']['components']['network']['match'])
                },
                'studio': {
                    'score': float(self.studio),
                    'max': float(scoring['production']['components']['studio']['primary'] + 
                          scoring['production']['components']['studio']['max_additional'])
                },
                'team': {
                    'score': float(self.team),
                    'max': float(scoring['production']['components']['team']['first'] + 
                          scoring['production']['components']['team']['max_additional'])
                }
            }
        }
        
        format_section = {
            'score': float(self.format_score()),
            'max': float(scoring['format']['total']),
            'components': {
                'episodes': {
                    'score': float(self.episodes),
                    'max': float(scoring['format']['components']['episodes']['within_2'])
                },
                'order_type': {
                    'score': float(self.order_type),
                    'max': float(scoring['format']['components']['order_type']['match'])
                }
            }
        }
        
        return {
            'content': content,
            'production': production,
            'format': format_section
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
                'genre': {'base': 15, 'overlap': 8},  # Primary genre more important
                'source_type': {'match': 10},
                'character_types': {'first': 9, 'second': 3},  # Reduced to 12 total
                'plot_elements': {'first': 7.5, 'second': 2.5},  # Reduced to 10 total
                'thematic_elements': {'first': 7.5, 'second': 2.5},  # Reduced to 10 total
                'tone': {'match': 9},
                'time_setting': {'match': 4},  # Increased to 4
                'location_setting': {'match': 4}  # Increased to 4
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
        score = CompScore(_scoring=self.SCORING)
        
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
                
        # Only calculate source type score if it was selected in criteria
        if source.get('source_type_id') is not None:
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
        
        score.thematic_elements = self._calculate_array_match(
            source.get('thematic_element_ids') or [],
            target.get('thematic_element_ids') or [],
            self.SCORING['content']['components']['thematic_elements']['first'],
            self.SCORING['content']['components']['thematic_elements']['second'],
            'thematic_elements'
        )
        
        # Direct matches
        # Only calculate tone score if it was selected in criteria
        if source.get('tone_id') is not None:
            if source.get('tone_id') == target.get('tone_id'):
                score.tone = self.SCORING['content']['components']['tone']['match']
            
        # Only calculate time setting score if it was selected in criteria
        if source.get('time_setting_id') is not None:
            if source.get('time_setting_id') == target.get('time_setting_id'):
                score.time_setting = self.SCORING['content']['components']['time_setting']['match']
            
        # Only calculate location setting score if it was selected in criteria
        if source.get('location_setting_id') is not None:
            if source.get('location_setting_id') == target.get('location_setting_id'):
                score.location = self.SCORING['content']['components']['location_setting']['match']
            
        # Only calculate network score if network was selected in criteria
        if source.get('network_id') is not None:
            if source.get('network_id') == target.get('network_id'):
                score.network = self.SCORING['production']['components']['network']['match']
        
        # Studio matching
        source_studios = source.get('studios')
        if source_studios:  # Check if studios were selected in criteria
            source_studios = set(source_studios)
            target_studios = set(target.get('studios') or [])
            matches = source_studios & target_studios
            if matches:  # If there are any matches
                score.studio = self.SCORING['production']['components']['studio']['primary']
                # Calculate additional points for matches beyond the first
                additional_matches = len(matches) - 1
                if additional_matches > 0:
                    additional_points = min(
                        additional_matches * self.SCORING['production']['components']['studio']['additional'],
                        self.SCORING['production']['components']['studio']['max_additional']
                    )
                    score.studio += additional_points
                    
        # Team matching
        source_team = source.get('team_member_ids')  # Match the mapped field name
        if source_team:  # Check if team members were selected in criteria
            target_team = target.get('team_member_ids') or []  # Match the mapped field name
            score.team = self._calculate_array_match(
                source_team,
                target_team,
                self.SCORING['production']['components']['team']['first'],
                self.SCORING['production']['components']['team']['additional'],
                'team_members'
            )
                    
        # Episode scoring
        # Only calculate episode score if episode count was selected in criteria and both shows have valid numeric counts
        source_eps = source.get('episode_count')
        if source_eps is not None:
            target_eps = target.get('episode_count')
            if pd.notna(source_eps) and pd.notna(target_eps) and \
               isinstance(source_eps, (int, float)) and isinstance(target_eps, (int, float)):
                diff = abs(source_eps - target_eps)
                if diff <= 2:
                    score.episodes = self.SCORING['format']['components']['episodes']['within_2']
                elif diff <= 4:
                    score.episodes = self.SCORING['format']['components']['episodes']['within_4']
                elif diff <= 6:
                    score.episodes = self.SCORING['format']['components']['episodes']['within_6']
                
        # Only calculate order type score if it was selected in criteria
        if source.get('order_type_id') is not None:
            if source.get('order_type_id') == target.get('order_type_id'):
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
        if field_name == 'team_member_ids':  # Match the mapped field name
            # Get all team member options with their grouped IDs
            team_options = self.field_manager.get_options('team_member_ids')  # Match the mapped field name
            
            # Create a map of ID -> name that includes all IDs for each name
            id_to_name = {}
            for opt in team_options:
                # Each ID for this name maps to the same name
                # We know all_ids is always set for team members
                for team_id in opt.all_ids:
                    id_to_name[team_id] = opt.name
            
            import streamlit as st
            st.write(f"DEBUG team_member_ids: source={source_arr}, target={target_arr}")
            st.write(f"DEBUG id_to_name: {id_to_name}")
                    
            # Get unique names for source and target using the lookup map
            source_names = {id_to_name.get(id) for id in source_arr if id in id_to_name}
            target_names = {id_to_name.get(id) for id in target_arr if id in id_to_name}
                    
            # Remove None values and count matches by unique names
            source_names = {name for name in source_names if name}
            target_names = {name for name in target_names if name}
            matches = source_names & target_names
            

            # Calculate points based on matches
            if len(matches) > 0:
                # First match gets first_points
                score = first_points
                # Additional matches get second_points up to max_additional
                additional_matches = len(matches) - 1
                if additional_matches > 0:
                    # Get max_additional from the SCORING config for team members
                    max_additional = self.SCORING['production']['components']['team']['max_additional']
                    # Calculate additional points capped by max_additional
                    additional_points = min(
                        additional_matches * second_points,
                        max_additional
                    )
                    score += additional_points
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
        self.success_analyzer = SuccessAnalyzer(self.shows_analyzer)
        self.comp_data = None
        self.reference_data = None
        self.field_manager = None
        self.score_engine = None  

    def initialize(self, force: bool = False):
        """Initialize or refresh the analyzer data."""
        # Only initialize if not already initialized or force refresh
        if self.comp_data is None or force:
            
            # Get fresh data from ShowsAnalyzer
            self.comp_data, self.reference_data = self.shows_analyzer.fetch_comp_data(force=force)
            
            # Validate data
            if self.comp_data is None or self.reference_data is None:
                raise ValueError("Failed to get comp data from ShowsAnalyzer")
                
            # Convert numeric fields for success calculation
            numeric_fields = ['tmdb_seasons', 'tmdb_avg_eps']
            for field in numeric_fields:
                try:
                    self.comp_data[field] = pd.to_numeric(self.comp_data[field], errors='coerce')
                except KeyError:
                    pass
            
        # Only recreate field manager if data changes
        if self.field_manager is None or force:
            self.field_manager = FieldManager(self.reference_data)
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
            import traceback
            traceback.print_exc()
            raise
            
    def get_field_score(self, field: str, match: Dict) -> float:
        """Get the score for a specific field from a match's comp_score.
        
        Args:
            field: Field to get score for
            match: Match data containing comp_score
            
        Returns:
            Score for the field
        """
        if not match or 'comp_score' not in match:
            return 0
            
        comp_score = match['comp_score']
        components = comp_score.get('components', {})
        
        if field == 'genre':
            return float(components.get('genre_base', 0) + components.get('genre_overlap', 0))
        
        return float(components.get(field, 0))
        
    def get_field_max_score(self, field: str) -> float:
        """Get the maximum possible score for a field based on scoring config.
        
        Args:
            field: Field to get max score for
            
        Returns:
            Maximum possible score for the field
        """
        if not self.score_engine:
            self.initialize()
            
        if field == 'content':
            return self.score_engine.SCORING['content']['total']
        elif field == 'production':
            return self.score_engine.SCORING['production']['total']
        elif field == 'format':
            return self.score_engine.SCORING['format']['total']
        
        # Individual field scores
        content = self.score_engine.SCORING['content']['components']
        production = self.score_engine.SCORING['production']['components']
        format = self.score_engine.SCORING['format']['components']
        
        if field == 'genre':
            return content['genre']['base'] + content['genre']['overlap']
        elif field == 'source_type':
            return content['source_type']['match']
        elif field in ['character_types', 'plot_elements', 'thematic_elements']:
            component = content[field]
            return component['first'] + component['second']
        elif field == 'tone':
            return content['tone']['match']
        elif field == 'time_setting':
            return content['time_setting']['match']
        elif field == 'location_setting':
            return content['location_setting']['match']
        elif field == 'network':
            return production['network']['match']
        elif field == 'studio':
            return production['studio']['primary'] + production['studio']['max_additional']
        elif field == 'team':
            return production['team']['first'] + production['team']['max_additional']
        elif field == 'episodes':
            return format['episodes']['within_2']
        elif field == 'order_type':
            return format['order_type']['match']
            
        return 0
        
    def get_field_display_name(self, field_name: str, id: int) -> str:
        """Get display name for a field value.
        
        Args:
            field_name: Name of the field to get name for
            id: ID to get name for
            
        Returns:
            Display name for the ID
        """
        try:
            if not self.field_manager:
                raise RuntimeError("CompAnalyzer not initialized")
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
            if not self.field_manager:
                raise RuntimeError("CompAnalyzer not initialized")
            return self.field_manager.get_display_options(field_name)
        except Exception as e:
            raise
        
    def find_by_criteria(self, criteria: Dict) -> List[Tuple[int, CompScore]]:
        """Find shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria to match against
            
        Returns:
            List of tuples containing (show_id, CompScore) for matching shows,
            sorted by total score descending.
        """
            
        if self.comp_data is None:
            raise RuntimeError("CompAnalyzer not initialized")
            
        # Create a dummy show with the criteria
        # Map field names to match database schema
        mapped_criteria = {}
        field_mapping = {
            'studios': 'studios',  # Keep same field name as in database
            'character_type_ids': 'character_type_ids',
            'plot_element_ids': 'plot_element_ids',
            'thematic_element_ids': 'thematic_element_ids',
            'team_members': 'team_member_ids',  # Map to database field name
            'episode_count': 'episode_count'  # First season episode count
        }
        
        # Convert criteria values to proper types
        for key, value in criteria.items():
            # Use mapped name if it exists, otherwise use original
            mapped_key = field_mapping.get(key, key)
            
            # Convert arrays to lists if they're not already
            if isinstance(value, (list, set)):
                mapped_criteria[mapped_key] = list(value)
            else:
                # Include None values and handle other types
                if value is None:
                    mapped_criteria[mapped_key] = None
                elif isinstance(value, (int, float, str)):
                    mapped_criteria[mapped_key] = value
                else:
                    # Try to convert to string if it's some other type
                    try:
                        mapped_criteria[mapped_key] = str(value)
                    except:
                        # Skip invalid values
                        continue
            
        # Create a dummy show with mapped criteria
        source = pd.Series(mapped_criteria)
        
        # Score each show
        results = []
        for _, target in self.comp_data.iterrows():
            score = self.score_engine.calculate_score(source, target)
            # Include results with any matching criteria
            if score.total() > 0 or score.character_types > 0 or score.plot_elements > 0 or score.thematic_elements > 0:
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

                # Convert CompScore to dict for consistent handling
                score_dict = {
                    'total': float(score.total()),
                    'content': float(score.content_score()),
                    'production': float(score.production_score()),
                    'format': float(score.format_score()),
                    'components': {
                        'genre_base': float(score.genre_base),
                        'genre_overlap': float(score.genre_overlap),
                        'source_type': float(score.source_type),
                        'character_types': float(score.character_types),
                        'plot_elements': float(score.plot_elements),
                        'thematic_elements': float(score.thematic_elements),
                        'tone': float(score.tone),
                        'time_setting': float(score.time_setting),
                        'location': float(score.location),
                        'network': float(score.network),
                        'studio': float(score.studio),
                        'team': float(score.team),
                        'episodes': float(score.episodes),
                        'order_type': float(score.order_type)
                    }
                }
                
                # Include all fields exactly as they appear in api_show_comp_data view
                result = {
                    # Core Show Data
                    'id': target['id'],
                    'title': target['title'],
                    'genre_id': target.get('genre_id'),
                    'subgenres': target.get('subgenres', []),
                    'source_type_id': target.get('source_type_id'),
                    'episode_count': target.get('episode_count'),
                    'order_type_id': target.get('order_type_id'),
                    
                    # Description Analysis
                    'time_setting_id': target.get('time_setting_id'),
                    'location_setting_id': target.get('location_setting_id'),
                    'tone_id': target.get('tone_id'),
                    'character_type_ids': target.get('character_type_ids', []),
                    'plot_element_ids': target.get('plot_element_ids', []),
                    'thematic_element_ids': target.get('thematic_element_ids', []),
                    
                    # Production Data
                    'network_id': target.get('network_id'),
                    'studios': target.get('studios', []),
                    'team_member_ids': target.get('team_member_ids', []),
                    'role_categories': target.get('role_categories', []),
                    
                    # Display Names
                    'genre_name': target.get('genre_name'),
                    'subgenre_names': target.get('subgenre_names', []),
                    'source_type_name': target.get('source_type_name'),
                    'order_type_name': target.get('order_type_name'),
                    'time_setting_name': target.get('time_setting_name'),
                    'location_setting_name': target.get('location_setting_name'),
                    'tone_name': target.get('tone_name'),
                    'character_type_names': target.get('character_type_names', []),
                    'plot_element_names': target.get('plot_element_names', []),
                    'thematic_element_names': target.get('thematic_element_names', []),
                    'network_name': target.get('network_name'),
                    'studio_names': target.get('studio_names', []),
                    'team_member_names': target.get('team_member_names', []),
                    
                    # TMDB Success Metrics
                    'tmdb_id': target.get('tmdb_id'),
                    'tmdb_seasons': target.get('tmdb_seasons'),
                    'tmdb_status': target.get('tmdb_status'),
                    'tmdb_total_episodes': target.get('tmdb_total_episodes'),
                    'tmdb_avg_eps': target.get('tmdb_avg_eps'),
                    
                    # Description
                    'description': target.get('description', ''),
                    'comp_score': score_dict,  # Store as dict with explicit float conversions
                    'success_score': success_score,  # Add success score
                    'score': float(score.total())  # Add total score for sorting
                }
                results.append(result)
                
        # Sort by total score descending, then by success score descending
        return sorted(results, key=lambda x: (x['score'], x.get('success_score', 0)), reverse=True)
        
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
                # Convert CompScore to dict for consistent handling
                score_dict = {
                    'total': float(score.total()),
                    'content': float(score.content_score()),
                    'production': float(score.production_score()),
                    'format': float(score.format_score()),
                    'components': {
                        'genre_base': float(score.genre_base),
                        'genre_overlap': float(score.genre_overlap),
                        'source_type': float(score.source_type),
                        'character_types': float(score.character_types),
                        'plot_elements': float(score.plot_elements),
                        'thematic_elements': float(score.thematic_elements),
                        'tone': float(score.tone),
                        'time_setting': float(score.time_setting),
                        'location': float(score.location),
                        'network': float(score.network),
                        'studio': float(score.studio),
                        'team': float(score.team),
                        'episodes': float(score.episodes),
                        'order_type': float(score.order_type)
                    }
                }
                result = {
                    'id': target['id'],
                    'title': target['title'],
                    'description': target.get('description', ''),
                    'comp_score': score_dict,
                    'score': float(score.total())
                }
                scores.append(result)
                
        # Sort by total score descending and limit results
        return sorted(scores, key=lambda x: x['score'], reverse=True)[:limit]
        
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
            'scores': score.get_match_details(),
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
            # Special handling for team members - map IDs to names
            if field == 'team_members':
                source_ids = source.get('team_member_ids', [])
                target_ids = target.get('team_member_ids', [])
                source_names = [self.field_manager.get_name('team_member_ids', id) for id in source_ids]
                target_names = [self.field_manager.get_name('team_member_ids', id) for id in target_ids]
                # Calculate matches based on IDs
                matches = set(source_ids) & set(target_ids)
                # Map matched IDs back to names
                matched_names = [self.field_manager.get_name('team_member_ids', id) for id in matches]
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
            raise
