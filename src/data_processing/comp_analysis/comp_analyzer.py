"""CompAnalyzer: Source of truth for show comparison scoring.

This component defines the scoring weights and rules that other components
(like CompBuilder) must reference. It uses ShowsAnalyzer as its data provider
to maintain consistency with our established component pattern.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from src.config.supabase_client import get_client
from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.success_analysis import SuccessAnalyzer


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

@dataclass
class CompScore:
    """Score breakdown for a comparable show match (100 points total)."""
    
    # Content Match (72 points)
    genre_base: float = field(default=0)     # Base genre match (9 points)
    genre_overlap: float = field(default=0)  # Subgenre overlap (8 points)
    source_type: float = field(default=0)   # Source type match (10 points)
    character_types: float = field(default=0)  # Character type matches (14 points)
    plot_elements: float = field(default=0)   # Plot element matches (12 points)
    theme_elements: float = field(default=0)  # Theme element matches (13 points)
    tone: float = field(default=0)          # Tone match (9 points)
    time_setting: float = field(default=0)  # Time period match (4 points)
    location: float = field(default=0)      # Location match (3 points)
    
    # Production Match (23 points)
    network: float = field(default=0)       # Network match (5 points)
    studio: float = field(default=0)        # Studio matches (3 points)
    team: float = field(default=0)         # Team overlap (5 points)
    
    # Format Match (5 points)
    episodes: float = field(default=0)      # Episode count similarity (4 points)
    order_type: float = field(default=0)   # Order type match (1 point)
    
    def __post_init__(self):
        """Validate all scores are non-negative."""
        for field_name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or value < 0:
                setattr(self, field_name, 0)
    
    def to_display_dict(self) -> Dict[str, Any]:
        """Convert scores to a display-friendly dictionary."""
        return {
            'genre_score': (self.genre_base or 0) + (self.genre_overlap or 0),
            'source_score': self.source_type or 0,
            'character_score': self.character_types or 0,
            'plot_score': self.plot_elements or 0,
            'theme_score': self.theme_elements or 0,
            'tone_score': self.tone or 0,
            'time_score': self.time_setting or 0,
            'location_score': self.location or 0,
            'team_score': self.team or 0,
            'episode_score': self.episodes or 0,
            'order_score': self.order_type or 0,
            'network_score': self.network or 0,
            'studio_score': self.studio or 0,
            'date_score': 0,  # Not tracked yet
            'content_total': self.content_score,
            'format_total': self.format_score,
            'setting_total': (self.time_setting or 0) + (self.location or 0)
        }
    
    def get_match_details(self) -> Dict[str, Any]:
        """Get match details for display."""
        return {
            'genre': {
                'primary_match': bool(self.genre_base),
                'subgenre_points': self.genre_overlap or 0,
                'primary': None,
                'shared_subgenres': []
            },
            'source': {
                'match': bool(self.source_type),
                'name': None
            },
            'characters': {
                'match': bool(self.character_types),
                'types': []
            },
            'plot': {
                'match': bool(self.plot_elements),
                'elements': []
            },
            'themes': {
                'match': bool(self.theme_elements),
                'elements': []
            },
            'tone': {
                'match': bool(self.tone),
                'name': None
            },
            'setting': {
                'time_match': bool(self.time_setting),
                'location_match': bool(self.location),
                'time': None,
                'location': None
            },
            'format': {
                'episode_match': bool(self.episodes),
                'order_match': bool(self.order_type),
                'episodes': None,
                'order': None
            }
        }

    @property
    def total(self) -> float:
        """Calculate total score across all components."""
        return sum([
            # Content (72 points)
            self.genre_base or 0,
            self.genre_overlap or 0,
            self.source_type or 0,
            self.character_types or 0,
            self.plot_elements or 0,
            self.theme_elements or 0,
            self.tone or 0,
            self.time_setting or 0,
            self.location or 0,
            
            # Production (23 points)
            self.network or 0,
            self.studio or 0,
            self.team or 0,
            
            # Format (5 points)
            self.episodes or 0,
            self.order_type or 0
        ])

    @property
    def content_score(self) -> float:
        """Calculate content match score (72 points)."""
        return sum([
            self.genre_base or 0,
            self.genre_overlap or 0,
            self.source_type or 0,
            self.character_types or 0,
            self.plot_elements or 0,
            self.theme_elements or 0,
            self.tone or 0,
            self.time_setting or 0,
            self.location or 0
        ])

    @property
    def production_score(self) -> float:
        """Calculate production match score (23 points)."""
        return sum([
            self.network or 0,
            self.studio or 0,
            self.team or 0
        ])

    @property
    def format_score(self) -> float:
        """Calculate format match score (5 points)."""
        return sum([
            self.episodes or 0,
            self.order_type or 0
        ])


class CompAnalyzer:
    """Analyzer for show comparisons and similarity scoring."""

    # Scoring configuration
    SCORING_CONFIG = {
        # Content Match (70 points total)
        'content': {
            'total_points': 70,
            'components': {
                # Genre (17 points)
                'genre': {
                    'points': 17,
                    'breakdown': {
                        'base_match': 9,       # Direct genre match
                        'subgenre_match': 8     # 1.6 points per subgenre match up to 5
                    }
                },
                # Source Type (10 points)
                'source_type': {
                    'points': 10,
                    'breakdown': {
                        'direct_match': 10      # Direct source type match
                    }
                },
                # Character Types (14 points)
                'character_types': {
                    'points': 14,
                    'breakdown': {
                        'base_match': 5,        # Primary character type
                        'additional_match': 1.8  # Per additional match up to 5
                    }
                },
                # Plot Elements (12 points)
                'plot_elements': {
                    'points': 12,
                    'breakdown': {
                        'per_match': 2.4        # 2.4 points per match up to 5
                    }
                },
                # Thematic Elements (13 points)
                'thematic_elements': {
                    'points': 13,
                    'breakdown': {
                        'per_match': 2.6        # 2.6 points per match up to 5
                    }
                },
                # Tone (9 points)
                'tone': {
                    'points': 9,
                    'breakdown': {
                        'direct_match': 9      # Direct tone match
                    }
                },
                # Setting (7 points)
                'setting': {
                    'points': 7,
                    'breakdown': {
                        'time_period': 4,       # Time period match
                        'location': 3           # Location type match
                    }
                }
            }
        },
        
        # Production Match (13 points total)
        'production': {
            'total_points': 13,
            'components': {
                # Network (5 points)
                'network': {
                    'points': 5,
                    'breakdown': {
                        'direct_match': 5       # Same network
                    }
                },
                # Studio (3 points)
                'studio': {
                    'points': 3,
                    'breakdown': {
                        'primary_match': 2,     # Primary studio match
                        'additional_match': 0.5  # Per additional match up to 2
                    }
                },
                # Team (5 points)
                'team': {
                    'points': 5,
                    'breakdown': {
                        'per_match': 1          # 1 point per team member match up to 5
                    }
                }
            }
        },
        
        # Format Match (3 points total)
        'format': {
            'total_points': 3,
            'components': {
                # Episode Count (4 points)
                'episodes': {
                    'points': 4,
                    'breakdown': {
                        'within_2': 4,          # Within 2 episodes
                        'within_4': 3,        # Within 4 episodes
                        'within_6': 2           # Within 6 episodes
                    }
                },
                # Order Type (1 point)
                'order_type': {
                    'points': 1,
                    'breakdown': {
                        'direct_match': 1        # Same order type
                    }
                }
            }
        }
    }

    def __init__(self, shows_analyzer: Optional[ShowsAnalyzer] = None, success_analyzer: Optional[SuccessAnalyzer] = None):
        """Initialize the analyzer.
        
        Args:
            shows_analyzer: Optional ShowsAnalyzer instance. If not provided,
                          a new instance will be created.
            success_analyzer: Optional SuccessAnalyzer instance. If not provided,
                          a new instance will be created.
        """
        self.shows_analyzer = shows_analyzer or ShowsAnalyzer()
        self.success_analyzer = success_analyzer or SuccessAnalyzer()
        self.comp_data = None
        self.field_options = {}
        
    def get_field_options(self, force: bool = False) -> Dict[str, List[Tuple[int, str]]]:
        """Get all unique values for dropdown fields.
        
        Args:
            force: If True, bypass cache and fetch fresh data
            
        Returns:
            Dictionary mapping field names to lists of (id, name) tuples
        """
        try:
            if self.comp_data is None or force:
                self.fetch_comp_data()
                
            if not self.field_options or force:
                self.field_options = {}
                
                # Extract field options from comp_data
                field_mappings = [
                    ('genres', 'genre_id', 'genre_name'),
                    ('source_types', 'source_type_id', 'source_type_name'),
                    ('character_types', 'character_type_ids', 'character_type_names'),
                    ('plot_elements', 'plot_element_ids', 'plot_element_names'),
                    ('thematic_elements', 'thematic_element_ids', 'thematic_element_names'),
                    ('tones', 'tone_id', 'tone_name'),
                    ('time_settings', 'time_setting_id', 'time_setting_name'),
                    ('locations', 'location_setting_id', 'location_setting_name'),
                    ('networks', 'network_id', 'network_name'),
                    ('studios', 'studios', 'studio_names'),
                    ('order_types', 'order_type_id', 'order_type_name')
                ]
                
                # Handle subgenres separately since they're a special case
                self.field_options['subgenre_names'] = []
                for _, row in self.comp_data.iterrows():
                    if isinstance(row['subgenre_names'], list):
                        for subgenre in row['subgenre_names']:
                            if pd.notna(subgenre):
                                self.field_options['subgenre_names'].append((len(self.field_options['subgenre_names']), str(subgenre)))
                
                # Extract all field options using a single consistent approach
                for field_name, id_col, name_col in field_mappings:
                    # Handle array fields (studios, character_types, plot_elements, thematic_elements)
                    if field_name in ['studios', 'character_types', 'plot_elements', 'thematic_elements']:
                        array_col = 'studios' if field_name == 'studios' else f"{field_name[:-1]}_ids"
                        tuples = []
                        for _, row in self.comp_data.iterrows():
                            # Get the array of IDs and names
                            ids = row[array_col]
                            names = row[name_col]
                            
                            # Skip if either is missing
                            if not isinstance(ids, list) or not isinstance(names, list):
                                continue
                                
                            # Create tuples from parallel arrays
                            for item_id, name in zip(ids, names):
                                if pd.notna(item_id) and pd.notna(name):
                                    tuples.append((int(item_id), str(name)))
                    else:
                        # Handle regular fields
                        tuples = [(int(row[id_col]), str(row[name_col]))
                                 for _, row in self.comp_data.iterrows()
                                 if pd.notna(row[id_col]) and pd.notna(row[name_col])]
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    self.field_options[field_name] = [x for x in tuples 
                                                    if not (x in seen or seen.add(x))]
                
            return self.field_options
        except Exception as e:
            print(f"Error in get_field_options: {str(e)}")
            print(f"Available columns: {self.comp_data.columns if self.comp_data is not None else 'No data'}")
            import traceback
            traceback.print_exc()
            raise
        
    def fetch_comp_data(self) -> pd.DataFrame:
        """Fetch show comparison data using ShowsAnalyzer.
        
        Returns:
            DataFrame containing all data needed for show comparisons.
        """
        try:
            # Get both comp data and market data
            self.comp_data = self.shows_analyzer.fetch_comp_data()
            titles_df, _, _ = self.shows_analyzer.fetch_market_data()
            
            # Calculate average episodes per season
            titles_df['tmdb_avg_eps'] = titles_df.apply(
                lambda x: x['tmdb_total_episodes'] / x['tmdb_seasons'] 
                if pd.notna(x['tmdb_total_episodes']) and pd.notna(x['tmdb_seasons']) and x['tmdb_seasons'] > 0
                else None,
                axis=1
            )
            
            # Initialize success analyzer with market data
            self.success_analyzer.initialize_data(titles_df)
            
            return self.comp_data
            
        except Exception as e:
            raise Exception(f"Error fetching comp data: {str(e)}")
            
    def find_by_criteria(self, criteria: Dict) -> List[Dict]:
        """Find shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria to match against
            
        Returns:
            List of shows matching the criteria with their scores
        """
        if self.comp_data is None:
            self.fetch_comp_data()
            
        # Create a "criteria show" with the selected criteria
        criteria_show = pd.Series({
            'id': -1,  # Dummy ID
            'genre_id': criteria.get('genre_id'),
            'subgenres': criteria.get('subgenres', []),
            'source_type_id': criteria.get('source_type_id'),
            'character_type_ids': criteria.get('character_type_ids', []),
            'plot_element_ids': criteria.get('plot_element_ids', []),
            'thematic_element_ids': criteria.get('thematic_element_ids', []),
            'tone_id': criteria.get('tone_id'),
            'time_setting_id': criteria.get('time_setting_id'),
            'location_setting_id': criteria.get('location_setting_id'),
            'network_id': criteria.get('network_id'),
            'studios': criteria.get('studios', []),
            'team_member_ids': criteria.get('team_member_ids', []),
            'episode_count': criteria.get('episode_count') if 'episode_count' in criteria else None,
            'order_type_id': criteria.get('order_type_id')
        })
        
        # Calculate scores for all shows against criteria
        results = []
        for _, show in self.comp_data.iterrows():
            # Calculate match score using existing scoring logic
            score = self._calculate_score(criteria_show, show)
            
            # Get success score
            success_score = 0
            show_data = self.success_analyzer.titles_df[
                self.success_analyzer.titles_df['id'] == show['id']
            ]
            if not show_data.empty:
                success_score = self.success_analyzer.calculate_success(show_data.iloc[0])
                
            # Add to results if score above threshold
            if score.total > 0:  # Only include shows with some match
                show_dict = show.to_dict()
                show_dict['comp_score'] = score
                show_dict['success_score'] = success_score
                results.append(show_dict)
            
        # Sort by weighted score
        results.sort(key=lambda x: 0.7 * x['comp_score'].total + 0.3 * x['success_score'], reverse=True)
        return results[:10]
        
    def get_similar_shows(self, show_id: int, limit: int = 10) -> List[Tuple[int, CompScore]]:
        """Get similar shows for the given show ID.
        
        Args:
            show_id: ID of the show to find similar shows for
            limit: Maximum number of similar shows to return
            
        Returns:
            List of tuples containing (show_id, CompScore) for similar shows,
            sorted by total score descending.
        """
        if self.comp_data is None:
            self.fetch_comp_data()
            
        # Get source show data
        source = self.comp_data[self.comp_data['id'] == show_id].iloc[0]
        
        # Calculate scores for all other shows
        scores = []
        for _, target in self.comp_data[self.comp_data['id'] != show_id].iterrows():
            score = self._calculate_score(source, target)
            scores.append((target['id'], score))
            
        # Calculate success scores for matches
        success_scores = {}
        for show_id, _ in scores:
            # Look up show in market data by show id
            show_data = self.success_analyzer.titles_df[
                self.success_analyzer.titles_df['id'] == show_id
            ]
            if not show_data.empty:
                success_scores[show_id] = self.success_analyzer.calculate_success(show_data.iloc[0])
            
        # Sort by weighted combination of match score and success score
        def score_key(item):
            show_id, comp_score = item
            success_score = success_scores.get(show_id, 0)
            return 0.7 * comp_score.total + 0.3 * success_score
            
        # Get top matches with full data
        top_matches = sorted(scores, key=score_key, reverse=True)[:limit]
        results = []
        for show_id, comp_score in top_matches:
            show_data = self.comp_data[self.comp_data['id'] == show_id].iloc[0]
            show_data = show_data.to_dict()
            show_data['comp_score'] = comp_score
            show_data['success_score'] = success_scores.get(show_id, 0)
            results.append(show_data)
            
        return results
    
    def _calculate_score(self, source: pd.Series, target: pd.Series) -> CompScore:
        """Calculate comparison score between two shows.
        
        Args:
            source: Source show data from comp_data
            target: Target show data from comp_data
            
        Returns:
            CompScore object containing score breakdown
        """
        try:
            # Content scores
            genre_base = (
                self.SCORING_CONFIG['content']['components']['genre']['breakdown']['base_match']
                if source['genre_id'] == target['genre_id']
                else 0
            )
            
            # Get subgenre names for comparison
            source_subgenres = [name for _, name in self.field_options['subgenre_names'] if _ in source['subgenres']]
            target_subgenres = [name for _, name in self.field_options['subgenre_names'] if _ in target['subgenres']]
            
            genre_overlap = min(
                len(set(source_subgenres).intersection(set(target_subgenres))) * 1.6,
                self.SCORING_CONFIG['content']['components']['genre']['breakdown']['subgenre_match']
            )
            
            source_type = (
                self.SCORING_CONFIG['content']['components']['source_type']['breakdown']['direct_match']
                if source['source_type_id'] == target['source_type_id']
                else 0
            )
            
            character_types = self._calculate_array_match(
                source['character_type_ids'],
                target['character_type_ids'],
                self.SCORING_CONFIG['content']['components']['character_types']['breakdown']['base_match'],
                self.SCORING_CONFIG['content']['components']['character_types']['breakdown']['additional_match'],
                5
            )
            
            plot_elements = self._calculate_array_match(
                source['plot_element_ids'],
                target['plot_element_ids'],
                0,
                self.SCORING_CONFIG['content']['components']['plot_elements']['breakdown']['per_match'],
                5
            )
            
            theme_elements = self._calculate_array_match(
                source['thematic_element_ids'],
                target['thematic_element_ids'],
                0,
                self.SCORING_CONFIG['content']['components']['thematic_elements']['breakdown']['per_match'],
                5
            )
            
            tone = (
                self.SCORING_CONFIG['content']['components']['tone']['breakdown']['direct_match']
                if source['tone_id'] == target['tone_id']
                else 0
            )
            
            time_setting = (
                self.SCORING_CONFIG['content']['components']['setting']['breakdown']['time_period']
                if source['time_setting_id'] == target['time_setting_id']
                else 0
            )
            
            location = (
                self.SCORING_CONFIG['content']['components']['setting']['breakdown']['location']
                if source['location_setting_id'] == target['location_setting_id']
                else 0
            )
            
            # Production scores
            network = (
                self.SCORING_CONFIG['production']['components']['network']['breakdown']['direct_match']
                if source['network_id'] == target['network_id']
                else 0
            )
            
            studio = self._calculate_array_match(
                source['studios'],
                target['studios'],
                self.SCORING_CONFIG['production']['components']['studio']['breakdown']['primary_match'],
                self.SCORING_CONFIG['production']['components']['studio']['breakdown']['additional_match'],
                2
            )
            
            # Team score based on shared team members
            team = 0  # TODO: Implement team scoring once team data is available
            
            # Format scores
            episodes = self._calculate_episode_score(
                source['episode_count'],
                target['episode_count']
            )
            
            order_type = (
                self.SCORING_CONFIG['format']['components']['order_type']['breakdown']['direct_match']
                if source['order_type_id'] == target['order_type_id']
                else 0
            )
            
            return CompScore(
                genre_base=genre_base,
                genre_overlap=genre_overlap,
                source_type=source_type,
                character_types=character_types,
                plot_elements=plot_elements,
                theme_elements=theme_elements,
                tone=tone,
                time_setting=time_setting,
                location=location,
                network=network,
                studio=studio,
                team=team,
                episodes=episodes,
                order_type=order_type
            )
        except Exception as e:
            print(f"Error in _calculate_score: {str(e)}")
            print(f"Source columns: {source.index}")
            print(f"Target columns: {target.index}")
            import traceback
            traceback.print_exc()
            raise
    
    def _calculate_array_match(
        self,
        source_arr: List,
        target_arr: List,
        base_points: float,
        per_match_points: float,
        max_matches: int
    ) -> float:
        """Calculate score for array field matches.
        
        Args:
            source_arr: Source array
            target_arr: Target array
            base_points: Points for first match
            per_match_points: Points per additional match
            max_matches: Maximum number of matches to count
            
        Returns:
            Total points for matches
        """
        if not source_arr or not target_arr:
            return 0
            
        matches = len(set(source_arr).intersection(set(target_arr)))
        if matches == 0:
            return 0
            
        if base_points > 0:
            # First match gets base points, rest get per_match
            additional = min(matches - 1, max_matches - 1)
            return base_points + (additional * per_match_points)
        else:
            # All matches get per_match points
            return min(matches, max_matches) * per_match_points
    
    def _calculate_episode_score(self, source_eps: Optional[int], target_eps: Optional[int]) -> float:
        """Calculate episode count similarity score.
        
        Args:
            source_eps: Source show episode count
            target_eps: Target show episode count
            
        Returns:
            Score based on episode count difference
        """
        # Handle None, strings, and other non-numeric values
        try:
            if source_eps is None or target_eps is None:
                return 0
            source_val = int(source_eps)
            target_val = int(target_eps)
        except (TypeError, ValueError):
            return 0
            
        diff = abs(source_val - target_val)
        if diff <= 2:
            return self.SCORING_CONFIG['format']['components']['episodes']['breakdown']['within_2']
        elif diff <= 4:
            return self.SCORING_CONFIG['format']['components']['episodes']['breakdown']['within_4']
        elif diff <= 6:
            return self.SCORING_CONFIG['format']['components']['episodes']['breakdown']['within_6']
        else:
            return 0
