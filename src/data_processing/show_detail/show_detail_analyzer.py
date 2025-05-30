"""Show Detail Analyzer

Analyzes shows to find similar content and network patterns.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.success_analysis.success_analyzer import SuccessAnalyzer
from src.config.supabase_client import get_client


@dataclass
class SimilarShow:
    show_id: int
    title: str
    network_name: str
    description: Optional[str]
    success_score: Optional[float]
    match_score: Dict[str, int] = field(default_factory=lambda: {
        # Content Match (85 points)
        'genre_score': 0,      # 45 points
        'team_score': 0,       # 25 points
        'source_score': 0,     # 15 points
        'studio_score': 0,     # 3 points
        
        # Format Match (15 points)
        'episode_score': 0,    # 8 points
        'order_score': 0,      # 4 points
        'date_score': 0,       # 3 points
        
        # Group totals
        'content_total': 0,    # Sum of content scores
        'format_total': 0,     # Sum of format scores
        'total': 0            # Overall total
    })


@dataclass
class NetworkAnalysis:
    similar_show_counts: Dict[str, int]  # network -> count
    success_scores: Dict[str, float]     # network -> average success score
    success_rates: Dict[str, float]      # network -> % of shows with high success


class ShowDetailAnalyzer:
    """Analyzer for finding similar shows and network patterns."""
    
    def __init__(self, success_analyzer: Optional[SuccessAnalyzer] = None):
        """Initialize the analyzer.
        
        Args:
            success_analyzer: Optional SuccessAnalyzer instance
        """
        # Create ShowsAnalyzer first if we need to create a SuccessAnalyzer
        self.shows_analyzer = ShowsAnalyzer()
        self.success_analyzer = success_analyzer or SuccessAnalyzer(self.shows_analyzer)
        self._shows_df = None
        
    @st.cache_data(ttl=3600)
    def fetch_show_data(_self, force: bool = False) -> pd.DataFrame:
        """Fetch show data needed for similarity analysis.
        
        Args:
            force: If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame with show details
        """
        try:
            # Get show data from api_show_details view
            supabase = get_client(use_service_key=True)
            result = supabase.table(_self.shows_analyzer.VIEWS['details']).select('*').execute()
            shows_df = pd.DataFrame(result.data)
            return shows_df
        except Exception as e:
            st.write(f"Error fetching show data: {str(e)}")
            raise

    def _ensure_data_loaded(self, force: bool = False):
        """Ensure show data is loaded.
        
        Args:
            force: If True, bypass cache and fetch fresh data
        """
        if force or self._shows_df is None:
            self._shows_df = self.fetch_show_data(force=force)
            
    def search_shows(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for shows by title.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching shows with id, title, and network
        """
        self._ensure_data_loaded()
        
        # Case-insensitive search
        matches = self._shows_df[self._shows_df['title'].str.contains(query, case=False, na=False)]
        
        # Sort by title and take top matches
        matches = matches.sort_values('title').head(limit)
        
        # Return relevant fields
        return [{
            'show_id': row['show_id'],
            'title': row['title'],
            'network_name': row['network_name'],
            'announced_date': row.get('announced_date'),
            'success_score': row.get('success_score')
        } for _, row in matches.iterrows()]
        
    def get_show_details(self, show_id: int) -> Optional[Dict]:
        """Get detailed information for a specific show.
        
        Args:
            show_id: ID of the show to get details for
            
        Returns:
            Dictionary with show details or None if not found
        """
        self._ensure_data_loaded()
        
        show = self._shows_df[self._shows_df['show_id'] == show_id]
        if show.empty:
            return None
            
        show = show.iloc[0]
        return {
            'show_id': show['show_id'],
            'title': show['title'],
            'network_name': show['network_name'],
            'genre_name': show['genre_name'],
            'subgenre_names': show.get('subgenre_names', []),
            'source_type_name': show['source_type_name'],
            'team_members': show.get('team_members', []),
            'success_score': show.get('success_score'),
            'tmdb_status': show.get('tmdb_status'),
            'episode_count': show.get('episode_count'),
            'tmdb_seasons': show.get('tmdb_seasons'),
            'tmdb_total_episodes': show.get('tmdb_total_episodes')
        }
    
    def compute_similarity(self, show1: pd.Series, show2: pd.Series) -> Dict:
        """Compute similarity score between two shows.

        Scoring System (100 points total):
        Content Match (85 points):
            - Genre (45 points):
                * Primary genre match: 30 points
                * Subgenre matches: Up to 15 points
                    - First match: 10 points
                    - Second match: 5 points
            - Team (25 points):
                * 8 points per shared team member
                * Maximum 25 points
            - Source (15 points):
                * Full match: 15 points
                * No match: 0 points

        Format Match (15 points):
            - Episodes (8 points):
                * ≤2 eps difference: 8 points
                * ≤4 eps difference: 5 points
                * ≤6 eps difference: 2 points
            - Order Type (4 points):
                * Same order type: 4 points
            - Timing (3 points):
                * Based on announcement date proximity

        Args:
            show1: First show to compare
            show2: Second show to compare

        Returns:
            Dictionary containing:
                - total: Overall similarity score (0-100)
                - content_total: Content match score (0-85)
                - format_total: Format match score (0-15)
                - Individual component scores and match details
        """
        scores = {
            # Content Match (85 points)
            'genre_score': 0,      # 40 points
            'team_score': 0,       # 20 points
            'source_score': 0,     # 15 points
            'network_score': 0,    # 7 points
            'studio_score': 0,     # 3 points
            
            # Format Match (15 points)
            'episode_score': 0,    # 8 points
            'order_score': 0,      # 4 points
            'date_score': 0,       # 3 points
            
            # Match Details
            'details': {
                'genre': {
                    'primary': show1['genre_name'],
                    'primary_match': False,
                    'primary_points': 0,
                    'shared_subgenres': [],
                    'subgenre_points': 0
                },
                'source': {
                    'type1': show1['source_type_name'],
                    'type2': show2['source_type_name'],
                    'match': False
                },
                'team': {
                    'shared_members': []
                },
                'format': {
                    'eps_per_season1': None,
                    'eps_per_season2': None,
                    'order_type1': show1['order_type_name'],
                    'order_type2': show2['order_type_name']
                },
                'network': {
                    'name1': show1['network_name'],
                    'name2': show2['network_name'],
                    'match': show1['network_name'] == show2['network_name']
                },
                'studio': {
                    'name1': show1.get('studio_names', [])[0] if show1.get('studio_names') else None,
                    'name2': show2.get('studio_names', [])[0] if show2.get('studio_names') else None,
                    'match': bool(show1.get('studio_names')) and bool(show2.get('studio_names')) and \
                            any(s1 == s2 for s1 in show1['studio_names'] for s2 in show2['studio_names'])
                }
            }
        }
        
        # Genre match (40 points)
        # Primary genre match (27 points)
        primary_match = show1['genre_name'] == show2['genre_name']
        if primary_match:
            scores['genre_score'] += 27
            scores['details']['genre']['primary_match'] = True
            scores['details']['genre']['primary_points'] = 27
        scores['details']['genre']['primary'] = show1['genre_name']
        
        # Subgenre matches (13 points)
        subgenre_names1 = set(show1.get('subgenre_names', []) or [])
        subgenre_names2 = set(show2.get('subgenre_names', []) or [])
        shared_subgenre_names = subgenre_names1 & subgenre_names2
        scores['details']['genre']['shared_subgenre_names'] = list(shared_subgenre_names)
        
        if len(shared_subgenre_names) >= 1:
            scores['genre_score'] += 8  # First subgenre match
            scores['details']['genre']['subgenre_points'] += 8
        if len(shared_subgenre_names) >= 2:
            scores['genre_score'] += 5   # Second subgenre match
            scores['details']['genre']['subgenre_points'] += 5
        
        # Team overlap (20 points)
        team1 = [(m['name'], m.get('role', 'Unknown')) for m in show1.get('team_members', []) or []]
        team2 = [(m['name'], m.get('role', 'Unknown')) for m in show2.get('team_members', []) or []]
        shared_members = set(team1) & set(team2)
        scores['details']['team']['shared_members'] = list(shared_members)
        # Round to nearest integer, max 20 points
        scores['team_score'] = min(round(len(shared_members) * 6.7), 20)
        
        # Network match (7 points)
        network_match = show1['network_name'] == show2['network_name']
        scores['network_score'] = 7 if network_match else 0
        
        # Studio match (3 points)
        studio_match = bool(show1.get('studio_names')) and bool(show2.get('studio_names')) and \
                      any(s1 == s2 for s1 in show1['studio_names'] for s2 in show2['studio_names'])
        scores['studio_score'] = 3 if studio_match else 0
        
        # Source match (15 points)
        source_match = show1['source_type_name'] == show2['source_type_name']
        scores['source_score'] = 15 if source_match else 0
        scores['details']['source']['type1'] = show1['source_type_name']
        scores['details']['source']['type2'] = show2['source_type_name']
        scores['details']['source']['match'] = source_match
        
        # Episode format match (8 points)
        eps1 = pd.to_numeric(show1['tmdb_total_episodes'], errors='coerce') / pd.to_numeric(show1['tmdb_seasons'], errors='coerce')
        eps2 = pd.to_numeric(show2['tmdb_total_episodes'], errors='coerce') / pd.to_numeric(show2['tmdb_seasons'], errors='coerce')
        scores['details']['format']['eps_per_season1'] = eps1 if pd.notna(eps1) else None
        scores['details']['format']['eps_per_season2'] = eps2 if pd.notna(eps2) else None
        
        if pd.notna(eps1) and pd.notna(eps2):
            eps_diff = abs(eps1 - eps2)
            if eps_diff <= 2:
                scores['episode_score'] = 8
            elif eps_diff <= 4:
                scores['episode_score'] = 5
            elif eps_diff <= 6:
                scores['episode_score'] = 2
        
        # Order type match (4 points)
        order_match = show1['order_type_name'] == show2['order_type_name']
        scores['order_score'] = 4 if order_match else 0
        scores['details']['format']['order_type1'] = show1['order_type_name']
        scores['details']['format']['order_type2'] = show2['order_type_name']
        
        # Date proximity (3 points)
        date1 = pd.to_datetime(show1['announced_date']).year if pd.notna(show1.get('announced_date')) else None
        date2 = pd.to_datetime(show2['announced_date']).year if pd.notna(show2.get('announced_date')) else None
        if date1 and date2:
            years_apart = abs(date1 - date2)
            scores['date_score'] = max(3 - years_apart, 0)  # -1 point per year apart
        
        # Calculate totals
        scores['content_total'] = (
            scores['genre_score'] + scores['team_score'] + 
            scores['source_score'] + scores['network_score'] + scores['studio_score']
        )
        scores['format_total'] = scores['episode_score'] + scores['order_score'] + scores['date_score']
        scores['total'] = scores['content_total'] + scores['format_total']
        
        return scores
    
    def find_similar_shows(self, show_id: int, min_score: int = 30, max_shows: int = 15) -> List[SimilarShow]:
        """Find shows similar to the given show.
        
        Args:
            show_id: ID of the show to find similar shows for
            min_score: Minimum similarity score to include (default: 30)
            max_shows: Maximum number of similar shows to return (default: 15)
            
        Returns:
            List of SimilarShow objects, sorted by total score
        """
        self._ensure_data_loaded()
        
        # Get target show
        target_show = self._shows_df[self._shows_df['id'] == show_id].iloc[0]
        
        # Calculate similarity scores for all other shows
        similar_shows = []
        for _, show in self._shows_df[self._shows_df['id'] != show_id].iterrows():
            scores = self.compute_similarity(target_show, show)
            if scores['total'] >= min_score:
                # Get success score from SuccessAnalyzer
                success_score = self.success_analyzer.calculate_success(show)
                
                similar_shows.append(SimilarShow(
                    show_id=show['id'],  # Use id from api_show_details
                    title=show['title'],
                    network_name=show['network_name'],
                    description=show.get('description'),
                    success_score=success_score,
                    match_score=scores
                ))
        
        # Sort by total score (descending) then title
        similar_shows.sort(key=lambda x: (-x.match_score['total'], x.title))
        return similar_shows[:max_shows]
    
    def get_success_metrics(self, show_id: int) -> Optional[Dict]:
        """Get success metrics for a specific show.
        
        Args:
            show_id: ID of the show to get metrics for
            
        Returns:
            Dict with success score and breakdown if available, None if not
        """
        print(f"Fetching success metrics for show {show_id}")
        success_df = self.success_analyzer.fetch_success_data()
        print(f"Success data shape: {success_df.shape}")
        print(f"Success data index: {success_df.index[:5]}")
        print(f"Success data columns: {success_df.columns.tolist()}")
        
        if show_id in success_df.index:
            print(f"Found show {show_id} in success data")
            show = success_df.loc[show_id]
            return {
                'score': show['success_score'],
                'breakdown': self.success_analyzer.get_score_breakdown(show)
            }
        print(f"Show {show_id} not found in success data")
        return None

    def analyze_network_patterns(self, similar_shows: List[SimilarShow]) -> NetworkAnalysis:
        """Analyze network patterns in similar shows.
        
        Args:
            similar_shows: List of similar shows to analyze
            
        Returns:
            NetworkAnalysis object with network stats
        """
        # Count shows by network
        network_counts = {}
        network_success_shows = {}  # For calculating success rates
        
        for show in similar_shows:
            network = show.network_name
            network_counts[network] = network_counts.get(network, 0) + 1
            
            # Track success scores for rate calculation
        
        # Calculate success metrics
        success_scores = {}
        success_rates = {}
        HIGH_SUCCESS_THRESHOLD = 70
        
        # Group shows by network
        network_shows = {}
        for show in similar_shows:
            network = show.network_name
            if network not in network_shows:
                network_shows[network] = []
            network_shows[network].append(show)
            
        for network, shows in network_shows.items():
            # Get overall network score from SuccessAnalyzer
            success_scores[network] = self.success_analyzer.calculate_network_success(network)
            
            # Debug success rate calculation
            st.write(f"\n{network} shows:")
            for show in shows:
                st.write(f"- {show.title}: score = {show.success_score}")
            
            # Calculate success rate based on stored success scores
            high_success = sum(1 for show in shows 
                             if show.success_score and show.success_score >= HIGH_SUCCESS_THRESHOLD)
            total_shows = len(shows)
            success_rates[network] = (high_success * 100.0) / total_shows
            
            st.write(f"High success shows: {high_success}, Total shows: {total_shows}")
            st.write(f"Success rate: {success_rates[network]}%\n")
        
        return NetworkAnalysis(
            similar_show_counts=network_counts,
            success_scores=success_scores,
            success_rates=success_rates
        )
