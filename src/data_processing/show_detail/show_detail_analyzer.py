"""Show Detail Analyzer

Analyzes shows to find similar content and network patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.success_analysis.success_analyzer import SuccessAnalyzer
from src.config.supabase_client import get_client


@dataclass
class SimilarShow:
    show_id: int
    title: str
    network_name: str
    success_score: Optional[float]
    match_score: Dict[str, int] = field(default_factory=lambda: {
        'genre_score': 0,
        'team_score': 0,
        'source_score': 0,
        'date_score': 0,
        'total': 0
    })


@dataclass
class NetworkAnalysis:
    similar_show_counts: Dict[str, int]  # network -> count
    success_rates: Dict[str, float]      # network -> success rate


class ShowDetailAnalyzer:
    """Analyzer for finding similar shows and network patterns."""
    
    def __init__(self, success_analyzer: Optional[SuccessAnalyzer] = None):
        """Initialize the analyzer.
        
        Args:
            success_analyzer: Optional SuccessAnalyzer instance
        """
        self.success_analyzer = success_analyzer or SuccessAnalyzer()
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
            # Get show data from ShowsAnalyzer
            shows_df = ShowsAnalyzer().fetch_show_data(force=force)
            
            # Calculate average episodes per season
            shows_df['tmdb_avg_eps'] = pd.to_numeric(shows_df['tmdb_total_episodes'], errors='coerce') / pd.to_numeric(shows_df['tmdb_seasons'], errors='coerce')
            
            # Calculate success scores
            success_analyzer = SuccessAnalyzer()
            success_analyzer.initialize_data(shows_df)
            
            # Add success scores to DataFrame
            shows_df['success_score'] = shows_df.apply(success_analyzer.calculate_success, axis=1)
            shows_df['success_score'] = shows_df['success_score'].fillna(0)
            
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
            'subgenres': show.get('subgenres', []),
            'source_name': show['source_name'],
            'team_members': show.get('team_members', []),
            'success_score': show.get('success_score'),
            'tmdb_status': show.get('tmdb_status'),
            'episode_count': show.get('episode_count'),
            'tmdb_seasons': show.get('tmdb_seasons'),
            'tmdb_total_episodes': show.get('tmdb_total_episodes')
        }
    
    def compute_similarity(self, show1: pd.Series, show2: pd.Series) -> Dict:
        """Compute similarity score between two shows.
        
        Args:
            show1: First show data
            show2: Second show data
            
        Returns:
            Dictionary with score components and total
        """
        # Genre match (40 points total)
        genre_score = 35 if show1['genre_name'] == show2['genre_name'] else 0
        # 5 points if they share any subgenre
        shared_subgenres = bool(set(show1.get('subgenres', []) or []) & set(show2.get('subgenres', []) or []))
        genre_score += 5 if shared_subgenres else 0
        
        # Team overlap (30 points)
        team1 = {member['name'] for member in show1.get('team_members', []) or []}
        team2 = {member['name'] for member in show2.get('team_members', []) or []}
        team_overlap = len(team1 & team2)
        # Each team member overlap worth 10 points, max 30
        team_score = min(team_overlap * 10, 30)
        
        # Source match (20 points)
        source_score = 20 if show1['source_name'] == show2['source_name'] else 0
        
        # Date proximity (10 points)
        date1 = pd.to_datetime(show1['announced_date']).year if pd.notna(show1.get('announced_date')) else None
        date2 = pd.to_datetime(show2['announced_date']).year if pd.notna(show2.get('announced_date')) else None
        years_apart = abs(date1 - date2) if date1 and date2 else 5
        # Lose 2 points per year apart, minimum 0
        date_score = max(10 - (years_apart * 2), 0)
        
        return {
            'total': genre_score + team_score + source_score + date_score,
            'genre_score': genre_score,
            'genre_details': {
                'main_genre_match': show1['genre_name'] == show2['genre_name'],
                'has_subgenre_match': shared_subgenres
            },
            'team_score': team_score,
            'source_score': source_score,
            'date_score': date_score,
            'details': {
                'team_overlap': team_overlap,
                'years_apart': years_apart
            }
        }
    
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
        target_show = self._shows_df[self._shows_df['show_id'] == show_id].iloc[0]
        
        # Calculate similarity scores for all other shows
        similar_shows = []
        for _, show in self._shows_df[self._shows_df['show_id'] != show_id].iterrows():
            scores = self.compute_similarity(target_show, show)
            if scores['total'] >= min_score:
                similar_shows.append(SimilarShow(
                    show_id=show['show_id'],
                    title=show['title'],
                    network_name=show['network_name'],
                    success_score=show.get('success_score'),
                    match_score=scores
                ))
        
        # Sort by total score (descending) then title
        similar_shows.sort(key=lambda x: (-x.match_score['total'], x.title))
        return similar_shows[:max_shows]
    
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
            if show.success_score is not None:
                if network not in network_success_shows:
                    network_success_shows[network] = []
                network_success_shows[network].append(show.success_score)
        
        # Calculate success rates
        success_rates = {}
        for network, scores in network_success_shows.items():
            success_rates[network] = sum(scores) / len(scores) if scores else 0
        
        return NetworkAnalysis(
            similar_show_counts=network_counts,
            success_rates=success_rates
        )
