"""
Success Analyzer Component.
Calculates success metrics for shows based on reliable data.

=== STANDARDIZED COLUMN NAMES ===
All views now use standardized column names:
- 'title' for show names
- 'network_name' for networks
- 'studio_names' for studios
- 'status_name' for status
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..external.tmdb.tmdb_models import ShowStatus


@dataclass
class SuccessConfig:
    """Configuration for success calculations."""
    # Season achievements (40%)
    SEASON2_VALUE: int = 40  # Show renewed for S2
    ADDITIONAL_SEASON_VALUE: int = 20  # Each season after S2
    
    # Episode volume scoring (40% of total)
    EPISODE_BASE_POINTS: int = 20     # Points awarded for reaching min threshold
    EPISODE_BONUS_POINTS: int = 20    # Additional points for reaching high threshold
    EPISODE_MIN_THRESHOLD: int = 8    # Minimum episodes needed for base points
    EPISODE_BONUS_THRESHOLD: int = 10  # Episodes needed for bonus points
    
    # Status modifiers
    STATUS_MODIFIERS: Dict[str, float] = None
    
    def __post_init__(self):
        self.STATUS_MODIFIERS = {
            'Returning Series': 1.2,  # 20% bonus for active shows
            'Ended': 1.0,            # Base multiplier for completed shows
            'Canceled': 0.8,         # 20% penalty for canceled shows
            'In Production': 0.0,    # No score for shows in production
            'Pilot': 0.0,           # No score for pilots
            'In Development': 0.0,   # No score for shows in development
        }


class SuccessAnalyzer:
    """
    Analyzes show success based on reliable metrics.
    Only calculates scores for shows with reliable data (Returning, Ended, Canceled).
    """
    def __init__(self, config: Optional[SuccessConfig] = None):
        self.config = config or SuccessConfig()
        self.titles_df = None
        
    def initialize_data(self, titles_df: pd.DataFrame):
        """Initialize analyzer with show data."""
        self.titles_df = titles_df.copy()
        
    def calculate_network_success(self, network: str) -> float:
        """Calculate success score for a specific network.
        
        Args:
            network: Name of the network
            
        Returns:
            Success score as percentage (0-100)
        """
        if self.titles_df is None:
            return 0  # No data available
            
        network_titles = self.titles_df[self.titles_df['network_name'].str.lower() == network.lower()]
        if len(network_titles) == 0:
            return 0  # Network not found
            
        # Calculate success for each show
        success_scores = []
        for _, title in network_titles.iterrows():
            score = self.calculate_success(title)
            if score > 0:  # Only include shows that have a valid score
                success_scores.append(score)
            
        return np.mean(success_scores) if success_scores else 0
        
    def calculate_overall_success(self, df: Optional[pd.DataFrame] = None) -> float:
        """Calculate overall success score for a set of shows.
        
        Args:
            df: Optional DataFrame to calculate success for. If None, uses all shows.
            
        Returns:
            Average success score (0-100)
        """
        if df is None:
            df = self.titles_df
            
        # Calculate success for each show
        success_metrics = self.analyze_market(df)
        
        # Get average success score
        total_score = 0
        num_shows = 0
        for show_id, show_data in success_metrics['titles'].items():
            if 'score' in show_data:
                total_score += show_data['score']
                num_shows += 1
                
        avg_score = total_score / num_shows if num_shows > 0 else 0
        return avg_score

    def calculate_renewal_rate(self, network: str) -> float:
        """Calculate renewal rate for a specific network.
        
        Args:
            network: Name of the network
            
        Returns:
            Renewal rate as percentage (0-100)
        """
        if self.titles_df is None:
            return 90.0  # Default if no data
            
        network_titles = self.titles_df[self.titles_df['network_name'] == network]
        if len(network_titles) == 0:
            return 90.0
            
        # Count shows that got renewed (2+ seasons)
        network_titles['seasons'] = pd.to_numeric(network_titles['tmdb_seasons'], errors='coerce')
        renewed = network_titles[network_titles['seasons'] >= 2]
        return (len(renewed) / len(network_titles)) * 100
        
    def analyze_market(self, titles_df: pd.DataFrame) -> Dict:
        """
        Calculate success metrics for all reliable shows in the market.
        Returns thresholds and tiers based on the highest scores.
        """
        # Filter to reliable shows only
        reliable_titles = titles_df[titles_df['tmdb_status'].isin(ShowStatus.RELIABLE)]
        if reliable_titles.empty:
            return {
                'max_score': 0,
                'high_threshold': 0,
                'medium_threshold': 0,
                'titles': {}
            }
            
        # Calculate scores for all reliable shows
        scores = []
        for _, title in reliable_titles.iterrows():
            score = self.calculate_success(title)
            # === CRITICAL: Column Name Difference ===
            # We're working with the titles sheet here, which uses 'title' column
            # Do NOT use 'show_name' which is only for the show_team sheet
            # Convert tmdb_id to string for dictionary keys
            scores.append({
                'title_id': str(title['tmdb_id']),
                'name': title['title'],  # Title column from standardized views
                'score': score
            })
            
        # Get max score and set tier thresholds
        max_score = max(s['score'] for s in scores) if scores else 0
        return {
            'max_score': max_score,
            'high_threshold': max_score * 0.8,  # High: Top 20%
            'medium_threshold': max_score * 0.5,  # Medium: Top 50%
            'titles': {
                s['title_id']: {
                    'name': s['name'],
                    'score': s['score'],
                    'tier': self._get_tier(s['score'], max_score)
                }
                for s in scores
            }
        }
        
    def calculate_success(self, show: pd.Series) -> float:
        """Calculate success score for a single show."""
        # Shows must have reliable status to get a score
        if show['tmdb_status'] not in ShowStatus.RELIABLE:
            return 0
            
        score = 0
        
        # Season achievements
        if pd.notna(show['tmdb_seasons']):
            seasons = int(show['tmdb_seasons'])
            if seasons >= 2:
                score += self.config.SEASON2_VALUE
                extra_seasons = seasons - 2
                if extra_seasons > 0:
                    score += min(extra_seasons * self.config.ADDITIONAL_SEASON_VALUE, 40)  # Cap extra seasons bonus at 40 points
                
        # Episode volume points (40% of total possible)
        if pd.notna(show['tmdb_avg_eps']):
            try:
                avg_eps = float(show['tmdb_avg_eps'])
                if avg_eps >= self.config.EPISODE_BONUS_THRESHOLD:
                    # High volume show (10+ episodes)
                    score += self.config.EPISODE_BASE_POINTS + self.config.EPISODE_BONUS_POINTS
                elif avg_eps >= self.config.EPISODE_MIN_THRESHOLD:
                    # Standard volume show (8-9 episodes)
                    score += self.config.EPISODE_BASE_POINTS
                # No points if below minimum
            except (ValueError, TypeError):
                # Skip invalid episode count
                pass
            
        # Apply status modifier
        modifier = self.config.STATUS_MODIFIERS.get(show['tmdb_status'], 1.0)
        score *= modifier
        
        return min(100, max(0, score))  # Cap at 100, don't allow negative
        
    def _get_tier(self, score: float, max_score: float) -> str:
        """Get success tier based on score relative to max."""
        if score >= max_score * 0.8:
            return 'high'
        elif score >= max_score * 0.5:
            return 'medium'
        return 'low'
