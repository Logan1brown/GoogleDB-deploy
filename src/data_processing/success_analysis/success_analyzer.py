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
    """Configuration for success calculations.
    
    Base metrics (70% total):
    - Seasons (30%)
    - Episodes (30%)
    - Status (10%)
    
    RT metrics (30% total):
    - Tomatometer (15%)
    - Popcornmeter (15%)
    
    When RT data is missing, base metrics are scaled up proportionally.
    """
    # Base metric weights
    SEASON_WEIGHT: float = 0.30
    EPISODE_WEIGHT: float = 0.30
    STATUS_WEIGHT: float = 0.10
    
    # RT metric weights
    TOMATOMETER_WEIGHT: float = 0.15
    POPCORNMETER_WEIGHT: float = 0.15    
    # Season scoring
    SEASON1_VALUE: int = 50   # Base points for first season
    SEASON2_VALUE: int = 50   # Bonus for renewal (S2)
    ADDITIONAL_SEASON_VALUE: int = 25  # Each season after S2 (max 100)
    
    # Episode scoring
    EPISODE_BASE_POINTS: int = 50     # Points for reaching min threshold
    EPISODE_BONUS_POINTS: int = 50    # Additional points for high threshold
    EPISODE_MIN_THRESHOLD: int = 8    # Minimum episodes needed
    EPISODE_BONUS_THRESHOLD: int = 10  # Episodes for bonus
    
    # Status points
    STATUS_POINTS: Dict[str, int] = None
    
    def __post_init__(self):
        self.STATUS_POINTS = {
            'Returning Series': 100,  # Full points for active shows
            'Ended': 50,            # Half points for completed shows
            'Canceled': 0,          # No points for canceled shows
            'In Production': 0,     # No score for shows in production
            'Pilot': 0,            # No score for pilots
            'In Development': 0,    # No score for shows in development
        }


class SuccessAnalyzer:
    """
    Analyzes show success based on reliable metrics.
    Only calculates scores for shows with reliable data (Returning, Ended, Canceled).
    
    Data is fetched from ShowsAnalyzer.
    """
    def __init__(self, shows_analyzer):
        """Initialize the analyzer.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for fetching data
        """
        self.shows_analyzer = shows_analyzer
        self.config = SuccessConfig()
        self._success_data = None
    
    @property
    def success_data(self) -> pd.DataFrame:
        """Cached success metrics data."""
        if self._success_data is None:
            self._success_data = self.fetch_success_data()
        return self._success_data

    def fetch_success_data(self) -> pd.DataFrame:
        """Fetch success metrics from ShowsAnalyzer.
        
        Returns:
            DataFrame with success metrics and calculated success scores
        """
        if self._success_data is None:
            try:
                # Use ShowsAnalyzer to fetch success metrics
                success_df = self.shows_analyzer.fetch_success_metrics()
                
                # Calculate success score for each show
                success_df['success_score'] = success_df.apply(self.calculate_success, axis=1)
                
                # Set show_id as index for easier lookups
                success_df.set_index('show_id', inplace=True)
                
                self._success_data = success_df
            except Exception as e:
                raise
                
        return self._success_data

    def calculate_network_success(self, network: str) -> float:
        """Calculate success score for a specific network.
        
        Args:
            network: Name of the network
            
        Returns:
            Success score as percentage (0-100)
        """
        # Get network shows from cached success data
        network_titles = self.success_data[
            self.success_data['network_name'].str.lower() == network.lower()
        ]
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
            df = self.success_data
            
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
        # Get network shows from cached success data
        network_titles = self.success_data[
            self.success_data['network_name'] == network
        ]
        if len(network_titles) == 0:
            return 90.0  # Default if no data
            
        # Count shows that got renewed (2+ seasons)
        renewed = network_titles[network_titles['tmdb_seasons'] >= 2]
        return (len(renewed) / len(network_titles)) * 100
        
    def analyze_market(self, titles_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate success metrics for all reliable shows in the market.
        Returns thresholds and tiers based on the highest scores.
        
        Args:
            titles_df: Optional DataFrame to analyze. If None, uses cached success data.
        """
        df = titles_df if titles_df is not None else self.success_data
        
        # Filter to reliable shows only
        reliable_titles = df[df['tmdb_status'].isin(ShowStatus.RELIABLE)]
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
            scores.append({
                'title_id': str(title['show_id']),
                'name': title['title'],
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
        """Calculate success score for a single show.
        
        Combines base metrics (seasons, episodes, status) with RT metrics.
        When RT data is missing, base metrics are scaled up proportionally.
        
        Args:
            show: Show data from api_success_metrics view
            
        Returns:
            Success score (0-100)
        """
        if show.get('tmdb_status') not in ShowStatus.RELIABLE:
            return 0
            
        # Calculate base components
        season_score = self._calculate_season_score(show)
        episode_score = self._calculate_episode_score(show)
        status_score = self._calculate_status_score(show)
        
        # Calculate RT components if available
        has_rt = show.get('has_rt', False)
        if has_rt:
            rt_score = self._calculate_rt_score(show)
            
            # Combine all components with their weights
            final_score = (
                season_score * self.config.SEASON_WEIGHT +
                episode_score * self.config.EPISODE_WEIGHT +
                status_score * self.config.STATUS_WEIGHT +
                rt_score * (self.config.TOMATOMETER_WEIGHT + self.config.POPCORNMETER_WEIGHT)
            )
        else:
            # Scale up base components proportionally when RT is missing
            scale = 1 / (self.config.SEASON_WEIGHT + self.config.EPISODE_WEIGHT + self.config.STATUS_WEIGHT)
            final_score = (
                season_score * (self.config.SEASON_WEIGHT * scale) +
                episode_score * (self.config.EPISODE_WEIGHT * scale) +
                status_score * (self.config.STATUS_WEIGHT * scale)
            )
            
        return min(100, max(0, final_score))  # Cap at 100, don't allow negative
        
    def _calculate_season_score(self, show: pd.Series) -> float:
        """Calculate score component from season count.
        
        Points breakdown:
        - First season: 50 points (base)
        - Second season: +50 points (renewal bonus)
        - Additional seasons: +25 points each (max 100)
        
        Max total: 200 points (scales to 60 with 30% weight)
        """
        seasons = show.get('tmdb_seasons')
        if not pd.notna(seasons):
            return 0
            
        score = self.config.SEASON1_VALUE  # Base points for first season
        
        if seasons >= 2:
            score += self.config.SEASON2_VALUE  # Renewal bonus
            extra_seasons = seasons - 2
            if extra_seasons > 0:
                extra_points = min(extra_seasons * self.config.ADDITIONAL_SEASON_VALUE, 100)  # Max 100 bonus points
                score += extra_points
        return score
        
    def _calculate_episode_score(self, show: pd.Series) -> float:
        """Calculate score component from episode count."""
        avg_eps = show.get('tmdb_avg_eps')
        if not pd.notna(avg_eps):
            return 0
            
        try:
            avg_eps = float(avg_eps)
            if avg_eps >= self.config.EPISODE_BONUS_THRESHOLD:
                return self.config.EPISODE_BASE_POINTS + self.config.EPISODE_BONUS_POINTS
            elif avg_eps >= self.config.EPISODE_MIN_THRESHOLD:
                return self.config.EPISODE_BASE_POINTS
        except (ValueError, TypeError):
            pass
            
        return 0
        
    def _calculate_status_score(self, show: pd.Series) -> float:
        """Calculate score component from show status."""
        status = show.get('tmdb_status')
        return self.config.STATUS_POINTS.get(status, 0)  # Get points based on status
        
    def _calculate_rt_score(self, show: pd.Series) -> float:
        """Calculate score component from RT metrics."""
        tomatometer = show.get('tomatometer')
        popcornmeter = show.get('popcornmeter')
        
        # Both metrics contribute equally to the RT component
        rt_score = 0
        if pd.notna(tomatometer):
            rt_score += tomatometer
        if pd.notna(popcornmeter):
            rt_score += popcornmeter
            
        # Average the available scores
        num_scores = (pd.notna(tomatometer) + pd.notna(popcornmeter))
        return rt_score / num_scores if num_scores > 0 else 0
        
    def _get_tier(self, score: float, max_score: float) -> str:
        """Get success tier based on score relative to max."""
        if score >= max_score * 0.8:
            return 'high'
        elif score >= max_score * 0.5:
            return 'medium'
        return 'low'
        
    def get_score_breakdown(self, show: pd.Series) -> Dict[str, float]:
        """Get detailed breakdown of success score components.
        
        Args:
            show: Show data from api_success_metrics view
            
        Returns:
            Dict with score components and their values
        """
        if show.get('tmdb_status') not in ShowStatus.RELIABLE:
            return {}
            
        # Calculate each component
        season_score = self._calculate_season_score(show)
        episode_score = self._calculate_episode_score(show)
        status_score = self._calculate_status_score(show)
        
        # Get RT score if available
        has_rt = show.get('has_rt', False)
        rt_score = self._calculate_rt_score(show) if has_rt else None
        
        # Build breakdown
        breakdown = {}
        
        # Season breakdown
        seasons = show.get('tmdb_seasons')
        if pd.notna(seasons):
            # Always add base points for first season
            breakdown['season1_base'] = self.config.SEASON1_VALUE
            if seasons >= 2:
                breakdown['season2_renewal'] = self.config.SEASON2_VALUE
                extra_seasons = seasons - 2
                if extra_seasons > 0:
                    breakdown['additional_seasons'] = min(extra_seasons * self.config.ADDITIONAL_SEASON_VALUE, 100)
                
        # Episode breakdown
        avg_eps = show.get('tmdb_avg_eps')
        if pd.notna(avg_eps):
            try:
                avg_eps = float(avg_eps)
                if avg_eps >= self.config.EPISODE_MIN_THRESHOLD:
                    breakdown['episode_base'] = self.config.EPISODE_BASE_POINTS
                if avg_eps >= self.config.EPISODE_BONUS_THRESHOLD:
                    breakdown['episode_bonus'] = self.config.EPISODE_BONUS_POINTS
            except (ValueError, TypeError):
                pass
                
        # Status points
        status = show.get('tmdb_status')
        status_points = self.config.STATUS_POINTS.get(status, 0)
        if status_points > 0:
            breakdown['status_points'] = status_points
            
        # RT scores
        if has_rt:
            tomatometer = show.get('tomatometer')
            if pd.notna(tomatometer):
                breakdown['tomatometer'] = tomatometer
            popcornmeter = show.get('popcornmeter')
            if pd.notna(popcornmeter):
                breakdown['popcornmeter'] = popcornmeter
                
        return breakdown

    def format_breakdown_for_display(self, breakdown: Dict[str, float]) -> List[str]:
        """Format success score breakdown for display.
        
        Args:
            breakdown: Score breakdown from get_score_breakdown
            
        Returns:
            List of formatted strings ready for display
        """
        display_items = []
        
        # Calculate base points (70% total)
        season_points = 0
        if 'season1_base' in breakdown:
            season_points += breakdown['season1_base']
        if 'season2_renewal' in breakdown:
            season_points += breakdown['season2_renewal']
        if 'additional_seasons' in breakdown:
            season_points += breakdown['additional_seasons']
        season_contribution = season_points * self.config.SEASON_WEIGHT
        display_items.append(f"**Seasons:** {season_points:.0f} points → **{season_contribution:.1f} pts** _(30% weight)_")
        
        episode_points = 0
        if 'episode_base' in breakdown:
            episode_points += breakdown['episode_base']
        if 'episode_bonus' in breakdown:
            episode_points += breakdown['episode_bonus']
        episode_contribution = episode_points * self.config.EPISODE_WEIGHT
        display_items.append(f"**Episodes:** {episode_points:.0f} points → **{episode_contribution:.1f} pts** _(30% weight)_")
        
        # Status points
        if 'status_points' in breakdown:
            status_points = breakdown['status_points']
            status_contribution = status_points * self.config.STATUS_WEIGHT
            if status_points == 100:
                display_items.append(f"**Active Series Bonus:** {status_points:.0f} points → **{status_contribution:.1f} pts** _(10% weight)_")
            elif status_points == 50:
                display_items.append(f"**Completed Series:** {status_points:.0f} points → **{status_contribution:.1f} pts** _(10% weight)_")
        
        # RT scores (30% total)
        if 'tomatometer' in breakdown:
            tomatometer = breakdown['tomatometer']
            tomatometer_contribution = tomatometer * self.config.TOMATOMETER_WEIGHT
            display_items.append(f"**Tomatometer:** {tomatometer:.0f}/100 → **{tomatometer_contribution:.1f} pts** _(15% weight)_")
            
        if 'popcornmeter' in breakdown:
            popcornmeter = breakdown['popcornmeter']
            popcornmeter_contribution = popcornmeter * self.config.POPCORNMETER_WEIGHT
            display_items.append(f"**Popcornmeter:** {popcornmeter:.0f}/100 → **{popcornmeter_contribution:.1f} pts** _(15% weight)_")
            
        return display_items
