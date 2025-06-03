from dataclasses import dataclass, field
from typing import Dict, Any
import pandas as pd
from abc import ABC, abstractmethod

from .optimizer_config import OptimizerConfig

__all__ = [
    'ComponentScore',
    'ScoreCalculationError',
    'ScoreCalculator',
    'SuccessScoreCalculator',
    'AudienceScoreCalculator',
    'CriticsScoreCalculator',
    'LongevityScoreCalculator'
]

@dataclass
class ComponentScore:
    """Success score for a component (audience, critics, longevity)."""
    component: str  # audience, critics, or longevity
    score: float  # 0-1 score
    sample_size: int
    confidence: str  # none, low, medium, high
    details: Dict[str, Any] = field(default_factory=dict)  # Detailed breakdown of score

class ScoreCalculationError(Exception):
    """Base exception for score calculation errors."""
    pass

class ScoreCalculator(ABC):
    """Abstract base class for score calculations."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
    
    @abstractmethod
    def calculate(self, shows: pd.DataFrame) -> 'ComponentScore': # Forward reference ComponentScore
        """Calculate the component score."""
        pass
    
    def _get_confidence(self, sample_size: int) -> str:
        """Get confidence level based on sample size."""
        return OptimizerConfig.get_confidence_level(sample_size)

class SuccessScoreCalculator(ScoreCalculator):
    """Calculate score based on success metrics."""
    
    def __init__(self):
        super().__init__(component_name='success')

    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError(f"Cannot calculate {self.component_name} score with empty shows DataFrame")
        
        if 'success_score' not in shows.columns:
            raise ScoreCalculationError(f"success_score column not found in shows data for {self.component_name} score")

        valid_shows = shows[shows['success_score'].notna()]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError(f"No shows with valid success_score data found for {self.component_name}")
        
        avg_score = valid_shows['success_score'].mean()
        confidence = self._get_confidence(sample_size)
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details={'success_score': avg_score}
        )

class AudienceScoreCalculator(ScoreCalculator):
    """Calculate audience score."""

    def __init__(self):
        super().__init__(component_name='audience')

    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError(f"Cannot calculate {self.component_name} score with empty shows DataFrame")
            
        if 'popcornmeter' not in shows.columns:
            raise ScoreCalculationError(f"popcornmeter column not found in shows data for {self.component_name} score")
            
        valid_shows = shows[shows['popcornmeter'].notna()]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError(f"No shows with valid popcornmeter data found for {self.component_name} score")
            
        avg_score = valid_shows['popcornmeter'].mean() / 100  # Normalize to 0-1
        confidence = self._get_confidence(sample_size)
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details={'popcornmeter': avg_score}
        )

class CriticsScoreCalculator(ScoreCalculator):
    """Calculate critics score using tomatometer."""

    def __init__(self):
        super().__init__(component_name='critics')

    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError(f"Cannot calculate {self.component_name} score with empty shows DataFrame")
            
        if 'tomatometer' not in shows.columns:
            raise ScoreCalculationError(f"tomatometer column not found in shows data for {self.component_name} score. Available columns: {list(shows.columns)}")
            
        valid_shows = shows[shows['tomatometer'].notna()]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError(f"No shows with valid tomatometer data found for {self.component_name} score")
            
        avg_score = valid_shows['tomatometer'].mean() / 100.0  # Normalize to 0-1
        confidence = self._get_confidence(sample_size)
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details={'tomatometer_avg_score': avg_score, 'sample_size': sample_size}
        )

class LongevityScoreCalculator(ScoreCalculator):
    """Calculate longevity score using TMDB metrics."""

    def __init__(self):
        super().__init__(component_name='longevity')

    def _calculate_season_score(self, seasons: float) -> float:
        if seasons >= 2:
            return min(100, 50 + (seasons - 2) * 10)
        return seasons * 25

    def _calculate_episode_score(self, episodes: float) -> float:
        if episodes >= 10:
            return 100
        elif episodes >= 5:
            return 50 + (episodes - 5) * 10
        return episodes * 10

    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError(f"Cannot calculate {self.component_name} score with empty shows DataFrame")
        
        required_columns = ['tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status']
        missing_columns = [col for col in required_columns if col not in shows.columns]
        if missing_columns:
            raise ScoreCalculationError(
                f"Missing required columns for {self.component_name} score: {', '.join(missing_columns)}. "
                f"Available columns: {list(shows.columns)}"
            )
        
        valid_shows = shows.dropna(subset=required_columns).copy() # Use .copy() to avoid SettingWithCopyWarning
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError(f"No shows with valid TMDB metrics found for {self.component_name} score")
        
        status_scores = {
            'Returning Series': 100, 'Ended': 75, 'Canceled': 25,
            'In Production': 50, 'Pilot': 10, 'In Development': 5
        }
        
        valid_shows['season_score_val'] = valid_shows['tmdb_seasons'].apply(self._calculate_season_score)
        valid_shows['episode_score_val'] = valid_shows['tmdb_total_episodes'].apply(self._calculate_episode_score)
        valid_shows['status_score_val'] = valid_shows['tmdb_status'].map(status_scores).fillna(0)
        
        valid_shows['longevity_score_calc'] = (
            valid_shows['season_score_val'] * 0.4 +
            valid_shows['episode_score_val'] * 0.4 +
            valid_shows['status_score_val'] * 0.2
        ) / 100  # Convert to 0-1 scale
        
        avg_score = valid_shows['longevity_score_calc'].mean()
        confidence = self._get_confidence(sample_size)
        
        details = {
            'avg_seasons': valid_shows['tmdb_seasons'].mean(),
            'avg_episodes': valid_shows['tmdb_total_episodes'].mean(),
            'status_distribution': valid_shows['tmdb_status'].value_counts(normalize=True).mul(100).round(1).to_dict(),
            'sample_size': sample_size
        }
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
