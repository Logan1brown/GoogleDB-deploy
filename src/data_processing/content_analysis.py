"""Content and Creative Analysis Module.

This module provides specialized analysis of TV show content patterns,
creative relationships, and network preferences.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Analyzer for content patterns and creative relationships."""
    
    def __init__(self, shows_df: pd.DataFrame, team_df: pd.DataFrame):
        """Initialize the analyzer.
        
        Args:
            shows_df: DataFrame containing show information
            team_df: DataFrame containing team member information
        """
        self.shows_df = shows_df
        self.team_df = team_df
        
    def analyze_genre_creative_patterns(self) -> Dict:
        """Analyze relationships between genres and creative teams.
        
        Returns:
            Dictionary containing:
            - genre_creator_matrix: DataFrame of creator success in each genre
            - top_creators_by_genre: Dict mapping genres to their most successful creators
            - genre_combinations: Common genre combinations that lead to sales
        """
        # Merge shows and team data
        combined_df = pd.merge(
            self.team_df,
            self.shows_df[['title', 'genre', 'network_name', 'source_type']],
            on='title'
        )
        
        # Create genre-creator matrix
        genre_creator = pd.crosstab(
            combined_df['name'],
            combined_df['genre']
        )
        
        # Find top creators for each genre
        top_creators = {}
        for genre in genre_creator.columns:
            top_in_genre = genre_creator[genre].nlargest(5)
            top_creators[genre] = top_in_genre.to_dict()
            
        return {
            'genre_creator_matrix': genre_creator,
            'top_creators_by_genre': top_creators
        }
    
    def analyze_network_preferences(self) -> Dict:
        """Analyze network preferences for genres and creative teams.
        
        Returns:
            Dictionary containing:
            - network_genre_prefs: Genre preferences by network
            - network_creator_prefs: Creator preferences by network
            - source_type_patterns: Source type patterns by network
        """
        # Genre preferences
        network_genre = pd.crosstab(
            self.shows_df['network_name'],
            self.shows_df['genre']
        )
        
        # Source type patterns
        source_patterns = pd.crosstab(
            self.shows_df['network_name'],
            self.shows_df['source_type_name']
        )
        
        # Creator preferences (using merged data)
        combined_df = pd.merge(
            self.team_df,
            self.shows_df[['title', 'network_name']],
            on='title'
        )
        creator_network = pd.crosstab(
            combined_df['name'],
            combined_df['network']
        )
        
        # Find repeat collaborators
        repeat_creators = {}
        for network in creator_network.columns:
            network_creators = creator_network[network]
            repeats = network_creators[network_creators > 1]
            if not repeats.empty:
                repeat_creators[network] = repeats.to_dict()
        
        return {
            'network_genre_prefs': network_genre,
            'source_type_patterns': source_patterns,
            'repeat_collaborators': repeat_creators
        }
    
    def visualize_genre_patterns(self) -> None:
        """Create visualizations for genre-related patterns."""
        # Genre distribution heatmap
        genre_net = pd.crosstab(
            self.shows_df['genre'],
            self.shows_df['network']
        )
        
        fig = px.imshow(
            genre_net,
            title='Genre Distribution by Network',
            labels=dict(x='Network', y='Genre', color='Number of Shows'),
            aspect='auto'
        )
        
        # Save to HTML file
        output_dir = Path('output/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_dir / 'genre_network_heatmap.html')
        
    def visualize_creative_patterns(self) -> None:
        """Create visualizations for creative team patterns."""
        # Merge data for analysis
        combined_df = pd.merge(
            self.team_df,
            self.shows_df[['title', 'genre', 'network_name']],
            on='title'
        )
        
        # Create creator success visualization
        creator_success = combined_df.groupby('name').size()
        top_creators = creator_success.nlargest(20)
        
        fig = px.bar(
            x=top_creators.index,
            y=top_creators.values,
            title='Most Active Creators',
            labels={'x': 'Creator', 'y': 'Number of Shows'}
        )
        
        # Save to HTML file
        output_dir = Path('output/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_dir / 'top_creators.html')
