"""Market Analysis Module.

This module provides market overview analytics including network distribution and key metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Analyzer for market overview and network patterns."""
    
    def __init__(self, shows_df: pd.DataFrame, team_df: pd.DataFrame):
        """Initialize the analyzer.
        
        Args:
            shows_df: DataFrame containing show information
            team_df: DataFrame containing team member information
        """
        self.shows_df = shows_df
        self.team_df = team_df
        
        # Log basic stats
        # Calculate market overview metrics
        total_shows = len(shows_df)
        total_networks = len(shows_df['network'].unique())
        total_creatives = len(team_df['name'].unique())
    
    def create_network_chart(self) -> go.Figure:
        """Create a bar chart showing the distribution of shows across networks.
        
        Returns:
            Plotly figure object
        """
        # Get network distribution
        shows_by_network = self.shows_df.groupby('network').size().sort_values(ascending=False)
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(shows_by_network.index),
            y=list(shows_by_network.values),
            name="Shows per Network"
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title="Network",
            yaxis_title="Number of Shows",
            font_family="Source Sans Pro",
            showlegend=False,
            margin=dict(t=20)
        )
        
        return fig
    
    def generate_market_insights(self) -> Dict:
        """Generate key insights about market patterns.
        
        Returns:
            Dictionary containing various market insights including:
            - total_shows: Total number of shows
            - total_networks: Number of unique networks
            - total_creatives: Number of unique creatives
            - total_roles: Number of unique role types
            - network_concentration: Share of shows from top 3 networks
            - top_genre_network: Network with most shows in top genre
            - top_genre: Most common genre
            - top_genre_count: Number of shows in top genre
        """
        # Basic counts
        total_shows = len(self.shows_df)
        total_networks = len(self.shows_df['network'].unique())
        total_creatives = len(self.team_df['name'].unique())
        
        # Get unique roles and normalize them
        roles = self.team_df['roles'].str.split(',').explode().str.strip()
        total_roles = len(roles.unique())
        
        # Network concentration
        shows_by_network = self.shows_df.groupby('network').size().sort_values(ascending=False)
        top_3_networks = shows_by_network.head(3)
        network_concentration = (top_3_networks.sum() / total_shows) * 100
        
        # Genre leadership
        genre_by_network = pd.crosstab(
            self.shows_df['genre'],
            self.shows_df['network']
        )
        top_genre = self.shows_df['genre'].mode()[0]
        top_genre_count = genre_by_network.loc[top_genre].max()
        top_genre_network = genre_by_network.loc[top_genre].idxmax()
        
        return {
            'total_shows': total_shows,
            'total_networks': total_networks,
            'total_creatives': total_creatives,
            'total_roles': total_roles,
            'network_concentration': network_concentration,
            'top_3_networks': list(top_3_networks.index),
            'top_genre_network': top_genre_network,
            'top_genre': top_genre,
            'top_genre_count': top_genre_count
        }
