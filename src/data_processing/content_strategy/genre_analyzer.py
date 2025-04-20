"""Genre Analysis Module.

Analyzes genre patterns and trends in TV shows data, providing:
- Genre distribution across networks
- Network specialization insights
- Genre diversity metrics
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

from dashboard.templates.defaults.heatmap import create_heatmap_defaults
from dashboard.templates.grids.chart_insights import create_chart_insights_grid

logger = logging.getLogger(__name__)

def analyze_genre_patterns(shows_df: pd.DataFrame) -> Dict:
    """Analyze genre distribution patterns across networks.
    
    Args:
        shows_df: DataFrame containing show information
        
    Returns:
        Dict containing:
        - genre_stats: Basic genre statistics
        - network_patterns: Network specialization patterns
        - diversity_metrics: Genre diversity by network
    """
    # Basic counts
    total_shows = len(shows_df)
    genre_counts = shows_df['genre'].value_counts()
    
    # Genre distribution
    genre_distribution = genre_counts.to_dict()
    
    # Genre stats
    genre_stats = {
        'total_shows': total_shows,
        'unique_genres': len(genre_counts),
        'genre_distribution': genre_distribution,
        'top_genre': {
            'name': genre_counts.index[0],
            'count': int(genre_counts.iloc[0]),
            'share': round((genre_counts.iloc[0] / total_shows) * 100, 1)
        }
    }
    
    # Network analysis
    network_genre = pd.crosstab(
        shows_df['network'],
        shows_df['genre']
    )
    network_genre_pct = network_genre.div(network_genre.sum(axis=1), axis=0) * 100
    
    # Find network specializations
    network_patterns = {}
    for network in network_genre_pct.index:
        # Only analyze networks with at least 5 shows
        if network_genre.loc[network].sum() >= 5:
            genres = network_genre_pct.loc[network]
            primary_genre = genres.idxmax()
            primary_share = genres[primary_genre]
            
            # Consider a genre secondary if it has >15% share
            secondary = genres[genres > 15].index.tolist()
            if primary_genre in secondary:
                secondary.remove(primary_genre)
            
            network_patterns[network] = {
                'primary': (primary_genre, primary_share),
                'secondary': [(g, genres[g]) for g in secondary],
                'show_count': int(network_genre.loc[network].sum()),
                'genre_shares': genres.to_dict()  # Full genre distribution
            }
    
    # Calculate genre diversity
    genre_diversity = network_genre_pct.apply(
        lambda row: (row > 10).sum(),  # Count genres with >10% share
        axis=1
    )
    
    # Only consider networks with at least 10 shows for diversity metrics
    valid_mask = network_genre.sum(axis=1) >= 10
    filtered_diversity = genre_diversity[valid_mask]
    
    # Calculate market averages for each genre
    market_averages = network_genre_pct.mean()
    
    # Find unique patterns (15% above market average)
    unique_patterns = []
    for network in network_genre_pct.index:
        if network_genre.loc[network].sum() >= 3:  # Min 3 shows
            for genre in network_genre_pct.columns:
                share = network_genre_pct.loc[network, genre]
                market_share = market_averages[genre]
                if share >= market_share + 15:
                    unique_patterns.append({
                        'network': network,
                        'genre': genre,
                        'share': share,
                        'market_share': market_share,
                        'difference': share - market_share
                    })
    
    # Calculate diversity metrics
    diversity_metrics = {
        'most_diverse': {
            k: int(v) for k, v in filtered_diversity.nlargest(3).items()
        },
        'least_diverse': {
            k: int(v) for k, v in filtered_diversity.nsmallest(3).items()
        },
        'avg_genres_per_network': float(filtered_diversity.mean()),
        'genre_diversity_score': float(filtered_diversity.std()),
        'unique_patterns': sorted(unique_patterns, key=lambda x: x['difference'], reverse=True)
    }
    
    return {
        'genre_distribution': genre_distribution,  # Top level for backward compatibility
        'genre_stats': genre_stats,
        'network_patterns': network_patterns,
        'diversity_metrics': diversity_metrics,
        'network_genre_pct': network_genre_pct  # For visualization
    }

def create_genre_visualization(analysis_results: Dict) -> go.Figure:
    """Create genre distribution visualization.
    
    Args:
        analysis_results: Results from analyze_genre_patterns()
        
    Returns:
        Plotly figure with genre heatmap and insights
    """
    # Extract key metrics for insights
    genre_stats = analysis_results['genre_stats']
    network_patterns = analysis_results['network_patterns']
    diversity_metrics = analysis_results['diversity_metrics']
    network_genre_pct = analysis_results['network_genre_pct']
    
    # Sort networks by total show count for better visualization
    network_totals = network_genre_pct.sum(axis=1)
    network_genre_pct = network_genre_pct.reindex(network_totals.sort_values(ascending=False).index)
    
    # Create heatmap trace
    heatmap = go.Heatmap(
        z=network_genre_pct.values,
        x=network_genre_pct.columns,
        y=network_genre_pct.index,
        colorscale='Viridis',
        colorbar=dict(title='% of Network Shows'),
        hoverongaps=False
    )
    
    # Get most diverse network for insights
    most_diverse = list(diversity_metrics['most_diverse'].keys())[0]
    most_diverse_count = diversity_metrics['most_diverse'][most_diverse]
    
    # Get network with strongest specialization
    max_specialization = 0
    specialist = None
    specialty = None
    specialty_share = 0
    
    for network, data in network_patterns.items():
        primary_genre, primary_share = data['primary']
        if primary_share > max_specialization:
            max_specialization = primary_share
            specialist = network
            specialty = primary_genre
            specialty_share = primary_share
    
    # Create figure with insights
    fig = create_chart_insights_grid(
        traces=[heatmap],
        title='Genre Distribution by Network',
        subtitle=f'Analysis of {genre_stats["total_shows"]} shows across {genre_stats["unique_genres"]} genres',
        insights=[
            f"Genre Diversity: {most_diverse} leads with {most_diverse_count} major genres",
            f"Network Specialization: {specialist} focuses on {specialty} ({specialty_share:.1f}%)"
        ],
        template=create_heatmap_defaults()
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Network',
        xaxis={'tickangle': 45}
    )
    
    return fig
