"""Source Type Analysis Module.

This module analyzes patterns in source types (original vs adaptations) across networks.
It provides metrics and insights for the dashboard's source analysis component.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def analyze_source_patterns(shows_df: pd.DataFrame) -> Dict:
    """Analyze source type patterns across networks.
    
    Args:
        shows_df: DataFrame containing show information with 'network' and 'source_type' columns
        
    Returns:
        Dictionary containing:
        - source_counts: Overall counts by source type
        - network_source_matrix: Source type distribution by network
        - network_source_pct: Percentage distribution by network
        - source_preferences: Networks with strong preferences for each source
        - diversity_scores: Source diversity score by network
    """
    logger.info("Analyzing source type patterns")
    
    # Get overall source type distribution and key metrics
    source_counts = shows_df['source_type_name'].value_counts()
    total_shows = len(shows_df)
    
    # Calculate top source metrics
    top_source = source_counts.index[0]
    top_source_count = source_counts[top_source]
    top_source_share = (top_source_count / total_shows) * 100
    
    # Calculate original content metrics
    original_count = source_counts.get('Original', 0)
    original_share = (original_count / total_shows) * 100
    
    # Network source type focus
    network_source = pd.crosstab(
        shows_df['network'],
        shows_df['source_type_name']
    )
    
    # Calculate percentages
    network_source_pct = network_source.div(network_source.sum(axis=1), axis=0) * 100
    
    # Filter out networks with too few shows
    MIN_SHOWS = 3
    valid_networks = network_source.sum(axis=1) >= MIN_SHOWS
    filtered_data = network_source_pct[valid_networks]
    filtered_counts = network_source[valid_networks]
    
    # Calculate source diversity (using Shannon entropy)
    def shannon_diversity(row: pd.Series) -> float:
        """Calculate Shannon diversity index for a row of percentages."""
        props = row[row > 0] / 100  # Convert percentages to proportions
        return -(props * np.log(props)).sum()
    
    diversity_scores = filtered_data.apply(shannon_diversity, axis=1)
    
    # Find networks with strong preferences for each source type
    source_preferences = {}
    for source_type in filtered_data.columns:
        # Get networks with highest percentage for this source
        source_data = filtered_data[source_type].sort_values(ascending=False)
        source_counts = filtered_counts[source_type]
        
        # Only include if percentage > 30% and at least 2 shows
        significant = source_data[(source_data > 30) & (source_counts >= 2)]
        
        if not significant.empty:
            source_preferences[source_type] = [
                {
                    'network': network,
                    'percentage': pct,
                    'count': filtered_counts.loc[network, source_type],
                    'total_shows': filtered_counts.loc[network].sum()
                }
                for network, pct in significant.items()
            ]
    
    # Sort source types by total volume for heatmap
    source_volumes = network_source.sum().sort_values(ascending=False)
    
    # Generate key insights
    insights = {
        # Overall distribution (use unfiltered data)
        'source_counts': network_source.sum().to_dict(),
        'total_shows': total_shows,
        
        # Top source metrics
        'top_source': {
            'type': top_source,
            'count': int(top_source_count),
            'share': round(top_source_share, 1)
        },
        
        # Original content metrics
        'original_content': {
            'count': int(original_count),
            'share': round(original_share, 1)
        },
        
        # Network analysis
        'network_source_matrix': network_source.to_dict(),
        'network_source_pct': filtered_data.round(1).to_dict(),
        'source_preferences': source_preferences,
        'diversity_scores': diversity_scores.round(3).to_dict(),
        'source_volumes': source_volumes.to_dict()
    }
    
    logger.info(f"Analyzed {total_shows} shows across {len(filtered_data)} networks")
    return insights
    

