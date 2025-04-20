"""Market Analysis Module.

This module provides market overview analytics including network distribution and key metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Analyzer for market overview and network patterns."""
    
    def __init__(self, shows_df: pd.DataFrame, team_df: pd.DataFrame = None):
        """Initialize the analyzer.
        
        Args:
            shows_df: DataFrame containing show information
            team_df: Optional DataFrame containing team member information
            
        Raises:
            ValueError: If required columns are missing from shows_df
        """
        # Validate required columns
        required_cols = ['network', 'studio', 'success_score']
        missing_cols = [col for col in required_cols if col not in shows_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in shows_df: {missing_cols}")
        
        # Create deep copies to avoid modifying original data
        self.shows_df = shows_df.copy(deep=True)
        self.team_df = pd.DataFrame() if team_df is None else team_df.copy(deep=True)
        
        # Reset indices to ensure clean data
        self.shows_df = self.shows_df.reset_index(drop=True)
        self.team_df = self.team_df.reset_index(drop=True)
        
        # Convert and validate success_score
        self.shows_df['success_score'] = pd.to_numeric(self.shows_df['success_score'], errors='coerce')
        valid_scores = self.shows_df['success_score'].dropna()
        if len(valid_scores) > 0:
            invalid_scores = valid_scores[(valid_scores < 0) | (valid_scores > 100)]
            if len(invalid_scores) > 0:
                logger.warning(f"Found {len(invalid_scores)} success scores outside valid range [0-100]")
                self.shows_df.loc[invalid_scores.index, 'success_score'] = None
        
        # Initialize studio data as None (lazy loading)
        self.studio_df = None
        
        # Log initial state
        logger.info(f"Initial shows_df shape: {self.shows_df.shape}")
        logger.info(f"Initial team_df shape: {self.team_df.shape}")
        logger.info("Market overview:")
        logger.info(f"  Total shows: {len(self.shows_df)}")
        logger.info(f"  Total networks: {len(self.shows_df['network'].unique())}")
        if not self.team_df.empty and 'name' in self.team_df.columns:
            logger.info(f"  Total creatives: {len(self.team_df['name'].unique())}")
    
            required_cols = ['studio_key', 'category']
            missing_cols = [col for col in required_cols if col not in self.studio_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in studio list: {missing_cols}")
                return None
            
            return self.studio_df
        except Exception as e:
            logger.error(f"Error loading studio list: {str(e)}")
            self.studio_df = None
            return None
    
    def _process_studio_data(self) -> bool:
        """Process studio data and merge with shows.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get studio list
            studio_list = self._get_studio_list()
            if studio_list is None:
                return False
            
            # Create show-studio mapping
            shows_with_studios = self._create_show_studio_mapping()
            
            # Merge studio categories
            shows_with_categories = self._merge_studio_categories(shows_with_studios)
            
            # Update shows DataFrame
            self._update_shows_with_categories(shows_with_categories)
            
            return True
        except Exception as e:
            logger.error(f"Error processing studio data: {str(e)}")
            self.studio_df = None
            return False
    
    def _merge_studio_categories(self, shows_with_studios: pd.DataFrame) -> pd.DataFrame:
        """Merge studio categories with shows.
        
        Args:
            shows_with_studios: DataFrame containing shows with studio information
        
        Returns:
            DataFrame with studio categories merged
        """
        try:
            # Get studio list
            studio_list = self._get_studio_list()
            if studio_list is None:
                return pd.DataFrame({
                    'show_uuid': shows_with_studios['show_uuid'].unique(),
                    'category': 'Unknown'
                })
            
            # Merge studio categories
            merged_df = pd.merge(
                shows_with_studios[['show_uuid', 'studio_key']],
                studio_list[['studio_key', 'category']],
                how='left',
                on='studio_key'
            )
            
            logger.info(f"Merged df shape: {merged_df.shape}")
            
            # Check for unmatched studios
            unmatched = merged_df[merged_df['category'].isna()]
            if len(unmatched) > 0:
                logger.warning(f"Found {len(unmatched)} shows with unmatched studios")
                logger.warning(f"Unmatched studios: {unmatched['studio_key'].unique().tolist()}")
            
            # Group by show to combine categories
            categories = merged_df.groupby('show_uuid', as_index=False).agg({
                'category': lambda x: ','.join(sorted(set(x.dropna()))) or 'Unknown'
            })
            
            return categories
        except Exception as e:
            logger.error(f"Error merging studio categories: {str(e)}")
            return pd.DataFrame({
                'show_uuid': shows_with_studios['show_uuid'].unique(),
                'category': 'Unknown'
            })

    def _create_show_studio_mapping(self) -> pd.DataFrame:
        """Create mapping between shows and their studios."""
        # Add UUID for stable joining
        self.shows_df['show_uuid'] = [str(uuid.uuid4()) for _ in range(len(self.shows_df))]
        
        # Split multiple studios
        shows_with_studios = self.shows_df.copy()
        
        # Handle missing values
        shows_with_studios['studio'] = shows_with_studios['studio'].fillna('Unknown')
        
        # Split and clean studio names
        shows_with_studios = shows_with_studios.assign(
            studio=shows_with_studios['studio'].str.split(',')
        ).explode('studio')
        
        # Clean and validate studios
        shows_with_studios['studio'] = shows_with_studios['studio'].str.strip()
        empty_studios = shows_with_studios['studio'] == ''
        if empty_studios.any():
            empty_count = empty_studios.sum()
            logger.warning(f"Found {empty_count} empty studio entries - marking as 'Unknown'")
            shows_with_studios.loc[empty_studios, 'studio'] = 'Unknown'
        
        # Clean studio names
        shows_with_studios['studio'] = shows_with_studios['studio'].str.strip()
        shows_with_studios['studio_key'] = shows_with_studios['studio'].str.lower()
        
        return shows_with_studios.reset_index(drop=True)
    
    def _update_shows_with_categories(self, categories: pd.DataFrame):
        """Update shows DataFrame with studio categories.
        
        Args:
            categories: DataFrame containing show categories
        """
        # Merge categories back to shows
        self.shows_df = pd.merge(
            self.shows_df,
            categories,
            how='left',
            on='show_uuid',
            validate='one_to_one'  # Each show should get exactly one category string
        )
        
        # Clean up
        self.shows_df = self.shows_df.drop('show_uuid', axis=1)
        self.shows_df = self.shows_df.reset_index(drop=True)
    
    def create_network_chart(self, shows_df: pd.DataFrame = None) -> go.Figure:
        """Create a bar chart showing the distribution of shows across networks.
        
        Args:
            shows_df: Optional DataFrame to use instead of self.shows_df
            
        Returns:
            Plotly figure object
        """
        df = shows_df if shows_df is not None else self.shows_df
        
        # Validate DataFrame
        if len(df) == 0:
            return go.Figure().add_annotation(
                text='No data available',
                xref='paper',
                yref='paper',
                x=0.5,
                y=0.5,
                showarrow=False
            )
        
        # Clean network data
        df['network'] = df['network'].fillna('Unknown')
        df['network'] = df['network'].str.strip()
        empty_networks = df['network'] == ''
        if empty_networks.any():
            logger.warning(f"Found {empty_networks.sum()} shows with empty networks")
            df.loc[empty_networks, 'network'] = 'Unknown'
        
        # Get show counts by network
        network_counts = df['network'].value_counts()
        
        # Get success scores by network if available
        valid_scores = df[df['success_score'].notna()]
        if len(valid_scores) > 0:
            network_success = valid_scores.groupby('network')['success_score'].mean().to_dict()
            # Create color scale based on success score
            colors = ['rgb(200, 200, 200)'] * len(network_counts)
            customdata = [0] * len(network_counts)
            for i, network in enumerate(network_counts.index):
                if network in network_success:
                    score = network_success[network]
                    colors[i] = f'rgb({int(255*(1-score/100))}, {int(255*score/100)}, 0)'
                    customdata[i] = score
        else:
            colors = ['rgb(200, 200, 200)'] * len(network_counts)
            customdata = [0] * len(network_counts)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=network_counts.index.tolist(),
                y=network_counts.values.tolist(),
                text=network_counts.values.tolist(),
                textposition='auto',
                marker_color=colors,
                hovertemplate='%{x}<br>Shows: %{y}<br>Success: %{customdata:.1f}<extra></extra>',
                customdata=customdata
            )
        ])
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Show Distribution by Network ({len(df)} shows)',
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.95,
                'yanchor': 'top'
            },
            xaxis_title='Network',
            yaxis_title='Number of Shows',
            showlegend=False,
            xaxis={'tickangle': -45},
            height=500,  # Fixed height
            width=800,   # Fixed width
            margin={
                'l': 50,   # Left margin
                'r': 50,   # Right margin
                't': 100,  # Top margin
                'b': 100   # Bottom margin for rotated labels
            },
            plot_bgcolor='white',  # White background
            paper_bgcolor='white'  # White paper
        )
        
        # Update bar appearance
        fig.update_traces(
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            opacity=0.8
        )
        
        return fig

    def generate_market_insights(self, shows_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate insights about market patterns.
        
        Args:
            shows_df: Optional DataFrame to use instead of self.shows_df
            
        Returns:
            Dictionary containing market insights
        """
        df = shows_df if shows_df is not None else self.shows_df
        
        # Validate DataFrame
        if len(df) == 0:
            return {
            }
            for _, row in top_networks_df.iterrows()
        ]
            
        logger.info(f"Calculated success rates for {len(network_success)} networks")
        logger.info(f"Top network: {top_networks[0]['network'] if top_networks else 'None'}")
        
    return {
        'total_shows': total_shows,
        'total_networks': total_networks,
        'network_success': network_success,
        'top_networks': top_networks
    }
            })
            network_success.columns = ['avg_score', 'total_shows', 'high_success_shows']
            network_success = network_success.reset_index()
            
            # Remove networks with 0 shows
            network_success = network_success[network_success['total_shows'] > 0]
            
            # Calculate success rates
            network_success_rates = {}
            for _, row in network_success.iterrows():
                network = row['network']
                total_shows = int(row['total_shows'])
                avg_score = float(row['avg_score'])
                high_success = float(row['high_success_shows'])
                logger.info(f"Network {network}: {total_shows} shows, avg score {avg_score:.1f}, {high_success} high success")
                
                network_success_rates[network] = {
                    'avg_score': avg_score,
                    'total_shows': total_shows,
                    'high_success_rate': float(high_success / total_shows * 100) if total_shows > 0 else 0.0
                }
            
            # Get networks with high success shows
            high_success_networks = len(network_success[network_success['high_success_shows'] > 0])
            
            # Get top network by success score (must have at least 3 shows)
            top_networks = network_success[network_success['total_shows'] >= 3]
            if len(top_networks) > 0:
                top_network_idx = top_networks['avg_score'].idxmax()
                top_network = top_networks.iloc[top_network_idx]
                top_success_network = top_network['network']
                top_success_score = top_network['avg_score']
                logger.info(f"Top network: {top_success_network} (score: {top_success_score:.1f}, shows: {int(top_network['total_shows'])}")
            else:
                top_success_network = 'None'
                top_success_score = 0
                logger.warning("No networks with 3+ shows found")
        
        # Calculate network volume metrics
        network_counts = df['network'].value_counts()
        top_3_networks = network_counts.nlargest(3)
        network_concentration = (top_3_networks.sum() / total_shows) * 100 if total_shows > 0 else 0
        top_3_names = top_3_networks.index.tolist()
        
        # Calculate vertical integration
        try:
            # Look for studio type in studio column
            if 'studio' in df.columns:
                # Handle NaN and empty strings
                df['studio'] = df['studio'].fillna('Unknown')
                df.loc[df['studio'].str.strip() == '', 'studio'] = 'Unknown'
                
                # Consider network-owned studios as vertically integrated
                network_owned = ['Disney Television', 'HBO Max', 'NBCUniversal', 'Paramount', 
                                'Warner Bros', 'ABC Studios', 'CBS Studios']
                vi_pattern = '|'.join(network_owned)
                vi_shows = df[df['studio'].str.contains(vi_pattern, na=False, case=False)]
                vertical_integration = (len(vi_shows) / total_shows) * 100 if total_shows > 0 else 0
                logger.info(f"Found {len(vi_shows)} vertically integrated shows out of {total_shows} total")
            else:
                logger.warning("No studio column found for vertical integration calculation")
                vertical_integration = 0
        except Exception as e:
            logger.error(f"Error calculating vertical integration: {str(e)}")
            vertical_integration = 0
        
        return {
            'total_shows': total_shows,
            'total_networks': total_networks,
            'total_creatives': total_creatives,
            'avg_success_score': avg_success,
            'network_concentration': network_concentration,
            'vertical_integration': vertical_integration,
            'top_success_network': top_success_network,
            'top_success_score': top_success_score,
            'high_success_networks': high_success_networks,
            'top_3_networks': top_3_names,
            'network_success_rates': network_success_rates
        }
