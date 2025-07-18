"""TV Titles Data Analysis Module.

This module provides comprehensive analysis of TV title data from Supabase,
using materialized views for efficient data retrieval and analysis.

Main components:
1. Data fetching from materialized views
2. Statistical analysis
3. Report generation using ydata-profiling
4. Results caching

=== CRITICAL COLUMN NAMES ===
Standardized column names used across all views:
1. Title Names: 'title' column
2. Network Names: 'network_name' column
3. Studio Names: 'studio_names' column
4. Status Names: 'status_name' column
"""

__all__ = ['ShowsAnalyzer']

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import os

import streamlit as st

import pandas as pd
try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
from supabase import create_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import the centralized Supabase client
from src.config.supabase_client import get_client

class ShowsAnalyzer:
    """Analyzer for TV titles data.
    
    This class handles fetching and analyzing TV title data from Supabase materialized views.
    Results are cached to avoid unnecessary recomputation.
    """
    
    # View names
    VIEWS = {
        'titles': 'api_market_analysis',  # Use market analysis view for market snapshot
        'networks': 'api_network_stats',
        'team': 'show_team',  # Use raw team table to get all team members
        'details': 'api_show_details',  # Additional show details for content analysis
        'summary': 'api_show_summary',  # Summary view for show detail page
        'comp': 'api_show_comp_data',  # Comparison data for show similarity scoring
        'success': 'api_success_metrics'  # Success metrics including RT data
    }
    
    # Reference tables for lookups
    REFERENCE_TABLES = {
        'character_types': 'character_type_types',
        'genre': 'genre_list',  # Primary genre
        'subgenres': 'genre_list',  # Secondary genres
        'source_type': 'source_types',
        'location_setting': 'location_setting_types',
        'network': 'network_list',
        'order_type': 'order_types',
        'plot_elements': 'plot_element_types',
        'studios': 'studio_list',
        'team_members': 'api_show_comp_data',  # Use view for processed team data
        'thematic_elements': 'thematic_element_types',
        'tone': 'tone_types',
        'time_setting': 'time_setting_types'
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the analyzer.
        
        Args:
            cache_dir: Directory to store cached results. Defaults to 'cache' in current dir.
        """
        try:
            self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / 'cache'
            self.cache_dir.mkdir(exist_ok=True)
            
            self.titles_df: Optional[pd.DataFrame] = None
            self.team_df: Optional[pd.DataFrame] = None
            self.network_df: Optional[pd.DataFrame] = None
            self.last_fetch: Optional[datetime] = None
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    @st.cache_data(ttl=3600)
    def fetch_optimizer_data(_self) -> pd.DataFrame:
        """Fetch data needed for the Show Optimizer.
        
        This method fetches show data specifically for the Show Optimizer component.
        It is kept separate from fetch_comp_data() to maintain decoupling between
        different application features.
        
        Returns:
            DataFrame containing show data for optimization
        """
        try:
            # Get Supabase client with service key for full access
            supabase = get_client(use_service_key=True)
            
            if supabase is None:
                st.error("Supabase client not initialized. Check your environment variables.")
                return pd.DataFrame()
                
            # Fetch comparison data
            st.write("Fetching show data for optimization...")
            comp_data = supabase.table(_self.VIEWS['comp']).select('*').execute()
            
            if not hasattr(comp_data, 'data') or not comp_data.data:
                st.error(f"No data returned from {_self.VIEWS['comp']} view")
                return pd.DataFrame()
                
            comp_df = pd.DataFrame(comp_data.data)
            
            # Convert array fields to Python lists
            array_fields = ['subgenres', 'character_type_ids', 'plot_element_ids', 
                          'thematic_element_ids', 'studios', 'team_member_ids', 'team_member_names']
            for field in array_fields:
                if field in comp_df.columns:
                    comp_df[field] = comp_df[field].apply(_self.convert_to_list)
                    
            # Ensure description field is present and handled
            if 'description' not in comp_df.columns:
                st.warning("Description field missing from show data")
            else:
                # Replace None with empty string for consistency
                comp_df['description'] = comp_df['description'].fillna('')
            
            return comp_df
            
        except Exception as e:
            st.error(f"Error fetching show data for optimization: {str(e)}")
            return pd.DataFrame()
            
    @st.cache_data(ttl=3600)
    def get_reference_data(_self) -> Dict[str, pd.DataFrame]:
        """Get reference data for the Show Optimizer.
        
        This method fetches reference data (genres, networks, etc.) specifically for
        the Show Optimizer component.
        
        Returns:
            Dictionary mapping reference table names to their DataFrames
        """
        try:
            # Get Supabase client with service key for full access
            supabase = get_client(use_service_key=True)
            
            if supabase is None:
                st.error("Supabase client not initialized. Check your environment variables.")
                return {}
                
            # Fetch reference tables
            st.write("Fetching reference data for optimization...")
            reference_data = {}
            for ref_name, table_name in _self.REFERENCE_TABLES.items():
                # For studio_list, only get actual studios (not production companies)
                if table_name == 'studio_list':
                    ref_data = supabase.table(table_name).select('*').eq('type', 'Studio').execute()
                else:
                    ref_data = supabase.table(table_name).select('*').execute()
                    
                if not hasattr(ref_data, 'data') or not ref_data.data:
                    st.warning(f"No data returned from {table_name}")
                    continue
                    
                # Special handling for different tables
                if table_name == 'genre_list':
                    # For genre_list table, we need to handle both genre and subgenres
                    df = pd.DataFrame(ref_data.data)
                    # For primary genre, only include rows where category='Main'
                    if ref_name == 'genre':
                        df = df[df['category'] == 'Main']
                    reference_data[ref_name] = df
                elif table_name == 'api_show_comp_data':
                    # For team members, use the comp data we already fetched
                    # We'll fetch this separately to avoid circular dependencies
                    comp_data = supabase.table(_self.VIEWS['comp']).select('*').execute()
                    if hasattr(comp_data, 'data') and comp_data.data:
                        comp_df = pd.DataFrame(comp_data.data)
                        reference_data[ref_name] = comp_df
                else:
                    reference_data[ref_name] = pd.DataFrame(ref_data.data)
            
            return reference_data
            
        except Exception as e:
            st.error(f"Error fetching reference data for optimization: {str(e)}")
            return {}
    
    def fetch_comp_data(_self, force: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetch data needed for show comparisons.
        
        Args:
            force (bool): If True, bypass cache and fetch fresh data

        Returns:
            Tuple containing:
            - DataFrame containing show comparison data
            - Dictionary mapping reference table names to their DataFrames
        """
        try:
            # Get Supabase client with service key for full access
            supabase = get_client(use_service_key=True)
            
            if supabase is None:
                raise ValueError("Supabase client not initialized. Check your environment variables.")
                
            # Fetch comparison data
            logger.info(f"Fetching data from {_self.VIEWS['comp']}...")
            comp_data = supabase.table(_self.VIEWS['comp']).select('*').execute()
            
            if not hasattr(comp_data, 'data') or not comp_data.data:
                raise ValueError(f"No data returned from {_self.VIEWS['comp']} view")
                
            comp_df = pd.DataFrame(comp_data.data)
            
            # Convert array fields to Python lists
            array_fields = ['subgenres', 'character_type_ids', 'plot_element_ids', 
                          'thematic_element_ids', 'studios', 'team_member_ids', 'team_member_names']
            for field in array_fields:
                if field in comp_df.columns:
                    comp_df[field] = comp_df[field].apply(_self.convert_to_list)
                    
            # Ensure description field is present and handled
            if 'description' not in comp_df.columns:
                logger.warning("Description field missing from comp_data")
            else:
                # Replace None with empty string for consistency
                comp_df['description'] = comp_df['description'].fillna('')
            
            # Fetch reference tables
            logger.info("Fetching reference tables...")
            reference_data = {}
            for ref_name, table_name in _self.REFERENCE_TABLES.items():
                logger.info(f"Fetching {table_name}...")
                # For studio_list, only get actual studios (not production companies)
                if table_name == 'studio_list':
                    ref_data = supabase.table(table_name).select('*').eq('type', 'Studio').execute()
                else:
                    ref_data = supabase.table(table_name).select('*').execute()
                if not hasattr(ref_data, 'data') or not ref_data.data:
                    raise ValueError(f"No data returned from {table_name}")
                    
                # Special handling for different tables
                if table_name == 'genre_list':
                    # For genre_list table, we need to handle both genre and subgenres
                    df = pd.DataFrame(ref_data.data)
                    # For primary genre, only include rows where category='Main'
                    if ref_name == 'genre':
                        df = df[df['category'] == 'Main']
                    reference_data[ref_name] = df
                elif table_name == 'api_show_comp_data':
                    # For team members, use the comp data we already fetched
                    reference_data[ref_name] = comp_df
                else:
                    reference_data[ref_name] = pd.DataFrame(ref_data.data)
                    
                pass  # Reference data loaded
            
            return comp_df, reference_data
            
        except Exception as e:
            logger.error(f"Error fetching comp data: {str(e)}")
            raise

    @st.cache_data(ttl=3600)
    def fetch_studio_data(_self, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch data needed for studio analysis.
        
        Args:
            force (bool): If True, bypass cache and fetch fresh data

        Returns:
            Tuple containing:
            - DataFrame with show data including genres
            - DataFrame with network stats
            - DataFrame with studio categories
        """
        try:
            # Get Supabase client with service key for full access
            supabase = get_client(use_service_key=True)
            
            if supabase is None:
                raise ValueError("Supabase client not initialized. Check your environment variables.")
                
            # Fetch titles data from market analysis
            logger.info(f"Fetching data from {_self.VIEWS['titles']}...")
            titles_data = supabase.table(_self.VIEWS['titles']).select('*').execute()
            
            if not hasattr(titles_data, 'data') or not titles_data.data:
                raise ValueError(f"No data returned from {_self.VIEWS['titles']}")
                
            titles_df = pd.DataFrame(titles_data.data)
            
            # Get show data including genres
            shows_data = supabase.table('shows').select('id,title,active,genre_id,subgenres').execute()
            if not hasattr(shows_data, 'data') or not shows_data.data:
                raise ValueError("No data returned from shows table")
            shows_df = pd.DataFrame(shows_data.data)
            
            # Get genre names
            genre_data = supabase.table('genre_list').select('id,genre').execute()
            if not hasattr(genre_data, 'data') or not genre_data.data:
                raise ValueError("No data returned from genre_list")
            genre_df = pd.DataFrame(genre_data.data)
            
            # Map genre IDs to names
            genre_map = genre_df.set_index('id')['genre'].to_dict()
            shows_df['genre_name'] = shows_df['genre_id'].map(genre_map)
            shows_df['subgenre_names'] = shows_df['subgenres'].apply(
                lambda x: [genre_map.get(g) for g in (x or []) if g in genre_map]
            )
            
            # Merge show data into titles
            titles_df = titles_df.merge(shows_df[['title', 'genre_name', 'subgenre_names']], on='title', how='left')
            
            # Fetch network stats
            logger.info(f"Fetching data from {_self.VIEWS['networks']}...")
            network_data = supabase.table(_self.VIEWS['networks']).select('*').execute()
            if not hasattr(network_data, 'data') or not network_data.data:
                raise ValueError(f"No data returned from {_self.VIEWS['networks']}")
            network_df = pd.DataFrame(network_data.data)
            
            # Fetch studio categories
            studio_list_data = supabase.table('studio_list').select('*').execute()
            if not hasattr(studio_list_data, 'data') or not studio_list_data.data:
                raise ValueError("No data returned from studio_list")
            studio_df = pd.DataFrame(studio_list_data.data)
            
            return titles_df, network_df, studio_df
            
        except Exception as e:
            logger.error(f"Error fetching studio data: {str(e)}")
            raise

    @st.cache_data(ttl=3600)
    def fetch_market_data(_self, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch data needed for market analysis.
        
        Args:
            force (bool): If True, bypass cache and fetch fresh data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: titles_df (with market data), network_df
        """
        try:
            # Get Supabase client with service key for full access
            supabase = get_client(use_service_key=True)
            
            if supabase is None:
                raise ValueError("Supabase client not initialized. Check your environment variables.")
                
            # Fetch titles data from market analysis
            logger.info(f"Fetching data from {_self.VIEWS['titles']}...")
            titles_data = supabase.table(_self.VIEWS['titles']).select('*').execute()
            logger.info(f"Raw response from {_self.VIEWS['titles']}: {titles_data}")
            
            if not hasattr(titles_data, 'data'):
                raise ValueError(f"Invalid response from {_self.VIEWS['titles']}: missing 'data' attribute")
            if not titles_data.data:
                raise ValueError(f"No data returned from {_self.VIEWS['titles']}")
                
            titles_df = pd.DataFrame(titles_data.data)
            # Verify required columns for market analysis
            required_cols = ['title', 'network_name', 'tmdb_id', 'tmdb_seasons', 'tmdb_total_episodes', 
                            'tmdb_status', 'status_name', 'studio_names']
            missing_cols = [col for col in required_cols if col not in titles_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for market analysis: {missing_cols}")
            
            # Verify title column has data
            if 'title' in titles_df.columns and titles_df['title'].isna().any():
                logger.warning("Some rows have missing titles")
                logger.warning(f"Rows with missing titles: {titles_df[titles_df['title'].isna()]}")
            
            # Get active status and IDs directly from shows table
            shows_data = supabase.table('shows').select('id,title,active').execute()
            if not hasattr(shows_data, 'data') or not shows_data.data:
                raise ValueError("No data returned from shows table")
            shows_df = pd.DataFrame(shows_data.data)

            # Verify shows_df has required columns
            if 'id' not in shows_df.columns or 'title' not in shows_df.columns:
                raise ValueError(f"Missing required columns in shows table. Available: {shows_df.columns.tolist()}")
            
            titles_df = titles_df.merge(shows_df[['id', 'title', 'active']], on='title', how='left')
            titles_df['active'] = titles_df['active'].fillna(False)  # Default to inactive for any shows not in shows table
            

            
            # Fetch team data with pagination
            page_size = 1000
            start = 0
            all_team_data = []
            while True:
                team_data = supabase.table(_self.VIEWS['team']).select('*').range(start, start + page_size - 1).execute()
                if not hasattr(team_data, 'data') or not team_data.data:
                    break
                all_team_data.extend(team_data.data)
                if len(team_data.data) < page_size:
                    break
                start += page_size
            
            if not all_team_data:
                raise ValueError(f"No data returned from {_self.VIEWS['team']}")
            team_df = pd.DataFrame(all_team_data)
            
            # Filter team_df to only include members from active shows
            active_show_ids = titles_df[titles_df['active'] == True]['id'].tolist()
            team_df = team_df[team_df['show_id'].isin(active_show_ids)]
            
            # Fetch network data
            network_data = supabase.table(_self.VIEWS['networks']).select('*').execute()
            if not hasattr(network_data, 'data') or not network_data.data:
                raise ValueError(f"No data returned from {_self.VIEWS['networks']}")
            network_df = pd.DataFrame(network_data.data)
            
            # Log success
            logger.info(f"Successfully fetched market data:")
            logger.info(f"- {len(titles_df)} shows")
            logger.info(f"- {len(team_df)} team members")
            logger.info(f"- {len(network_df)} networks")
            
            return titles_df, team_df, network_df
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise  # Re-raise to let MarketAnalyzer handle the error
    
    def fetch_content_data(self, force: bool = False) -> pd.DataFrame:
        """Fetch data needed for content analysis.
        
        Args:
            force (bool): If True, bypass cache and fetch fresh data

        Returns:
            pd.DataFrame: Details DataFrame with content analysis fields
        """
        try:
            # Get Supabase client with service key for full access
            supabase = get_client(use_service_key=True)
            
            if supabase is None:
                raise ValueError("Supabase client not initialized. Check your environment variables.")
                
            # Fetch show details
            details_data = supabase.table(self.VIEWS['details']).select('*').execute()
            details_df = pd.DataFrame(details_data.data)
            logger.info(f"Fetched {len(details_df)} rows from {self.VIEWS['details']}")
            
            # Get active status directly from shows table
            shows_data = supabase.table('shows').select('title,active').execute()
            shows_df = pd.DataFrame(shows_data.data)
            details_df = details_df.merge(shows_df[['title', 'active']], on='title', how='left')
            details_df['active'] = details_df['active'].fillna(False)  # Default to inactive for any shows not in shows table
            
            # Fetch team data with pagination
            all_team_rows = []
            start = 0
            page_size = 1000
            while True:
                # Join show_team with shows to get title
                page = supabase.table(self.VIEWS['team']).select(
                    'name,role_type_id,shows(title)'
                ).range(start, start + page_size - 1).execute()
                
                # Extract title from the nested shows object
                for row in page.data:
                    row['title'] = row['shows']['title']
                    del row['shows']
                if not page.data:
                    break
                all_team_rows.extend(page.data)
                if len(page.data) < page_size:
                    break
                start += page_size
            team_df = pd.DataFrame(all_team_rows)
            logger.info(f"Fetched {len(team_df)} team members")
            
            # Validate required columns for details_df
            required_details_cols = [
                'source_type_name', 'genre_name', 'network_name', 'title',
                'episode_count', 'order_type_name'
            ]
            missing_details_cols = [col for col in required_details_cols if col not in details_df.columns]
            if missing_details_cols:
                raise ValueError(f"Missing required columns in details_df: {missing_details_cols}")
            
            # Validate required columns for team_df
            required_team_cols = ['title', 'name', 'role_type_id']
            if len(team_df) > 0:
                missing_team_cols = [col for col in required_team_cols if col not in team_df.columns]
                if missing_team_cols:
                    raise ValueError(f"Missing required columns in team_df: {missing_team_cols}")
                
                # Merge details with team data
                details_df['team'] = details_df['title'].map(team_df.groupby('title').apply(
                    lambda x: x[['name', 'role_type_id']].to_dict('records')
                ))
            else:
                details_df['team'] = None
            
            return details_df
        except Exception as e:
            logger.error(f"Error fetching content data: {str(e)}")
            raise  # Re-raise the error so UnifiedAnalyzer can handle it properly
                        
    @st.cache_data(ttl=3600)
    def fetch_show_data(_self, force: bool = False) -> pd.DataFrame:
        """Fetch show data from api_show_summary view.
        
        Args:
            force: If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame with show details from api_show_summary
        """
        try:
            # Get Supabase client with service key for full access
            supabase = get_client(use_service_key=True)
            if not supabase:
                raise ValueError("Supabase client not initialized")
                
            # Fetch from api_show_details view
            result = supabase.table(_self.VIEWS['details']).select('*').execute()
            if not hasattr(result, 'data') or not result.data:
                raise ValueError("No data returned from api_show_details")
                
            shows_df = pd.DataFrame(result.data)
            return shows_df
            
        except Exception as e:
            st.write(f"Error fetching show data: {str(e)}")
            raise
            
            # Helper function to normalize studio names
            def normalize_studio_names(studios):
                if not isinstance(studios, list):
                    return []
                normalized = []
                for studio in studios:
                    # Use the primary studio name if this is an alias
                    normalized.append(alias_to_studio.get(studio, studio))
                return normalized
            
            # Ensure studio_names is a list and normalize using aliases
            result_df['studio_names'] = result_df['studio_names'].apply(self.convert_to_list).apply(normalize_studio_names)
            
            return result_df, studio_list_df
        except Exception as e:
            logger.error(f"Error fetching studio data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame(), pd.DataFrame()
            
    def convert_to_list(self, x):
        """Convert a value to a list.
        
        Args:
            x: Value to convert. Could be:
            - Python list
            - JSON array string
            - PostgreSQL array string
            - Single value
            
        Returns:
            List version of the value
        """
        try:
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                # Handle PostgreSQL array format: {"item1","item2"}
                if x.startswith('{') and x.endswith('}'):
                    # Remove {} and split on commas, handling escaped quotes
                    items = x[1:-1].split(',')
                    return [item.strip('"') for item in items if item.strip()]
                # Handle JSON array format
                if x.startswith('[') and x.endswith(']'):
                    import ast
                    return ast.literal_eval(x)
                return [x]  # Single value
            if pd.isna(x) or x is None:
                return []
            return [str(x)]  # Convert other types to string
        except Exception as e:
            logger.warning(f"Error converting {x} to list: {e}")
            return []

    def generate_basic_stats(self) -> Dict[str, Any]:
        """Generate basic statistics about the shows.
        
        Returns:
            Dictionary containing basic statistics:
            - Total number of shows
            - Active shows
            - Shows by network
            - Shows by genre
            - Average team size
            etc.
        """
        if self.titles_df is None:
            self.fetch_data()
            
        # Clean data before generating stats
        # Data is already clean from the view
        
        try:
            # Basic show statistics from market analysis view
            stats = {
                'total_shows': len(self.titles_df),
                'active_shows': len(self.titles_df[self.titles_df['status_name'].str.lower() == 'active']),
                'genres': self.titles_df['genre'].value_counts().to_dict() if 'genre' in self.titles_df.columns else {},
                'status_breakdown': self.titles_df['status_name'].value_counts().to_dict(),
                'source_types': self.titles_df['source_type'].value_counts().to_dict() if 'source_type' in self.titles_df.columns else {},
                'avg_episodes': self.titles_df['tmdb_total_episodes'].mean() if 'tmdb_total_episodes' in self.titles_df.columns else 0
            }
            
            # Add network statistics
            # Get network statistics from network_df
            if self.network_df is not None and len(self.network_df) > 0:
                # Get unique networks from network_df since it has the complete list
                unique_networks = self.network_df['network_name'].unique()
                stats['networks'] = {}
                
                # For each unique network, calculate stats from titles_df
                for network in unique_networks:
                    network_titles = self.titles_df[self.titles_df['network_name'] == network]
                    
                    if len(network_titles) > 0:
                        status_counts = network_titles['status_name'].value_counts().to_dict()
                        genre_counts = network_titles['genre'].value_counts().to_dict() if 'genre' in network_titles.columns else {}
                        source_type_counts = network_titles['source_type'].value_counts().to_dict() if 'source_type' in network_titles.columns else {}
                        
                        stats['networks'][network] = {
                            'total_shows': len(network_titles),
                            'active_shows': status_counts.get('Active', 0),
                            'ended_shows': status_counts.get('Ended', 0),
                            'genres': genre_counts,
                            'source_types': source_type_counts
                        }
                    else:
                        # Network exists but has no shows in current filter
                        stats['networks'][network] = {
                            'total_shows': 0,
                            'active_shows': 0,
                            'ended_shows': 0,
                            'genres': {},
                            'source_types': {}
                        }
                        
                # For backward compatibility, provide simple network counts
                stats['network_counts'] = {net: data['total_shows'] 
                                         for net, data in stats['networks'].items()}
            
            # Add studio stats if available
            if 'studio_names' in self.titles_df.columns:
                studio_counts = self.titles_df['studio_names'].explode().value_counts()
                stats['studios'] = studio_counts.to_dict() if not studio_counts.empty else {}
            
            # Add team stats if available
            if not self.team_df.empty:
                # Get unique team members and their networks
                team_networks = self.team_df.groupby('name')['network_name'].unique().to_dict()
                all_members = self.team_df['name'].unique()
                
                stats['team_stats'] = {
                    'total_members': len(team_networks),
                    'team_networks': team_networks  # Map of team member -> list of networks they work with
                }
                
                # Calculate team size metrics
                stats['avg_team_size'] = len(all_members) / len(self.titles_df) if len(self.titles_df) > 0 else 0
                stats['total_team_members'] = len(all_members)
            else:
                stats['avg_team_size'] = 0
                stats['total_team_members'] = 0
                stats['roles'] = {}
        except Exception as e:
            logger.error(f"Error calculating basic stats: {str(e)}")
            stats = {
                'total_titles': 0,
                'active_shows': 0,
                'networks': {},
                'genres': {},
                'studios': {},
                'avg_episodes': 0,
                'avg_team_size': 0,
                'status_breakdown': {},
                'source_types': {}
            }
        
        # Add team role statistics
        if self.team_roles_df is not None and len(self.team_roles_df) > 0:
            stats['roles'] = self.team_roles_df['role_type'].value_counts().to_dict()
            stats['avg_team_size'] = len(self.team_roles_df) / len(self.titles_df) if len(self.titles_df) > 0 else 0
            stats['total_team_members'] = len(self.team_roles_df['name'].unique())
        else:
            stats['roles'] = {}
            stats['avg_team_size'] = 0
            stats['total_team_members'] = 0
            
        # Add date-based statistics
        if 'announced_date' in self.titles_df.columns:
            try:
                last_month = pd.Timestamp.now() - pd.DateOffset(months=1)
                stats['new_shows_last_month'] = len(self.titles_df[self.titles_df['announced_date'] >= last_month])
                
                # Titles by year
                stats['shows_by_year'] = self.titles_df['announced_date'].dt.year.value_counts().sort_index().to_dict()
                
                # Recent trends (last 12 months)
                last_year = datetime.now() - pd.DateOffset(months=12)
                recent_shows = self.titles_df[self.titles_df['announced_date'] >= last_year]
                stats['recent_trends'] = {
                    'total_shows': len(recent_shows),
                    'top_networks': recent_shows['network_name'].value_counts().head(5).to_dict(),
                    'top_genres': recent_shows['genre'].value_counts().head(5).to_dict()
                }
            except Exception as e:
                logger.error(f"Error processing dates: {e}")
                stats['new_titles_last_month'] = 0
                stats['titles_by_year'] = {}
                stats['recent_trends'] = {
                    'total_titles': 0,
                    'top_networks': {},
                    'top_genres': {}
                }
        else:
            stats['new_titles_last_month'] = 0
            stats['titles_by_year'] = {}
            stats['recent_trends'] = {
                'total_titles': 0,
                'top_networks': {},
                'top_genres': {}
            }
        
        logger.info(f"Analysis complete - {stats['total_shows']} shows processed")
        return stats


    def generate_profile_report(self, output_file: Optional[str] = None) -> None:
        """Generate comprehensive profile reports using ydata-profiling.
        
        This generates a report with:
        1. Show statistics and distributions
        2. Team member analysis
        3. Network and genre trends
        
        Args:
            output_file: Path to save the HTML report. If None, uses default path in cache_dir.
        """
        if self.titles_df is None or self.team_df is None:
            self.fetch_data()
            # Data is already clean from the view
            
        # Default output paths
        if output_file is None:
            base_path = self.cache_dir / f'profile_{datetime.now():%Y%m%d}'
            titles_output = base_path.with_name(f'{base_path.name}_titles.html')
            team_output = base_path.with_name(f'{base_path.name}_team.html')
        else:
            titles_output = Path(output_file)
            team_output = titles_output.with_name(f'{titles_output.stem}_team.html')
        
        logger.info('Generating profile reports...')
        
        try:
            # Prepare data for profile report
            shows_with_team = self.titles_df.copy()
            
            # Add team metrics if team data is available
            if self.team_df is not None and len(self.team_df) > 0:
                team_metrics = self.team_df.groupby('title').agg({
                    'name': 'count',
                    'role_name': lambda x: len(set([role for roles in x for role in roles]))
                }).rename(columns={
                    'name': 'team_size',
                    'role_name': 'unique_roles'
                })
                titles_with_team = titles_with_team.join(team_metrics, on='title')
            
            # Fill NaN values for better reporting
            numeric_cols = titles_with_team.select_dtypes(include=['int64', 'float64']).columns
            titles_with_team[numeric_cols] = titles_with_team[numeric_cols].fillna(0)
            
            # Create shows profile report
            if not PROFILING_AVAILABLE:
                logger.warning("ydata_profiling not installed. Profile report generation skipped.")
                return
                
            shows_profile = ProfileReport(
                shows_with_team,
                title="TV Shows Analysis Report",
                explorative=True
            )
            
            # Save report if output file is specified
            if output_file:
                logger.info(f'Saving shows profile report to {output_file}')
                shows_profile.to_file(output_file)
            
            logger.info('Profile report generation completed')
            
        except Exception as e:
            logger.error(f'Error generating profile reports: {str(e)}')
            raise
            
    @st.cache_data(ttl=3600)
    def fetch_success_metrics(_self, force: bool = False) -> pd.DataFrame:
        """Fetch success metrics data including RT scores.
        
        Args:
            force (bool): If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame with success metrics from api_success_metrics view
        """
        try:
            # Get Supabase client with service key for full access
            supabase = get_client(use_service_key=True)
            
            if supabase is None:
                raise ValueError("Supabase client not initialized")
                
            # Fetch from success metrics view
            logger.info(f"Fetching data from {_self.VIEWS['success']}...")
            result = supabase.table(_self.VIEWS['success']).select('*').execute()
            
            if not hasattr(result, 'data') or not result.data:
                raise ValueError(f"No data returned from {_self.VIEWS['success']}")
                
            success_df = pd.DataFrame(result.data)
            
            # Validate required columns
            required_cols = [
                'show_id', 'title', 'network_name', 'tmdb_status',
                'tmdb_seasons', 'tmdb_avg_eps',
                'has_rt', 'tomatometer', 'popcornmeter'
            ]
            missing_cols = [col for col in required_cols if col not in success_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in success_df: {missing_cols}")
                
            logger.info(f"Fetched {len(success_df)} shows with success metrics")
            return success_df
            
        except Exception as e:
            logger.error(f"Error fetching success metrics: {str(e)}")
            raise

