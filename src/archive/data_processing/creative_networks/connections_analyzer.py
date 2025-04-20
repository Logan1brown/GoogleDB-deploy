"""Network Connection Analysis.

This module analyzes relationships between networks through shared creators,
with filtering capabilities for networks, genres, and source types.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataFields(Enum):
    """Constants for data field names."""
    SHOW_NAME = 'show_name'
    SHOWS = 'shows'
    NETWORK = 'network'
    GENRE = 'genre'
    SOURCE_TYPE = 'source_type'
    NAME = 'name'

@dataclass
class CreatorProfile:
    """Profile of a creator's work across networks."""
    def __init__(self, name: str):
        self.name = name
        self.networks = set()
        self.genres = set()
        self.source_types = set()
        self.total_shows = 0
        self.shows = set()  # Track specific shows for overlap calculation

class ConnectionsAnalyzer:
    """Analyzer for network relationships and creator filtering."""
    
    UNKNOWN_VALUE = 'Unknown'
    REQUIRED_SHOW_COLUMNS = {DataFields.NETWORK.value, DataFields.GENRE.value, 
                           DataFields.SOURCE_TYPE.value}
    REQUIRED_TEAM_COLUMNS = {DataFields.NAME.value, DataFields.SHOW_NAME.value}
    
    def __init__(self, shows_df: pd.DataFrame, team_df: pd.DataFrame) -> None:
        """Initialize the analyzer.
        
        Args:
            shows_df: DataFrame with show information (network, genre, source_type)
            team_df: DataFrame with creator information
            
        Raises:
            ValueError: If required columns are missing
        """
        self._validate_dataframes(shows_df, team_df)
        self.shows_df = self._prepare_shows_data(shows_df)
        self.team_df = team_df
        self.combined_df = self._merge_data()
        self.creator_profiles = self._build_creator_profiles()
        self._log_stats()
    
    def _validate_dataframes(self, shows_df: pd.DataFrame, team_df: pd.DataFrame) -> None:
        """Validate required columns exist in dataframes."""
        missing_show_cols = self.REQUIRED_SHOW_COLUMNS - set(shows_df.columns)
        missing_team_cols = self.REQUIRED_TEAM_COLUMNS - set(team_df.columns)
        
        if missing_show_cols:
            raise ValueError(f"Missing required show columns: {missing_show_cols}")
        if missing_team_cols:
            raise ValueError(f"Missing required team columns: {missing_team_cols}")
    
    def _prepare_shows_data(self, shows_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare shows data by handling column names and missing values."""
        df = shows_df.copy()
        
        # Handle column name differences
        if DataFields.SHOWS.value in df.columns:
            df = df.rename(columns={DataFields.SHOWS.value: DataFields.SHOW_NAME.value})
        
        # Handle missing and empty values
        for col in [DataFields.GENRE.value, DataFields.SOURCE_TYPE.value]:
            # Replace empty strings with NaN
            df[col] = df[col].replace('', pd.NA)
            # Then replace NaN with Unknown
            df[col] = df[col].fillna(self.UNKNOWN_VALUE)
            
        return df
    
    def _merge_data(self) -> pd.DataFrame:
        """Merge show and creator data safely."""
        try:
            return pd.merge(
                self.team_df,
                self.shows_df[[DataFields.SHOW_NAME.value, DataFields.NETWORK.value,
                              DataFields.GENRE.value, DataFields.SOURCE_TYPE.value]],
                on=DataFields.SHOW_NAME.value
            )
        except Exception as e:
            raise ValueError(f"Failed to merge show and creator data: {e}")
    
    def _log_stats(self) -> None:
        """Log basic statistics about the data."""
        logger.info("Network connection stats:")
        logger.info(f"  Networks: {self.combined_df[DataFields.NETWORK.value].nunique()}")
        logger.info(f"  Genres: {self.combined_df[DataFields.GENRE.value].nunique()}")
        logger.info(f"  Source types: {self.combined_df[DataFields.SOURCE_TYPE.value].nunique()}")
        logger.info(f"  Unique creators: {self.combined_df[DataFields.NAME.value].nunique()}")
    
    def _build_creator_profiles(self) -> Dict[str, CreatorProfile]:
        """Build cached profiles of creators and their work.
        
        Returns:
            Dict mapping creator names to their profiles
        """
        profiles = {}
        
        for name, group in self.combined_df.groupby(DataFields.NAME.value):
            profile = CreatorProfile(name)
            for _, row in group.iterrows():
                profile.networks.add(row[DataFields.NETWORK.value])
                profile.genres.add(row[DataFields.GENRE.value])
                profile.source_types.add(row[DataFields.SOURCE_TYPE.value])
                profile.shows.add(row[DataFields.SHOW_NAME.value])
                profile.total_shows += 1
            profiles[name] = profile
            
        return profiles
    
    def get_shared_creators_matrix(self) -> np.ndarray:
        """Get matrix of shared creator counts between networks.
        Returns:
            Dict mapping creator names to their profiles
        """
        profiles = {}
        
        for name, group in self.combined_df.groupby(DataFields.NAME.value):
            profiles[name] = CreatorProfile(
                name=name,
                networks=set(group[DataFields.NETWORK.value].unique()),
                genres=set(group[DataFields.GENRE.value].unique()),
                source_types=set(group[DataFields.SOURCE_TYPE.value].unique()),
                total_shows=len(group[DataFields.SHOW_NAME.value].unique())
            )
        
        return profiles
    
    def get_filter_options(self) -> Dict[str, List[str]]:
        """Get available filter options.
        
        Returns:
            Dict with lists of available networks, genres, and source types
        """
        return {
            'networks': sorted(self.combined_df[DataFields.NETWORK.value].unique()),
            'genres': sorted(self.combined_df[DataFields.GENRE.value].unique()),
            'sources': sorted(self.combined_df[DataFields.SOURCE_TYPE.value].unique())
        }
    
    def filter_creators(
        self,
        networks: Optional[List[str]] = None,
        genre: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> List[CreatorProfile]:
        """Filter creators based on specified criteria.
        
        Args:
            networks: List of networks to filter by
            genre: Genre to filter by
            source_type: Source type to filter by
            
        Returns:
            List of filtered creator profiles sorted by total shows
        """
        filtered_profiles = []
        
        for profile in self.creator_profiles.values():
            # Check all specified criteria must match (AND logic)
            if networks and not all(net in profile.networks for net in networks):
                continue
            
            if genre and genre not in profile.genres:
                continue
                
            if source_type and source_type not in profile.source_types:
                continue
            
            filtered_profiles.append(profile)
        
        # Sort by total shows descending
        return sorted(filtered_profiles, key=lambda x: x.total_shows, reverse=True)
    
    def get_shared_creators_matrix(
        self,
        network1: Optional[str] = None,
        network2: Optional[str] = None,
        genre: Optional[str] = None,
        source: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """Generate matrix of shared creators between networks.
        
        Args:
            network1: First network to compare (optional)
            network2: Second network to compare (optional)
            genre: Filter by genre (optional)
            source: Filter by source type (optional)
            
        Returns:
            Tuple of:
            - matrix: where matrix[i,j] is the number of creators shared between networks[i] and networks[j]
            - network_labels: list of network names
            - selected_indices: indices of selected networks (if any filters applied)
        """
        # Get filtered data first
        filtered_df = self.combined_df.copy()
        if genre:
            filtered_df = filtered_df[filtered_df[DataFields.GENRE.value] == genre]
        if source:
            filtered_df = filtered_df[filtered_df[DataFields.SOURCE_TYPE.value] == source]
            
        # Get all networks and initialize matrix
        all_networks = sorted(self.combined_df[DataFields.NETWORK.value].unique())
        n = len(all_networks)
        full_matrix = np.zeros((n, n), dtype=int)
        
        # Build matrix using filtered data
        for i, net1 in enumerate(all_networks):
            net1_creators = set(filtered_df[filtered_df[DataFields.NETWORK.value] == net1][DataFields.NAME.value])
            for j, net2 in enumerate(all_networks[i:], i):
                if i == j:
                    full_matrix[i, j] = len(net1_creators)
                else:
                    net2_creators = set(filtered_df[filtered_df[DataFields.NETWORK.value] == net2][DataFields.NAME.value])
                    shared = len(net1_creators & net2_creators)
                    full_matrix[i, j] = full_matrix[j, i] = shared
            
        # Get selected networks
        selected_networks = set()
        if network1:
            selected_networks.add(network1)
        if network2:
            selected_networks.add(network2)
        if not selected_networks and (genre or source):
            # When filtering by genre/source but no networks selected,
            # highlight networks that have any creators in that genre/source
            active_networks = set(filtered_df[DataFields.NETWORK.value].unique())
            selected_networks.update(active_networks)
            
        # Get indices of selected networks
        selected_indices = [i for i, net in enumerate(all_networks) if net in selected_networks] if selected_networks else []
            
        return full_matrix, all_networks, selected_indices
        
        # Initialize matrix
        n = len(networks)
        matrix = np.zeros((n, n))
        net_to_idx = {net: i for i, net in enumerate(networks)}
        
        # Count shared creators
        for name, group in filtered_df.groupby(DataFields.NAME.value):
            creator_networks = list(group[DataFields.NETWORK.value].unique())
            for i, net1 in enumerate(creator_networks):
                for net2 in creator_networks[i+1:]:
                    idx1 = net_to_idx[net1]
                    idx2 = net_to_idx[net2]
                    matrix[idx1][idx2] += 1
                    matrix[idx2][idx1] += 1  # Symmetric
        
        return matrix, networks
    
    def get_success_stories(
        self,
        network: Optional[str] = None,
        genre: Optional[str] = None,
        source: Optional[str] = None,
        min_networks: int = 2,
        top_k: int = 5,
        overlap_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Get creators who have worked across multiple networks.
        
        Args:
            min_networks: Minimum number of networks
            top_k: Number of creators to return
            overlap_threshold: Minimum show overlap to group creators (0.0-1.0)
            
        Returns:
            List of creator profiles with network counts, sorted by network count
            and total shows
        """
        # First, identify creator teams based on show overlap
        teams = []
        used_creators = set()
        
        profiles = list(self.creator_profiles.values())
        for i, profile1 in enumerate(profiles):
            if profile1.name in used_creators:
                continue
                
            # Start a new team
            team = [profile1]
            used_creators.add(profile1.name)
            
            # Look for team members with high show overlap
            for profile2 in profiles[i+1:]:
                if profile2.name in used_creators:
                    continue
                    
                # Calculate show overlap in both directions
                if len(profile1.shows) == 0 or len(profile2.shows) == 0:
                    continue
                    
                intersection = profile1.shows & profile2.shows
                overlap1 = len(intersection) / len(profile1.shows)  # % of profile1's shows
                overlap2 = len(intersection) / len(profile2.shows)  # % of profile2's shows
                
                # Only team up if they appear in enough of each other's shows
                if overlap1 >= overlap_threshold and overlap2 >= overlap_threshold:
                    team.append(profile2)
                    used_creators.add(profile2.name)
            
            teams.append(team)
        
        # Convert teams to stories
        stories = []
        for team in teams:
            # Combine team stats
            networks = set()
            genres = set()
            source_types = set()
            total_shows = 0
            shows = set()
            
            for member in team:
                networks.update(member.networks)
                genres.update(member.genres)
                source_types.update(member.source_types)
                shows.update(member.shows)
            
            total_shows = len(shows)  # Use unique show count
            network_count = len(networks)
            
            # Apply filters
            if network and network not in networks:
                continue
            if genre and genre not in genres:
                continue
            if source and source not in source_types:
                continue
                
            if network_count >= min_networks:
                stories.append({
                    'creator_team': ' & '.join(p.name for p in team),
                    'networks': sorted(list(networks)),
                    'network_count': network_count,
                    'total_shows': total_shows,
                    'roles': sorted(list(source_types))
                })
        
        return sorted(stories,
                     key=lambda x: (x['network_count'], x['total_shows']),
                     reverse=True)[:top_k]
        return {
            'networks': sorted(self.combined_df[DataFields.NETWORK.value].unique()),
            'genres': sorted(self.combined_df[DataFields.GENRE.value].unique()),
            'sources': sorted(self.combined_df[DataFields.SOURCE_TYPE.value].unique())
        }

def analyze_network_connections(shows_df: pd.DataFrame, team_df: pd.DataFrame) -> ConnectionsAnalyzer:
    """Initialize and return a ConnectionsAnalyzer instance.
    
    Args:
        shows_df: DataFrame with show information
        team_df: DataFrame with creator information
        
    Returns:
        ConnectionsAnalyzer instance ready for analysis
    """
    return ConnectionsAnalyzer(shows_df, team_df)
