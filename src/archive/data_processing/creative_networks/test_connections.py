"""Test script for ConnectionsAnalyzer."""

import logging
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from connections_analyzer import ConnectionsAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run test cases for ConnectionsAnalyzer."""
    # Load environment
    load_dotenv()
    
    # Get sheet names from env
    shows_sheet = os.getenv('SHOWS_SHEET_NAME')
    team_sheet = os.getenv('TEAM_SHEET_NAME')
    
    # Get data from shows_analyzer
    from src.data_processing.analyze_shows import shows_analyzer
    shows_df, team_df = shows_analyzer.fetch_data(force=True)
    
    logger.info(f"Loaded {len(shows_df)} shows and {len(team_df)} team records")
    
    # Initialize analyzer
    analyzer = ConnectionsAnalyzer(shows_df, team_df)
    
    # Test 1: Get filter options
    logger.info("\nTest 1: Available Filter Options")
    options = analyzer.get_filter_options()
    logger.info(f"Networks: {len(options['networks'])}")
    logger.info(f"Genres: {options['genres']}")
    logger.info(f"Sources: {options['sources']}")
    
    # Test 2: Matrix for heatmap
    logger.info("\nTest 2: Network Matrix")
    matrix, networks = analyzer.get_shared_creators_matrix()
    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"First few networks: {networks[:3]}")
    logger.info(f"Sample values:\n{matrix[:3,:3]}")
    
    # Test 3: Filter creators
    logger.info("\nTest 3: Filter Creators")
    # Try finding creators who worked at first two networks
    test_networks = networks[:2]
    creators = analyzer.filter_creators(networks=test_networks)
    logger.info(f"Creators who worked at both {test_networks}:")
    for creator in creators[:3]:  # Show first 3
        logger.info(f"  {creator['name']}: {creator['total_shows']} shows")
    
    # Test 4: Success stories
    logger.info("\nTest 4: Success Stories")
    stories = analyzer.get_success_stories(min_networks=2, top_k=3)
    logger.info("Top cross-network creators:")
    for story in stories:
        logger.info(f"  {story['name']}: {story['network_count']} networks, {story['total_shows']} shows")

if __name__ == '__main__':
    main()
