#!/usr/bin/env python3
"""Local script to collect RT data for shows"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from sync_collector import RTCollector

# Load environment variables from source directory
source_dir = Path('/Users/loganbrown/Desktop/GoogleDB')
env_file = source_dir / '.env'
if not env_file.exists():
    print(f"Error: .env file not found at {env_file}")
    sys.exit(1)
load_dotenv(env_file)

def collect_for_show(collector: RTCollector, show_id: int) -> bool:
    """Collect RT data for a single show."""
    result = collector.collect_show_data(show_id)
    if result['success']:
        print(f"✓ Show {show_id}: RT scores collected successfully")
        scores = result['scores']
        print(f"  Tomatometer: {scores['tomatometer']}%")
        print(f"  Audience: {scores['audience_score']}%")
        return True
    else:
        print(f"✗ Show {show_id}: {result.get('error', 'Failed to collect data')}")
        return False

def get_unmatched_shows(supabase) -> list:
    """Get all shows that need RT scores collected."""
    # Get shows that have matched RT scores
    matched_response = supabase.table('rt_success_metrics')\
        .select('show_id')\
        .execute()
    matched_ids = [row.get('show_id') for row in matched_response.data] if matched_response.data else []
    
    # Get shows without RT matches
    response = supabase.table('shows')\
        .select('id, title')\
        .not_.in_('id', matched_ids)\
        .eq('active', True)\
        .order('title')\
        .execute()
    
    return response.data

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rt_collector.log')
        ]
    )
    
    # Check required env vars
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file")
        sys.exit(1)
    
    # Initialize collector
    collector = RTCollector()
    
    # Get all unmatched shows if no IDs provided
    show_ids = []
    if len(sys.argv) > 1:
        show_ids = [int(id) for id in sys.argv[1:]]
    else:
        shows = get_unmatched_shows(collector.supabase)
        if not shows:
            print("No shows found that need RT scores")
            sys.exit(0)
        print(f"Found {len(shows)} shows that need RT scores:")
        for show in shows:
            print(f"  {show['id']}: {show['title']}")
        show_ids = [show['id'] for show in shows]
    
    # Process each show
    success_count = 0
    with collector:
        for show_id in show_ids:
            if collect_for_show(collector, show_id):
                success_count += 1
    
    # Print summary
    total = len(show_ids)
    print(f"\nCollection complete: {success_count}/{total} shows processed successfully")
    print(f"See rt_collector.log for details")

if __name__ == "__main__":
    main()
