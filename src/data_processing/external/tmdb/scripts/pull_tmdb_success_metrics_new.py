"""Pull TMDB success metrics and create CSVs for sheet updates."""
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data_processing.external.tmdb.tmdb_client import TMDBClient
from src.dashboard.utils.sheets_client import SheetsClient

def get_show_details(client: TMDBClient, tmdb_id: int) -> Dict:
    """Get all required TMDB data for a show."""
    show = client.get_tv_show_details(tmdb_id)
    
    # Get season 1 episode count if available
    season_1 = None
    if show.number_of_seasons > 0:
        try:
            season_1 = client.get_tv_show_season(tmdb_id, 1)
        except Exception as e:
            print(f"Warning: Could not fetch season 1 for show {tmdb_id}: {e}")
    
    season_1_episodes = season_1.get_episode_count() if season_1 else 0
    
    # Calculate average episodes per season
    avg_eps = round(show.number_of_episodes / show.number_of_seasons, 2) if show.number_of_seasons > 0 else 0
    
    # Get episode counts for all seasons
    episode_counts = []
    total_episodes = 0
    for s in range(1, show.number_of_seasons + 1):
        try:
            season = client.get_tv_show_season(tmdb_id, s)
            eps = len(season.episodes)
            episode_counts.append(eps)
            total_episodes += eps
        except Exception as e:
            print(f"Warning: Could not fetch season {s} for show {tmdb_id}: {e}")
            episode_counts.append(0)
    
    # Format episode counts as comma-separated string
    eps_str = ', '.join(str(x) for x in episode_counts)
    
    return {
        'tmdb_seasons': show.number_of_seasons,
        'tmdb_eps': eps_str,
        'tmdb_total_eps': total_episodes,
        'tmdb_avg_eps': avg_eps,
        'tmdb_status': show.status,
        'tmdb_last_air': show.last_air_date,
        'notes': show.overview,
        'order_type': 'limited' if (show.number_of_seasons == 1 and show.status != 'Returning Series') else 'ongoing',
        'status': map_tmdb_status(show),
        'episode_count': season_1_episodes
    }

def map_tmdb_status(show) -> str:
    """Map TMDB status to our status values."""
    if show.status == 'Returning Series':
        return 'active'
    elif show.status in ['Canceled', 'Cancelled']:
        return 'cancelled'
    elif show.status == 'Ended':
        return 'ended'  # Show completed its run
    elif show.status in ['Pilot', 'In Production', 'Planned']:
        return 'development'
    else:
        return 'development'  # Default to development for unknown statuses

def calculate_success_score(show_data: Dict) -> int:
    """Calculate success score (0-100) based on our metrics."""
    score = 0
    
    # Base success - Season 2 renewal
    if show_data['tmdb_seasons'] >= 2:
        score += 40
    
    # Sustained success - Additional seasons
    if show_data['tmdb_seasons'] >= 3:
        score += 20
    if show_data['tmdb_seasons'] >= 4:
        score += 20
    if show_data['tmdb_seasons'] >= 5:
        score += 20
    
    # Ending status
    if show_data['tmdb_status'] == 'Ended':
        score += 10  # Planned ending
    elif show_data['tmdb_status'] == 'Returning Series':
        score += 5   # Still going
    
    # Quick cancel penalty
    if show_data['tmdb_status'] in ['Canceled', 'Cancelled'] and show_data['tmdb_total_eps'] < 11:
        score = max(0, score - 20)
    
    return min(100, score)

def main(test_mode: bool = False):
    """Pull TMDB data and create update CSVs."""
    client = TMDBClient()
    
    if test_mode:
        # Test with a mix of shows in different states
        test_data = [
            (85271, 'WandaVision'),
            (88396, 'The Falcon and the Winter Soldier'),
            (198178, 'Wonder Man'),
            (71912, 'The Morning Show'),
            (1396, 'Breaking Bad'),
            (209366, 'Monarch: Legacy of Monsters'),
            (82856, 'The Mandalorian'),
            (114461, 'Ahsoka')
        ]
        shows_with_tmdb = pd.DataFrame(test_data, columns=['TMDB_ID', 'Title'])
    else:
        sheets_client = SheetsClient()
        shows_sheet = sheets_client.get_worksheet(sheets_client.config.shows_sheet)
        
        # Use actual sheet headers
        expected_headers = ['shows', 'key_creatives', 'network', 'studio', 'date', 'genre', 
                          'subgenre', 'episode_count', 'source_type', 'status', 'order_type', 
                          'notes', 'TMDB_ID']
        shows_data = pd.DataFrame(shows_sheet.get_all_records(expected_headers=expected_headers))
        
        # Filter to shows with TMDB IDs and rename for consistency
        shows_with_tmdb = shows_data[shows_data['TMDB_ID'].notna()].copy()
        shows_with_tmdb = shows_with_tmdb.rename(columns={'shows': 'Title'})
        
        print(f"\nProcessing {len(shows_with_tmdb)} shows with TMDB IDs:")
        print(shows_with_tmdb[['Title', 'TMDB_ID']].to_string())
    
    # Process each show
    all_show_data = []
    for _, show in shows_with_tmdb.iterrows():
        try:
            tmdb_id = int(show['TMDB_ID'])
            show_data = get_show_details(client, tmdb_id)
            show_data['success_score'] = calculate_success_score(show_data)
            show_data['TMDB_ID'] = tmdb_id
            show_data['Title'] = show.get('Title', '')
            all_show_data.append(show_data)
        except Exception as e:
            print(f"Error processing show {show.get('Title', '')} (ID: {show.get('TMDB_ID', '')}): {e}")
    
    if not all_show_data:
        print("No show data was processed successfully!")
        return
    
    # Create success metrics DataFrame
    metrics_df = pd.DataFrame([{
        'TMDB_ID': d['TMDB_ID'],
        'Title': d['Title'],
        'tmdb_seasons': d['tmdb_seasons'],
        'tmdb_eps': d['tmdb_eps'],
        'tmdb_total_eps': d['tmdb_total_eps'],
        'tmdb_avg_eps': d['tmdb_avg_eps'],
        'tmdb_status': d['tmdb_status'],
        'tmdb_last_air': d['tmdb_last_air'],
        'success_score': d['success_score']
    } for d in all_show_data])
    
    # Create shows updates DataFrame
    updates_df = pd.DataFrame([{
        'TMDB_ID': d['TMDB_ID'],
        'Title': d['Title'],
        'notes': d['notes'],
        'order_type': d['order_type'],
        'status': d['status'],
        'episode_count': d['episode_count']
    } for d in all_show_data])
    
    # Ensure output directory exists
    output_dir = Path('/Users/loganbrown/Desktop/GoogleDB/docs/sheets/TMDB csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSVs
    metrics_file = output_dir / 'success_metrics.csv'
    updates_file = output_dir / 'shows_updates.csv'
    
    metrics_df.to_csv(metrics_file, index=False)
    updates_df.to_csv(updates_file, index=False)
    
    print(f"\nFiles saved successfully:")
    print(f"1. Success Metrics: {metrics_file}")
    print(f"2. Shows Updates: {updates_file}")
    
    print("\nSuccess Metrics Preview:")
    print(metrics_df.to_string())
    print("\nShows Updates Preview:")
    print(updates_df.to_string())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode with sample shows')
    args = parser.parse_args()
    main(test_mode=args.test)
