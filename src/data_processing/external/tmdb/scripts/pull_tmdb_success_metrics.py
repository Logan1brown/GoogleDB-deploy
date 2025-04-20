"""Pull TMDB success metrics and create CSVs for sheet updates."""
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data_processing.external.tmdb.tmdb_client import TMDBClient
from src.dashboard.utils.sheets_client import SheetsClient

def get_show_details(client: TMDBClient, tmdb_id: int):
    """Get all required TMDB data for a show."""
    show = client.get_tv_show_details(tmdb_id)
    
    # Get season 1 episode count
    season_1 = client.get_tv_show_season(tmdb_id, 1) if show.number_of_seasons > 0 else None
    season_1_episodes = season_1.get_episode_count() if season_1 else 0
    
    # Calculate average episodes per season
    avg_eps = round(show.number_of_episodes / show.number_of_seasons, 2) if show.number_of_seasons > 0 else 0
    
    return {
        # TMDB Success Metrics
        'tmdb_seasons': show.number_of_seasons,
        'tmdb_eps': str([len(client.get_tv_show_season(tmdb_id, s).episodes) 
                        for s in range(1, show.number_of_seasons + 1)]),  # List of episodes per season
        'tmdb_total_eps': show.number_of_episodes,
        'tmdb_avg_eps': avg_eps,
        'tmdb_status': show.status,
        'tmdb_last_air': show.last_air_date,
        
        # Shows Sheet Updates
        'notes': show.overview,
        'order_type': 'limited' if (show.number_of_seasons == 1 and show.status != 'Returning Series') else 'ongoing',
        'status': map_tmdb_status(show),
        'episode_count': season_1_episodes
    }

def map_tmdb_status(show) -> str:
    """Map TMDB status to our status values.
    
    Rules:
    - If show has aired any episodes -> production
    - If show is planned/announced but no episodes -> development
    - If show is ended/cancelled -> cancelled
    - If show is returning -> active
    """
    if show.status == 'Returning Series':
        return 'active'
    elif show.status in ['Ended', 'Canceled', 'Cancelled']:
        return 'cancelled'
    elif show.number_of_episodes > 0:
        return 'production'
    elif show.status in ['Planned', 'In Production']:
        return 'development'
    return 'unknown'

def calculate_success_score(show_data: dict) -> int:
    """Calculate success score based on our metrics."""
    score = 0
    
    # Base success - Season 2
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
        score += 10
    elif show_data['tmdb_status'] == 'Returning Series':
        score += 5
    
    # Quick cancel penalty
    if show_data['tmdb_status'] in ['Canceled', 'Cancelled'] and show_data['tmdb_total_eps'] < 11:
        score = max(0, score - 20)
    
    return min(100, score)

def main(test_mode=False):
    """Pull TMDB data and create update CSVs."""
    client = TMDBClient()
    
    if test_mode:
        # Test with a mix of shows in different states
        test_ids = [
            85271,   # WandaVision (Limited, Production)
            88396,   # Falcon and Winter Soldier (Limited, Production)
            198178,  # Wonder Man (Development)
            71912,   # The Morning Show (Active)
            1396,    # Breaking Bad (Cancelled/Ended)
            209366,  # Monarch: Legacy of Monsters (New Production)
            82856,   # The Mandalorian (Active, Multiple Seasons)
            114461   # Ahsoka (Production, Single Season)
        ]
        shows_with_tmdb = pd.DataFrame({'TMDB_ID': test_ids})
    else:
        sheets_client = SheetsClient()
        shows_sheet = sheets_client.get_worksheet(sheets_client.config.shows_sheet)
        # Use actual sheet headers
        expected_headers = ['shows', 'key_creatives', 'network', 'studio', 'date', 'genre', 
                          'subgenre', 'episode_count', 'source_type', 'status', 'order_type', 
                          'notes', 'TMDB_ID']
        shows_data = pd.DataFrame(shows_sheet.get_all_records(expected_headers=expected_headers))
        
        # Rename 'shows' column to 'Title' for consistency
        shows_data = shows_data.rename(columns={'shows': 'Title'})
        shows_with_tmdb = shows_data[shows_data['TMDB_ID'].notna()].head(10)
        print(f"\nTesting with {len(shows_with_tmdb)} shows:")
        print(shows_with_tmdb[['Title', 'TMDB_ID']].to_string())
    
    all_show_data = []
    for _, show in shows_with_tmdb.iterrows():
        tmdb_id = int(show['TMDB_ID'])
        show_data = get_show_details(client, tmdb_id)
        show_data['success_score'] = calculate_success_score(show_data)
        all_show_data.append(show_data)
    
    # Create success metrics DataFrame with exact column order
    metrics_df = pd.DataFrame([{
        'TMDB_ID': show['TMDB_ID'],
        'link': show['Title'],
        'tmdb_seasons': d['tmdb_seasons'],
        'tmdb_eps': d['tmdb_eps'],
        'tmdb_total_eps': d['tmdb_total_eps'],
        'tmdb_avg_eps': d['tmdb_avg_eps'],
        'tmdb_status': d['tmdb_status'],
        'tmdb_last_air': d['tmdb_last_air'],
        'success_score': d['success_score']
    } for d, show in zip(all_show_data, shows_with_tmdb.to_dict('records'))])[[
        'TMDB_ID',
        'link',
        'tmdb_seasons',
        'tmdb_eps',
        'tmdb_total_eps',
        'tmdb_avg_eps',
        'tmdb_status',
        'tmdb_last_air',
        'success_score'
    ]]
    
    # Create shows updates DataFrame
    updates_df = pd.DataFrame([{
        'notes': d['notes'],
        'order_type': d['order_type'],
        'status': d['status'],
        'episode_count': d['episode_count'],
        'success_score': d['success_score']
    } for d in all_show_data])
    
    # Save to CSVs in docs/sheets/TMDB csv
    # Use absolute path to avoid src/ prefix issue
    output_dir = Path('/Users/loganbrown/Desktop/GoogleDB/docs/sheets/TMDB csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_df.to_csv(output_dir / 'success_metrics.csv', index=False)
    updates_df.to_csv(output_dir / 'shows_updates.csv', index=False)
    print(f"\nCSVs saved to {output_dir}")
    
    print("\nTMDB Success Metrics:")
    print(metrics_df)
    print("\nShows Updates:")
    print(updates_df)

if __name__ == '__main__':
    main()
