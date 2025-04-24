"""Test script to check TMDB API responses directly."""
import os
import requests
import json

TMDB_API_KEY = '8ceb057922d27a34c69e4d3efb8442d2'

def get_tv_show_details(show_id: int):
    """Get TV show details from TMDB."""
    base_url = "https://api.themoviedb.org/3"
    endpoint = f"/tv/{show_id}"
    
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US'
    }
    
    print("Making initial request...")
    response = requests.get(f"{base_url}{endpoint}", params=params)
    response.raise_for_status()
    details = response.json()
    
    print("\nBasic show details:")
    print(json.dumps({
        'id': details.get('id'),
        'name': details.get('name'),
        'number_of_seasons': details.get('number_of_seasons'),
        'status': details.get('status')
    }, indent=2))
    
    print("\nSeasons from initial response:")
    for season in details.get('seasons', []):
        print(json.dumps({
            'season_number': season.get('season_number'),
            'episode_count': season.get('episode_count'),
            'air_date': season.get('air_date')
        }, indent=2))
    
    # Try getting season details separately
    print("\nTrying separate season request...")
    season_num = 1
    season_endpoint = f"/tv/{show_id}/season/{season_num}"
    season_response = requests.get(f"{base_url}{season_endpoint}", params=params)
    season_response.raise_for_status()
    season_details = season_response.json()
    
    print(f"\nSeason {season_num} details from separate request:")
    print(json.dumps({
        'id': season_details.get('id'),
        'name': season_details.get('name'),
        'episode_count': len(season_details.get('episodes', [])),
        'episodes': [{'episode_number': ep.get('episode_number')} for ep in season_details.get('episodes', [])]
    }, indent=2))

if __name__ == "__main__":
    get_tv_show_details(82856)  # The Mandalorian
