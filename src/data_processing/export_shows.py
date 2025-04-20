#!/usr/bin/env python3

import json
import pandas as pd
from datetime import datetime

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def normalize_network(network):
    """Normalize network names to match lookup table network column exactly"""
    # First normalize to lowercase for case-insensitive matching
    if not network:
        return ''
    
    input_lower = network.lower()
    
    # Map of lowercase network names to their correct casing
    network_map = {
        # Broadcast Networks
        'abc network': 'ABC',
        'american broadcasting company': 'ABC',
        'abc': 'ABC',
        'cbs network': 'CBS',
        'columbia broadcasting': 'CBS',
        'cbs': 'CBS',
        'fox': 'Fox',
        'fox network': 'Fox',
        'fox broadcasting': 'Fox',
        'nbc network': 'NBC',
        'national broadcasting': 'NBC',
        'nbc': 'NBC',
        'the cw': 'CW',
        'cw network': 'CW',
        'the cw network': 'CW',
        'cw': 'CW',
        
        # Cable Networks
        'amc+': 'AMC',
        'amc plus': 'AMC',
        'amc': 'AMC',
        'fx': 'FX',
        'fx network': 'FX',
        'fx on hulu': 'FX',
        'fx original': 'FX',
        'usa network': 'USA',
        'usa original': 'USA',
        'usa': 'USA',
        'hallmark channel': 'Hallmark+ ',  # Note the space
        'hallmark media': 'Hallmark+ ',
        'hallmark': 'Hallmark+ ',
        'hallmark+': 'Hallmark+ ',
        'crown media': 'Hallmark+ ',
        'tbs': 'TBS',
        'turner broadcasting': 'TBS',
        'tnt': 'TNT',
        'turner network television': 'TNT',
        
        # Premium Networks
        'canal+': 'Canal Plus',
        'canal plus': 'Canal Plus',
        'canal plus group': 'Canal Plus',
        'studiocanal': 'Canal Plus',
        'epix': 'Epix',
        'mgm+': 'Epix',
        'hbo max': 'HBO',
        'hbomax': 'HBO',
        'max': 'HBO',
        'hbo original': 'HBO',
        'hbo': 'HBO',
        'sho': 'Showtime',
        'showtime original': 'Showtime',
        'showtime': 'Showtime',
        'starz': 'Starz',
        
        # Streaming Services
        'apple': 'Apple TV+',
        'apple tv': 'Apple TV+',
        'apple+': 'Apple TV+',
        'apple original': 'Apple TV+',
        'apple tv plus': 'Apple TV+',
        'apple tv+': 'Apple TV+',
        'disney plus': 'Disney+',
        'd+': 'Disney+',
        'disney original': 'Disney+',
        'disney+': 'Disney+',
        'hulu+': 'Hulu',
        'hulu original': 'Hulu',
        'hulu': 'Hulu',
        'nflx': 'Netflix',
        'netflix+': 'Netflix',
        'netflix': 'Netflix',
        'paramount plus': 'Paramount+',
        'p+': 'Paramount+',
        'paramount+': 'Paramount+',
        'peacock+': 'Peacock',
        'peacock original': 'Peacock',
        'peacock': 'Peacock',
        'amazon': 'Prime Video',
        'amazon prime': 'Prime Video',
        'amazon prime video': 'Prime Video',
        'amazon original': 'Prime Video',
        'amazon studios': 'Prime Video',
        'prime video': 'Prime Video',
        'bet+': 'BET',
        'bet plus': 'BET',
        'bet network': 'BET',
        'black entertainment television': 'BET',
        'paramount bet plus': 'BET',
        'bet': 'BET',
        
        # Digital Platforms
        'blackpills': 'Blackpills',
        'black pills': 'Blackpills',
        
        # Other
        'nacelle': 'Nacelle',
        'the nacelle company': 'Nacelle',
        'nacelle company': 'Nacelle'
    }
    
    # Look up using lowercase input
    return network_map.get(input_lower, network)  # Keep original if not in map

def normalize_genre(genre):
    """Normalize genre names to match lookup table"""
    genre_map = {
        'Sci-Fi': 'Science Fiction',  # Must be full name
        'Science Fiction': 'Science Fiction',
        'Comic Book': 'Superhero',
        'True Crime': 'Crime',
        'Post-Apocalyptic': 'Science Fiction',
        'Techno': 'Science Fiction',
        'Espionage': 'Spy',
        'Dark Comedy': 'Comedy',
        'Live-Action': '',
        'Limited Series': '',
        'Contemporary': '',
        'Gaming': 'Science Fiction',
        'Young Adult': 'Drama',
        'Political': 'Drama',
        'Adventure': 'Action',
        'Western': 'Drama',  # Merge into main genres
        'Horror': 'Drama',
        'Spy': 'Drama',
        'Family': 'Comedy'
    }
    normalized = genre_map.get(genre, genre)
    return normalized if normalized else ''

def normalize_source_type(source):
    """Normalize source material types to match lookup table column A exactly"""
    source = source.lower()
    source_map = {
        # Original
        'new ip': 'Original',
        'original content': 'Original',
        'original': 'Original',
        
        # Literary
        'novel': 'Book',
        'fiction': 'Book',
        'non-fiction': 'Book',
        'book': 'Book',
        'story': 'Short Story',
        'novella': 'Short Story',
        'short story': 'Short Story',
        
        # Graphic
        'comic book': 'Comic',
        'graphic novel': 'Comic',
        'comic': 'Comic',
        'japanese comic': 'Manga',
        'manga': 'Manga',
        'web comic': 'Webtoon',
        'digital comic': 'Webtoon',
        'webtoon': 'Webtoon',
        
        # Adaptation
        'movie': 'Film',
        'feature film': 'Film',
        'film': 'Film',
        'television series': 'TV Show',
        'tv series': 'TV Show',
        'tv show': 'TV Show',
        'reboot': 'TV Show',
        'remake': 'TV Show',
        'spinoff': 'TV Show',
        'animation series': 'TV Show',
        'video game': 'Game',
        'mobile game': 'Game',
        'game': 'Game',
        'audio series': 'Podcast',
        'audio show': 'Podcast',
        'podcast': 'Podcast',
        
        # Real Events
        'biography': 'True Story',
        'real life event': 'True Story',
        'true story': 'True Story',
        'current events': 'News Story',
        'news adaptation': 'News Story',
        'news story': 'News Story',
        'magazine story': 'Article',
        'news article': 'Article',
        'article': 'Article'
    }
    return source_map.get(source, source)  # Keep original if not in map

def map_show_to_sheet_format(show):
    """Map a single show to the sheets format"""
    # Get original network name and normalize it
    network_name = show['network'][0] if show['network'] else ''
    network = normalize_network(network_name)
    
    # Merge creators and talent for key_creatives
    creatives = []
    if show.get('creators'):
        creatives.extend(show['creators'])
    if show.get('talent'):
        creatives.extend(show['talent'])
    key_creatives = ', '.join(creatives) if creatives else ''
    
    # Handle genres
    genres = [normalize_genre(g) for g in show.get('genres', []) if normalize_genre(g)]
    main_genre = genres[0] if genres else ''
    subgenres = ', '.join(genres[1:]) if len(genres) > 1 else ''
    
    # Get source type from source_material
    source_material = show.get('source_material', ['Original'])[0]
    source_type = normalize_source_type(source_material)
    
    return {
        'show_name': show['show_name'],
        'key_creatives': key_creatives,
        'network': network,  # Use normalized network name
        'studio': show.get('studio_seller', [''])[0] if show.get('studio_seller') else '',
        'date': show.get('date', ''),
        'genre': main_genre,
        'subgenre': subgenres,
        'episode_count': show.get('episode_count', ''),
        'source_type': source_type,
        'status': '',  # Empty by default
        'order_type': '',  # Empty by default
        'notes': ''
    }

def main():
    # Load data
    data = load_json_data('sample_data/processed/structured_deals.json')
    
    # Process all shows
    test_shows = data['shows']
    
    # Map shows to sheet format
    sheet_data = [map_show_to_sheet_format(show) for show in test_shows]
    
    # Convert to DataFrame
    df = pd.DataFrame(sheet_data)
    
    # Replace NaN with empty string
    df = df.fillna('')
    
    # Save to CSV
    output_file = 'sample_data/processed/shows_to_import.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(test_shows)} shows to {output_file}")
    print("\nPreview of data:")
    print(df.to_string())

if __name__ == "__main__":
    main()
