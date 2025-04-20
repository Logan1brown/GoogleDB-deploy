import pandas as pd
from pathlib import Path

def merge_validated_matches():
    # Load CSVs
    base_path = Path("/Users/loganbrown/Desktop/GoogleDB/docs/sheets")
    print(f"Looking for files in: {base_path}")
    shows_path = base_path / "STS Sales Database - shows.csv"
    matches_path = base_path / "tmdb_matches.csv"
    
    shows_df = pd.read_csv(shows_path)
    matches_df = pd.read_csv(matches_path)
    
    print("Shows columns:", shows_df.columns.tolist())
    print("Matches columns:", matches_df.columns.tolist())
    
    # Filter to only validated matches
    validated_matches = matches_df[matches_df['validated'].astype(str).str.lower() == 'true'].copy()
    
    # Create a mapping of show names to their TMDB info
    tmdb_info = {}
    for _, row in validated_matches.iterrows():
        tmdb_info[row['show_name']] = {
            'tmdb_id': row['tmdb_id'],
            'tmdb_genre': row['tmdb_genres'] if pd.notna(row['tmdb_genres']) else ''
        }
    
    # Update shows DataFrame
    updated_count = 0
    for idx, row in shows_df.iterrows():
        show_name = row['shows']
        if show_name in tmdb_info:
            shows_df.at[idx, 'tmdb_id'] = tmdb_info[show_name]['tmdb_id']
            
            # Split TMDB genres into main genre and subgenres
            genres = tmdb_info[show_name]['tmdb_genre'].split(',') if tmdb_info[show_name]['tmdb_genre'] else []
            if genres:
                shows_df.at[idx, 'genre'] = genres[0].strip()
                if len(genres) > 1:
                    shows_df.at[idx, 'subgenre'] = ', '.join(g.strip() for g in genres[1:])
            shows_df.at[idx, 'tmdb_id'] = tmdb_info[show_name]['tmdb_id']
            updated_count += 1
    
    # Drop unnamed/empty columns
    unnamed_cols = [col for col in shows_df.columns if col.startswith('Unnamed:')]
    shows_df = shows_df.drop(columns=unnamed_cols)
    
    # Save updated shows CSV
    output_path = base_path / "shows_with_tmdb.csv"
    shows_df.to_csv(output_path, index=False)
    
    print(f"Updated {updated_count} shows with TMDB information")
    print(f"Saved to: {output_path}")
    
    # Print shows that weren't matched
    unmatched = set(shows_df['shows']) - set(tmdb_info.keys())
    if unmatched:
        print("\nShows without validated TMDB matches:")
        for show in sorted(unmatched):
            print(f"- {show}")

if __name__ == "__main__":
    merge_validated_matches()
