"""Match shows against TMDB with confidence scoring."""
import csv
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

from dotenv import load_dotenv
# Using Streamlit's built-in string matching now
from .tmdb_client import TMDBClient
from .tmdb_models import TVShow, TVShowDetails, Network

class MatchConfidence(Enum):
    HIGH = "HIGH"      # 75-100: Auto-approve
    MEDIUM = "MEDIUM"  # 50-74: Manual review
    LOW = "LOW"        # <50: Manual review + research

class ShowMatch(NamedTuple):
    our_show: Dict
    tmdb_show: Optional[TVShowDetails]
    confidence: MatchConfidence
    score: int
    notes: List[str]
    our_genres: str
    tmdb_genres: str

def load_shows(csv_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load shows from CSV file."""
    shows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            shows.append(row)
    return shows

def load_show_eps(show_name: str, team_csv_path: str) -> List[str]:
    """Get executive producers for a show from show_team.csv.
    
    If no roles are specified in show_team.csv, we'll assume they're all EPs
    since that's what we have in the shows.csv key_creatives field.
    """
    eps = []
    has_roles = False
    show_team = []
    
    with open(team_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['show_name'].lower() == show_name.lower():
                show_team.append(row)
                if row.get('roles'):  # Track if any roles are specified
                    has_roles = True
    
    # If no roles specified, assume all are EPs
    if not has_roles and show_team:
        eps = [row['name'].split(',')[0].strip() for row in show_team]
    else:
        # Otherwise look for EP role
        for row in show_team:
            roles = row.get('roles', '').lower()
            if ('executive producer' in roles or 
                'executive producer' in row.get('name', '').lower()):
                name = row['name'].split(',')[0]  # Remove any role suffix
                eps.append(name.strip())
    
    return sorted(list(set(eps)))  # Remove duplicates and sort

def get_tmdb_eps(credits: Dict) -> List[str]:
    """Extract executive producer names from TMDB credits."""
    return [
        person['name'] for person in credits.get('crew', [])
        if person['job'] in ['Executive Producer']
    ]

def score_title_match(our_title: str, tmdb_title: str) -> int:
    """Score title match (0-60 points)."""
    ratio = fuzz.ratio(our_title.lower(), tmdb_title.lower())
    if ratio == 100:
        return 60  # Exact match
    elif ratio >= 90:
        return 50  # Very close
    elif ratio >= 80:
        return 30  # Similar
    return 0

def score_network_match(our_network: str, tmdb_networks: List[Network]) -> int:
    """Score network match (0-25 points)."""
    our_network = our_network.lower()
    for network in tmdb_networks:
        if (our_network in network.name.lower() or 
            network.name.lower() in our_network):
            return 25
    return 0

def score_ep_matches(our_eps: List[str], tmdb_eps: List[str], max_points: int = 15) -> Tuple[int, List[str]]:
    """Score executive producer matches (0-15 points)."""
    score = 0
    matches = []
    
    # Clean and normalize names
    our_eps = [name.lower().strip() for name in our_eps]
    tmdb_eps = [name.lower().strip() for name in tmdb_eps]
    
    # Find matches using fuzzy matching
    for our_ep in our_eps:
        for tmdb_ep in tmdb_eps:
            if fuzz.ratio(our_ep, tmdb_ep) > 85:
                matches.append(f"EP Match: {our_ep} ≈ {tmdb_ep}")
                if len(matches) == 1:
                    score += 7  # First match worth more
                else:
                    score += 4  # Additional matches
                break
                
    # Bonus if we match all our EPs
    if len(matches) == len(our_eps) and our_eps:
        score += 2
        matches.append("Matched all EPs!")
        
    return min(score, max_points), matches

def get_confidence_level(score: int) -> MatchConfidence:
    """Convert score to confidence level."""
    if score >= 75:
        return MatchConfidence.HIGH
    elif score >= 50:
        return MatchConfidence.MEDIUM
    return MatchConfidence.LOW

def get_search_variations(title: str) -> List[str]:
    """Get search variations for a title."""
    variations = [title]
    
    # Handle "The" prefix/suffix
    if title.lower().startswith('the '):
        variations.append(title[4:])
    elif not title.lower().startswith('the '):
        variations.append(f'The {title}')
        
    # Remove special characters
    clean = re.sub(r'[^\w\s]', '', title)
    if clean != title:
        variations.append(clean)
        
    return list(set(variations))  # Remove duplicates

def match_show(client: TMDBClient, show: Dict, team_csv_path: str) -> ShowMatch:
    """Match a single show against TMDB."""
    title = show['shows']
    best_match = None
    best_score = -1
    best_details = None
    best_notes = []
    
    # Try variations of the title
    for search_title in get_search_variations(title):
        results = client.search_tv_show(search_title)
        if not results:
            continue
            
        # Score all results from this search
        for result in results:
            try:
                details = client.get_tv_show_details(result.id)
                credits = client.get_tv_show_credits(result.id)
                
                # Score components
                title_score = score_title_match(title, details.name)
                network_score = score_network_match(show['network'], details.networks)
                
                our_eps = load_show_eps(show['shows'], team_csv_path)
                tmdb_eps = get_tmdb_eps(credits)
                ep_score, ep_notes = score_ep_matches(our_eps, tmdb_eps)
                
                total_score = title_score + network_score + ep_score
                notes = [
                    f"Title score: {title_score}/60",
                    f"Network score: {network_score}/25",
                    f"EP score: {ep_score}/15",
                    f"Matched using search term: {search_title}"
                ] + ep_notes
                
                # Update if this is the best match so far
                if total_score > best_score:
                    best_score = total_score
                    best_match = result
                    best_details = details
                    best_notes = notes
            except Exception as e:
                print(f"Error processing result {result.id}: {e}")
                continue
    
    if not best_match:
        return ShowMatch(
            show, None, MatchConfidence.LOW, 0,
            ["No results found"], show.get('genre', ''), ''
        )
    
    # Extract genres
    our_genres = show.get('genre', '')
    tmdb_genres = ','.join([g.name for g in best_details.genres]) if best_details else ''
    
    return ShowMatch(
        show,
        best_details,
        get_confidence_level(best_score),
        best_score,
        best_notes,
        our_genres,
        tmdb_genres
    )

def main():
    # Load environment variables
    load_dotenv()
    
    # Set up paths
    base_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent  # Go up to project root
    shows_csv = base_path / "docs/sheets/STS Sales Database - shows.csv"
    team_csv = base_path / "docs/sheets/STS Sales Database - show_team.csv"
    matches_csv = base_path / "docs/sheets/tmdb_matches.csv"
    
    # Backup existing matches if they exist
    if matches_csv.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = matches_csv.parent / f"tmdb_matches_backup_{timestamp}.csv"
        import shutil
        shutil.copy2(matches_csv, backup_path)
        print(f"Backed up existing matches to {backup_path}")
    
    # Initialize client
    client = TMDBClient()
    
    # Load and process shows
    shows = load_shows(shows_csv)  # Process all shows
    
    # Process each show
    results = []
    for show in shows:
        print(f"\nProcessing: {show['shows']}")
        match = match_show(client, show, str(team_csv))
        
        # Convert match to dict for CSV
        match_dict = {
            'show_name': show['shows'],
            'tmdb_id': match.tmdb_show.id if match.tmdb_show else None,
            'confidence': match.confidence.value,
            'score': match.score,
            'match_notes': '; '.join(match.notes),
            'our_genres': match.our_genres,
            'tmdb_genres': match.tmdb_genres,
            'validated': match.score >= 75,  # Auto-validate high confidence matches
            # Add comparison fields
            'our_network': show['network'],
            'tmdb_network': ','.join(n.name for n in match.tmdb_show.networks) if match.tmdb_show else '',
            'our_eps': ','.join(load_show_eps(show['shows'], str(team_csv))),
            'tmdb_eps': ','.join(get_tmdb_eps(client.get_tv_show_credits(match.tmdb_show.id))) if match.tmdb_show else '',
            'tmdb_name': match.tmdb_show.name if match.tmdb_show else ''
        }
        results.append(match_dict)
        
        # Show results
        print(f"Confidence: {match.confidence.value}")
        print(f"Score: {match.score}/100")
        if match.tmdb_show:
            print(f"Matched to: {match.tmdb_show.name} ({match.tmdb_show.first_air_date.year if match.tmdb_show.first_air_date else 'N/A'})")
        for note in match.notes:
            print(f"- {note}")
    
    # Save results
    fieldnames = [
        'show_name', 'tmdb_id', 'confidence', 'score', 'match_notes',
        'our_genres', 'tmdb_genres', 'validated', 'our_network', 'tmdb_network',
        'our_eps', 'tmdb_eps', 'tmdb_name'
    ]
    with open(matches_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved matches to {matches_csv}")

if __name__ == "__main__":
    main()
