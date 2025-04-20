"""
Analyze show summaries grouped by network (read-only analysis).
"""
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from collections import defaultdict

def analyze_tone(text):
    """Analyze the tone of a piece of text."""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    # Determine primary tone
    if scores['compound'] < -0.1:
        primary = 'dark'
    elif scores['compound'] > 0.1:
        primary = 'light'
    else:
        primary = 'neutral'
    
    return {
        'primary_tone': primary,
        'intensity': abs(scores['compound']),
        'compound': scores['compound'],
        'pos': scores['pos'],
        'neg': scores['neg'],
        'neu': scores['neu']
    }

def main():
    # Download required NLTK data if needed
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('vader_lexicon')
    
    # Read shows data
    shows_path = 'docs/sheets/STS Sales Database - shows.csv'
    df = pd.read_csv(shows_path)
    
    # Filter for non-empty summaries
    df = df[df['notes'].notna()]
    
    # Group by network and analyze
    network_stats = defaultdict(lambda: {
        'show_count': 0,
        'dark_count': 0,
        'light_count': 0,
        'neutral_count': 0,
        'total_intensity': 0,
        'avg_pos': 0,
        'avg_neg': 0,
        'avg_neu': 0,
        'shows': []
    })
    
    for _, show in df.iterrows():
        network = show['network']
        analysis = analyze_tone(show['notes'])
        
        stats = network_stats[network]
        stats['show_count'] += 1
        stats['dark_count'] += analysis['primary_tone'] == 'dark'
        stats['light_count'] += analysis['primary_tone'] == 'light'
        stats['neutral_count'] += analysis['primary_tone'] == 'neutral'
        stats['total_intensity'] += analysis['intensity']
        stats['avg_pos'] += analysis['pos']
        stats['avg_neg'] += analysis['neg']
        stats['avg_neu'] += analysis['neu']
        stats['shows'].append({
            'title': show['shows'],
            'tone': analysis['primary_tone'],
            'intensity': analysis['intensity']
        })
    
    # Calculate averages and print results
    print("\nNetwork Tone Analysis\n")
    
    for network, stats in network_stats.items():
        if stats['show_count'] < 3:  # Skip networks with too few shows
            continue
            
        print(f"\nNetwork: {network}")
        print(f"Total Shows Analyzed: {stats['show_count']}")
        print("\nTone Distribution:")
        print(f"Dark: {stats['dark_count']} ({stats['dark_count']/stats['show_count']*100:.1f}%)")
        print(f"Light: {stats['light_count']} ({stats['light_count']/stats['show_count']*100:.1f}%)")
        print(f"Neutral: {stats['neutral_count']} ({stats['neutral_count']/stats['show_count']*100:.1f}%)")
        
        print("\nSentiment Averages:")
        print(f"Average Intensity: {stats['total_intensity']/stats['show_count']:.2f}")
        print(f"Average Positive: {stats['avg_pos']/stats['show_count']:.2f}")
        print(f"Average Negative: {stats['avg_neg']/stats['show_count']:.2f}")
        print(f"Average Neutral: {stats['avg_neu']/stats['show_count']:.2f}")
        
        # Show most intense shows in each category
        dark_shows = sorted([s for s in stats['shows'] if s['tone'] == 'dark'], 
                          key=lambda x: x['intensity'], reverse=True)
        light_shows = sorted([s for s in stats['shows'] if s['tone'] == 'light'],
                           key=lambda x: x['intensity'], reverse=True)
        
        if dark_shows:
            print(f"\nDarkest Show: {dark_shows[0]['title']} (intensity: {dark_shows[0]['intensity']:.2f})")
        if light_shows:
            print(f"Lightest Show: {light_shows[0]['title']} (intensity: {light_shows[0]['intensity']:.2f})")
        
        print("-" * 80)

if __name__ == "__main__":
    main()
