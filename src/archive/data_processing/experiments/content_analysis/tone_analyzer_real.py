"""
Analyze show summaries from actual data (read-only analysis).
"""
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

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
    
    # Check for complex tone (both positive and negative elements)
    tone_mix = 'complex' if scores['pos'] > 0.2 and scores['neg'] > 0.2 else 'simple'
    
    return {
        'primary_tone': primary,
        'tone_mix': tone_mix,
        'intensity': abs(scores['compound']),
        'neutrality': scores['neu'],
        'raw_scores': scores
    }

def main():
    # Download required NLTK data if needed
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('vader_lexicon')
    
    # Read shows data (just the first 5 shows with non-empty summaries)
    shows_path = 'docs/sheets/STS Sales Database - shows.csv'
    df = pd.read_csv(shows_path)
    
    # Filter for non-empty summaries and get first 5
    shows = df[df['notes'].notna()].head(5)
    
    print(f"\nAnalyzing {len(shows)} show summaries...\n")
    
    for _, show in shows.iterrows():
        print(f"\nShow: {show['shows']}")  # Using 'shows' as the title column
        print(f"Summary: {show['notes']}")
        
        analysis = analyze_tone(show['notes'])
        
        print("\nAnalysis:")
        print(f"Primary Tone: {analysis['primary_tone']}")
        print(f"Tone Mix: {analysis['tone_mix']}")
        print(f"Intensity: {analysis['intensity']:.2f}")
        print(f"Neutrality: {analysis['neutrality']:.2f}")
        print(f"Raw Scores: {analysis['raw_scores']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
