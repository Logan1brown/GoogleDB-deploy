"""
Quick proof of concept for analyzing show summary tones using NLTK.
"""
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Test summaries - we can replace these with real ones later
test_summaries = [
    {
        "title": "Dark Mystery",
        "summary": "A cynical detective returns to her small hometown to investigate a series of mysterious disappearances."
    },
    {
        "title": "Light Comedy",
        "summary": "A heartwarming story about a struggling baker who finds love and success in an unexpected place."
    },
    {
        "title": "Mixed Drama",
        "summary": "Despite tragic losses, a determined family fights to keep their dream alive while uncovering shocking secrets."
    }
]

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
    # Download required NLTK data
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('vader_lexicon')
    
    print("\nAnalyzing show summaries...\n")
    
    for show in test_summaries:
        print(f"\nShow: {show['title']}")
        print(f"Summary: {show['summary']}")
        
        analysis = analyze_tone(show['summary'])
        
        print("\nAnalysis:")
        print(f"Primary Tone: {analysis['primary_tone']}")
        print(f"Tone Mix: {analysis['tone_mix']}")
        print(f"Intensity: {analysis['intensity']:.2f}")
        print(f"Neutrality: {analysis['neutrality']:.2f}")
        print(f"Raw Scores: {analysis['raw_scores']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
