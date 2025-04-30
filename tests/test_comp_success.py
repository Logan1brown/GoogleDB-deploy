"""Test the integration of CompAnalyzer with SuccessAnalyzer."""

import streamlit as st
from src.data_processing.comp_analysis import CompAnalyzer

def test_comp_success():
    """Test comp analyzer with success metrics."""
    
    # Initialize analyzer
    analyzer = CompAnalyzer()
    
    # Fetch data and check columns
    comp_data = analyzer.shows_analyzer.fetch_comp_data()
    print("\nColumns in comp_data:")
    print(comp_data.columns.tolist())
    
    # Check market data columns
    titles_df, _, _ = analyzer.shows_analyzer.fetch_market_data()
    print("\nColumns in market data:")
    print(titles_df.columns.tolist())
    print(f"\nLoaded {len(comp_data)} shows")
    
    # Get some test shows
    test_shows = comp_data.head(3)
    print("\nTest shows:")
    for _, show in test_shows.iterrows():
        print(f"- {show['title']}")
        
    # Test similar shows with success metrics
    for _, show in test_shows.iterrows():
        print(f"\nFinding similar shows for: {show['title']}")
        similar = analyzer.get_similar_shows(show['id'], limit=5)
        
        print("\nTop 5 matches:")
        for match in similar:
            print(f"- {match['title']}")
            print(f"  Comp Score: {match['comp_score'].total:.1f}")
            print(f"  Success Score: {match['success_score']:.1f}")
            print(f"  Content Match: {match['comp_score'].content_score:.1f}")
            print(f"  Production Match: {match['comp_score'].production_score:.1f}")
            print(f"  Format Match: {match['comp_score'].format_score:.1f}")
            
if __name__ == "__main__":
    test_comp_success()
