"""Test page for match details rendering."""

import streamlit as st
from data_processing.comp_analysis.comp_analyzer import CompAnalyzer
from dashboard.components import MatchDetailsManager, render_match_details
from data_processing.supabase_client import initialize_supabase

def main():
    """Main test function."""
    st.title("Match Details Test")
    
    # Initialize Supabase
    initialize_supabase()
    
    # Initialize analyzer
    comp_analyzer = CompAnalyzer()
    comp_analyzer.initialize()
    
    # Create test criteria
    criteria = {
        'genre': 1,  # Example genre ID
        'characters': [1, 2],  # Example character type IDs
        'plot': [1, 2],  # Example plot element IDs
        'source': 1,  # Example source ID
        'themes': [1, 2],  # Example theme IDs
        'tone': 1,  # Example tone ID
        'studio': [1],  # Example studio ID
        'team': [1],  # Example team ID
        'setting': {
            'time': 1,  # Example time setting ID
            'location': 1  # Example location setting ID
        },
        'format': {
            'episodes': 1,  # Example episode count ID
            'order_type': 1  # Example order type ID
        }
    }
    
    # Get test results
    results = comp_analyzer.find_by_criteria(criteria)
    if not results:
        st.error("No test results found")
        return
        
    # Create match details manager and show details
    details_manager = MatchDetailsManager(comp_analyzer)
    render_match_details(results, details_manager, criteria)

if __name__ == "__main__":
    main()
