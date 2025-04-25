"""Component for displaying unmatched show information."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Callable, List

def render_unmatched_shows_table(
    shows: List[Dict[str, Any]], 
    on_find_matches: Callable[[Dict[str, Any]], None]
):
    """Render all unmatched shows in a scrollable table.
    
    Args:
        shows: List of show dictionaries
        on_find_matches: Callback when Find Matches is clicked
    """
    # Convert shows to DataFrame for display
    df = pd.DataFrame([
        {
            'Title': show['title'],
            'Network': show.get('network_name', ''),
            'Date': show.get('year', ''),
            'Actions': 'Find Matches',  # Placeholder for button
            '_show_data': show  # Hidden column with full show data
        } for show in shows
    ])
    
    # Display the dataframe
    st.dataframe(
        df[['Title', 'Network', 'Date', 'Actions']],  # Don't show _show_data
        hide_index=True,
        column_config={
            'Actions': st.column_config.ButtonColumn(
                'Actions',
                help='Click to find TMDB matches',
                width='small'
            )
        }
    )
    
    # Handle button clicks
    last_clicked = st.session_state.get('last_clicked_row')
    if last_clicked is not None:
        show_data = df.iloc[last_clicked]['_show_data']
        on_find_matches(show_data)
        st.session_state.last_clicked_row = None  # Reset after handling
