"""Test page for RT matches component."""

import streamlit as st
from rt_matches import RTMatches

# Test data
test_shows = [
    {"id": 1, "title": "The Last of Us"},
    {"id": 2, "title": "House of the Dragon"},
    {"id": 3, "title": "Andor"}
]

def handle_scores(data):
    """Handle collected scores."""
    st.session_state.collected_scores = data

# Page config
st.set_page_config(page_title="Test RT Matches")

# Initialize state
if "collected_scores" not in st.session_state:
    st.session_state.collected_scores = None

# Create and render component
rt_matches = RTMatches(test_shows, handle_scores)
rt_matches.render()

# Show collected data
if st.session_state.collected_scores:
    st.markdown("### Last Collected Scores")
    st.json(st.session_state.collected_scores)
