"""Market Snapshot Page

Displays market analysis and insights for TV series data using secure Supabase views.
"""

import streamlit as st
from dataclasses import asdict, dataclass, field
from src.shared.auth import auth_required


@dataclass
class MarketState:
    """State for market snapshot page."""
    selected_shows: list[str] = field(default_factory=list)
    selected_creatives: list[str] = field(default_factory=list)
    selected_networks: list[str] = field(default_factory=list)
    success_filter: str = "All"

@auth_required
def show():
    """Display the market snapshot page."""
    st.title("Market Snapshot")
    
    # Initialize state
    if 'market_state' not in st.session_state:
        st.session_state.market_state = MarketState()
    
    # Market overview
    st.header("Market Overview")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Platform Performance")
        st.bar_chart({"Netflix": 45, "Amazon": 30, "Hulu": 25})
    
    with col2:
        st.subheader("Genre Distribution")
        st.bar_chart({"Drama": 40, "Comedy": 35, "Action": 25})
    
    # Market trends
    st.header("Market Trends")
    st.line_chart({"Q1": 100, "Q2": 110, "Q3": 130, "Q4": 150})
