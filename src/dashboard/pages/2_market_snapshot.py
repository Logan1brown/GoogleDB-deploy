"""Market Snapshot Page

Displays market analysis and insights for TV series data using secure Supabase views.
"""

import os
import traceback
from pathlib import Path
from dataclasses import asdict, dataclass, field

import streamlit as st
import sys
import os

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from dashboard.auth.auth_required import auth_required
from src.dashboard.utils.timing import time_page
from dotenv import load_dotenv

from src.dashboard.utils.style_config import COLORS, FONTS
from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.market_analysis.market_analyzer import MarketAnalyzer
from src.dashboard.components.market_view import render_market_snapshot
from src.dashboard.state.session import get_page_state

# Load environment variables
env_path = Path(__file__).parents[3] / '.env'
load_dotenv(env_path)

@dataclass
class MarketState:
    """State for market snapshot page."""
    selected_shows: list[str] = field(default_factory=list)
    selected_creatives: list[str] = field(default_factory=list)
    selected_networks: list[str] = field(default_factory=list)
    success_filter: str = "All"

@time_page
@auth_required()
def show():
    try:
        # Verify required environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
        for var in required_vars:
            if not os.getenv(var):
                st.error(f"Missing required environment variable: {var}")
                st.stop()

        # Initialize ShowsAnalyzer and fetch data
        shows_analyzer = ShowsAnalyzer()
        titles_df, team_df, network_df = shows_analyzer.fetch_market_data(force=True)
        
        # Verify DataFrames
        if titles_df.empty or team_df.empty or network_df.empty:
            st.error("No data available from Supabase. Please check your connection and try again.")
            st.stop()

        # Page title using style from style_config
        st.markdown(
            f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; '
            f'text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">'
            f'Market Snapshot</p>', 
            unsafe_allow_html=True
        )

        # Initialize page state
        state = get_page_state("market_snapshot")
        if "market" not in state:
            state["market"] = asdict(MarketState())

        # Initialize MarketAnalyzer and render view
        render_market_snapshot(MarketAnalyzer(
            titles_df=titles_df,
            team_df=team_df,
            network_df=network_df
        ))
        
        # Update state with filter values
        market_state = state["market"]
        for filter_type in ["shows", "creatives", "networks"]:
            key = f"market_filter_{filter_type}"
            if key in st.session_state:
                market_state[f"selected_{filter_type}"] = st.session_state[key]

    except Exception as e:
        st.error(f"Error initializing market analysis: {str(e)}")
        st.stop()

if __name__ == "__main__":
    show()
