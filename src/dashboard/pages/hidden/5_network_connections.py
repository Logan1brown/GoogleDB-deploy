"""
Network Connections Page

Analyzes and visualizes network connection patterns.
"""

import os
import streamlit as st
import pandas as pd
from supabase import create_client
from src.dashboard.utils.style_config import COLORS, FONTS
from src.data_processing.market_analysis.market_analyzer import MarketAnalyzer
from src.dashboard.components.connections_view import render_network_connections_dashboard
from src.data_processing.creative_networks.connections_analyzer import analyze_network_connections
from src.dashboard.state.session import get_page_state, FilterState

# Page title
st.markdown('<p class="section-header">Network Connections Analysis</p>', unsafe_allow_html=True)

try:
    # Get page state
    state = get_page_state("network_connections")
    
    # Initialize Supabase client
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_anon_key:
        st.error("Missing Supabase configuration. Please check your environment variables.")
    else:
        # Initialize Supabase client with anon key
        supabase = create_client(supabase_url, supabase_anon_key)
        
        # Fetch data from secure view
        response = supabase.table('api_market_analysis').select('*').execute()
        market_data = pd.DataFrame(response.data)
        
        # Initialize analyzer with secure data
        market_analyzer = MarketAnalyzer(market_data)
        
        # Render view with state
        render_network_connections_dashboard(market_analyzer.shows_df)
    
except Exception as e:
    st.error(f"Error analyzing network connections: {str(e)}")
    st.info("Please ensure Supabase configuration is properly set up.")
