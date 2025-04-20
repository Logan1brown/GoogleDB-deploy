"""
Market Intelligence Page (Prototype)

Prototype view for market intelligence analysis.
"""

import os
import streamlit as st
import pandas as pd
from supabase import create_client
from src.dashboard.utils.style_config import COLORS, FONTS
from src.data_processing.market_analysis.market_analyzer import MarketAnalyzer
from src.dashboard.components.prototype_market_intel_view import render_market_intel
from src.dashboard.state.session import get_page_state, FilterState

# Page title
st.markdown('<p class="section-header">Market Intelligence (Prototype)</p>', unsafe_allow_html=True)

try:
    # Get page state
    state = get_page_state("market_intel")
    
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
        render_market_intel(market_analyzer.shows_df)
    
except Exception as e:
    st.error(f"Error displaying market intel: {str(e)}")
    st.info("Please ensure Supabase configuration is properly set up.")
