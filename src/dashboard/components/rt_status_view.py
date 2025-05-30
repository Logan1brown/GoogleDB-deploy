"""RT Status Dashboard

Shows current status of RT data collection across all shows.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from ..services.supabase import get_supabase_client

def get_rt_metrics():
    """Get RT matching metrics."""
    supabase = get_supabase_client()
    
    # Get total shows
    total_response = supabase.table('shows') \
        .select('id', count='exact') \
        .execute()
    total_shows = total_response.count if total_response.count is not None else 0

    # Get matched shows
    matched_response = supabase.table('rt_success_metrics') \
        .select('show_id', count='exact') \
        .eq('is_matched', True) \
        .execute()
    matched_count = matched_response.count if matched_response.count is not None else 0

    # Get failed matches
    failed_response = supabase.table('rt_match_status') \
        .select('show_id', count='exact') \
        .eq('status', 'error') \
        .execute()
    failed_count = failed_response.count if failed_response.count is not None else 0

    # Calculate metrics
    unmatched_count = total_shows - matched_count
    match_rate = (matched_count / total_shows * 100) if total_shows > 0 else 0

    return {
        'total_shows': total_shows,
        'matched_count': matched_count,
        'unmatched_count': unmatched_count,
        'failed_count': failed_count,
        'match_rate': match_rate
    }

def get_show_statuses():
    """Get detailed RT status for all shows."""
    supabase = get_supabase_client()
    
    # Get all shows
    shows = supabase.table('shows') \
        .select('id,title') \
        .execute()
    
    # Get RT success metrics
    metrics = supabase.table('rt_success_metrics') \
        .select('*') \
        .execute()
    
    # Get match status
    status = supabase.table('rt_match_status') \
        .select('*') \
        .execute()
    
    # Convert to dataframes
    shows_df = pd.DataFrame(shows.data)
    metrics_df = pd.DataFrame(metrics.data) if metrics.data else pd.DataFrame()
    status_df = pd.DataFrame(status.data) if status.data else pd.DataFrame()
    
    # Merge data
    result = shows_df.merge(metrics_df, how='left', left_on='id', right_on='show_id')
    result = result.merge(status_df, how='left', left_on='id', right_on='show_id')
    
    # Add status column
    def get_status(row):
        if pd.notna(row['tomatometer']):
            return 'Matched'
        elif row['status'] == 'error':
            return 'Failed'
        elif pd.notna(row['attempts']) and row['attempts'] > 0:
            return 'Not Found'
        else:
            return 'Pending'
            
    result['rt_status'] = result.apply(get_status, axis=1)
    
    return result

def render_rt_status():
    """Render the RT status view component."""
    st.subheader("RT Data Collection Status")
    
    # Info about standalone app
    st.info(
        "⚠️ This is a view-only dashboard. To collect RT scores, use the standalone RT collector app:\n\n"
        "```bash\nstreamlit run src/data_processing/external/rt/collect_rt_app.py\n```"
    )
    
    # Get metrics
    metrics = get_rt_metrics()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shows", metrics['total_shows'])
    with col2:
        st.metric("Matched", metrics['matched_count'])
    with col3:
        st.metric("Unmatched", metrics['unmatched_count'])
    with col4:
        st.metric("Failed", metrics['failed_count'])
        
    st.metric("Match Rate", f"{metrics['match_rate']:.1f}%")
    
    # Get show statuses
    shows = get_show_statuses()
    
    # Status filter
    status_filter = st.selectbox(
        "Filter by Status",
        ["All Shows", "Pending", "Matched", "Failed", "Not Found"]
    )
    
    # Filter data
    if status_filter != "All Shows":
        shows = shows[shows['rt_status'] == status_filter]
    
    # Display shows table
    if len(shows) > 0:
        st.dataframe(
            shows[[
                'title', 'rt_status', 'tomatometer', 'popcornmeter',
                'attempts', 'error_details', 'manual_url'
            ]].sort_values('title'),
            hide_index=True
        )
    else:
        st.info("No shows found with selected status.")


