"""
Data Processing Utility
Handles data cleaning, transformation, and analysis.
"""

import pandas as pd

from data_processing.export_shows import normalize_network, normalize_source_type, normalize_genre

def clean_data(df):
    """Clean and prepare data for analysis."""
    # Convert date strings to datetime
    df['Announcement Date'] = pd.to_datetime(df['Announcement Date'])
    
    # Use existing normalization functions
    df['Network'] = df['Network'].apply(normalize_network)
    df['Source Type'] = df['Source Type'].apply(normalize_source_type)
    
    # Handle genres if present
    if 'Genre' in df.columns:
        df['Genre'] = df['Genre'].apply(normalize_genre)
    
    # Convert episodes to numeric
    df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
    
    return df

def process_sheets_data(raw_data):
    """Convert raw sheets data (list of lists) to DataFrame."""
    # First row is headers
    headers = raw_data[0]
    data_rows = raw_data[1:]
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=headers)
    
    # Clean the data
    df = clean_data(df)
    
    return df

def generate_insights(df):
    """Generate automatic insights from the data."""
    # Basic statistics and summaries
    insights = {
        'total_shows': len(df),
        'networks': df['Network'].nunique(),
        'avg_episodes': df['Episodes'].mean()
    }
    return insights

def calculate_trends(df):
    """Calculate trend data for visualizations."""
    # TODO: Implement trend calculations
    pass
