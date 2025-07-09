#!/usr/bin/env python3
"""
Test script to measure the performance of the find_matches_with_fallback method
with the new optimizations.
"""

import time
import pandas as pd
import numpy as np
from src.data_processing.show_optimizer.optimizer_matcher import Matcher
from src.data_processing.show_optimizer.optimizer_config import OptimizerConfig

def generate_test_data(size=1000):
    """Generate test data for performance testing."""
    np.random.seed(42)
    
    # Create a DataFrame with test data
    data = {
        'show_id': range(1, size + 1),
        'title': [f'Show {i}' for i in range(1, size + 1)],
        'genre': np.random.choice(['Drama', 'Comedy', 'Action', 'Sci-Fi', 'Horror'], size),
        'network': np.random.choice(['ABC', 'NBC', 'CBS', 'HBO', 'Netflix'], size),
        'runtime': np.random.randint(20, 60, size),
        'rt_score': np.random.randint(0, 100, size),
        'tmdb_score': np.random.randint(0, 10, size),
    }
    
    # Add some NaN values to test handling of missing data
    data_df = pd.DataFrame(data)
    mask = np.random.random(size) < 0.2
    data_df.loc[mask, 'rt_score'] = np.nan
    
    mask = np.random.random(size) < 0.2
    data_df.loc[mask, 'tmdb_score'] = np.nan
    
    return data_df

def test_performance():
    """Test the performance of find_matches_with_fallback."""
    # Generate test data
    test_data = generate_test_data(5000)
    
    # Create a matcher instance
    matcher = Matcher()
    matcher.set_criteria_data(test_data)
    
    # Define test criteria
    test_criteria = {
        'genre': 'Drama',
        'network': 'HBO',
        'runtime': 30
    }
    
    # Measure execution time
    start_time = time.time()
    
    # Run the method multiple times to get a good average
    iterations = 10
    for _ in range(iterations):
        matches, confidence = matcher.find_matches_with_fallback(test_criteria)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    
    # Print results
    print(f"Average execution time: {avg_time:.4f} seconds")
    print(f"Number of matches found: {len(matches)}")
    print(f"Match levels: {confidence.get('match_counts_by_level', {})}")
    
    return avg_time, matches, confidence

if __name__ == "__main__":
    print("Testing find_matches_with_fallback performance...")
    test_performance()
