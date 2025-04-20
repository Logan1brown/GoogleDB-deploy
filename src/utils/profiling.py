"""Profiling utilities for performance monitoring."""
import time
import functools
import streamlit as st
from typing import Dict, List, Optional

class ViewProfiler:
    """Tracks execution time of view components."""
    
    def __init__(self):
        if 'profiling_data' not in st.session_state:
            st.session_state.profiling_data = {
                'function_times': {},  # function name -> list of execution times
                'query_times': {},     # query name -> list of execution times
                'component_times': {}   # component name -> list of render times
            }
    
    @staticmethod
    def profile(component_name: str):
        """Decorator to profile a function's execution time."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Store timing data
                times = st.session_state.profiling_data['function_times'].setdefault(component_name, [])
                times.append(duration)
                
                # Log if slow
                if duration > 1.0:  # More than 1 second
                    print(f"⚠️ Slow component detected: {component_name} took {duration:.2f}s")
                
                return result
            return wrapper
        return decorator

    @staticmethod
    def profile_query(query_name: str):
        """Decorator to profile database query execution time."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Store query timing
                times = st.session_state.profiling_data['query_times'].setdefault(query_name, [])
                times.append(duration)
                
                # Log if slow
                if duration > 2.0:  # More than 2 seconds
                    print(f"🔴 Slow query detected: {query_name} took {duration:.2f}s")
                
                return result
            return wrapper
        return decorator

    @staticmethod
    def show_profiling_data():
        """Display profiling data in Streamlit."""
        if st.checkbox("Show Performance Data"):
            st.write("### Performance Profile")
            
            # Show function times
            st.write("#### Component Times")
            for func_name, times in st.session_state.profiling_data['function_times'].items():
                avg_time = sum(times) / len(times)
                max_time = max(times)
                st.write(f"- {func_name}: avg={avg_time:.2f}s, max={max_time:.2f}s")
            
            # Show query times
            st.write("#### Query Times")
            for query_name, times in st.session_state.profiling_data['query_times'].items():
                avg_time = sum(times) / len(times)
                max_time = max(times)
                st.write(f"- {query_name}: avg={avg_time:.2f}s, max={max_time:.2f}s")

# Global profiler instance
profiler = ViewProfiler()

# Example usage:
"""
from src.utils.profiling import profiler

@profiler.profile("market_view")
def render_market_view():
    ...

@profiler.profile_query("fetch_market_data")
def get_market_data():
    ...

# At bottom of page:
profiler.show_profiling_data()
"""
