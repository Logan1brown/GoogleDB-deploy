"""Performance timing utilities for the dashboard."""

import streamlit as st
import time

def time_page(func):
    """Decorator to time page load performance."""
    def wrapper(*args, **kwargs):
        if 'page_metrics' not in st.session_state:
            st.session_state.page_metrics = {}
            
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        # Get page name from script path
        import inspect
        script_path = inspect.getfile(func)
        page_name = script_path.split('/')[-1].replace('.py', '').replace('_', ' ').title()
        if page_name == 'Main':
            page_name = 'Overview'
        
        # Keep only last 5 timings per page to avoid memory bloat
        if page_name not in st.session_state.page_metrics:
            st.session_state.page_metrics[page_name] = []
        st.session_state.page_metrics[page_name].append(duration)
        if len(st.session_state.page_metrics[page_name]) > 5:
            st.session_state.page_metrics[page_name].pop(0)
        
        # Show metrics in sidebar
        with st.sidebar:
            if st.checkbox("Show Page Performance"):
                st.markdown("### Page Load Times")
                for name, times in sorted(st.session_state.page_metrics.items()):
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    if avg_time < 3.0:
                        status = "GOOD"
                    elif avg_time < 8.0:
                        status = "ACCEPTABLE"
                    else:
                        status = "SLOW"
                    st.write(f"**{name}** ({status})")
                    st.write(f"avg={avg_time:.2f}s, max={max_time:.2f}s")
        
        return result
    return wrapper
