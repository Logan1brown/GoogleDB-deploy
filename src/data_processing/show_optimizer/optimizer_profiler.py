"""
Simple Profiler for Show Optimizer

A lightweight profiling tool to track method execution times and call stacks.
Designed to work in Streamlit Cloud environment.
"""

import time
import functools
import streamlit as st
from typing import Dict, List, Callable, Any
from datetime import datetime

# Function to safely initialize session state
def ensure_session_state_initialized():
    """Ensure that all required session state variables are initialized."""
    # Global profiling data storage
    if 'profiling_data' not in st.session_state:
        st.session_state.profiling_data = {
            'calls': {},        # Method name -> call count
            'times': {},        # Method name -> list of execution times
            'callers': {},      # Method name -> dict of caller methods and counts
            'start_time': datetime.now()
        }

    # Call stack to track caller-callee relationships
    if 'call_stack' not in st.session_state:
        st.session_state.call_stack = []

# Initialize session state when module is loaded
ensure_session_state_initialized()

def profile(method_name: str = None):
    """Decorator to profile a method's execution time and track its callers.
    
    Args:
        method_name: Optional name for the method. If None, uses the method's __name__
        
    Returns:
        Decorated method with profiling
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal method_name
            
            # Ensure session state is initialized
            ensure_session_state_initialized()
            
            # Use function name if method_name not provided
            if method_name is None:
                method_name = func.__name__
                
            # Get caller from call stack
            caller = "root"
            if hasattr(st.session_state, 'call_stack') and st.session_state.call_stack:
                caller = st.session_state.call_stack[-1]
                
            # Record call in profiling data
            if not hasattr(st.session_state, 'profiling_data'):
                ensure_session_state_initialized()
                
            if method_name not in st.session_state.profiling_data['calls']:
                st.session_state.profiling_data['calls'][method_name] = 0
                st.session_state.profiling_data['times'][method_name] = []
                st.session_state.profiling_data['callers'][method_name] = {}
                
            st.session_state.profiling_data['calls'][method_name] += 1
            
            # Record caller
            if caller not in st.session_state.profiling_data['callers'][method_name]:
                st.session_state.profiling_data['callers'][method_name][caller] = 0
            st.session_state.profiling_data['callers'][method_name][caller] += 1
            
            # Add to call stack
            if hasattr(st.session_state, 'call_stack'):
                st.session_state.call_stack.append(method_name)
            
            # Execute the method and time it
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Record execution time
                execution_time = time.time() - start_time
                if hasattr(st.session_state, 'profiling_data'):
                    st.session_state.profiling_data['times'][method_name].append(execution_time)
                
                # Remove from call stack
                if hasattr(st.session_state, 'call_stack') and st.session_state.call_stack and st.session_state.call_stack[-1] == method_name:
                    st.session_state.call_stack.pop()
                    
        return wrapper
    return decorator

def reset_profiling():
    """Reset all profiling data."""
    # Ensure session state is initialized first
    ensure_session_state_initialized()
    
    # Reset the profiling data
    st.session_state.profiling_data = {
        'calls': {},
        'times': {},
        'callers': {},
        'start_time': datetime.now()
    }
    st.session_state.call_stack = []

def display_profiling_results():
    """Display profiling results in the Streamlit UI."""
    # Ensure session state is initialized first
    ensure_session_state_initialized()
    
    if not hasattr(st.session_state, 'profiling_data') or not st.session_state.profiling_data:
        st.write("No profiling data available.")
        return
    
    data = st.session_state.profiling_data
    
    # Calculate runtime
    runtime = (datetime.now() - data['start_time']).total_seconds()
    
    st.write("## Profiling Results")
    st.write(f"Total runtime: {runtime:.2f} seconds")
    
    # Create a list of methods sorted by total time
    methods = []
    total_method_time = 0
    for name in data['calls'].keys():
        if data['calls'][name] > 0:
            times = data['times'][name]
            avg_time = sum(times) / len(times) if times else 0
            total_time = sum(times)
            total_method_time += total_time
            methods.append({
                'name': name,
                'calls': data['calls'][name],
                'avg_time': avg_time,
                'total_time': total_time,
                'raw_time': total_time  # Store raw time for percentage calculation later
            })
    
    # Sort by total time
    methods.sort(key=lambda x: x['total_time'], reverse=True)
    
    # Calculate percentages based on total method time instead of wall clock time
    # This ensures percentages add up to 100% and are more meaningful
    for m in methods:
        if total_method_time > 0:
            m['pct_runtime'] = (m['raw_time'] / total_method_time) * 100
        else:
            m['pct_runtime'] = 0
    
    # Display top methods table
    st.write("### Methods by Total Time")
    method_data = []
    for m in methods:
        method_data.append({
            "Method": m['name'],
            "Calls": m['calls'],
            "Avg Time (s)": f"{m['avg_time']:.4f}",
            "Total Time (s)": f"{m['total_time']:.4f}",
            "% of Method Time": f"{m['pct_runtime']:.1f}%"
        })
    
    st.table(method_data)
    
    # Display caller relationships for top methods
    st.write("### Caller Analysis")
    for m in methods[:5]:  # Top 5 methods
        name = m['name']
        if name in data['callers'] and data['callers'][name]:
            st.write(f"**{name}** was called by:")
            
            # Sort callers by call count
            callers = []
            for caller, count in data['callers'][name].items():
                callers.append({
                    "Caller": caller,
                    "Calls": count,
                    "% of Calls": f"{(count / m['calls']) * 100:.1f}%"
                })
            
            callers.sort(key=lambda x: x['Calls'], reverse=True)
            st.table(callers)
    
    # Reset button is now in the sidebar Developer Options panel