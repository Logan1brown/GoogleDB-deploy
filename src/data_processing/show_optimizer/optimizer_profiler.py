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

def profile(method_name: str = None):
    """Decorator to profile a method's execution time and track its callers.
    
    Args:
        method_name: Optional name to use for the method (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use provided name or function name
            name = method_name or func.__name__
            
            # Track caller if we have something in the call stack
            if st.session_state.call_stack:
                caller = st.session_state.call_stack[-1]
                if name not in st.session_state.profiling_data['callers']:
                    st.session_state.profiling_data['callers'][name] = {}
                
                if caller not in st.session_state.profiling_data['callers'][name]:
                    st.session_state.profiling_data['callers'][name][caller] = 0
                
                st.session_state.profiling_data['callers'][name][caller] += 1
            
            # Initialize tracking for this method if needed
            if name not in st.session_state.profiling_data['calls']:
                st.session_state.profiling_data['calls'][name] = 0
                st.session_state.profiling_data['times'][name] = []
            
            # Update call count
            st.session_state.profiling_data['calls'][name] += 1
            
            # Push this method to call stack
            st.session_state.call_stack.append(name)
            
            # Time the execution
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                # Record execution time
                execution_time = time.time() - start_time
                st.session_state.profiling_data['times'][name].append(execution_time)
                
                # Pop from call stack
                if st.session_state.call_stack and st.session_state.call_stack[-1] == name:
                    st.session_state.call_stack.pop()
        
        return wrapper
    return decorator

def reset_profiling():
    """Reset all profiling data."""
    st.session_state.profiling_data = {
        'calls': {},
        'times': {},
        'callers': {},
        'start_time': datetime.now()
    }
    st.session_state.call_stack = []

def display_profiling_results():
    """Display profiling results in the Streamlit UI."""
    if 'profiling_data' not in st.session_state:
        st.write("No profiling data available.")
        return
    
    data = st.session_state.profiling_data
    
    # Calculate runtime
    runtime = (datetime.now() - data['start_time']).total_seconds()
    
    st.write("## Profiling Results")
    st.write(f"Total runtime: {runtime:.2f} seconds")
    
    # Create a list of methods sorted by total time
    methods = []
    for name in data['calls'].keys():
        if data['calls'][name] > 0:
            times = data['times'][name]
            avg_time = sum(times) / len(times) if times else 0
            total_time = sum(times)
            methods.append({
                'name': name,
                'calls': data['calls'][name],
                'avg_time': avg_time,
                'total_time': total_time,
                'pct_runtime': (total_time / runtime) * 100 if runtime > 0 else 0
            })
    
    # Sort by total time
    methods.sort(key=lambda x: x['total_time'], reverse=True)
    
    # Display top methods table
    st.write("### Methods by Total Time")
    method_data = []
    for m in methods:
        method_data.append({
            "Method": m['name'],
            "Calls": m['calls'],
            "Avg Time (s)": f"{m['avg_time']:.4f}",
            "Total Time (s)": f"{m['total_time']:.4f}",
            "% of Runtime": f"{m['pct_runtime']:.1f}%"
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