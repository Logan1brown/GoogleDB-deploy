"""Vertical stack grid layout.

This module provides a function to create a figure with multiple charts stacked vertically
using plotly.subplots.make_subplots.
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dashboard.templates.base import create_base_template

def create_stacked_grid(title=None, subplot_titles=None, heights=None, vertical_spacing=0.1):
    """Create a figure with vertically stacked layout.
    
    Args:
        title: Optional title for entire figure
        subplot_titles: Optional list of titles for each subplot
        heights: Optional list of proportional heights for each subplot.
                Must sum to 1. If None, equal heights are used.
        vertical_spacing: Spacing between subplots (default 0.1)
    
    Returns:
        go.Figure: Figure with stacked grid layout:
        - One column
        - Multiple rows based on number of titles/heights
        - Configurable spacing and proportions
        - Base template applied
    """
    # Determine number of rows
    n_rows = len(subplot_titles) if subplot_titles else len(heights) if heights else 1
    
    # Validate heights if provided
    if heights:
        if len(heights) != n_rows:
            raise ValueError("Number of heights must match number of subplots")
        if abs(sum(heights) - 1.0) > 0.001:  # Allow small float error
            raise ValueError("Heights must sum to 1")
    else:
        # Default to equal heights
        heights = [1/n_rows] * n_rows
    
    # Create subplot structure
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
        row_heights=heights
    )
    
    # Apply base template and layout settings
    fig.update_layout(
        template=create_base_template(),
        height=300 * n_rows,  # Scale height by number of rows
        margin=dict(t=50, b=30, l=50, r=50),  # Space for titles and labels
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=0.95  # Keep legend inside figure area
        )
    )
    
    # Add title if provided
    if title:
        fig.update_layout(title=title)
    
    return fig
