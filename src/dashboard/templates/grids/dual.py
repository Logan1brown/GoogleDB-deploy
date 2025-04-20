"""Side-by-side grid layout.

This module provides a function to create a figure with two charts side by side
using plotly.subplots.make_subplots.

Usage:
    from src.dashboard.templates.grids import create_dual_grid
    from src.dashboard.templates.defaults import create_bar_defaults
    
    # Create figure with side-by-side layout
    fig = create_dual_grid(
        title="My Dashboard",
        left_title="Left Chart",
        right_title="Right Chart"
    )
    
    # Add traces to specific grid positions
    fig.add_bar(x=x1, y=y1, row=1, col=1)
    fig.add_scatter(x=x2, y=y2, row=1, col=2)
    
    # Apply any additional styling if needed
    fig.update_layout(height=600)
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.dashboard.templates.base import create_base_template

def create_dual_grid(title=None, left_title=None, right_title=None):
    """Create a figure with side-by-side layout.
    
    Args:
        title: Optional title for entire figure
        left_title: Optional title for left subplot
        right_title: Optional title for right subplot
    
    Returns:
        go.Figure: Figure with dual grid layout:
        - Two equal width columns
        - 10% horizontal spacing
        - Optional subplot titles
        - Base template applied
    """
    # Create subplot structure
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.1,
        subplot_titles=(left_title, right_title) if left_title or right_title else None
    )
    
    # Apply base template and layout settings
    fig.update_layout(
        template=create_base_template(),
        height=500,  # Standard height
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
