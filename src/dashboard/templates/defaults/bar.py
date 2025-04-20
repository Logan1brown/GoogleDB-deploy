"""Default styles for bar charts.

This module provides default styling for bar charts, including:
- Colors from style_config
- Hover templates
- Legend settings

Usage:
    from src.dashboard.templates.defaults import create_bar_defaults
    
    fig = go.Figure(template=create_bar_defaults())
    fig.add_bar(x=categories, y=values)
"""

import plotly.graph_objects as go
from src.dashboard.utils.style_config import COLORS
from src.dashboard.templates.base import create_base_template

def create_bar_defaults():
    """Create template with bar chart defaults.
    
    Returns:
        go.layout.Template: Template with bar chart defaults:
        - Accent color for bars
        - Hover template with counts and percentages
        - Auto-sorting by value
        - Vertical orientation
    """
    # Start with base template
    template = create_base_template()
    
    # Add bar trace defaults
    template.data.bar = [
        go.Bar(
            marker_color=COLORS['accent'],
            hovertemplate="%{x}<br>" +
                         "Value: %{y:,}<br>" +
                         "<extra></extra>",  # Hide secondary box
            textposition="auto",
            texttemplate="%{y:,}",
            showlegend=False  # Single color bars don't need legend
        )
    ]
    
    return template
