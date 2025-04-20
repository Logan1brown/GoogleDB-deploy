"""Default template for scatter plots.

This module provides the default template for scatter plots following our style guide:
- Accent color for markers and lines
- Consistent marker size
- Hover text with x,y values
"""

import plotly.graph_objects as go
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.templates.base import create_base_template

def create_scatter_defaults():
    """Create default scatter template.
    
    Returns:
        go.layout.Template: Template with scatter defaults:
        - Marker and line styling
        - Hover template
        - Legend position
    """
    template = create_base_template()
    
    # Add scatter trace defaults
    template.data.scatter = [
        go.Scatter(
            mode="markers+lines",  # Default to both markers and lines
            marker=dict(
                color=COLORS['accent'],
                size=8,
                line=dict(
                    color='white',
                    width=1
                )
            ),
            line=dict(
                color=COLORS['accent'],
                width=2
            ),
            hovertemplate=(
                "%{x}<br>" +
                "Value: %{y:,}<br>" +
                "<extra></extra>"  # Hide secondary box
            )
        )
    ]
    
    # Update layout for proper spacing
    template.layout.update(
        margin=dict(l=50, r=50, t=50, b=50),  # Increase margins for labels
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05,  # Position legend outside plot area
        )
    )
    
    return template
