"""Default template for heatmaps.

This module provides the default template for heatmaps following our style guide:
- Primary colorscale: 'Viridis'
- Consistent axis styling
- Hover text with values
"""

import plotly.graph_objects as go
from src.dashboard.utils.style_config import COLORS, CHART_DEFAULTS
from src.dashboard.templates.base import create_base_template

def create_heatmap_defaults():
    """Create default heatmap template.
    
    Returns:
        go.layout.Template: Template with heatmap defaults:
        - Viridis colorscale
        - Axis styling
        - Hover template
    """
    template = create_base_template()
    
    # Add heatmap trace defaults
    template.data.heatmap = [
        go.Heatmap(
            colorscale=CHART_DEFAULTS['colorscales']['primary'],
            hoverongaps=False,  # Don't show hover for missing values
            hovertemplate=(
                "%{x}<br>" +
                "%{y}<br>" +
                "Value: %{z}<br>" +
                "<extra></extra>"  # Hide secondary box
            )
        )
    ]
    
    return template
