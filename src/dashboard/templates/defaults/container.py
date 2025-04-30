"""Default template for containers.

This module provides the default template for containers following our style guide:
- Consistent padding and margins
- Border styling
- Background colors
"""

import plotly.graph_objects as go
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.templates.base import create_base_template


def create_container_defaults():
    """Create default container template.
    
    Returns:
        go.layout.Template: Template with container defaults:
        - Border styling
        - Background colors
        - Padding and margins
    """
    template = create_base_template()
    
    # Add container trace defaults
    template.layout.update(
        paper_bgcolor=COLORS['background']['primary'],
        plot_bgcolor=COLORS['background']['secondary'],
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(
            family=FONTS['primary']['family'],
            size=FONTS['primary']['sizes']['body'],
            color=COLORS['text']['primary']
        ),
        # Add border
        shapes=[{
            'type': 'rect',
            'xref': 'paper',
            'yref': 'paper',
            'x0': 0,
            'y0': 0,
            'x1': 1,
            'y1': 1,
            'line': {
                'color': COLORS['border'],
                'width': 1,
            },
            'fillcolor': 'rgba(0,0,0,0)'
        }]
    )
    
    return template
