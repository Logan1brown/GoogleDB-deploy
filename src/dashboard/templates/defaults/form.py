"""Default template for forms.

This module provides the default template for forms following our style guide:
- Consistent input styling
- Section headers
- Help text formatting
"""

import plotly.graph_objects as go
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.templates.base import create_base_template


def create_form_defaults():
    """Create default form template.
    
    Returns:
        go.layout.Template: Template with form defaults:
        - Input styling
        - Section headers
        - Help text
    """
    template = create_base_template()
    
    # Add form trace defaults
    template.layout.update(
        # Form container styling
        paper_bgcolor=COLORS['background']['primary'],
        plot_bgcolor=COLORS['background']['secondary'],
        margin=dict(l=20, r=20, t=20, b=20),
        
        # Text styling
        font=dict(
            family=FONTS['primary']['family'],
            size=FONTS['primary']['sizes']['body'],
            color=COLORS['text']['primary']
        ),
        
        # Section headers
        annotations=[{
            'font': dict(
                family=FONTS['primary']['family'],
                size=FONTS['primary']['sizes']['header'],
                color=COLORS['text']['primary']
            ),
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False
        }]
    )
    
    return template
