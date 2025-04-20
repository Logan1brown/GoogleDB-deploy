"""Default template for chord diagrams.

This module provides the default template for chord diagrams following our style guide:
- Network-specific colors
- Consistent node padding
- Hover text with connection details
"""

import plotly.graph_objects as go
from src.dashboard.templates.base import create_base_template
from src.dashboard.utils.style_config import COLORS, FONTS

def create_chord_defaults():
    """Create default chord template.
    
    Returns:
        go.layout.Template: Template with chord defaults:
        - Node styling
        - Link styling
        - Hover template
        - Layout settings
    """
    template = create_base_template()
    
    # Add sankey trace defaults (used for chord diagrams)
    template.data.sankey = [
        go.Sankey(
            node=dict(
                pad=20,  # Padding between nodes
                thickness=20,  # Node thickness
                line=dict(
                    color='white',  # Node border color
                    width=0.5  # Node border width
                ),
                # Use accent color for nodes
                color=COLORS['accent'],
                # Hover template for nodes
                hoverlabel=dict(
                    bgcolor=COLORS['text']['secondary'],  # Use secondary text color for hover bg
                    font=dict(
                        family=FONTS['primary']['family'],
                        size=FONTS['primary']['sizes']['small'],
                        color=COLORS['background']  # Use background color for text
                    )
                ),
                hovertemplate='%{label}<br>%{value} creators<extra></extra>'
            ),
            link=dict(
                # Use semi-transparent accent color for links
                color=f"rgba({COLORS['accent'].strip('rgb()')}, 0.4)",
                # Hover template for links
                hoverlabel=dict(
                    bgcolor=COLORS['text']['secondary'],  # Use secondary text color for hover bg
                    font=dict(
                        family=FONTS['primary']['family'],
                        size=FONTS['primary']['sizes']['small'],
                        color=COLORS['background']  # Use background color for text
                    )
                ),
                hovertemplate='%{source.label} → %{target.label}<br>%{value} creators<extra></extra>'
            )
        )
    ]
    
    # Update layout for proper spacing
    template.layout.update(
        margin=dict(l=50, r=50, t=50, b=50),  # Increase margins
        paper_bgcolor=COLORS['background'],  # Match dashboard background
        plot_bgcolor=COLORS['background']
    )
    
    return template
