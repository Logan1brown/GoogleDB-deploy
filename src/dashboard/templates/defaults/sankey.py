"""Default template for Sankey diagrams.

This module provides the default template for Sankey diagrams following our style guide:
- Node styling (colors, padding)
- Link styling (opacity, hover)
- Consistent layout spacing
"""

import plotly.graph_objects as go
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.templates.base import create_base_template

def create_sankey_defaults():
    """Create default Sankey template.
    
    Returns:
        go.layout.Template: Template with Sankey defaults:
        - Node styling
        - Link styling
        - Layout spacing
    """
    template = create_base_template()
    
    # Add Sankey trace defaults
    template.data.sankey = [
        go.Sankey(
            node=dict(
                pad=15,  # Padding between nodes
                thickness=20,  # Node thickness
                line=dict(
                    color='white',  # Node outline color
                    width=0.5
                ),
                color=COLORS['accent'],  # Default node color
                hovertemplate=(
                    "%{label}<br>" +
                    "Value: %{value:,}<br>" +
                    "<extra></extra>"  # Hide secondary box
                )
            ),
            link=dict(
                color='rgba(0,0,0,0.2)',  # Light gray links
                hovertemplate=(
                    "%{source.label} → %{target.label}<br>" +
                    "Value: %{value:,}<br>" +
                    "<extra></extra>"  # Hide secondary box
                )
            )
        )
    ]
    
    # Update layout for Sankey
    template.layout.update(
        margin=dict(t=50, b=20, l=20, r=20),  # More space for title
        height=500  # Default height
    )
    
    return template
