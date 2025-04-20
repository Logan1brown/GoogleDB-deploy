"""Default template for tables.

This module provides the default template for tables following our style guide:
- Left-aligned text
- Source Sans Pro font
- Header: 14px, bold
- Cells: 12px, regular
"""

import plotly.graph_objects as go
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.templates.base import create_base_template

def create_table_defaults():
    """Create default table template.
    
    Returns:
        go.layout.Template: Template with table defaults:
        - Font styles (header/cells)
        - Cell alignment
        - Background colors
    """
    template = create_base_template()
    
    # Add table trace defaults
    template.data.table = [
        go.Table(
            header=dict(
                align='left',
                font=dict(
                    family=FONTS['primary']['family'],
                    size=FONTS['primary']['sizes']['body'],  # 14px
                    color=COLORS['text']['primary']
                ),
                fill_color='rgba(0,0,0,0.05)'  # Light gray background
            ),
            cells=dict(
                align='left',
                font=dict(
                    family=FONTS['primary']['family'],
                    size=FONTS['primary']['sizes']['small'],  # 12px
                    color=COLORS['text']['primary']
                ),
                fill_color='white'
            )
        )
    ]
    
    # Update layout for tables
    template.layout.update(
        margin=dict(t=20, b=20, l=20, r=20),  # Tighter margins for tables
        height=None  # Let height be determined by content
    )
    
    return template
