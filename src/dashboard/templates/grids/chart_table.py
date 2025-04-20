"""Grid layout for chart with data table.

This module provides a function to create a figure that combines:
- Main chart section
- Data table below
"""

from plotly.subplots import make_subplots


def create_chart_table_grid(
    title=None,
    chart_title=None,
    table_title=None,
    table_height=0.3  # Proportion of height for table
):
    """Create a figure with chart and table.

    Args:
        title: Optional title for entire figure
        chart_title: Optional title for chart section
        table_title: Optional title for table section
        table_height: Height proportion for table section (default 0.3)

    Returns:
        go.Figure: Figure with grid layout:
        - Chart in top section
        - Data table in bottom section
        - Configurable proportions
    """
    # Calculate row heights (give more space to chart)
    row_heights = [0.75, 0.25]  # Fixed ratio instead of variable
    
    # Build subplot titles
    subplot_titles = []
    if chart_title:
        subplot_titles.append(chart_title)
    if table_title:
        subplot_titles.append(table_title)
    
    # Create subplot structure with proper specs
    specs = [
        [{'type': 'xy'}],      # Chart
        [{'type': 'table'}]    # Table
    ]
    
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=row_heights,
        vertical_spacing=0.15,  # Increased spacing between subplots
        subplot_titles=subplot_titles if subplot_titles else None,
        specs=specs
    )
    
    # Apply layout settings
    fig.update_layout(
        title=title,
        showlegend=True,
        margin=dict(l=50, r=50, t=100, b=50),  # More top margin
        height=800  # Fixed height to prevent squishing
    )
    
    # Configure table section
    fig.update_xaxes(showgrid=False, showticklabels=False, row=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=2)
    
    return fig
