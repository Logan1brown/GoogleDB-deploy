"""Grid layout for chart with insights and data table.

This module provides a function to create a figure that combines:
- Chart in main section
- Key insights in side panel
- Data table below (optional)

Usage:
    from src.dashboard.templates.grids import create_with_table_grid
    
    # Create figure with insights and table
    fig = create_with_table_grid(
        title="Genre Analysis",
        chart_title="Distribution by Genre",
        insights_title="Key Findings",
        table_title="Raw Data",
        include_table=True
    )
    
    # Add chart to main section
    fig.add_bar(x=genres, y=counts, row=1, col=1)
    
    # Add insights text
    fig.add_annotation(
        text="Drama dominates with 45% share",
        row=1, col=2, y=0.9
    )
    
    # Add data table
    fig.add_table(
        cells=dict(values=[genres, counts]),
        row=2, col="all"
    )
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.dashboard.templates.base import create_base_template

def create_with_table_grid(
    title=None,
    chart_title=None,
    insights_title=None,
    table_title=None,
    include_table=True,
    chart_width=0.7,  # Proportion of width for chart vs insights
    table_height=0.3  # Proportion of height for table
):
    """Create a figure with chart, insights, and optional table.
    
    Args:
        title: Optional title for entire figure
        chart_title: Optional title for chart section
        insights_title: Optional title for insights section
        table_title: Optional title for table section
        include_table: Whether to include table section (default True)
        chart_width: Width proportion for chart vs insights (default 0.7)
        table_height: Height proportion for table section (default 0.3)
    
    Returns:
        go.Figure: Figure with grid layout:
        - Chart (left) and insights (right) in main section
        - Optional data table in bottom section
        - Configurable proportions and spacing
        - Base template applied
    """
    # Validate proportions
    if not 0 < chart_width < 1:
        raise ValueError("chart_width must be between 0 and 1")
    if not 0 < table_height < 1:
        raise ValueError("table_height must be between 0 and 1")
    
    # Configure grid
    rows = 2 if include_table else 1
    row_heights = [1 - table_height, table_height] if include_table else None
    subplot_titles = []
    if chart_title:
        subplot_titles.append(chart_title)
    if insights_title:
        subplot_titles.append(insights_title)
    if include_table and table_title:
        subplot_titles.append(table_title)
    
    # Create subplot structure with proper specs
    specs = [
        # First row: Chart (xy) and Insights (xy)
        [{'type': 'xy'}, {'type': 'xy'}]
    ]
    if include_table:
        # Second row: Table spanning both columns
        specs.append([{'type': 'table', 'colspan': 2}, None])
    
    fig = make_subplots(
        rows=rows,
        cols=2,
        column_widths=[chart_width, 1 - chart_width],
        row_heights=row_heights,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=subplot_titles if subplot_titles else None,
        specs=specs
    )
    
    # Apply base template and layout settings
    fig.update_layout(
        template=create_base_template(),
        height=800 if include_table else 500,
        margin=dict(t=50, b=30, l=50, r=50),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=0.95
        )
    )
    
    # Add title if provided
    if title:
        fig.update_layout(title=title)
    
    # Configure insights panel with border
    fig.update_xaxes(
        showgrid=False,
        showticklabels=False,
        col=2
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        col=2
    )
    
    # Add border to insights section
    fig.add_shape(
        type="rect",
        x0=-0.1, y0=-0.1,  # Extend slightly beyond axes
        x1=1.1, y1=1.1,
        line=dict(
            color="lightgray",
            width=2
        ),
        fillcolor="rgba(255,255,255,0.95)",  # Slightly transparent white
        layer="below",
        row=1, col=2,
        xref="paper",
        yref="paper"
    )
    
    # If table included, make it span both columns
    if include_table:
        fig.update_xaxes(showgrid=False, showticklabels=False, row=2)
        fig.update_yaxes(showgrid=False, showticklabels=False, row=2)
    
    return fig
