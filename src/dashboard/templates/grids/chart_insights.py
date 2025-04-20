"""Grid layout for chart with key findings panel.

This module provides a function to create a figure that combines:
- Chart on the left
- Key findings/insights panel on the right
"""

from plotly.subplots import make_subplots


def create_chart_insights_grid(
    title=None,
    chart_title=None,
    insights_title=None,
    chart_width=0.7  # Proportion of width for chart vs insights
):
    """Create a figure with chart and insights panel.

    Args:
        title: Optional title for entire figure
        chart_title: Optional title for chart section
        insights_title: Optional title for insights section
        chart_width: Width proportion for chart section (default 0.7)

    Returns:
        go.Figure: Figure with grid layout:
        - Chart on left side
        - Key findings panel on right side
        - Configurable proportions
    """
    # Build subplot titles
    subplot_titles = []
    if chart_title:
        subplot_titles.append(chart_title)
    if insights_title:
        subplot_titles.append(insights_title)
    
    # Create subplot structure
    specs = [
        [{'type': 'xy'}, {'type': 'xy'}]  # Chart and Insights side by side
    ]
    
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[chart_width, 1 - chart_width],
        horizontal_spacing=0.05,
        subplot_titles=subplot_titles if subplot_titles else None,
        specs=specs
    )
    
    # Apply layout settings
    fig.update_layout(
        title=title,
        showlegend=True,
        margin=dict(l=50, r=50, t=100, b=50),
        height=600  # Slightly shorter than layouts with tables
    )
    
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
    
    return fig
