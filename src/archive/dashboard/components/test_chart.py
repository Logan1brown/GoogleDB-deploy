"""Test file to isolate chart display issue."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ...data_processing.market_analysis.market_analyzer import MarketAnalyzer
from ..templates.grids.chart_only import create_chart_grid
from ..utils.style_config import COLORS

# Create test data
test_data = pd.DataFrame({
    'network': ['ABC', 'NBC', 'CBS', 'ABC', 'NBC', 'FOX', 'ABC'],
    'studio': ['Studio A', 'Studio B', 'Studio C', 'Studio A', 'Studio B', 'Studio D', 'Studio A'],
    'success_score': [80, 70, 90, 85, 75, 95, 82]
})

# Create analyzer and get success metrics
analyzer = MarketAnalyzer(test_data)
success_metrics = analyzer.success_analyzer.analyze_market(test_data)

# Get network distribution
shows_by_network = test_data.groupby('network').size().sort_values(ascending=False)

# Create colors array and hover text
colors = [COLORS['success']['none']] * len(shows_by_network)  # Default grey
hover_text = [f'{network}<br>Shows: {count}' for network, count in shows_by_network.items()]

# Update colors and hover text for networks with success data
for i, (network, count) in enumerate(shows_by_network.items()):
    network_scores = test_data[test_data['network'] == network]['success_score'].tolist()
    if network_scores:
        avg_score = sum(network_scores) / len(network_scores)
        if avg_score > 80:
            colors[i] = COLORS['success']['high']
        elif avg_score > 50:
            colors[i] = COLORS['success']['medium']
        else:
            colors[i] = COLORS['success']['low']
        hover_text[i] += f'<br>Avg Success Score: {avg_score:.1f}'

# Create direct chart
direct_chart = go.Figure()
direct_chart.add_trace(go.Bar(
    x=list(shows_by_network.index),
    y=list(shows_by_network.values),
    name="Shows per Network",
    marker_color=colors,
    hovertext=hover_text,
    hoverinfo='text'
))

# Update layout
direct_chart.update_layout(
    xaxis_title="Network",
    yaxis_title="Number of Shows",
    font_family="Source Sans Pro",
    showlegend=False,
    margin=dict(t=20)
)

st.write("Direct Chart (no grid):")
st.plotly_chart(direct_chart)

# Create chart with grid
grid = create_chart_grid(
    title="Show Distribution by Network",
    margin=dict(l=50, r=50, t=80, b=50)
)
grid.add_trace(direct_chart.data[0])
st.write("Chart with Grid:")
st.plotly_chart(grid)
