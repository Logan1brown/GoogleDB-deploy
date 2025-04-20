"""Preview test for network chord diagram.

This preview helps test the network connections chord diagram template
without needing the full dashboard running.
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from src.dashboard.templates.defaults.heatmap import create_heatmap_defaults

# Sample data for testing
networks = ['Netflix', 'HBO', 'Amazon', 'Apple TV+', 'Disney+']

# Create bidirectional connections for chord-like appearance
shared_creators = [
    # Original connections
    ('Netflix', 'HBO', 12),
    ('Netflix', 'Amazon', 8),
    ('HBO', 'Amazon', 5),
    ('Disney+', 'Netflix', 7),
    ('Apple TV+', 'Netflix', 4),
    # Mirror connections to make it bidirectional
    ('HBO', 'Netflix', 12),
    ('Amazon', 'Netflix', 8),
    ('Amazon', 'HBO', 5),
    ('Netflix', 'Disney+', 7),
    ('Netflix', 'Apple TV+', 4)
]

# Set up Streamlit page
st.set_page_config(page_title="Network Chord Preview", layout="wide")
st.title("Network Connections Chord Diagram")

# Add description
st.markdown("""
This preview shows network relationships using a chord diagram.
The width of each connection represents the number of creators shared between networks.
""")

# Create matrix of shared creators between networks
matrix = np.zeros((len(networks), len(networks)))
for source, target, value in shared_creators:
    source_idx = networks.index(source)
    target_idx = networks.index(target)
    # Only fill upper triangle since relationships are symmetric
    if source_idx < target_idx:
        matrix[source_idx][target_idx] = value
    elif source_idx > target_idx:
        matrix[target_idx][source_idx] = value

# Mask the lower triangle with None
mask = np.tril(np.ones_like(matrix))
matrix = np.ma.masked_array(matrix, mask=mask)

# Create figure with heatmap template
fig = go.Figure()
fig.update_layout(template=create_heatmap_defaults())

# Add heatmap trace
fig.add_trace(
    go.Heatmap(
        x=networks,
        y=networks,
        z=matrix,
        text=matrix,
        texttemplate='%{text:.0f}',
        textfont={"size": 11},
        hovertemplate='%{x} → %{y}<br>Shared Creators: %{z}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title='Shared<br>Creators',
            titleside='right',
            ticks='outside',
            thickness=15,
            len=0.75,  # Shorter colorbar
            y=0.5  # Center vertically
        )
    )
)

# Update layout
fig.update_layout(
    width=500,
    height=500,
    showlegend=False,
    margin=dict(t=50, l=100, r=50, b=100),  # More space for labels
    xaxis=dict(
        title='',  # Remove redundant title
        side='bottom',
        showgrid=True,
        tickangle=90,  # Vertical labels
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='',  # Remove redundant title
        showgrid=True,
        tickfont=dict(size=12)
    ),
    title=dict(
        text='Network Talent Sharing',
        x=0.5,
        y=0.95
    )
)

# Display the figure
st.plotly_chart(fig, use_container_width=True)

# Add data explanation
st.subheader("Sample Data Used")
st.markdown("### Network Connections:")
for from_net, to_net, count in shared_creators:
    st.write(f"- {from_net} → {to_net}: {count} shared creators")
