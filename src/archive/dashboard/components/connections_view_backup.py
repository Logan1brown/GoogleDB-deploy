"""Network Connections View.

Renders the network connections overview dashboard including:
- Force-directed network graph
- Success stories
- High-level metrics
"""

from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import networkx as nx

def render_metrics(metrics: Dict) -> None:
    """Render high-level network metrics.
    
    Args:
        metrics: Dictionary containing network metrics
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Networks",
            len(metrics['network_sizes']),
            help="Total number of networks in analysis"
        )
    
    with col2:
        total_talent = sum(metrics['talent_counts'].values())
        st.metric(
            "Total Creators",
            total_talent,
            help="Total number of unique creators"
        )
    
    with col3:
        st.metric(
            "Cross-Network %",
            f"{metrics['cross_network_activity']:.1f}%",
            help="Percentage of creators working with multiple networks"
        )

def create_network_graph(shows_df: pd.DataFrame, team_df: pd.DataFrame) -> go.Figure:
    """Create force-directed network graph.
    
    Args:
        shows_df: DataFrame with show information
        team_df: DataFrame with team member information
    
    Returns:
        Plotly figure with network graph
    """
    # Create network graph
    G = nx.Graph()
    
    # Add nodes for each network
    for network in shows_df['network'].unique():
        show_count = shows_df[shows_df['network'] == network].shape[0]
        G.add_node(network, size=show_count)
    
    # Add edges for shared talent
    merged = pd.merge(team_df, shows_df[['show_name', 'network']], on='show_name')
    for name, group in merged.groupby('name'):
        networks = group['network'].unique()
        if len(networks) > 1:
            for i in range(len(networks)):
                for j in range(i+1, len(networks)):
                    if G.has_edge(networks[i], networks[j]):
                        G[networks[i]][networks[j]]['weight'] += 1
                    else:
                        G.add_edge(networks[i], networks[j], weight=1)
    
    # Get node positions using force-directed layout
    pos = nx.spring_layout(G)
    
    # Create edges trace
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create nodes trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>{G.nodes[node]['size']} shows")
        node_size.append(G.nodes[node]['size'] * 20)  # Scale node size
    
    nodes_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            size=node_size,
            colorscale='Viridis',
            line_width=2
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edges_trace, nodes_trace],
        layout=go.Layout(
            title='Network Relationships',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

def render_success_stories(success_stories: List[Dict]) -> None:
    """Render multi-network success stories.
    
    Args:
        success_stories: List of success story dictionaries
    """
    st.subheader("Multi-Network Success Stories")
    
    for story in success_stories:
        with st.expander(f"{story['creator_team']} ({len(story['networks'])} networks)"):
            st.write(f"**Networks:** {', '.join(story['networks'])}")
            st.write(f"**Shows:** {story['show_count']}")
            st.write(f"**Roles:** {', '.join(story['roles'])}")

def render_network_connections_dashboard(connections_analyzer) -> None:
    """Render the complete network connections dashboard.
    
    Args:
        connections_analyzer: ConnectionsAnalyzer instance with processed data
    """
    st.title("Network Connections Overview")
    
    # Get data from analyzer
    shows_df = connections_analyzer.shows_df
    team_df = connections_analyzer.team_df
    
    # Get analysis results
    analysis_results = {
        'metrics': connections_analyzer.get_filter_options(),
        'success_stories': connections_analyzer.get_success_stories()
    }
    
    # Render metrics
    render_metrics(analysis_results['metrics'])
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Network Graph", "Success Stories"])
    
    with tab1:
        fig = create_network_graph(shows_df, team_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(
            "💡 The size of each node represents the number of shows. "
            "Connections indicate shared talent between networks."
        )
    
    with tab2:
        render_success_stories(analysis_results['success_stories'])
