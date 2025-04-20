"""Network Graph Visualization.

This module handles the creation of network graph visualizations showing relationships
between networks based on shared talent.
"""

import logging
from typing import Dict, List

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class NetworkGraphVisualizer:
    """Creates network graph visualizations showing relationships between networks."""
    
    def __init__(self, combined_df, talent_insights):
        """Initialize the visualizer.
        
        Args:
            combined_df: DataFrame with merged show and team data
            talent_insights: Dictionary containing network talent analysis results
        """
        self.combined_df = combined_df
        self.talent_insights = talent_insights
    
    def create_visualization(self) -> None:
        """Create network graph visualization showing network relationships."""
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Network Relationships'],
            specs=[[{"type": "scatter"}]],
            vertical_spacing=0.08
        )

        # Get network pairs data
        network_pairs = self.talent_insights['network_overlap']
        
        # Create network graph data
        unique_networks = set()
        edge_weights = {}
        
        # Collect networks and edge weights
        for pair in network_pairs:
            unique_networks.add(pair['network1'])
            unique_networks.add(pair['network2'])
            edge_weights[(pair['network1'], pair['network2'])] = pair['count']
        
        networks = sorted(list(unique_networks))
        
        # Create edges and nodes for force-directed graph
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=1.5, color='#888'),
            mode='lines',
            hoverinfo='text',
            text=[]
        )
        
        node_trace = go.Scatter(
            x=[], y=[],
            text=[],
            mode='markers',
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                reversescale=True,
                color=[],
                size=[],
                colorbar=dict(
                    thickness=15,
                    title='Shared %',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add edges with weights
        for (network1, network2), weight in edge_weights.items():
            G.add_edge(network1, network2, weight=weight)
        
        # Get spring layout positions
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Add edges to trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_text.append(f'{edge[0]} and {edge[1]} share {weight} creators')
        
        edge_trace.x = edge_x
        edge_trace.y = edge_y
        edge_trace.text = edge_text
        
        # Add nodes to trace
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get network details for hover text and node size
            network_data = next(
                (d for d in self.talent_insights['network_details'] if d['network'] == node),
                None
            )
            
            if network_data:
                total = network_data['total_creators']
                exclusive = network_data['exclusive_creators']
                shared = network_data['shared_creators']
                shared_pct = float(network_data['shared_pct'].replace('%', ''))
                
                node_text.append(
                    f"{node}<br>"
                    f"Total Creators: {total}<br>"
                    f"Exclusive: {exclusive}<br>"
                    f"Shared: {shared}<br>"
                    f"Shared %: {shared_pct:.1f}%"
                )
                
                # Size nodes by total creators
                node_sizes.append(np.sqrt(total) * 10)
                
                # Color nodes by shared percentage
                node_colors.append(shared_pct)
        
        node_trace.x = node_x
        node_trace.y = node_y
        node_trace.text = node_text
        node_trace.marker.size = node_sizes
        node_trace.marker.color = node_colors
        
        # Add traces to figure
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            title='Network Relationships',
            title_x=0.5,
            annotations=[
                dict(
                    text="Node size = Total creators<br>Color = % Shared creators",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0, y=0, align="left"
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # Save the figure
        fig.write_html("output/network_analysis/creative_networks/network_graph.html")
