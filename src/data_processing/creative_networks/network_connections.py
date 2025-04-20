"""Network Connection Analysis.

This module focuses on analyzing and visualizing relationships between networks,
including shared talent pools and cross-network success stories.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class NetworkConnectionAnalyzer:
    """Analyzer for network-to-network relationships and talent flow."""
    
    def __init__(self, shows_df: pd.DataFrame, team_df: pd.DataFrame):
        """Initialize the analyzer.
        
        Args:
            shows_df: DataFrame containing show information
            team_df: DataFrame containing team member information
        """
        self.shows_df = shows_df
        self.team_df = team_df
        
        # Log columns before merge
        logger.info(f"Shows DataFrame columns: {self.shows_df.columns.tolist()}")
        logger.info(f"Team DataFrame columns: {self.team_df.columns.tolist()}")
        
        # Merge shows and team data
        self.combined_df = pd.merge(
            self.team_df,
            self.shows_df[['title', 'network_name']],
            on='title'
        )
        
        # Log columns after merge
        logger.info(f"Combined DataFrame columns: {self.combined_df.columns.tolist()}")
        
        # Log basic stats
        logger.info("Network connection stats:")
        network_counts = self.combined_df['network_name_x'].nunique()  # Use network_name from team_df
        creator_counts = self.combined_df['name'].nunique()
        logger.info(f"  Networks: {network_counts}")
        logger.info(f"  Unique creators: {creator_counts}")
    
    def analyze_talent_pools(self) -> Dict:
        """Analyze network talent pools and relationships.
        
        Returns:
            Dictionary containing:
            - exclusive_talent: Creators who work with only one network
            - shared_talent: Creators who work with multiple networks
            - network_overlap: Pairs of networks that share talent
        """
        # Analyze creator network relationships
        creator_networks = {}
        creator_shows = {}
        
        for _, row in self.combined_df.iterrows():
            if row['name'] not in creator_networks:
                creator_networks[row['name']] = set()
                creator_shows[row['name']] = set()
            creator_networks[row['name']].add(row['network_name_x'])
            creator_shows[row['name']].add(row['title'])
        
        # Find exclusive and shared talent
        exclusive_talent = []
        shared_talent = []
        
        for creator, networks in creator_networks.items():
            if len(networks) == 1:
                exclusive_talent.append({
                    'name': creator,
                    'network': next(iter(networks)),
                    'shows': len(creator_shows[creator])
                })
            else:
                shared_talent.append({
                    'name': creator,
                    'networks': len(networks),
                    'network_list': sorted(networks),
                    'shows': len(creator_shows[creator])
                })
        
        # Analyze network overlap
        network_overlap = []
        networks = sorted(self.combined_df['network_name_x'].unique())
        
        for i, net1 in enumerate(networks):
            for net2 in networks[i+1:]:
                shared_creators = [
                    creator for creator, nets in creator_networks.items()
                    if net1 in nets and net2 in nets
                ]
                if shared_creators:
                    network_overlap.append({
                        'network1': net1,
                        'network2': net2,
                        'count': len(shared_creators),
                        'creators': shared_creators
                    })
        
        # Sort results
        exclusive_talent.sort(key=lambda x: x['shows'], reverse=True)
        shared_talent.sort(key=lambda x: (x['networks'], x['shows']), reverse=True)
        network_overlap.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'exclusive_talent': exclusive_talent,
            'shared_talent': shared_talent,
            'network_overlap': network_overlap
        }
    
    def create_visualization(self) -> None:
        """Create visualizations for network connections and talent flow."""
        talent_insights = self.analyze_talent_pools()
        
        # Create sunburst data using shared talent insights
        shared_talent = talent_insights['shared_talent']
        
        # Debug log shared talent data
        logger.info("Shared talent data structure:")
        logger.info(f"Total shared talent entries: {len(shared_talent)}")
        for talent in shared_talent[:2]:
            logger.info(f"Talent entry: {talent}")
            logger.info(f"Networks: {talent['network_list'] if 'network_list' in talent else 'Not found'}")        
        
        # Get all unique networks and creators
        all_networks = set()
        all_creators = set()
        network_creators = {}
        
        # First pass - collect all creators per network
        for talent in shared_talent:
            creator = talent['name']
            all_creators.add(creator)
            
            # Get networks from network_list field
            for network in talent['network_list']:
                all_networks.add(network)
                if network not in network_creators:
                    network_creators[network] = set()
                network_creators[network].add(creator)
        
        # Log data for verification
        logger.info(f"Total shared creators: {len(all_creators)}")
        logger.info(f"Total networks with shared creators: {len(all_networks)}")
        for network, creators in network_creators.items():
            logger.info(f"Network {network}: {len(creators)} shared creators")
        
        # Prepare sunburst data
        labels = ['Shared Creators']  # Root
        parents = ['']  # Root has no parent
        values = [len(all_creators)]  # Total shared creators
        
        # Add network level
        for network in sorted(all_networks):
            labels.append(network)
            parents.append('Shared Creators')
            values.append(len(network_creators[network]))
            
            # Add creator level for each network
            for creator in sorted(network_creators[network]):
                labels.append(f"{creator} ({network})")
                parents.append(network)
                values.append(1)  # Each creator counts as 1
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Multi-Network Success Stories', 'Network Connections', 'Network Details'),
            specs=[
                [{'type': 'table'}, {'type': 'domain'}],  # Top row: table and sunburst
                [{'type': 'table', 'colspan': 2}, None]   # Bottom row: full-width table
            ],
            column_widths=[0.6, 0.4],
            vertical_spacing=0.15,
            horizontal_spacing=0.05
        )
        
        # 1. Multi-Network Success Stories Table
        success_stories = []
        
        # Known partnerships to combine
        partnerships = {
            'Seth Rogen': ('Evan Goldberg', 'Seth Rogen & Evan Goldberg'),
            'Evan Goldberg': ('Seth Rogen', 'Seth Rogen & Evan Goldberg'),
            'John Davis': ('John Fox', 'John Davis & John Fox'),
            'John Fox': ('John Davis', 'John Davis & John Fox'),
            'Michael Costigan': ('Jason Bateman', 'Michael Costigan & Jason Bateman'),
            'Jason Bateman': ('Michael Costigan', 'Michael Costigan & Jason Bateman'),
            'Steve Hutensky': ('Bruna Papandrea', 'Steve Hutensky & Bruna Papandrea'),
            'Bruna Papandrea': ('Steve Hutensky', 'Steve Hutensky & Bruna Papandrea'),
            'Lauren Neustadter': ('Reese Witherspoon', 'Lauren Neustadter & Reese Witherspoon'),
            'Reese Witherspoon': ('Lauren Neustadter', 'Lauren Neustadter & Reese Witherspoon'),
            'Per Saari': ('Nicole Kidman', 'Per Saari & Nicole Kidman'),
            'Nicole Kidman': ('Per Saari', 'Per Saari & Nicole Kidman')
        }
        
        processed_creators = set()  # Track which creators we've processed
        
        for creator in talent_insights['shared_talent'][:15]:  # Look at top 15 to account for partnerships
            # Skip if we've already processed this creator as part of a partnership
            if creator['name'] in processed_creators:
                continue
                
            # Check if this creator is part of a partnership
            if creator['name'] in partnerships:
                partner_name, display_name = partnerships[creator['name']]
                # Skip if we can't find the partner in our data
                if not any(t['name'] == partner_name for t in talent_insights['shared_talent']):
                    continue
                    
                # Mark both creators as processed
                processed_creators.add(creator['name'])
                processed_creators.add(partner_name)
                
                # Get combined show details
                creator_df = self.combined_df[
                    (self.combined_df['name'] == creator['name']) |
                    (self.combined_df['name'] == partner_name)
                ]
            else:
                creator_df = self.combined_df[self.combined_df['name'] == creator['name']]
                display_name = creator['name']
            
            # Get show details
            shows = creator_df.groupby('title').agg({
                'network': 'first',
                'roles': lambda x: ', '.join(set(x))
            }).reset_index()
            
            show_details = []
            for _, show in shows.iterrows():
                # Make roles more compact
                roles = show['roles']
                roles = roles.replace('executive producer', 'EP')
                roles = roles.replace('co-producer', 'Co-P')
                roles = roles.replace('showrunner', 'SR')
                roles = roles.replace('writer', 'W')
                roles = roles.replace('director', 'D')
                show_details.append(f"{show['show_name']} ({show['network']}) - {roles}")
            
            success_stories.append({
                'creator': display_name,
                'networks': f"{creator['networks']} ({', '.join(creator['network_list'])})",
                'portfolio': show_details[:5]  # Limit to top 5 shows
            })
        
        # Debug log the data we'll use
        logger.info(f"About to create visualization with:")
        logger.info(f"- {len(success_stories)} success stories")
        logger.info(f"- {len(labels)} sunburst labels")
        logger.info(f"Sample labels: {labels[:5]}")
        logger.info(f"Sample parents: {parents[:5]}")
        logger.info(f"Sample values: {values[:5]}")
        
        fig.add_trace(
            go.Table(
                columnwidth=[2, 2, 4],  # Make Show Portfolio column wider
                header=dict(
                    values=['Creator', 'Networks', 'Show Portfolio'],
                    align='left',
                    font=dict(size=12, color='white', weight='bold'),
                    fill_color='rgb(0, 75, 150)',
                    height=40
                ),
                cells=dict(
                    values=[
                        [s['creator'] for s in success_stories],
                        [s['networks'] for s in success_stories],
                        ['<br>'.join(s['portfolio']) for s in success_stories]
                    ],
                    align=['left', 'left', 'left'],
                    font=dict(size=11),
                    fill_color=['rgb(245, 245, 250)'],
                    height=None,  # Let height adjust to content
                    line=dict(color='white', width=1)  # Subtle grid lines
                )
            ),
            row=1, col=1
        )
        
        # Add sunburst trace
        fig.add_trace(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues='total',
                hovertemplate='<b>%{label}</b><br>Creators: %{value}<br><extra></extra>',
                maxdepth=2,  # Limit to 2 levels deep
                insidetextorientation='radial',  # Text follows arc
                marker=dict(
                    colors=['rgb(0, 75, 150)'] * len(labels),  # Base color
                    colorscale='Blues',  # Use blues scale for variation
                    showscale=False
                ),
                rotation=90  # Rotate to make text more readable
            ),
            row=1, col=2  # Place in top right
        )
        
        # Network Details
        network_details = []
        # Get all unique networks from the combined data
        all_networks = self.combined_df['network'].unique()
        
        for network in all_networks:
            network_df = self.combined_df[self.combined_df['network'] == network]
            total_creators = network_df['name'].nunique()
            exclusive_creators = sum(1 for t in talent_insights['exclusive_talent'] if t['network'] == network)
            shared_creators = total_creators - exclusive_creators
            
            network_details.append({
                'network': network,
                'total_creators': total_creators,
                'exclusive_creators': exclusive_creators,
                'shared_creators': shared_creators,
                'shared_pct': f"{(shared_creators / total_creators * 100):.1f}%"
            })
        
        # First filter out networks with 0 shared creators
        network_details = [d for d in network_details if d['shared_creators'] > 0]
        
        # Then compute sort values and sort
        for d in network_details:
            # Extract just the number from the percentage string for sorting
            d['sort_value'] = float(d['shared_pct'].replace('%', ''))
            logger.info(f"Network: {d['network']}, Shared %: {d['shared_pct']}, Sort value: {d['sort_value']}")
        
        # Sort by shared percentage (high to low) and then by network name
        network_details = sorted(
            network_details,
            key=lambda x: (x['sort_value'], x['network']),
            reverse=True
        )
        
        # Add network details table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Network', 'Shared %', 'Shared', 'Total', 'Exclusive'],
                    align=['left', 'right', 'right', 'right', 'right'],
                    font=dict(size=12, color='white', weight='bold'),
                    fill_color='rgb(0, 75, 150)',
                    height=40
                ),
                cells=dict(
                    values=[
                        [d['network'] for d in network_details],
                        [d['shared_pct'] for d in network_details],
                        [d['shared_creators'] for d in network_details],
                        [d['total_creators'] for d in network_details],
                        [d['exclusive_creators'] for d in network_details]
                    ],
                    align=['left', 'right', 'right', 'right', 'right'],
                    font=dict(size=11),
                    fill_color=['rgb(245, 245, 250)'],
                    height=30
                )
            ),
            row=2, col=1
        )
        
        # Create sunburst data using shared talent insights
        shared_talent = talent_insights['shared_talent']
        
        # Debug log shared talent data
        logger.info("Shared talent data structure:")
        logger.info(f"Total shared talent entries: {len(shared_talent)}")
        for talent in shared_talent[:2]:
            logger.info(f"Talent entry: {talent}")
            logger.info(f"Networks: {talent['network_list'] if 'network_list' in talent else 'Not found'}")
        
        # Get all unique networks and creators
        all_networks = set()
        all_creators = set()
        network_creators = {}
        
        # First pass - collect all creators per network
        for talent in shared_talent:
            creator = talent['name']
            all_creators.add(creator)
            
            # Get networks from network_list field
            for network in talent['network_list']:
                all_networks.add(network)
                if network not in network_creators:
                    network_creators[network] = set()
                network_creators[network].add(creator)
        
        # Log data for verification
        logger.info(f"Total shared creators: {len(all_creators)}")
        logger.info(f"Total networks with shared creators: {len(all_networks)}")
        for network, creators in network_creators.items():
            logger.info(f"Network {network}: {len(creators)} shared creators")
        
        # Prepare sunburst data
        labels = ['Shared Creators']  # Root
        parents = ['']  # Root has no parent
        values = [len(all_creators)]  # Total shared creators
        
        # Add network level
        for network in sorted(all_networks):
            labels.append(network)
            parents.append('Shared Creators')
            values.append(len(network_creators[network]))
            
            # Add creator level for each network
            for creator in sorted(network_creators[network]):
                labels.append(f"{creator} ({network})")
                parents.append(network)
                values.append(1)  # Each creator counts as 1
        
        logger.info(f"Sunburst data: {len(labels)} labels, {len(parents)} parents, {len(values)} values")
        logger.info(f"Sample labels: {labels[:5]}")
        logger.info(f"Sample values: {values[:5]}")
        logger.info(f"Total creators: {len(all_creators)}")
        logger.info(f"Networks: {list(network_creators.keys())}")
        for network, creators in network_creators.items():
            logger.info(f"{network}: {len(creators)} creators")
        
        # Debug logging
        logger.info(f"Sunburst data structure:")
        logger.info(f"Labels ({len(labels)}): {labels[:10]}...")
        logger.info(f"Parents ({len(parents)}): {parents[:10]}...")
        logger.info(f"Values ({len(values)}): {values[:10]}...")
        
        # Add sunburst trace
        fig.add_trace(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues='total',
                hovertemplate='<b>%{label}</b><br>Creators: %{value}<br><extra></extra>',
                maxdepth=2,  # Limit to 2 levels deep
                insidetextorientation='radial',  # Text follows arc
                marker=dict(
                    colors=['rgb(0, 75, 150)'] * len(labels),  # Base color
                    colorscale='Blues',  # Use blues scale for variation
                    showscale=False
                ),
                rotation=90  # Rotate to make text more readable
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,  # Adjusted height
            showlegend=False,
            title=dict(
                text="Network Connection Analysis",
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top'
            ),
            margin=dict(t=100, b=20, l=20, r=20)  # Margins for better spacing
        )
        
        # Save the figure
        output_dir = Path('output/network_analysis/creative_networks')
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_dir / 'network_connections.html')
        
        # Update subplot titles font and position
        for annotation in fig.layout.annotations:
            annotation.update(font=dict(size=14, color='rgb(0, 75, 150)', weight='bold'))
        
        # Save visualization
        output_dir = Path("output/network_analysis/creative_networks")
        output_dir.mkdir(exist_ok=True, parents=True)
        fig.write_html(output_dir / "network_connections.html")
