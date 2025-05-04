"""Studio Performance View.

Renders the studio performance dashboard including:
- Force-directed studio relationship graph
- Studio filtering and analysis interface
- Studio insights and metrics
"""

from typing import Dict
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.studio_performance.studio_analyzer import (
    analyze_studio_relationships,
    get_studio_insights
)

# Define a consistent color scheme for networks
NETWORK_COLORS = {
    'Netflix': '#E50914',  # Netflix red
    'Prime Video': '#00A8E1',  # Amazon blue
    'Hulu': '#1CE783',  # Hulu green
    'Apple TV+': '#000000',  # Apple black
    'Disney+': '#113CCF',  # Disney blue
    'HBO Max': '#741B47',  # HBO purple
    'Paramount+': '#0064FF',  # Paramount blue
    'Peacock': '#000000',  # Peacock black
    'ABC': '#000000',  # ABC black
    'CBS': '#0037A5',  # CBS blue
    'NBC': '#F37E2E',  # NBC orange
    'Fox': '#E50914',  # Fox red
    'The CW': '#00A8E1',  # CW blue
    'FX': '#000000',  # FX black
    'AMC': '#000000',  # AMC black
    'Showtime': '#B50000',  # Showtime red
    'Starz': '#000000',  # Starz black
    'Adult Swim': '#000000'  # Adult Swim black
}

def create_studio_graph(shows_df: pd.DataFrame, studio_categories_df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart showing studio-network distribution.
    
    Args:
        shows_df: DataFrame with show information
        
    Returns:
        Plotly figure with studio-network distribution
    """
    # Get cached analysis results
    analysis = get_cached_analysis(shows_df, studio_categories_df)
    
    def get_studio_data(use_indies=False):
        if use_indies:
            # Get indie studios with at least 2 shows
            studio_data = {}
            for studio, data in analysis['indie_studios'].items():
                if data['show_count'] >= 2:
                    # Clean studio name
                    studio_name = studio.split(' - ')[0] if ' - ' in studio else studio
                    studio_data[studio_name] = data['show_count']
            # Return sorted series by show count
            return pd.Series(studio_data).sort_values(ascending=False)
        else:
            # Get top 15 studios with at least 2 shows (including indies)
            return pd.Series({s: count for s, count in analysis['studio_sizes'].items() 
                            if s and not s.isspace() and count >= 2}).nlargest(15)
    
    # Start with major studios
    top_studios = get_studio_data(use_indies=False)
    
    # Get network distribution for each studio
    data = []
    networks = set()
    studio_network_counts = {}
    network_total_counts = {}
    
    for studio in top_studios.index:
        # Get show counts by network for this studio
        # Explode studio_names to get one row per show-studio combination
        studio_shows = shows_df.explode('studio_names')
        # Filter for this studio
        studio_shows = studio_shows[studio_shows['studio_names'] == studio]
        # Drop duplicates to count each show once
        studio_shows = studio_shows.drop_duplicates(subset=['title'])
        # Filter out null/empty networks
        studio_shows = studio_shows[studio_shows['network_name'].notna() & (studio_shows['network_name'] != '')]
        network_counts = studio_shows['network_name'].value_counts()
        
        # Store counts and track unique networks
        studio_network_counts[studio] = network_counts
        networks.update(network_counts.index)
        
        # Update total shows per network
        for network, count in network_counts.items():
            network_total_counts[network] = network_total_counts.get(network, 0) + count
    
    # Filter out networks with only one show total
    networks = {net for net in networks if network_total_counts[net] > 1}
    
    # Create traces for each studio-network combination
    data = []
    
    # Process each studio
    for studio in top_studios.index:
        # Get network counts for this studio and sort by count (descending)
        studio_nets = studio_network_counts[studio]
        sorted_nets = sorted([(net, studio_nets.get(net, 0)) 
                            for net in networks],
                           key=lambda x: x[1], reverse=True)
        
        # Create a trace for each network in this studio
        for network, count in sorted_nets:
            data.append(go.Bar(
                name=network,
                x=[studio],  # Only this studio
                y=[count],
                text=[count] if count > 0 else None,
                textposition='auto',
                hovertext=[f"{studio} on {network}: {count} shows"],
                hoverinfo='text',
                legendgroup=network,  # Group bars with same network
                showlegend=studio == top_studios.index[0],  # Show legend once
                marker_color=NETWORK_COLORS.get(network, '#808080')  # Use gray for unknown networks
            ))
    
    # Create figure
    fig = go.Figure(data=data)
    
    # Create indie studio data (but don't show it yet)
    indie_studios = get_studio_data(use_indies=True)
    indie_data = []
    networks = set()
    
    # Use network relationships from analysis
    for studio in indie_studios.index:
        if studio in analysis['indie_studios']:
            network_counts = analysis['indie_studios'][studio]['networks']
            networks.update(network_counts.keys())
    
    # Create bar data for each studio
    for studio in indie_studios.index:
        if studio in analysis['indie_studios']:
            studio_nets = analysis['indie_studios'][studio]['networks']
            sorted_nets = sorted([(net, studio_nets.get(net, 0)) 
                                for net in networks],
                               key=lambda x: x[1], reverse=True)
        
        for network, count in sorted_nets:
            indie_data.append(go.Bar(
                name=network,
                x=[studio],
                y=[count],
                text=[count] if count > 0 else None,
                textposition='auto',
                hovertext=[f"{studio} on {network}: {count} shows"],
                hoverinfo='text',
                legendgroup=network,
                showlegend=False,  # Don't duplicate legend entries
                visible=False  # Start hidden
            ))
    
    # Add indie traces to figure
    for trace in indie_data:
        fig.add_trace(trace)
    
    # Update layout with all settings including buttons
    fig.update_layout(
        title='Studio Distribution Across Networks',
        barmode='stack',
        showlegend=True,
        height=600,
        margin=dict(
            t=30,  # top margin
            r=150,  # right margin for legend
            b=100,  # increased bottom margin for buttons
            l=50,  # left margin
            pad=10  # padding between elements
        ),
        legend=dict(
            title_text='Networks',
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Position legend just outside the plot
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,  # center buttons
                y=-0.25,  # move buttons further down
                xanchor='center',  # center alignment
                buttons=[
                    dict(label="Top Studios",
                         method="update",
                         args=[{"visible": [True] * len(data) + [False] * len(indie_data)}]),
                    dict(label="Indie Studios",
                         method="update",
                         args=[{"visible": [False] * len(data) + [True] * len(indie_data)}])
                ])
        ],
        xaxis=dict(
            title=None,  # No need for title since studio names are self-explanatory
            tickfont=dict(size=10),
            tickangle=45  # Angle the studio names for better readability
        ),
        yaxis=dict(
            title="Number of Shows",
            range=[0, 45],  # Set max just above highest value (41)
            tickfont=dict(size=10)
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def render_studio_metrics(analysis_results: Dict) -> None:
    """Render key metrics about studios.
    
    Args:
        analysis_results: Results from analyze_studio_relationships
    """
    # Adjust column widths to prevent truncation in the Most Active Studio metric
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col1:
        st.metric(
            "Total Studios",
            analysis_results['total_studios'],
            help="Total number of unique studios in database"
        )
    
    # Get top studio by show count
    top_studio = analysis_results['top_studios'][0]
    top_count = analysis_results['studio_sizes'][top_studio]
    
    with col2:
        st.metric(
            "Most Active Studio",
            top_studio,
            help=f"Studio with the most shows ({top_count}) in database"
        )
    
    # Calculate network diversity (avg networks per studio)
    network_counts = [len(networks) for networks in analysis_results['network_relationships'].values()]
    avg_networks = sum(network_counts) / len(network_counts) if network_counts else 0
    
    with col3:
        st.metric(
            "Avg Networks per Studio",
            f"{avg_networks:.1f}",
            help="Average number of different networks each studio works with"
        )

def render_studio_filter(shows_df: pd.DataFrame, studio_categories_df: pd.DataFrame, analysis_results: Dict) -> None:
    """Render studio filtering and analysis interface.
    
    Args:
        shows_df: DataFrame with show information
        analysis_results: Results from analyze_studio_relationships
    """
    # Get indie studios and their show counts
    indie_counts = {studio: data['show_count'] for studio, data in analysis_results['indie_studios'].items()}
    
    # Filter out empty/whitespace studios and sort by count
    valid_studios = pd.Series({k: v for k, v in indie_counts.items() 
                              if k and not str(k).isspace()})
    valid_studios = valid_studios.sort_values(ascending=False)
    
    # Create studio options with show counts
    studio_options = [f"{studio} ({count} shows)" for studio, count in valid_studios.items()]
    
    # Create a row for filters
    filter_cols = st.columns(3)
    
    # Studio selection in first column
    with filter_cols[0]:
        # Initialize session state if needed
        if 'selected_studio_option' not in st.session_state:
            st.session_state.selected_studio_option = studio_options[0] if studio_options else None
            
        selected_option = st.selectbox(
            "Select Studio",
            studio_options,
            help="Choose a studio to analyze",
            key="selected_studio_option"
        )
    
    # Extract studio name from selection and handle 'Other:' prefix
    if selected_option:
        selected_studio = selected_option.split(' (')[0].strip()
        # If it's an 'Other:' studio, keep just the studio name
        if selected_studio.startswith('Other: '):
            selected_studio = selected_studio[7:].strip()
    else:
        selected_studio = None
    
    if selected_studio:
        # Get insights for selected studio
        insights = get_studio_insights(shows_df, selected_studio, studio_categories_df)
        
        # Cache insights in session state
        if 'studio_insights' not in st.session_state:
            st.session_state.studio_insights = {}
        st.session_state.studio_insights[selected_studio] = insights
        
        
        # Show total valid shows
        st.write(f"### {insights.get('show_count', 0)} Shows")
        
        # Display insights in expandable sections
        if insights.get('top_genres'):
            with st.expander("Genre Distribution", expanded=False):
                for genre, count in sorted(insights['top_genres'].items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{genre}: {count} shows")
        
        if insights.get('network_partners'):
            with st.expander("Network Partners", expanded=False):
                for network, count in sorted(insights['network_partners'].items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{network}: {count} shows")
        
        if insights.get('show_details'):
            with st.expander("Show List", expanded=False):
                # Create a table of shows
                show_data = []
                for show in sorted(insights['show_details'], key=lambda x: x['title']):
                    show_data.append([show['title'], show['network_name'], show['genre']])
                
                if show_data:
                    df = pd.DataFrame(show_data, columns=['Title', 'Network', 'Genre'])
                    st.dataframe(df, hide_index=True)

def render_studio_success_stories(analysis_results: Dict) -> None:
    """Render studio success stories and collaborations.
    
    Args:
        analysis_results: Results from analyze_studio_relationships
    """
    # Show indie studio success stories
    st.subheader("Independent Studio Performance")
    indie_studios = analysis_results.get('indie_studios', {})
    
    if indie_studios:
        # Show top indie studios table
        indie_data = []
        all_genres = set()
        
        # Get indie studios with at least 2 shows (same as chart)
        studio_data = {}
        for studio, data in indie_studios.items():
            if data['show_count'] >= 2:
                # Clean studio name
                studio_name = studio.split(' - ')[0] if ' - ' in studio else studio
                studio_data[studio_name] = data['show_count']
        
        # Use pd.Series to combine studios with same name and sort
        studio_series = pd.Series(studio_data).sort_values(ascending=False)
        
        # Now create indie_data
        for studio_name, show_count in studio_series.items():
            # Get all data for this studio name
            networks = set()
            genres = set()
            for studio, data in indie_studios.items():
                if data['show_count'] >= 2:
                    clean_name = studio.split(' - ')[0] if ' - ' in studio else studio
                    if clean_name == studio_name:
                        networks.update(data['networks'].keys())
                        genres.update(data['genres'])
            
            indie_data.append({
                'Studio': studio_name,
                'Shows': show_count,
                'Networks': len(networks),
                'Genres': len(genres)
            })
            all_genres.update(genres)
        
        if indie_data:
            # Sort by show count descending
            df = pd.DataFrame(indie_data)
            df = df.sort_values('Shows', ascending=False)
            st.dataframe(df, hide_index=True)
            
            # Use the already sorted studio_series
            sorted_indies = studio_series.to_dict()
            for studio_name, show_count in sorted_indies.items():
                with st.expander(studio_name):
                    st.write(f"**Shows:** {show_count}")
                    
                    # Get all data for this studio name
                    networks = set()
                    genres = set()
                    for studio, data in indie_studios.items():
                        if data['show_count'] >= 2:
                            clean_name = studio.split(' - ')[0] if ' - ' in studio else studio
                            if clean_name == studio_name:
                                networks.update(data['networks'].keys())
                                genres.update(data['genres'])
                    
                    # Format networks nicely
                    if networks:
                        st.write("**Networks:**")
                        st.write(", ".join(sorted(networks)))
                    
                    # Format genres nicely
                    if genres:
                        st.write("**Genres:**")
                        st.write(", ".join(sorted(genres)))

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_analysis(df, cat_df):
    return analyze_studio_relationships(df, cat_df)

def render_studio_performance_dashboard(shows_df: pd.DataFrame, studio_categories_df: pd.DataFrame) -> None:
    """Render the complete studio performance dashboard.
    
    Args:
        shows_df: DataFrame with show information
        studio_categories_df: DataFrame with studio categories
    """
    
    try:
        # Get analysis results once for all components
        analysis_results = get_cached_analysis(shows_df, studio_categories_df)
        
        # Render metrics at the top
        render_studio_metrics(analysis_results)
        
        # Create tabs for main content
        graph_tab, filter_tab, stories_tab = st.tabs(["Studio Graph", "Studio Filter", "Success Stories"])
        
        with graph_tab:
            # Studio relationship graph
            fig = create_studio_graph(shows_df, studio_categories_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with filter_tab:
            # Create two columns - narrow one for filters, wide one for content
            filter_col, content_col = st.columns([1, 3])
            
            with filter_col:
                # Studio selection
                studio_counts = pd.Series(analysis_results['studio_sizes'])
                valid_studios = pd.Series({k: v for k, v in studio_counts.items() 
                                        if k and not str(k).isspace()})
                valid_studios = valid_studios.sort_values(ascending=False)
                studio_options = [f"{studio} ({count} shows)" for studio, count in valid_studios.items()]
                
                # Use same session state key as filter
                selected_option = st.selectbox(
                    "Select Studio",
                    studio_options,
                    help="Choose a studio to analyze",
                    key="selected_studio_option"
                )
                
            with content_col:
                # Extract studio name and show insights
                if selected_option:
                    selected_studio = selected_option.split(' (')[0].strip()
                    if selected_studio.startswith('Other: '):
                        selected_studio = selected_studio[7:].strip()
                        
                    # Get insights for selected studio
                    insights = get_studio_insights(shows_df, selected_studio, studio_categories_df)
                    
                    # Cache insights in session state
                    if 'studio_insights' not in st.session_state:
                        st.session_state.studio_insights = {}
                    st.session_state.studio_insights[selected_studio] = insights
                    
                    # Show total valid shows
                    st.write(f"### {insights.get('show_count', 0)} Shows")
                    
                    # Display insights in expandable sections
                    if insights.get('top_genres'):
                        with st.expander("Genre Distribution", expanded=False):
                            for genre, count in sorted(insights['top_genres'].items(), key=lambda x: x[1], reverse=True):
                                st.write(f"{genre}: {count} shows")
                    
                    if insights.get('network_partners'):
                        with st.expander("Network Partners", expanded=False):
                            for network, count in sorted(insights['network_partners'].items(), key=lambda x: x[1], reverse=True):
                                st.write(f"{network}: {count} shows")
                    
                    if insights.get('show_details'):
                        with st.expander("Show List", expanded=False):
                            # Create a table of shows
                            show_data = []
                            for show in sorted(insights['show_details'], key=lambda x: x['title']):
                                show_data.append([show['title'], show['network_name'], show['genre']])
                            
                            if show_data:
                                df = pd.DataFrame(show_data, columns=['Title', 'Network', 'Genre'])
                                st.dataframe(df, hide_index=True)
        
        with stories_tab:
            render_studio_success_stories(analysis_results)
                
    except Exception as e:
        st.error(f"Error in studio analysis: {str(e)}")
        st.exception(e)
