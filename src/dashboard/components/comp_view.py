"""View component for comp builder functionality."""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
from src.dashboard.utils.style_config import COLORS, FONTS
# Lazy imports
from . import get_match_details_manager, get_render_match_details_section


def render_comp_builder(state: Dict) -> None:
    """Render the comp builder interface."""
    # Add intro text and help
    st.write("The Comp Builder helps you find shows similar to your concept by matching specific creative elements. Select criteria across genre, characters, plot, and more to see how your show compares to others in the market.")
    
    with st.expander("How to Get the Best Matches"):
        st.markdown("""
        1. Start with Character Dynamics: Combine two character types that capture key relationships (e.g., duo_partners + mentor_student for journey shows, or ensemble_workplace + rivals for office dynamics)

        2. Build on Your Genre: Genre is your foundation - then use source type and tone to refine the style (e.g., Comedy + Original + Satirical, or Drama + Adaptation + Gritty)

        3. Tell a Story with Plot & Themes: Choose plot elements and themes that work together (e.g., survival_stakes + quest_journey, or conflict_power + political_intrigue)

        4. Ground it with Setting: Use time and location settings to narrow the focus (e.g., workplace + contemporary, or post_apocalyptic + multiple_settings)

        The more specific and coherent your criteria, the better your matches will be!
        """)
    
    if 'comp_analyzer' not in st.session_state:
        from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer
        st.session_state.comp_analyzer = CompAnalyzer()
        st.session_state.comp_analyzer.initialize()
        
        # Also cache field options
        field_options = st.session_state.comp_analyzer.get_field_options()
        st.session_state.field_options = field_options
        st.session_state.display_options = {field: st.session_state.comp_analyzer.get_field_display_options(field) 
                                          for field in field_options.keys()}  # Initialize once when created
    comp_analyzer = st.session_state.comp_analyzer
    
    # Initialize state with all possible criteria fields
    if 'criteria' not in state:
        state['criteria'] = {
            # Content
            'genre_id': None,
            'subgenres': [],
            'source_type_id': None,
            'character_type_ids': [],
            'plot_element_ids': [],
            'thematic_element_ids': [],
            'tone_id': None,
            'time_setting_id': None,
            'location_setting_id': None,
            # Production
            'network_id': None,
            'studio_ids': [],
            'team_members': [],  # Use consistent field name
            # Format
            'episode_count': None,
            'order_type_id': None
        }
    
    # Set up columns
    col1, col2 = st.columns([1, 2])
    
    # Render sections
    with col1:
        render_criteria_section(comp_analyzer, state)
    with col2:
        render_results_section(comp_analyzer, state)


def get_id_for_name(name: Optional[str], options: List[Tuple[int, str]]) -> Optional[int]:
    """Get ID for a display name from options list."""
    if not name:
        return None
    for id, option_name in options:
        if option_name == name:
            return id
    return None

def get_ids_for_names(names: List[str], options: List[Tuple[int, str]], field_name: str = None, comp_analyzer: Optional['CompAnalyzer'] = None) -> List[int]:
    """Get IDs for display names from options list.
    
    For team members, we need to get all IDs for each name since a person
    can have multiple entries. For other fields, we just take the first ID.
    
    Args:
        names: List of names to get IDs for
        options: List of (id, name) tuples from field_manager
        field_name: Name of the field (used to identify team members)
        comp_analyzer: Optional CompAnalyzer instance to use for team member lookups
        
    Returns:
        List of IDs for the given names
    """
    # For team members, get all IDs for each name
    if field_name == 'team_members' and comp_analyzer:
        # Get all IDs for selected names
        all_ids = []
        for name in names:
            # Find the option with this name
            opt = next((opt for opt in comp_analyzer.field_manager.get_options('team_members') 
                       if opt.name == name), None)
            if opt and hasattr(opt, 'all_ids'):
                all_ids.extend(opt.all_ids)
            elif opt:
                all_ids.append(opt.id)
        return all_ids
    
    # For team members without analyzer or other fields, just take first ID
    id_map = {}
    for id, name in options:
        if name not in id_map:  # Only take first ID for each name
            id_map[name] = id
    # Map names to IDs in the original order
    return [id_map[name] for name in names if name in id_map]

def render_criteria_section(comp_analyzer: 'CompAnalyzer', state: Dict) -> None:
    """Render criteria selection panel."""
    # Get field options from session state
    field_options = st.session_state.field_options
    display_options = st.session_state.display_options
    with st.expander("Content Match Criteria (82 pts)", expanded=True):
        # Content criteria
        st.markdown("### Content")
        
        # Genre
        genre_name = st.selectbox("Genre", 
            options=[name for _, name in display_options['genre'] if name and name.strip()],
            key="genre_id", index=None, placeholder="Select genre...")
        state["criteria"]["genre_id"] = get_id_for_name(genre_name, display_options['genre']) if genre_name else None
        
        # Subgenres
        subgenre_names = st.multiselect("Subgenres",
            options=[name for _, name in display_options['subgenres'] if name and name.strip()],
            key="subgenres", placeholder="Select subgenres...")
        state["criteria"]["subgenres"] = get_ids_for_names(subgenre_names, display_options['subgenres'])
        
        # Source Type
        source_name = st.selectbox("Source Type",
            options=[name for _, name in display_options['source_type'] if name and name.strip()],
            key="source_type_id", index=None, placeholder="Select source type...")
        state["criteria"]["source_type_id"] = get_id_for_name(source_name, display_options['source_type']) if source_name else None
        
        # Array-type fields
        # Character Types (special case - uses character_type_ids in database)
        names = st.multiselect('Character Types',
            options=[name for _, name in display_options['character_types'] if name and name.strip()],
            key="character_type_ids", placeholder="Select character types...")
        state["criteria"]["character_type_ids"] = get_ids_for_names(names, display_options['character_types'])
        
        # Other array fields
        # Plot Elements
        names = st.multiselect('Plot Elements',
            options=[name for _, name in display_options['plot_elements'] if name and name.strip()],
            key="plot_element_ids", placeholder="Select plot elements...")
        state["criteria"]["plot_element_ids"] = get_ids_for_names(names, display_options['plot_elements'])
        
        # Theme Elements
        names = st.multiselect('Theme Elements',
            options=[name for _, name in display_options['thematic_elements'] if name and name.strip()],
            key="thematic_element_ids", placeholder="Select theme elements...")
        state["criteria"]["thematic_element_ids"] = get_ids_for_names(names, display_options['thematic_elements'])
        
        # Single-select fields
        for field, label in [('tone', 'Tone'),
                           ('time_setting', 'Time Setting'),
                           ('location_setting', 'Location')]:
            name = st.selectbox(label,
                options=[name for _, name in display_options[field] if name and name.strip()],
                key=field, index=None, placeholder=f"Select {label.lower()}...")
            state["criteria"][f"{field}_id"] = get_id_for_name(name, display_options[field]) if name else None
        
    # Production criteria
    with st.expander("Production Match Criteria (13 pts)", expanded=True):
        st.markdown("### Production")
        
        # Network
        network_name = st.selectbox("Network",
            options=[name for _, name in display_options['network'] if name and name.strip()],
            key="network_id", index=None, placeholder="Select network...")
        state["criteria"]["network_id"] = get_id_for_name(network_name, display_options['network']) if network_name else None
        
        # Studios
        studio_names = st.multiselect("Studios",
            options=[name for _, name in display_options['studios'] if name and name.strip()],
            key="studios", placeholder="Select studios...")
        state["criteria"]["studios"] = get_ids_for_names(studio_names, display_options['studios'])
        
        # Team Members
        team_names = st.multiselect("Team Members",
            options=[name for _, name in display_options['team_members'] if name and name.strip()],
            key="team_members", placeholder="Select team members...")
        # Pass field_name and comp_analyzer to get all IDs for each team member
        team_member_ids = get_ids_for_names(
            team_names, 
            display_options['team_members'], 
            'team_members',
            comp_analyzer
        )
        # Store IDs in both fields for compatibility
        state["criteria"]["team_members"] = team_member_ids
        state["criteria"]["team_member_ids"] = team_member_ids
        # Also store the names for display
        state["criteria"]["team_member_names"] = team_names
    
    # Format criteria
    with st.expander("Format Match Criteria (5 pts)", expanded=True):
        st.markdown("### Format")
        
        # Episode Count
        eps = st.number_input("Episode Count", min_value=1, max_value=100, value=None,
            key="episode_count", help="Episode count proximity (2 within ±2, 1.5 within ±4, 1 within ±6)")
        state["criteria"]["episode_count"] = eps if eps is not None and eps > 0 else None
        
        # Order Type
        order_name = st.selectbox("Order Type",
            options=[name for _, name in display_options['order_type']],
            key="order_type_id", index=None, placeholder="Select order type...")
        state["criteria"]["order_type_id"] = get_id_for_name(order_name, display_options['order_type']) if order_name else None


def create_results_df(results: List[Dict]) -> pd.DataFrame:
    """Create results DataFrame with consistent formatting."""
    # Create DataFrame using standardized score format
    rows = []
    for r in results:
        scores = r.get('scores', {})
        try:
            total = scores.get('total', {})
            content = scores.get('content', {})
            production = scores.get('production', {})
            format_score = scores.get('format', {})
            success = int(r.get('success_score', 0) or 0)
            
            rows.append({
                'Show': r['title'],
                'Success': success,
                'Total Score': f"{int(total.get('score', 0))}/{total.get('max_score', 0)}",
                'Content': f"{int(content.get('score', 0))}/{content.get('max_score', 0)}",
                'Production': f"{int(production.get('score', 0))}/{production.get('max_score', 0)}",
                'Format': f"{int(format_score.get('score', 0))}/{format_score.get('max_score', 0)}"
            })
        except (TypeError, AttributeError):
            # Handle case where scores are not in expected format
            rows.append({
                'Show': r['title'],
                'Success': 0,
                'Total Score': '0/0',
                'Content': '0/0',
                'Production': '0/0',
                'Format': '0/0'
            })
    
    df = pd.DataFrame(rows)
    
    # Sort by Total Score descending, then by Success descending
    # Extract numeric score from 'score/max' format for sorting
    df['_sort_score'] = df['Total Score'].apply(lambda x: int(x.split('/')[0]))
    df = df.sort_values(['_sort_score', 'Success'], ascending=[False, False])
    df = df.drop('_sort_score', axis=1)
    
    return df

def apply_table_styling(s: pd.Series) -> List[str]:
    """Apply consistent styling to results table.
    
    Args:
        s: Series to style (passed by pandas)
        
    Returns:
        List of CSS styles for each cell
    """
    # Make success score bold and colored
    if s.name == 'Success':
        return [f'color: {COLORS["accent"]}; font-weight: bold'] * len(s)
    return [''] * len(s)

def apply_table_css() -> None:
    """Apply CSS styling to the results table."""
    st.markdown(f"""
        <style>
        .stDataFrame table {{ color: {COLORS['text']['primary']}; }}
        .stDataFrame th {{ 
            background-color: {COLORS['accent']};
            color: white;
            font-family: {FONTS['primary']['family']};
            font-size: {FONTS['primary']['sizes']['body']}px;
            font-weight: bold;
        }}
        .stDataFrame td {{ 
            font-family: {FONTS['primary']['family']};
            font-size: {FONTS['primary']['sizes']['small']}px;
        }}
        
        /* Hide index column */
        .stDataFrame .index {{ display: none !important; }}
        </style>
    """, unsafe_allow_html=True)

def render_results_section(comp_analyzer: 'CompAnalyzer', state: Dict) -> None:
    """Render comp results panel."""
    if not state.get('criteria'):
        st.info("Select criteria to find matches.")
        return
        
    try:
        results = comp_analyzer.find_by_criteria(state['criteria'])
        

    except Exception as e:
        st.error(f"Error finding matches: {str(e)}")
        return
    if not results:
        st.info("No matches found for the selected criteria.")
        return
        
    # Create match details manager
    MatchDetailsManager = get_match_details_manager()
    details_manager = MatchDetailsManager(comp_analyzer)
    
    # Get criteria for comparison
    criteria = state.get('criteria', {})
    
    # Transform results into expected format and include score details
    match_results = []
    for r in results:
        # Get detailed score breakdown
        details = details_manager.create_match_details(r, criteria)
        match_results.append({
            **r,  # Keep all original fields
            'scores': {
                'total': {'score': r['score'], 'max_score': 100},
                'content': details['content'],
                'production': details['production'],
                'format': details['format']
            }
        })
    
    # Sort by total score descending
    match_results.sort(key=lambda x: x['scores']['total']['score'], reverse=True)
    
    # Create results table
    if match_results:
        df = create_results_df(match_results)
        apply_table_css()
        st.dataframe(df.style.apply(apply_table_styling))
    
    # Process top 10 matches
    for match in match_results[:10]:
        comp_score = match['comp_score']
        
        with st.expander(
            f"#### #{match['id']}: {match['title']} (Match: {comp_score['total']:.1f})", 
            expanded=match == match_results[0]
        ):
            # Get display names for all fields
            details = {
                'comp_score': comp_score,
                'score_details': comp_score['components'],  # Add score details for base_match_breakdown
                # Content fields
                'genre_name': details_manager.get_field_name('genre', match.get('genre_id'), match),
                'selected_genre_name': details_manager.get_field_name('genre', criteria.get('genre_id'), match),
                'genre_match': match.get('genre_id') == criteria.get('genre_id'),
                'subgenre_names': details_manager.get_field_names('subgenres', match.get('subgenres', [])),
                'selected_subgenre_names': details_manager.get_field_names('subgenres', criteria.get('subgenres', [])),
                'subgenre_matches': [name for name in details_manager.get_field_names('subgenres', match.get('subgenres', [])) 
                                    if name in details_manager.get_field_names('subgenres', criteria.get('subgenres', []))],
                'source_type_name': details_manager.get_field_name('source_type', match.get('source_type_id'), match),
                'selected_source_type_name': details_manager.get_field_name('source_type', criteria.get('source_type_id'), match),
                'source_type_match': match.get('source_type_id') == criteria.get('source_type_id'),
                'character_type_names': details_manager.get_field_names('character_types', match.get('character_type_ids', [])),
                'selected_character_type_names': details_manager.get_field_names('character_types', criteria.get('character_type_ids', [])),
                'character_type_matches': [name for name in details_manager.get_field_names('character_types', match.get('character_type_ids', [])) 
                                         if name in details_manager.get_field_names('character_types', criteria.get('character_type_ids', []))],
                'plot_element_names': details_manager.get_field_names('plot_elements', match.get('plot_element_ids', [])),
                'selected_plot_element_names': details_manager.get_field_names('plot_elements', criteria.get('plot_element_ids', [])),
                'plot_element_matches': [name for name in details_manager.get_field_names('plot_elements', match.get('plot_element_ids', [])) 
                                       if name in details_manager.get_field_names('plot_elements', criteria.get('plot_element_ids', []))],
                'theme_element_names': details_manager.get_field_names('thematic_elements', match.get('thematic_element_ids', [])),
                'selected_theme_element_names': details_manager.get_field_names('thematic_elements', criteria.get('thematic_element_ids', [])),
                'theme_element_matches': [name for name in details_manager.get_field_names('thematic_elements', match.get('thematic_element_ids', [])) 
                                        if name in details_manager.get_field_names('thematic_elements', criteria.get('thematic_element_ids', []))],
                'tone_name': details_manager.get_field_name('tone', match.get('tone_id'), match),
                'selected_tone_name': details_manager.get_field_name('tone', criteria.get('tone_id'), match),
                'tone_match': match.get('tone_id') == criteria.get('tone_id'),
                # Setting
                'time_setting_name': details_manager.get_field_name('time_setting', match.get('time_setting_id'), match),
                'selected_time_setting_name': details_manager.get_field_name('time_setting', criteria.get('time_setting_id'), match),
                'time_setting_match': match.get('time_setting_id') == criteria.get('time_setting_id'),
                'location_setting_name': details_manager.get_field_name('location_setting', match.get('location_setting_id'), match),
                'selected_location_setting_name': details_manager.get_field_name('location_setting', criteria.get('location_setting_id'), match),
                'location_setting_match': match.get('location_setting_id') == criteria.get('location_setting_id'),
                # Production
                'network_name': details_manager.get_field_name('network', match.get('network_id'), match),
                'selected_network_name': details_manager.get_field_name('network', criteria.get('network_id'), match),
                'network_match': match.get('network_id') == criteria.get('network_id'),
                'studio_names': details_manager.get_field_names('studios', match.get('studios', [])),
                'selected_studio_names': details_manager.get_field_names('studios', criteria.get('studio_ids', [])),
                'studio_matches': [name for name in details_manager.get_field_names('studios', match.get('studios', [])) 
                                  if name in details_manager.get_field_names('studios', criteria.get('studio_ids', []))],
                'team_member_names': match.get('team_member_names', []),
                'selected_team_member_names': criteria.get('team_member_names', []),
                'team_member_matches': [name for name in match.get('team_member_names', []) 
                                      if name in criteria.get('team_member_names', [])],
                # Format
                'episode_count': match.get('episode_count'),
                'selected_episode_count': criteria.get('episode_count'),
                'episode_count_match': match.get('episode_count') == criteria.get('episode_count'),
                'order_type_name': details_manager.get_field_name('order_type', match.get('order_type_id'), match),
                'selected_order_type_name': details_manager.get_field_name('order_type', criteria.get('order_type_id'), match),
                'order_type_match': match.get('order_type_id') == criteria.get('order_type_id')
            }
            # Get total score for sorting
            total_score = match['comp_score']['total']  # Total score is in comp_score dict
            
            # Transform raw match data into proper UI components using MatchDetailsManager
            match_details = details_manager.create_match_details(match, criteria)
            
            # Store total score for sorting
            match_details['total_score'] = total_score
            
            render_section = get_render_match_details_section()
            render_section(match_details, success_score=match.get('success_score'), description=match.get('description', ''))
