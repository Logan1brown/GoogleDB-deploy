import streamlit as st
import pandas as pd

def render_market_intel():
    # Sample data (we'll replace with real data)
    sample_shows = pd.DataFrame({
        'shows': ['The Last of Us', 'House of the Dragon', 'Wednesday', 'The Bear'],
        'network': ['HBO', 'HBO', 'Netflix', 'FX'],
        'success_score': [8.8, 8.6, 8.2, 8.9],
        'renewal_status': ['Renewed', 'Renewed', 'Renewed', 'Renewed'],
        'genre': ['Drama', 'Fantasy', 'Comedy', 'Drama']
    })
    
    sample_creators = pd.DataFrame({
        'name': ['Craig Mazin', 'Neil Druckmann', 'Ryan Condal', 'Miles Millar'],
        'role': ['Showrunner', 'EP/Writer', 'Showrunner', 'Creator'],
        'show_name': ['The Last of Us', 'The Last of Us', 'House of the Dragon', 'Wednesday'],
        'success_rate': ['92%', '95%', '85%', '88%']
    })
    """Prototype of the market intel unified view."""
    
    # Apply standard margins and spacing
    st.markdown("""
        <style>
        .stMarkdown { margin-top: 20px; }
        .stDataFrame { margin: 20px 0; }
        .stTabs { margin: 30px 0; }
        .section-header { 
            font-family: 'Source Sans Pro', sans-serif;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.1em;
            color: #1E4D8C;
            margin-bottom: 1em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sample data
    dummy_shows = pd.DataFrame({
        'shows': ['The Last of Us', 'House of the Dragon', 'Wednesday', 'The Bear', 'Abbott Elementary'],
        'success_score': [8.8, 8.6, 8.2, 8.9, 8.5],
        'network': ['HBO', 'HBO', 'Netflix', 'FX', 'ABC'],
        'genre': ['Drama', 'Fantasy', 'Comedy', 'Drama', 'Comedy'],
        'source': ['Game', 'Book', 'TV Show', 'Original', 'Original'],
        'episodes': ['9 eps', '10 eps', '8 eps', '8 eps', '13 eps'],
        'status': ['Renewed', 'Renewed', 'Renewed', 'Renewed', 'Renewed']
    })
    
    dummy_creators = pd.DataFrame({
        'name': ['Craig Mazin', 'Neil Druckmann', 'Ryan Condal', 'Miles Millar', 'Christopher Storer'],
        'role': ['Showrunner', 'EP/Writer', 'Showrunner', 'Creator', 'Creator'],
        'show_name': ['The Last of Us', 'The Last of Us', 'House of the Dragon', 'Wednesday', 'The Bear'],
        'past_hits': ['Chernobyl', 'TLOU Game', 'Colony', 'Smallville', 'Ramy'],
        'success_rate': ['92%', '95%', '85%', '88%', '90%']
    })

    # Top: Analysis Type
    analysis_type = st.radio(
        "",
        ["Acquisition", "Packaging", "Development"],
        horizontal=True
    )
    
    if analysis_type == "Acquisition":
        col1, col2 = st.columns([1, 3])
        
        with col1:  # Input Panel
            source = st.selectbox("Source Type", ["Original", "Book", "IP"])
            genre = st.selectbox("Genre", ["Drama", "Comedy", "Thriller", "Fantasy"])
            
        with col2:  # Results Panel
            tab1, tab2, tab3, tab4 = st.tabs(["Networks", "Creators", "Pairings", "Insights"])
            
            with tab1:  # Network Analysis
                st.markdown('### Network Performance')
                if source and genre:
                    st.markdown('''
                    HBO<br>
                    Shows in Genre: 5<br>
                    Success Rate: 85%<br>
                    Renewal Rate: 90%<br>
                    <br>
                    <br>
                    Netflix<br>
                    Shows in Genre: 4<br>
                    Success Rate: 82%<br>
                    Renewal Rate: 85%<br>
                    <br>
                    <br>
                    FX<br>
                    Shows in Genre: 3<br>
                    Success Rate: 88%<br>
                    Renewal Rate: 92%<br>
                    <br>
                    <br>
                    ABC<br>
                    Shows in Genre: 2<br>
                    Success Rate: 75%<br>
                    Renewal Rate: 80%
                    ''', unsafe_allow_html=True)
                
            with tab2:  # Creator Analysis
                st.markdown('### Top Creators')
                if source and genre:
                    st.markdown('''
                    Craig Mazin<br>
                    Recent Show: The Last of Us<br>
                    Success Rate: 92%<br>
                    <br>
                    <br>
                    Mike Flanagan<br>
                    Recent Show: The Fall of House of Usher<br>
                    Success Rate: 88%<br>
                    <br>
                    <br>
                    Tony Gilroy<br>
                    Recent Show: Andor<br>
                    Success Rate: 85%
                    ''', unsafe_allow_html=True)
            
            with tab3:  # Creator / Network Pairings
                st.markdown('### Successful Pairings')
                if source and genre:
                    st.markdown('''
                    HBO + Craig Mazin<br>
                    (The Last of Us)<br>
                    <br>
                    <br>
                    Netflix + Mike Flanagan<br>
                    (Midnight Series)<br>
                    <br>
                    <br>
                    FX + Christopher Storer<br>
                    (The Bear)
                    ''', unsafe_allow_html=True)
              
            with tab4:  # Pattern Analysis
                st.markdown('### Success Patterns')
                if source and genre:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Episodes", "8-10")
                        st.metric("Order Type", "Limited")
                    with col2:
                        st.metric("Success Rate", "85%")
                        st.metric("Renewal Rate", "80%")
    
    elif analysis_type == "Packaging":
        col1, col2 = st.columns([1, 3])
        
        with col1:  # Input Panel
            source = st.selectbox("Source Type", ["Original", "Book", "IP"])
            genre = st.selectbox("Genre", ["Drama", "Comedy", "Thriller"])
            
        with col2:  # Package Recommendations
            if source and genre:
                st.markdown('<p class="section-header">Recommended Packages</p>', unsafe_allow_html=True)
                
                if source == "Book":
                    if genre == "Drama":
                        st.markdown('''
                        ### Package A<br>
                        Network: HBO Max<br>
                        Shows: House of the Dragon, His Dark Materials<br>
                        Team: David E. Kelley, Melissa James Gibson<br>
                        <br>
                        ### Package B<br>
                        Network: Prime Video<br>
                        Shows: The Wheel of Time, Good Omens<br>
                        Team: Rafe Judkins, Neil Gaiman
                        ''', unsafe_allow_html=True)
                    else:  # Other genres
                        st.markdown('''
                        ### Package A<br>
                        Network: Netflix<br>
                        Shows: Shadow and Bone, Bridgerton<br>
                        Team: Eric Heisserer, Chris Van Dusen<br>
                        <br>
                        ### Package B<br>
                        Network: Apple TV+<br>
                        Shows: Foundation, Pachinko<br>
                        Team: David S. Goyer, Soo Hugh
                        ''', unsafe_allow_html=True)
                        
                elif source == "IP":
                    if genre == "Drama":
                        st.markdown('''
                        ### Package A<br>
                        Network: HBO Max<br>
                        Shows: The Last of Us, House of the Dragon<br>
                        Team: Craig Mazin, Ryan Condal<br>
                        <br>
                        ### Package B<br>
                        Network: Disney+<br>
                        Shows: Andor, The Mandalorian<br>
                        Team: Tony Gilroy, Jon Favreau
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown('''
                        ### Package A<br>
                        Network: Netflix<br>
                        Shows: Arcane, Wednesday<br>
                        Team: Christian Linke, Alfred Gough<br>
                        <br>
                        ### Package B<br>
                        Network: Prime Video<br>
                        Shows: The Boys, Reacher<br>
                        Team: Eric Kripke, Nick Santora
                        ''', unsafe_allow_html=True)
                        
                else:  # Original
                    if genre == "Drama":
                        st.markdown('''
                        ### Package A<br>
                        Network: FX<br>
                        Shows: The Bear, The Patient<br>
                        Team: Christopher Storer, Joel Fields<br>
                        <br>
                        ### Package B<br>
                        Network: Apple TV+<br>
                        Shows: Severance, The Morning Show<br>
                        Team: Dan Erickson, Kerry Ehrin
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown('''
                        ### Package A<br>
                        Network: ABC<br>
                        Shows: Abbott Elementary, The Good Doctor<br>
                        Team: Quinta Brunson, David Shore<br>
                        <br>
                        ### Package B<br>
                        Network: Hulu<br>
                        Shows: Only Murders, The Great<br>
                        Team: John Hoffman, Tony McNamara
                        ''', unsafe_allow_html=True)
    
    else:  # Development
        col1, col2 = st.columns([1, 3])
        
        with col1:  # Input Panel
            ip_type = st.selectbox("IP Type", ["Book", "Game", "Film"])
            genre = st.selectbox("Genre", ["Drama", "Comedy", "Thriller"])
            
        with col2:  # Strategy Panel
            if ip_type and genre:
                st.markdown('<p class="section-header">Network Alignment</p>', unsafe_allow_html=True)
                
                # Sample data - replace with real data from shows.csv
                if ip_type == "Book":
                    networks = {
                        "HBO Max": ["House of the Dragon", "The Last of Us", "His Dark Materials"],
                        "Prime Video": ["The Lord of the Rings", "Good Omens", "The Wheel of Time"],
                        "Netflix": ["Shadow and Bone", "The Witcher", "Bridgerton"]
                    }
                elif ip_type == "Game":
                    networks = {
                        "HBO Max": ["The Last of Us", "Fallout"],
                        "Netflix": ["Arcane", "Castlevania", "DOTA: Dragon's Blood"],
                        "Paramount+": ["Halo", "Sonic Prime"]
                    }
                else:  # Film
                    networks = {
                        "Disney+": ["Andor", "The Mandalorian", "Ahsoka"],
                        "Paramount+": ["Yellowstone", "1923", "Tulsa King"],
                        "HBO Max": ["Peacemaker", "Penguin", "Welcome to Derry"]
                    }
                
                for network, shows in networks.items():
                    with st.expander(f"{network} - {len(shows)} Shows"):
                        for show in shows:
                            st.markdown(f"• {show}")
                
                st.markdown('<p class="section-header">Market Insights</p>', unsafe_allow_html=True)
                
                # Success Metrics
                col1, col2 = st.columns(2)
                if ip_type == "Book":
                    col1.metric("Success Rate", "75%")
                    col2.metric("Renewal Rate", "65%")
                elif ip_type == "Game":
                    col1.metric("Success Rate", "85%")
                    col2.metric("Renewal Rate", "80%")
                else:  # Film
                    col1.metric("Success Rate", "60%")
                    col2.metric("Renewal Rate", "50%")
                
                st.markdown('<p class="section-header">Format Strategy</p>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                if ip_type == "Book":
                    col1.metric("Optimal Episodes", "8-10")
                    col2.metric("Best Format", "Limited")
                elif ip_type == "Game":
                    col1.metric("Optimal Episodes", "10-13")
                    col2.metric("Best Format", "Ongoing")
                else:  # Film
                    col1.metric("Optimal Episodes", "6-8")
                    col2.metric("Best Format", "Limited")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_market_intel()
