"""Simple Streamlit app for validating TMDB matches."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.append(str(root_dir))

from src.data_processing.external.tmdb.tmdb_client import TMDBClient
from src.data_processing.external.tmdb.scripts.match_shows import (
    score_title_match,
    score_network_match,
    score_ep_matches,
    get_tmdb_eps,
    get_confidence_level
)

def load_matches(csv_path: str) -> pd.DataFrame:
    """Load matches from CSV."""
    df = pd.read_csv(csv_path)
    # Fill NaN values with -1 for TMDB IDs
    df['tmdb_id'] = df['tmdb_id'].fillna(-1).astype(int)
    # Convert validated to bool
    # Read validation status directly from CSV
    df['validated'] = df['validated'].fillna(False)
    return df

def save_matches(df: pd.DataFrame, csv_path: str):
    """Save matches back to CSV."""
    # Print file content before save
    print("\nBefore save:")
    if Path(csv_path).exists():
        with open(csv_path) as f:
            print(f.read())
    
    # Save new content
    print(f"\nSaving matches to {csv_path}")
    print(f"DataFrame has {len(df)} rows")
    df.to_csv(csv_path, index=False)
    
    # Print file content after save
    print("\nAfter save:")
    with open(csv_path) as f:
        print(f.read())
    
    print("Save completed")
    
def search_tmdb(client: TMDBClient, show_name: str):
    """Search TMDB for alternative matches."""
    results = client.search_tv_show(show_name)
    if not results:
        return []
    
    # Get details for each result
    details = []
    for show in results[:5]:  # Limit to top 5 matches
        try:
            detail = client.get_tv_show_details(show.id)
            details.append(detail)
        except Exception as e:
            st.error(f"Error getting details for {show.name}: {e}")
    return details

def update_match(show_name: str, tmdb_id: int):
    """Callback for updating a match."""
    csv_path = Path(__file__).parent.parent.parent.parent.parent.parent / "docs/sheets/tmdb_matches.csv"
    
    try:
        # Get TMDB details
        client = TMDBClient(api_key=st.session_state.tmdb_api_key)
        details = client.get_tv_show_details(tmdb_id)
        credits = client.get_tv_show_credits(tmdb_id)
        
        # Score the match
        show_idx = st.session_state.matches_df.index[st.session_state.matches_df['show_name'] == show_name].item()
        row = st.session_state.matches_df.loc[show_idx]
        
        title_score = score_title_match(row['show_name'], details.name)
        network_score = score_network_match(row['our_network'], details.networks)
        our_eps = row['our_eps'].split(',') if pd.notna(row['our_eps']) else []
        tmdb_eps = get_tmdb_eps(credits)
        ep_score, ep_notes = score_ep_matches(our_eps, tmdb_eps)
        total_score = title_score + network_score + ep_score
        
        # Store old values for feedback
        old_name = row['tmdb_name'] if pd.notna(row['tmdb_name']) else 'No previous match'
        
        # Update both DataFrames
        updates = {
            'tmdb_id': tmdb_id,
            'tmdb_name': details.name,
            'tmdb_network': ','.join(n.name for n in details.networks) if details.networks else '',
            'tmdb_genres': ','.join(g.name for g in details.genres) if details.genres else '',
            'score': total_score,
            'confidence': get_confidence_level(total_score).value,
            'validated': False,
            'match_notes': f'Updated match to {details.name} (ID: {tmdb_id})'
        }
        
        for df in [st.session_state.matches_df, st.session_state.filtered_df]:
            for col, value in updates.items():
                df.at[show_idx, col] = value
        
        # Save changes
        save_matches(st.session_state.matches_df, csv_path)
        
        # Store update info in session state
        st.session_state.last_update = {
            'show_name': show_name,
            'old_match': old_name,
            'new_match': details.name,
            'new_score': total_score,
            'new_confidence': get_confidence_level(total_score).value
        }
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error updating match: {str(e)}")

def main():
    st.title("TMDB Match Validation")
    
    # Get TMDB API key from environment or .env file
    env_path = Path(__file__).parent.parent.parent.parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith('TMDB_API_KEY='):
                    st.session_state.tmdb_api_key = line.split('=')[1].strip()
                    break
    
    # Load matches
    csv_path = Path(__file__).parent.parent.parent.parent.parent.parent / "docs/sheets/tmdb_matches.csv"
    if not csv_path.exists():
        st.error("No matches file found. Please run the matcher first.")
        return
    
    # Initialize session state
    if 'matches_df' not in st.session_state:
        st.session_state.matches_df = load_matches(str(csv_path))
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        confidence_filter = st.selectbox(
            "Filter by confidence:",
            ["All"] + list(st.session_state.matches_df["confidence"].unique())
        )
    with col2:
        validated_filter = st.selectbox(
            "Filter by validation:",
            ["All", "Validated", "Not Validated"]
        )
    
    # Apply filters
    filtered_df = st.session_state.matches_df.copy()
    if confidence_filter != "All":
        filtered_df = filtered_df[filtered_df["confidence"] == confidence_filter]
    if validated_filter != "All":
        is_validated = validated_filter == "Validated"
        filtered_df = filtered_df[filtered_df["validated"] == is_validated]
    
    # Store filtered_df in session state
    st.session_state.filtered_df = filtered_df
    
    # Show progress
    total = len(st.session_state.matches_df)
    validated = len(st.session_state.matches_df[st.session_state.matches_df["validated"] == True])
    not_validated = len(st.session_state.matches_df[st.session_state.matches_df["validated"] == False])
    st.progress(validated / total)
    st.write(f"Validated (True): {validated}/{total} shows")
    st.write(f"Not Validated (False): {not_validated}/{total} shows")
    
    # Show matches
    for idx, row in filtered_df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2,2,1])
            
            # Show title comparison
            with col1:
                st.subheader("Show Name Comparison (60 pts)")
                st.write("**Our Title:**", row["show_name"])
                st.write("**TMDB Title:**", row["tmdb_name"])
                st.write(f"Score: {row['score']} ({row['confidence']})")
                st.write(f"TMDB ID: {row['tmdb_id']}")
            
            # Show network and EP comparison
            with col2:
                st.subheader("Network (25 pts) & EPs (15 pts)")
                st.write("**Our Network:**", row["our_network"])
                st.write("**TMDB Network:**", row["tmdb_network"])
                st.write("")
                st.write("**Our EPs:**")
                our_eps = row["our_eps"].split(",") if pd.notna(row["our_eps"]) else []
                for ep in our_eps:
                    st.write(f"- {ep.strip()}")
                st.write("")
                st.write("**TMDB EPs:**")
                tmdb_eps = row["tmdb_eps"].split(",") if pd.notna(row["tmdb_eps"]) else []
                for ep in tmdb_eps:
                    st.write(f"- {ep.strip()}")
            
            # Validation and TMDB Search
            with col3:
                # Make key unique by including row index
                key = f"validate_{idx}_{row['tmdb_id']}"
                current = bool(row["validated"])  # Convert to bool explicitly
                
                # Show current validation status with color
                status_color = "green" if current else "red"
                st.markdown(f"<p style='color: {status_color}'>Current status: {'✓ Validated' if current else '✗ Not Validated'}</p>", unsafe_allow_html=True)
                
                # Add validation checkbox
                validated = st.checkbox("Mark as Validated", value=current, key=key, help="Check this box to validate the match")
                
                if validated != current:
                    # Update both DataFrames
                    show_idx = st.session_state.matches_df.index[st.session_state.matches_df['show_name'] == row['show_name']].item()
                    for df in [st.session_state.matches_df, st.session_state.filtered_df]:
                        df.at[show_idx, 'validated'] = validated
                    
                    # Save changes
                    save_matches(st.session_state.matches_df, csv_path)
                    
                    # Show clear feedback
                    if validated:
                        st.success(f"✓ Successfully validated '{row['show_name']}' and saved to CSV")
                    else:
                        st.warning(f"✗ Removed validation for '{row['show_name']}' and saved to CSV")
                    
                    st.rerun()
                
                # Add TMDB search button
                if st.button("Search TMDB", key=f"search_{idx}"):
                    st.write("Alternative matches from TMDB:")
                    try:
                        client = TMDBClient(api_key=st.session_state.tmdb_api_key)
                        results = search_tmdb(client, row['show_name'])
                        
                        if results:
                            for result in results:
                                air_year = result.first_air_date.year if result.first_air_date else 'N/A'
                                with st.expander(f"{result.name} ({air_year})"):
                                    st.write(f"TMDB ID: {result.id}")
                                    st.write(f"Overview: {result.overview}")
                                    st.write(f"First Air Date: {result.first_air_date}")
                                    st.write(f"Networks: {', '.join(n.name for n in result.networks) if result.networks else 'N/A'}")
                                    if st.button(f"Use this match", key=f"use_{idx}_{result.id}"):
                                        # Update in the main DataFrame
                                        show_idx = st.session_state.matches_df.index[st.session_state.matches_df['show_name'] == row['show_name']].item()
                                        
                                        # Get TMDB details
                                        client = TMDBClient(api_key=st.session_state.tmdb_api_key)
                                        details = client.get_tv_show_details(result.id)
                                        credits = client.get_tv_show_credits(result.id)
                                        
                                        # Score the match
                                        title_score = score_title_match(row['show_name'], details.name)
                                        network_score = score_network_match(row['our_network'], details.networks)
                                        our_eps = row['our_eps'].split(',') if pd.notna(row['our_eps']) else []
                                        tmdb_eps = get_tmdb_eps(credits)
                                        ep_score, ep_notes = score_ep_matches(our_eps, tmdb_eps)
                                        total_score = title_score + network_score + ep_score
                                        
                                        # Update DataFrame
                                        updates = {
                                            'tmdb_id': result.id,
                                            'tmdb_name': details.name,
                                            'tmdb_network': ','.join(n.name for n in details.networks) if details.networks else '',
                                            'tmdb_genres': ','.join(g.name for g in details.genres) if details.genres else '',
                                            'score': total_score,
                                            'confidence': get_confidence_level(total_score).value,
                                            'validated': False,
                                            'match_notes': f'Updated match to {details.name} (ID: {result.id})'
                                        }
                                        
                                        # Update the main DataFrame
                                        for col, value in updates.items():
                                            st.session_state.matches_df.at[show_idx, col] = value
                                        
                                        # Save changes
                                        save_matches(st.session_state.matches_df, csv_path)
                                        st.success(f"✓ Updated '{row['show_name']}' to match '{details.name}' with score {total_score}")
                                        st.rerun()
                        else:
                            st.write("No matches found")
                    except Exception as e:
                        st.error(f"Error searching TMDB: {e}")
    
    # Save button (backup)
    if st.button("Force Save"):
        save_matches(matches_df, csv_path)
        st.success("Saved!")

if __name__ == "__main__":
    main()
