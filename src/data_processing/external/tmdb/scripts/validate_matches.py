"""Simple Streamlit app for validating TMDB matches."""
import streamlit as st
import pandas as pd
from pathlib import Path

def load_matches(csv_path: str) -> pd.DataFrame:
    """Load matches from CSV."""
    df = pd.read_csv(csv_path)
    # Fill NaN values with -1 for TMDB IDs
    df['tmdb_id'] = df['tmdb_id'].fillna(-1).astype(int)
    return df

def save_matches(df: pd.DataFrame, csv_path: str):
    """Save matches back to CSV."""
    df.to_csv(csv_path, index=False)
    
def main():
    st.title("TMDB Match Validation")
    
    # Load matches
    csv_path = Path(__file__).parent.parent.parent.parent.parent.parent / "docs/sheets/tmdb_matches.csv"
    if not csv_path.exists():
        st.error("No matches file found. Please run the matcher first.")
        return
        
    matches_df = load_matches(str(csv_path))
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        confidence_filter = st.selectbox(
            "Filter by confidence:",
            ["All"] + list(matches_df["confidence"].unique())
        )
    with col2:
        validated_filter = st.selectbox(
            "Filter by validation:",
            ["All", "Validated", "Not Validated"]
        )
    
    # Apply filters
    filtered_df = matches_df.copy()
    if confidence_filter != "All":
        filtered_df = filtered_df[filtered_df["confidence"] == confidence_filter]
    if validated_filter != "All":
        is_validated = validated_filter == "Validated"
        filtered_df = filtered_df[filtered_df["validated"] == is_validated]
    
    # Show progress
    total = len(matches_df)
    validated = len(matches_df[matches_df["validated"] == True])
    st.progress(validated / total)
    st.write(f"Validated: {validated}/{total} shows")
    
    # Show matches
    for _, row in filtered_df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2,2,1])
            
            # Show info
            with col1:
                st.subheader(row["show_name"])
                st.write(f"Score: {row['score']} ({row['confidence']})")
                st.write(f"TMDB ID: {row['tmdb_id']}")
                st.write(f"Match notes: {row['match_notes']}")
            
            # Show genres
            with col2:
                st.write("**Current Genres:**")
                st.write(row["our_genres"])
                st.write("**TMDB Genres:**")
                st.write(row["tmdb_genres"])
                
                # Highlight differences
                our_genres = set(row["our_genres"].split(",")) if pd.notna(row["our_genres"]) else set()
                tmdb_genres = set(row["tmdb_genres"].split(",")) if pd.notna(row["tmdb_genres"]) else set()
                if our_genres != tmdb_genres:
                    added = tmdb_genres - our_genres
                    removed = our_genres - tmdb_genres
                    if added:
                        st.write("✨ Added:", ", ".join(added))
                    if removed:
                        st.write("❌ Removed:", ", ".join(removed))
            
            # Validation
            with col3:
                key = f"validate_{row['tmdb_id']}"
                current = row["validated"] == True
                validated = st.checkbox("Validate", value=current, key=key)
                if validated != current:
                    idx = matches_df.index[matches_df['tmdb_id'] == int(row['tmdb_id'])].tolist()
                    if idx:
                        matches_df.loc[idx[0], 'validated'] = validated
                        save_matches(matches_df, csv_path)
            
            st.divider()
    
    # Save button (backup)
    if st.button("Force Save"):
        save_matches(matches_df, csv_path)
        st.success("Saved!")

if __name__ == "__main__":
    main()
