"""TMDB match service for managing show matches."""

import asyncio
from typing import List, Optional
import streamlit as st
from ...state.admin_state import TMDBMatch, MatchStatus
from .tmdb_client import TMDBClient
from .tmdb_models import TVShow, TVShowDetails
from ..supabase import get_supabase_client
from .match_shows import (
    match_show, ShowMatch, get_search_variations,
    score_title_match, score_ep_matches, get_tmdb_eps,
    get_confidence_level, score_network_match
)

class TMDBMatchService:
    """Service for managing TMDB show matches."""
    
    def __init__(self, client: Optional[TMDBClient] = None):
        """Initialize service with optional client."""
        self.client = client or TMDBClient()
    
    def propose_match(self, match: TMDBMatch) -> bool:
        """Store a proposed match in tmdb_match_attempts with all TMDB data for review.
        
        Args:
            match: TMDBMatch object containing all match data
            
        Returns:
            True if match was successfully proposed, False otherwise
        """
        try:
            data = {
                'show_id': match.show_id,
                'tmdb_id': match.tmdb_id,
                
                # Confidence scores
                'confidence_score': match.confidence,
                'title_match_score': match.title_score,
                'network_match_score': match.network_score,
                'year_match_score': match.year_score,
                
                # TMDB data for review
                'tmdb_name': match.name,
                'tmdb_seasons': len(match.episodes_per_season),
                'tmdb_episodes_per_season': match.episodes_per_season,
                'tmdb_total_episodes': sum(match.episodes_per_season),
                'tmdb_average_episodes': sum(match.episodes_per_season) / len(match.episodes_per_season),
                'tmdb_status': match.status,
                'tmdb_last_air_date': match.last_air_date,
                
                # Review-only fields
                'tmdb_executive_producers': match.executive_producers,
                'tmdb_network_name': match.networks,
                
                # Default status
                'status': 'pending'
            }
            
            result = self.client.table('tmdb_match_attempts').insert(data).execute()
            return bool(result.data)
            
        except Exception as e:
            print(f"Failed to propose match: {str(e)}")
            return False
            
    def search_and_match(self, query: str, confidence_threshold: float = 0.8) -> List[TMDBMatch]:
        """Search TMDB and find potential matches for our shows.
        
        Args:
            query: Show title to search for
            confidence_threshold: Minimum confidence score for automatic matches
            
        Returns:
            List of potential matches with confidence scores
        """
        matches = []
        st.write(f"Searching for: {query}")
        
        # Search TMDB
        variations = get_search_variations(query)
        st.write(f"Search variations: {variations}")
        
        for search_title in variations:
            st.write(f"Trying variation: {search_title}")
            results = self.client.search_tv_show(search_title)
            st.write(f"Found {len(results) if results else 0} results")
            
            if not results:
                continue
                
            # Get our show data first
            supabase = get_supabase_client()
            response = supabase.table('api_tmdb_match') \
                .select('*') \
                .eq('title', query) \
                .execute()
            
            our_show = response.data[0] if response.data else None
            
            # Extract our executive producers
            team_members = our_show.get('team_members', []) if our_show else []
            our_eps = [member['name'] for member in team_members 
                     if member['role'].lower() == 'executive producer']
            
            # Score and convert each result
            for result in results:
                try:
                    st.write(f"Processing result: {result.name} (ID: {result.id})")
                    
                    # Get full details
                    details = self.client.get_tv_show_details(result.id)
                    st.write(f"Got details: {details.name} ({details.first_air_date})")
                    
                    credits = self.client.get_tv_show_credits(result.id)
                    tmdb_eps = get_tmdb_eps(credits)
                    st.write(f"Found {len(tmdb_eps)} executive producers")
                    
                    # Calculate scores
                    title_score = score_title_match(query, details.name)
                    network_score = score_network_match(our_show['network_name'] if our_show else None, details.networks)
                    ep_score, _ = score_ep_matches(our_eps, tmdb_eps)
                    
                    total_score = title_score + network_score + ep_score
                    confidence = get_confidence_level(total_score)
                    st.write(f"Scores - Title: {title_score}, Network: {network_score}, EPs: {ep_score}, Total: {total_score}")
                    
                    try:
                        # Create TMDBMatch
                        st.write("Creating TMDBMatch object...")
                        
                        match = TMDBMatch(
                            our_show_id=our_show['show_id'] if our_show else 0,
                            our_show_title=our_show['title'] if our_show else query,
                            our_network=our_show['network_name'] if our_show else None,
                            our_year=our_show['date'] if our_show else None,
                            tmdb_id=details.id,
                            name=details.name,
                            first_air_date=str(details.first_air_date) if details.first_air_date else None,
                            episodes_per_season=[s.episode_count for s in details.seasons if s.episode_count],
                            status=details.status,
                            networks=[n.name for n in details.networks],
                            our_eps=our_eps,
                            tmdb_eps=tmdb_eps,
                            confidence=total_score,
                            title_score=title_score,
                            network_score=network_score,
                            year_score=0  # We don't have show's year yet
                        )
                        st.write("Successfully created TMDBMatch")
                        matches.append(match)
                    except Exception as e:
                        st.error(f"Error creating match: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing result: {str(e)}")
                    continue
            
            return matches
        
        # Sort by confidence score
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return matches
    
    def update_match_status(self, match_ids: List[int], status: MatchStatus, notes: str = "") -> None:
        """Update status for multiple matches.
        
        Args:
            match_ids: List of match IDs to update
            status: New status to set
            notes: Optional notes about the status change
        """
        # TODO: Implement status update
        # 1. Update match status in database
        # 2. Log status change with notes
        # 3. Update related show records
        pass
    
    def get_integration_progress(self) -> dict:
        """Get current integration progress metrics.
        
        Returns:
            Dict with progress metrics:
            - total_shows: Total shows to match
            - matched_shows: Number of shows with approved matches
            - pending_reviews: Number of matches needing review
            - success_rate: Percentage of successful matches
        """
        # TODO: Implement progress tracking
        return {
            "total_shows": 0,
            "matched_shows": 0,
            "pending_reviews": 0,
            "success_rate": 0.0
        }
