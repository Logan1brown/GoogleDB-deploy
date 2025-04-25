"""TMDB match service for managing show matches."""

import asyncio
from typing import List, Optional
import streamlit as st
from ...state.admin_state import TMDBMatchState, MatchStatus
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
    
    def propose_match(self, match: TMDBMatchState) -> bool:
        """Store a proposed match in tmdb_match_attempts with all TMDB data for review.
        
        Args:
            match: TMDBMatchState object containing match data and UI state
            
        Returns:
            True if match was successfully proposed, False otherwise
            
        Note:
            Only the match data is stored, UI state is not persisted
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
            
    def search_and_match(self, show_data: dict, confidence_threshold: float = 0.8) -> List[TMDBMatchState]:
        """Search TMDB and find potential matches for our shows.
        
        Args:
            show_data: Dictionary containing our show data (title, network_name, etc)
            confidence_threshold: Minimum confidence score for automatic matches
            
        Returns:
            List of potential matches with confidence scores and initial UI state
        """
        matches = []
        # Search TMDB with variations
        variations = get_search_variations(show_data['title'])
        
        for search_title in variations:
            results = self.client.search_tv_show(search_title)
            if not results:
                continue
            
            # Extract our executive producers
            team_members = show_data.get('team_members', [])
            our_eps = [member['name'] for member in team_members 
                     if member.get('role', '').lower() == 'executive producer']
            
            # Score and convert each result
            for result in results:
                try:
                    # Get full details
                    details = self.client.get_tv_show_details(result.id)
                    credits = self.client.get_tv_show_credits(result.id)
                    tmdb_eps = get_tmdb_eps(credits)
                    
                    # Calculate scores
                    title_score = score_title_match(show_data['title'], details.name)
                    network_score = score_network_match(show_data.get('search_network'), details.networks)
                    ep_score, _ = score_ep_matches(our_eps, tmdb_eps)
                    
                    total_score = title_score + network_score + ep_score
                    confidence = get_confidence_level(total_score)
                    
                    # Create TMDBMatchState with initial UI state
                    match = TMDBMatchState(
                        our_show_id=show_data['show_id'],
                        our_show_title=show_data['title'],
                        our_network=show_data.get('network_name'),  # Use display name
                        our_year=show_data.get('year'),
                        tmdb_id=details.id,
                        name=details.name,
                        first_air_date=str(details.first_air_date) if details.first_air_date else None,
                        episodes_per_season=[s.episode_count for s in details.seasons if s.episode_count],
                        status=details.status,
                        networks=[n.name for n in details.networks],
                        executive_producers=tmdb_eps,
                        confidence=total_score,
                        title_score=title_score,
                        network_score=network_score,
                        ep_score=ep_score,
                        # Initial UI state
                        expanded=False,
                        validation_error=None
                    )
                    matches.append(match)
                except Exception as e:
                    st.warning(f"Skipping result due to error: {str(e)}")
                    continue
        
        # Deduplicate matches by TMDB ID and sort by confidence score
        seen_ids = set()
        unique_matches = []
        for match in matches:
            if match.tmdb_id not in seen_ids:
                seen_ids.add(match.tmdb_id)
                unique_matches.append(match)
        
        # Sort by confidence score
        unique_matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return unique_matches
    
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
