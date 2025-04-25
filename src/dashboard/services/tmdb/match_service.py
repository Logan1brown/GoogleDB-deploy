"""TMDB match service for managing show matches."""

import asyncio
from typing import List, Optional
import streamlit as st
from datetime import datetime
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
    
    def __init__(self, client: Optional[TMDBClient] = None, supabase_client=None):
        """Initialize service with optional clients."""
        self.client = client or TMDBClient()
        self.supabase = supabase_client or get_supabase_client()
    
    def validate_match(self, match: TMDBMatchState) -> bool:
        """Validate a TMDB match and save the data.
        
        Args:
            match: TMDBMatchState object containing match data and UI state
            
        Returns:
            True if validation succeeded, False if there were any errors
        """
        try:
            # Debug logging
            st.write("Debug - Match object:")
            st.write(f"our_show_id: {match.our_show_id} ({type(match.our_show_id)})")
            st.write(f"tmdb_id: {match.tmdb_id} ({type(match.tmdb_id)})")
            st.write(f"our_show_title: {match.our_show_title}")
            
            # Input validation
            if not match.our_show_id or not isinstance(match.our_show_id, int):
                raise ValueError("Invalid show ID")
            if not isinstance(match.tmdb_id, int):
                raise ValueError("Invalid TMDB ID")
            
            # Check if show exists and doesn't have TMDB ID
            st.write("Debug - Looking up show in database...")
            show_response = self.supabase.table('shows')\
                .select('*')\
                .eq('id', match.our_show_id)\
                .execute()
            
            st.write(f"Debug - Show response: {show_response.data}")
            
            if not show_response.data:
                raise ValueError(f"Show {match.our_show_title} not found")
            
            existing_show = show_response.data[0]
            st.write(f"Debug - Existing show: {existing_show}")
            
            if existing_show.get('tmdb_id'):
                raise ValueError(f"Show {match.our_show_title} already has TMDB ID {existing_show['tmdb_id']}")
            
            if match.tmdb_id == -1:
                # Insert into no_tmdb_matches
                self.supabase.table('no_tmdb_matches').insert({
                    'show_id': match.our_show_id,
                    'reason': 'Manual validation - No match found',
                    'created_at': datetime.now().isoformat()
                }).execute()
                return True
            
            # Check if TMDB ID already exists in success metrics
            metrics_response = self.supabase.table('tmdb_success_metrics').select('id').eq('tmdb_id', match.tmdb_id).execute()
            if metrics_response.data:
                raise ValueError(f"TMDB ID {match.tmdb_id} already exists in success metrics")
            
            # Get full show details from TMDB
            try:
                st.write("Debug - Getting TMDB details...")
                details = self.client.get_tv_show_details(match.tmdb_id)
                st.write(f"Debug - TMDB details: {details}")
            except Exception as e:
                raise ValueError(f"Failed to get TMDB details: {str(e)}")
            
            if not details:
                raise ValueError(f"No TMDB details found for ID {match.tmdb_id}")
            
            # Map TMDB data to our format
            st.write("Debug - Mapping TMDB data...")
            from .tmdb_data_mapper import map_tmdb_success_metrics, map_tmdb_show_data
            
            # Map success metrics
            metrics_data = map_tmdb_success_metrics(details)
            st.write(f"Debug - Metrics data: {metrics_data}")
            
            # Validate required fields
            if not all(key in metrics_data for key in ['tmdb_id', 'seasons', 'episodes_per_season', 'status']):
                raise ValueError("Missing required fields in TMDB data")
            
            # Map show updates
            show_updates = map_tmdb_show_data(details, existing_show)
            st.write(f"Debug - Show updates: {show_updates}")
            
            # Begin transaction
            try:
                # Update show with TMDB data first
                st.write("Debug - Preparing show update...")
                show_updates.update({
                    'tmdb_id': match.tmdb_id,
                    'updated_at': datetime.now().isoformat()
                })
                
                # Remove any None values and tmdb_ prefixed fields
                show_updates = {k: v for k, v in show_updates.items() 
                              if v is not None and not k.startswith('tmdb_')}
                st.write(f"Debug - Final show updates: {show_updates}")
                
                st.write("Debug - Executing show update...")
                show_response = self.supabase.table('shows')\
                    .update(show_updates)\
                    .eq('id', match.our_show_id)\
                    .execute()
                st.write(f"Debug - Show update response: {show_response.data}")
                
                if not show_response.data:
                    raise ValueError("Failed to update show with TMDB data")
                
                # Now insert metrics since show has tmdb_id
                st.write("Debug - Inserting metrics...")
                metrics_response = self.supabase.table('tmdb_success_metrics')\
                    .insert(metrics_data)\
                    .execute()
                st.write(f"Debug - Metrics response: {metrics_response.data}")
                
                if not metrics_response.data:
                    # Rollback show update if metrics fail
                    st.write("Debug - Failed to insert metrics, rolling back show update...")
                    self.supabase.table('shows').update({"tmdb_id": None}).eq('id', match.our_show_id).execute()
                    raise ValueError("Failed to insert TMDB metrics")
                
                st.write("Debug - Successfully completed validation!")
                return True
                
            except Exception as e:
                raise ValueError(f"Database error: {str(e)}")
                
        except Exception as e:
            raise ValueError(str(e))
    
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
        st.write(f"Debug - Search variations: {variations}")
        
        for search_title in variations:
            results = self.client.search_tv_show(search_title)
            st.write(f"Debug - TMDB results for '{search_title}': {len(results) if results else 0}")
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
                        our_eps=our_eps,  # Add our EPs for comparison
                        confidence=total_score,
                        title_score=title_score,
                        network_score=network_score,
                        ep_score=ep_score,
                        # Initial UI state
                        expanded=True,  # Show side-by-side by default
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
