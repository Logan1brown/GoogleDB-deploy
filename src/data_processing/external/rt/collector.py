"""RT Collector

Handles fetching shows for RT matching, generating search URLs,
and saving RT metrics to the database.
"""

from typing import Dict, List
from urllib.parse import quote
from postgrest import APIResponse
from supabase import Client

class RTCollector:
    def __init__(self, supabase: Client):
        self.supabase = supabase

    def get_unmatched_shows(self, limit: int = 5) -> List[Dict]:
        """Get shows without RT metrics"""
        response: APIResponse = self.supabase.from_('shows')\
            .select('id, title, network_list(network)')\
            .is_('rt_success_metrics.rt_id', 'null')\
            .order('id')\
            .limit(limit)\
            .execute()
        
        return response.data if response else []

    def generate_search_url(self, title: str) -> str:
        """Generate Google search URL for RT show"""
        query = f"site:rottentomatoes.com tv {title}"
        return f"https://www.google.com/search?q={quote(query)}"

    def get_next_batch(self) -> List[str]:
        """Get search URLs for next batch of shows"""
        shows = self.get_unmatched_shows()
        return [self.generate_search_url(s['title']) for s in shows]

    def save_rt_metrics(self, show_id: int, tomatometer: int, 
                       popcornmeter: int) -> bool:
        """Save RT metrics for a show
        
        Args:
            show_id: ID of the show
            tomatometer: RT critic score (0-100)
            popcornmeter: RT audience score (0-100)
            
        Returns:
            bool: True if save successful
        """
        # Validate scores
        if not isinstance(tomatometer, int) or not isinstance(popcornmeter, int):
            return False
        if not (0 <= tomatometer <= 100 and 0 <= popcornmeter <= 100):
            return False
            
        try:
            response = self.supabase.table('rt_success_metrics')\
                .upsert({
                    'show_id': show_id,
                    'tomatometer': tomatometer,
                    'popcornmeter': popcornmeter,
                    'is_matched': True,
                    'last_updated': 'now()'
                }).execute()
            return bool(response)
        except Exception:
            return False
