"""Supabase Client Configuration.

This module provides a centralized configuration for Supabase client initialization.
It handles both local development and production environments.
"""

import os
import logging
import traceback
from typing import Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class SupabaseConfig:
    """Supabase configuration singleton."""
    _instance = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Supabase client based on environment."""
        # Check all possible environment variables
        env_vars = {
            'SUPABASE_URL': os.getenv('SUPABASE_URL'),
            'SUPABASE_SERVICE_KEY': os.getenv('SUPABASE_SERVICE_KEY'),
            'SUPABASE_ANON_KEY': os.getenv('SUPABASE_ANON_KEY')
        }
        
        url = env_vars['SUPABASE_URL']
        key = env_vars['SUPABASE_SERVICE_KEY']  # Default to service key
        
        if not url or not key:
            logger.error("Missing Supabase configuration")
            logger.error("Missing Supabase configuration. Please check your environment variables.")
            return
        
        try:
            self._client = create_client(url, key)
            
            # Test the client
            try:
                test_query = self._client.table('api_market_analysis').select('*').limit(1)
                test_result = test_query.execute()
                logger.info("Supabase client initialized and tested successfully")
            except Exception as e:
                logger.error(f"Failed to test Supabase client: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            self._client = None
    
    @property
    def client(self) -> Optional[Client]:
        """Get the Supabase client instance."""
        return self._client
    
    def reinitialize(self, use_service_key: bool = True):
        """Reinitialize the client, optionally switching between service and anon keys."""
        url = os.getenv('SUPABASE_URL')
        service_key = os.getenv('SUPABASE_SERVICE_KEY')
        anon_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not url or not service_key:
            logger.error("Missing required Supabase environment variables")
            self._client = None
            return
        
        logger.info(f"Initializing Supabase client with URL: {url}")
        
        # Try to create client with service key
        try:
            self._client = create_client(url, service_key)
            # Test access
            test = self._client.table('api_market_analysis').select('*').limit(1).execute()
            logger.info("Service key access successful")
        except Exception as e:
            logger.warning(f"Service key access failed: {str(e)}")
            if anon_key:
                try:
                    self._client = create_client(url, anon_key)
                    test = self._client.table('api_market_analysis').select('*').limit(1).execute()
                    logger.info("Anon key access successful")
                except Exception as e:
                    logger.error("Anon key access also failed")
                    raise
            else:
                logger.error("No anon key available")
                raise
        
        logger.info("Supabase client initialized successfully")

# Create a global instance
supabase_config = SupabaseConfig()

def get_client(use_service_key: bool = True) -> Optional[Client]:
    """Get the Supabase client instance.
    
    Args:
        use_service_key: If True, use service key for full access.
                        If False, use anon key for limited access.
    
    Returns:
        Optional[Client]: The Supabase client instance or None if initialization failed.
    """
    # Check environment variables
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY' if use_service_key else 'SUPABASE_ANON_KEY')
    
    try:
        if use_service_key:
            client = supabase_config.client
            
            if client is None:
                supabase_config.reinitialize(use_service_key=True)
                client = supabase_config.client
            return client
        else:
            # For anon access, always reinitialize
            supabase_config.reinitialize(use_service_key=False)
            client = supabase_config.client
            return client
    except Exception as e:
        logger.error(f"Error getting Supabase client: {str(e)}")
        raise
