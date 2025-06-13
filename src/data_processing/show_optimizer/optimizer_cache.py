"""
Optimizer Cache Manager

This module provides caching functionality for the Show Optimizer components.
It handles:
1. Timestamp-based cache invalidation
2. Integrated data caching
3. Component initialization caching
4. Cache status reporting and visualization

The OptimizerCache class serves as a central cache manager that can be used
by ShowOptimizer to reduce code duplication and improve maintainability.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from .optimizer_config import OptimizerConfig


class OptimizerCache:
    """Cache manager for Show Optimizer components and data."""
    
    def __init__(self):
        """Initialize the cache manager."""
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        
        # Component initialization cache
        self.components_initialized = False
        self.components_last_update = None
        
        # Integrated data cache
        self.integrated_data = None
        self.data_last_update = None
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get information about the cache status.
        
        This method provides details about the cache status, including when components
        were last initialized and when data was last fetched.
        
        Returns:
            Dictionary with cache status information
        """
        current_time = datetime.now()
        
        # Calculate component cache status
        components_cache_valid = False
        components_cache_age = None
        components_cache_expiry = None
        
        if self.components_last_update is not None:
            components_cache_age = (current_time - self.components_last_update).total_seconds()
            components_cache_valid = components_cache_age <= self.cache_duration
            components_cache_expiry = self.components_last_update + timedelta(seconds=self.cache_duration)
        
        # Calculate data cache status
        data_cache_valid = False
        data_cache_age = None
        data_cache_expiry = None
        
        if self.data_last_update is not None:
            data_cache_age = (current_time - self.data_last_update).total_seconds()
            data_cache_valid = data_cache_age <= self.cache_duration
            data_cache_expiry = self.data_last_update + timedelta(seconds=self.cache_duration)
        
        # Compile status information
        return {
            'components': {
                'initialized': self.components_initialized,
                'last_update': self.components_last_update.isoformat() if self.components_last_update else None,
                'cache_valid': components_cache_valid,
                'cache_age_seconds': round(components_cache_age) if components_cache_age else None,
                'cache_expiry': components_cache_expiry.isoformat() if components_cache_expiry else None
            },
            'data': {
                'available': self.integrated_data is not None,
                'last_update': self.data_last_update.isoformat() if self.data_last_update else None,
                'cache_valid': data_cache_valid,
                'cache_age_seconds': round(data_cache_age) if data_cache_age else None,
                'cache_expiry': data_cache_expiry.isoformat() if data_cache_expiry else None,
                'record_count': len(self.integrated_data['shows']) if self.integrated_data and 'shows' in self.integrated_data else 0
            },
            'settings': {
                'cache_duration': self.cache_duration
            }
        }
    
    # Cache display and invalidation methods removed
    
    def is_components_cache_valid(self, force_refresh: bool = False) -> bool:
        """Check if the components cache is valid.
        
        Args:
            force_refresh: If True, ignore cache and return False
            
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            if force_refresh:
                return False
                
            # Check if components are initialized
            if self.components_initialized and self.components_last_update:
                current_time = datetime.now()
                components_cache_age = (current_time - self.components_last_update).total_seconds()
                components_cache_valid = components_cache_age <= self.cache_duration
                components_cache_expiry = self.components_last_update + timedelta(seconds=self.cache_duration)
                return components_cache_valid
            
            return False
        except Exception as e:
            st.error(f"Cache validation error: {str(e)}")
            # If there's an error, assume cache is invalid to force refresh
            return False
    
    def is_data_cache_valid(self, force_refresh: bool = False) -> bool:
        """Check if the data cache is valid.
        
        Args:
            force_refresh: If True, ignore cache and return False
            
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            if force_refresh:
                return False
                
            # Check if data is cached
            if self.integrated_data and self.data_last_update:
                current_time = datetime.now()
                data_cache_age = (current_time - self.data_last_update).total_seconds()
                data_cache_valid = data_cache_age <= self.cache_duration
                data_cache_expiry = self.data_last_update + timedelta(seconds=self.cache_duration)
                return data_cache_valid
            
            return False
        except Exception as e:
            st.error(f"Data cache validation error: {str(e)}")
            # If there's an error, assume cache is invalid to force refresh
            return False
    
    def update_components_cache(self) -> None:
        """Mark the components cache as updated."""
        self.components_initialized = True
        self.components_last_update = datetime.now()
    
    def update_data_cache(self, integrated_data: Dict[str, Any]) -> None:
        """Update the integrated data cache.
        
        Args:
            integrated_data: The integrated data to cache
        """
        self.integrated_data = integrated_data
        self.data_last_update = datetime.now()
    
    def get_integrated_data(self) -> Dict[str, Any]:
        """Get the cached integrated data.
        
        Returns:
            The cached integrated data or None if not available
        """
        return self.integrated_data
