"""Models for announcement data."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class ShowAnnouncement:
    """Represents a new show announcement."""
    title: str  # e.g. "CIA"
    network: str  # e.g. "CBS"
    season: str  # e.g. "2025-26"
    order_type: str  # e.g. "straight-to-series"
    author: str  # e.g. "Nellie Andreeva"
    published_date: datetime
    url: str
    description: str
    
    # Additional metadata
    talent: List[str] = None  # e.g. ["Tom Ellis"]
    franchise: Optional[str] = None  # e.g. "FBI"
    working_title: Optional[str] = None  # e.g. "FBI: CIA"
    tags: List[str] = None  # e.g. ["CBS", "FBI", "Tom Ellis"]
