"""
Data entry state management.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

@dataclass
class ShowFormState:
    """State for show entry form."""
    id: Optional[int] = None  # Show ID for edit/remove operations
    title: str = ""
    original_title: Optional[str] = None  # Original title for edit operations
    description: str = ""
    network_id: Optional[int] = None
    genre_id: Optional[int] = None
    subgenre_id: Optional[int] = None
    subgenres: List[int] = field(default_factory=list)
    source_type_id: Optional[int] = None
    order_type_id: Optional[int] = None
    status_id: Optional[int] = None
    date: Optional[date] = None
    
    def __post_init__(self):
        # Set default date for new shows
        if self.date is None:
            self.date = date.today()
    episode_count: int = 0
    studio_ids: List[int] = field(default_factory=list)  # List of studio IDs for matching
    new_studios: List[str] = field(default_factory=list)  # List of studio names
    active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    team_members: List[dict] = field(default_factory=list)

@dataclass
class DataEntryState:
    """Overall state for data entry pages."""
    operation: str = "Add Show"  # Current operation (Add Show, Edit Show, Remove Show)
    form_started: bool = False  # Whether we're past the landing page
    read_only: bool = False  # For Remove operation
    success_message: Optional[str] = None  # Message to show on landing page after successful operation
    form_error: Optional[str] = None  # Error message to show in form
    show_form: ShowFormState = field(default_factory=ShowFormState)
    active_tab: int = 0
    search_results: list = field(default_factory=list)  # Search results
    num_new_studios: int = 0
    num_team_members: int = 0
    lookups: dict = field(default_factory=dict)
