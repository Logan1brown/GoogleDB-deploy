# TV Series Database Development Guide

## Table of Contents

### I. Getting Started
1. [Quick Links & Access](#quick-links--access)
   - [Production Environment](#production-environment)
   - [Development Environment](#development-environment)
2. [Development Setup](#development-setup)
   - [Environment Setup](#environment-setup)
   - [Database Connection](#database-connection)
   - [Direct Database Access](#direct-database-access)
3. [Authentication & Permissions](#authentication--permissions)
   - [Role-Based Access](#role-based-access)
   - [Security Rules](#security-rules)

### II. Core Architecture
1. [Database](#database)
   - [Schema Inspection](#schema-inspection)
   - [Database Schema](#database-schema)
     - [Core Tables](#core-tables)
     - [Support Tables](#support-tables)
     - [Views](#views)
   - [Querying](#querying)
2. [Services](#services)
   - [TMDB Integration](#tmdb-integration)
   - [Data Services](#data-services)
   - [Error Handling](#error-handling)
3. [Pages](#pages)
   - [Admin Dashboard](#admin-dashboard)
   - [Data Entry](#data-entry)
   - [Market Analysis](#market-analysis)

### III. Integration & APIs
1. [TMDB Integration](#tmdb-integration-1)
   - [TMDBClient](#tmdbclient)
   - [TMDBMatchService](#tmdbmatchservice)
   - [TMDBDataMapper](#tmdbdatamapper)
2. [External Services](#external-services)
3. [API Rate Limiting](#api-rate-limiting)

### IV. Development Workflow
1. [Best Practices](#best-practices)
2. [Common Tasks](#common-tasks)
3. [Troubleshooting](#troubleshooting)

---

## I. Getting Started

### Quick Links & Access

#### Production Environment
- **Live App**: https://appdb-deploy-nrsfx8wrmuajjww5qrkwwy.streamlit.app/
- **Admin Dashboard**: https://appdb-deploy-nrsfx8wrmuajjww5qrkwwy.streamlit.app/admin

#### Development Environment
- Uses live database for consistency
- Requires Supabase credentials
- Local Streamlit server

### Development Setup

1. **Environment Setup**
```bash
# Clone repository
git clone <repo_url>

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

2. **Database Connection**

Three connection modes available:

#### a. Session Mode (Recommended for Development)
```bash
# .env
DATABASE_URL=postgres://postgres.[project_ref]:[password]@aws-0-us-west-1.pooler.supabase.com:5432/postgres
```
- Uses Supavisor pooler
- Best for local development
- Maintains persistent connections

#### b. Transaction Mode
```bash
# .env
DATABASE_URL=postgres://postgres.[project_ref]:[password]@aws-0-us-west-1.pooler.supabase.com:6543/postgres
```
- Uses connection pooling
- Best for serverless functions
- Good for high-concurrency

#### c. Direct Connection
```bash
# .env
DATABASE_URL=postgresql://postgres:[password]@db.[project_ref].supabase.co:5432/postgres
```
- Direct to Postgres
- Requires IPv6 support
- No connection pooling

3. **Direct Database Access**

```python
# Python Client
from supabase import create_client
import os

supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')  # For admin access
)

# Query Examples

# Basic select
response = supabase.table('shows').select('*').limit(1).execute()

# Joins with foreign keys
response = supabase.table('shows')\
    .select('title, network_list!inner(name)')\
    .limit(3)\
    .execute()

# Filters and sorting
response = supabase.table('shows')\
    .select('title, network_list(name)')\
    .filter('title', 'ilike', '%the%')\
    .order('title')\
    .limit(3)\
    .execute()

# Schema Inspection

```sql
-- List all tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public';

-- Get table columns
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default,
    character_maximum_length
FROM information_schema.columns
WHERE table_schema = 'public' 
AND table_name = 'shows';

-- Get foreign keys
SELECT
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_schema = 'public';

-- Get indexes
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- Get table row counts
SELECT
    relname as table_name,
    n_live_tup as row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;
```

You can run these queries using:
1. Supabase Dashboard SQL Editor
2. Direct psql connection:
```bash
psql $DATABASE_URL
```
3. Python client:
```python
# Using raw SQL with supabase-py
response = supabase.raw('''
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
''').execute()
```
```

4. **Required Credentials**
- Supabase URL & Service Key
- TMDB API Key
- Admin credentials

5. **Running Locally**
```bash
streamlit run src/dashboard/app.py
```

### Authentication & Permissions

#### Role-Based Access
- **Admin**: Full system access
- **Editor**: Data entry and updates
- **Viewer**: Read-only analytics

#### Security Rules
- Row Level Security (RLS) in Supabase
- Service role for admin operations
- JWT validation for API requests

## II. Core Architecture

### Database Schema

#### Core Tables

##### Shows
```sql
CREATE TABLE shows (
    id bigint PRIMARY KEY,
    title text NOT NULL,
    search_title text GENERATED ALWAYS AS (lower(title)) STORED,
    description text,
    status_id bigint REFERENCES status_types(id),
    network_id bigint NOT NULL REFERENCES network_list(id),
    genre_id bigint REFERENCES genre_list(id),
    subgenres bigint[],
    source_type_id bigint REFERENCES source_types(id),
    order_type_id bigint REFERENCES order_types(id),
    date date,
    episode_count integer,
    tmdb_id integer UNIQUE,
    active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    studios bigint[] DEFAULT '{}'
);

CREATE TABLE show_team (
    id bigint PRIMARY KEY,
    show_id bigint NOT NULL REFERENCES shows(id),
    name text NOT NULL,
    search_name text GENERATED ALWAYS AS (lower(name)) STORED,
    role_type_id bigint NOT NULL REFERENCES role_types(id),
    team_order integer,
    notes text,
    active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT show_team_unique UNIQUE (show_id, name, role_type_id)
);

CREATE TABLE tmdb_success_metrics (
    id bigint PRIMARY KEY,
    tmdb_id integer UNIQUE REFERENCES shows(tmdb_id),
    seasons integer,
    episodes_per_season integer[],
    total_episodes integer,
    average_episodes double precision,
    status text,
    last_air_date date,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE VIEW api_team_summary AS
SELECT 
    st.show_id,
    s.title,
    array_agg(DISTINCT st.name) FILTER (WHERE role_type_id = 1) AS writers,
    array_agg(DISTINCT st.name) FILTER (WHERE role_type_id = 2) AS producers,
    array_agg(DISTINCT st.name) FILTER (WHERE role_type_id = 3) AS directors,
    array_agg(DISTINCT st.name) FILTER (WHERE role_type_id = 4) AS creators
FROM shows s
JOIN show_team st ON s.id = st.show_id
WHERE st.active = true
GROUP BY st.show_id, s.title;

CREATE MATERIALIZED VIEW show_details AS
SELECT 
    s.id,
    s.title,
    s.description,
    nl.network AS network_name,
    gl.genre AS genre_name,
    array_agg(DISTINCT sgl.genre) AS subgenre_names,
    array_agg(DISTINCT stl.studio) AS studio_names,
    st.status AS status_name,
    ot.type AS order_type_name,
    srt.type AS source_type_name,
    s.date,
    s.episode_count,
    s.tmdb_id,
    tsm.seasons AS tmdb_seasons,
    tsm.total_episodes AS tmdb_total_episodes,
    tsm.status AS tmdb_status
FROM shows s
LEFT JOIN network_list nl ON s.network_id = nl.id
LEFT JOIN genre_list gl ON s.genre_id = gl.id
LEFT JOIN genre_list sgl ON sgl.id = ANY(s.subgenres)
LEFT JOIN studio_list stl ON stl.id = ANY(s.studios)
LEFT JOIN status_types st ON s.status_id = st.id
LEFT JOIN order_types ot ON s.order_type_id = ot.id
LEFT JOIN source_types srt ON s.source_type_id = srt.id
LEFT JOIN tmdb_success_metrics tsm ON s.tmdb_id = tsm.tmdb_id;

CREATE MATERIALIZED VIEW network_stats AS
SELECT
    nl.id AS network_id,
    nl.network AS network_name,
    count(s.id) AS total_shows,
    count(s.id) FILTER (WHERE st.status = 'Active') AS active_shows,
    count(s.id) FILTER (WHERE st.status = 'Ended') AS ended_shows,
    array_agg(DISTINCT gl.genre) AS genres,
    array_agg(DISTINCT srt.type) AS source_types
FROM network_list nl
LEFT JOIN shows s ON nl.id = s.network_id AND s.active = true
LEFT JOIN status_types st ON s.status_id = st.id
LEFT JOIN genre_list gl ON s.genre_id = gl.id
LEFT JOIN source_types srt ON s.source_type_id = srt.id
WHERE nl.active = true
GROUP BY nl.id, nl.network;

### Schema Inspection

To inspect the database schema:

```bash
# Connect and explore schema
psql $DATABASE_URL

# List tables
\dt

# Describe table
\d shows

# List views
\dv

# List materialized views
\dm
```

Or use SQL queries:

```sql
-- List all tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public';

-- Get table columns
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public' 
AND table_name = 'shows';

-- Get foreign keys
SELECT
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_schema = 'public';
```

### Database Schema

#### Core Tables

##### Shows
```sql
CREATE TABLE shows (
    id bigint PRIMARY KEY,
    title text NOT NULL,
    search_title text GENERATED ALWAYS AS (lower(title)) STORED,
    description text,
    status_id bigint REFERENCES status_types(id),
    network_id bigint NOT NULL REFERENCES network_list(id),
    genre_id bigint REFERENCES genre_list(id),
    subgenres bigint[],
    source_type_id bigint REFERENCES source_types(id),
    order_type_id bigint REFERENCES order_types(id),
    order_type_id bigint,
    date date,
    episode_count integer,
    tmdb_id integer UNIQUE,
    active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    studios bigint[] DEFAULT '{}'
);
```

**Show Team (show_team)**
```sql
CREATE TABLE show_team (
    id bigint PRIMARY KEY,
    show_id bigint REFERENCES shows(id),
    name text NOT NULL,
    role_type_id bigint REFERENCES role_types(id),
    active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
```

**TMDB Success Metrics (tmdb_success_metrics)**
```sql
CREATE TABLE tmdb_success_metrics (
    id BIGSERIAL PRIMARY KEY,
    tmdb_id INTEGER REFERENCES shows(tmdb_id),
    seasons INTEGER,
    episodes_per_season INTEGER[],
    total_episodes INTEGER,
    average_episodes FLOAT,
    status TEXT,
    last_air_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

##### Support Tables

**Network List (network_list)**
```sql
CREATE TABLE network_list (
    id bigint PRIMARY KEY,
    network text NOT NULL,
    type text NOT NULL,
    parent_company text,
    aliases text[],
    search_network text GENERATED ALWAYS AS (lower(network)) STORED,
    active boolean NOT NULL DEFAULT true
);
```

**Genre List (genre_list)**
```sql
CREATE TABLE genre_list (
    id bigint PRIMARY KEY,
    genre text NOT NULL,
    search_genre text GENERATED ALWAYS AS (lower(genre)) STORED,
    active boolean NOT NULL DEFAULT true
);
```

**Studio List (studio_list)**
```sql
CREATE TABLE studio_list (
    id bigint PRIMARY KEY,
    studio text NOT NULL,
    search_studio text GENERATED ALWAYS AS (lower(studio)) STORED,
    type text,
    parent_company text,
    division text,
    platform text,
    category text,
    aliases text[],
    active boolean NOT NULL DEFAULT true
);
```

**Role Types (role_types)**
```sql
CREATE TABLE role_types (
    id bigint PRIMARY KEY,
    role text NOT NULL,
    active boolean NOT NULL DEFAULT true
);
```

##### Views

**Show Details**
- Denormalized view of shows with names instead of IDs
- Includes network_name, genre_name, studio_names
- Used for most read operations

**Team Summary**
- Aggregates show team data
- Groups by show and role type
- Used for team analytics

#### Querying

##### Client Setup
```python
# Regular client (viewer/editor roles)
client = get_supabase_client()

# Admin client (service role)
admin_client = get_admin_client()
```

##### Reading Data
```python
# Get all active shows with network names
response = client.table('show_details').select(
    'id,title,network_name,genre_name,date'
).eq('active', True).execute()

# Get show by ID with full details
response = client.table('show_details').select('*').eq('id', show_id).single().execute()

# Search shows by title
response = client.table('shows').select('*').ilike('search_title', f'%{query.lower()}%').execute()

# Get shows without TMDB matches
response = client.table('shows').select('*').is_('tmdb_id', None).execute()
```

##### Writing Data
```python
# Insert new show
show_data = {
    'title': 'New Show',
    'network_id': network_id,
    'genre_id': genre_id,
    'date': '2025-01-01',
    'studios': [studio_id1, studio_id2]
}
response = admin_client.table('shows').insert(show_data).execute()

# Update show
update_data = {
    'status_id': status_id,
    'episode_count': 10,
    'updated_at': 'now()'
}
response = admin_client.table('shows').update(update_data).eq('id', show_id).execute()

# Soft delete
response = admin_client.table('shows').update({'active': False}).eq('id', show_id).execute()
```

##### Complex Queries
```python
# Get shows with team info
response = client.table('shows').select(
    '*,show_team(name,role_types(role))'
).eq('active', True).execute()

# Get network stats
response = client.table('show_details').select(
    'network_name,count(*)',
    count='exact'
).eq('active', True).group_by('network_name').execute()

# Get shows with TMDB metrics
response = client.table('show_details').select(
    '*,tmdb_success_metrics(seasons,total_episodes,status)'
).not_.is_('tmdb_id', None).execute()
```
```

### 3. Services

#### TMDB Integration

##### TMDBClient
```python
class TMDBClient:
    """Handles raw TMDB API requests with rate limiting."""
    def search_tv(self, query: str) -> List[TVShow]:
        """Search TMDB for TV shows."""
    
    def get_tv_details(self, tmdb_id: int) -> TVShowDetails:
        """Get full show details including seasons, credits."""

    def get_episode_groups(self, tmdb_id: int) -> List[EpisodeGroup]:
        """Get episode grouping info."""
```

##### TMDBMatchService
```python
class TMDBMatchService:
    """Handles show matching and validation logic."""
    def search_and_match(self, show_data: dict) -> List[TMDBMatchState]:
        """Find potential TMDB matches for a show.
        
        Scoring:
        - Title: 50 points (fuzzy match)
        - Network: 25 points (exact or alias match)
        - Executive Producers: 25 points (% matching)
        """
    
    def validate_match(self, match: TMDBMatchState) -> bool:
        """Validate and save a TMDB match."""
    
    def propose_match(self, match: TMDBMatchState) -> bool:
        """Store match data for review."""
```

##### TMDBDataMapper
```python
class TMDBDataMapper:
    """Maps TMDB API data to our schema."""
    def map_tv_show(self, show: TVShowDetails) -> dict:
        """Map TMDB show to our format."""
    
    def map_credits(self, credits: Credits) -> List[dict]:
        """Map TMDB credits to our team format."""
```

#### Data Services

##### ShowService
```python
class ShowService:
    """Handles show CRUD operations."""
    def get_shows(self, filters: dict = None) -> List[dict]:
        """Get shows with optional filtering."""
    
    def create_show(self, show_data: dict) -> dict:
        """Create new show with validation."""
    
    def update_show(self, show_id: int, data: dict) -> dict:
        """Update show with validation."""
    
    def delete_show(self, show_id: int) -> bool:
        """Soft delete a show."""
```

##### TeamService
```python
class TeamService:
    """Handles show team operations."""
    def get_team(self, show_id: int) -> List[dict]:
        """Get show team members."""
    
    def add_team_member(self, show_id: int, member_data: dict) -> dict:
        """Add team member with role."""
```

##### NetworkService
```python
class NetworkService:
    """Handles network operations."""
    def get_networks(self) -> List[dict]:
        """Get all active networks."""
    
    def search_networks(self, query: str) -> List[dict]:
        """Search networks by name/alias."""
    
    def get_network_stats(self) -> List[dict]:
        """Get network show statistics."""
```

#### Error Handling
```python
class ServiceError(Exception):
    """Base error for service layer."""
    pass

class ValidationError(ServiceError):
    """Data validation errors."""
    pass

class TMDBError(ServiceError):
    """TMDB API errors."""
    pass

# Usage
try:
    match = match_service.validate_match(match_data)
except ValidationError as e:
    state.error_message = str(e)
except TMDBError as e:
    state.error_message = f"TMDB Error: {str(e)}"
```
```

### 4. Pages

#### Admin Dashboard (/admin)
```python
@auth_required(['admin'])
def admin_show():
    """Main admin dashboard."""
    state = get_admin_state()
    
    # View selection
    state.current_view = st.radio(
        "Select Function",
        ["User Management", "Announcements", "TMDB Matches"]
    )
    
    if state.current_view == "User Management":
        render_user_management()
    elif state.current_view == "TMDB Matches":
        render_tmdb_matches()
```

##### User Management
- Role-based access control
- User invitations
- Role updates
- Activity tracking

##### TMDB Matching
- Search unmatched shows
- Review potential matches
- Validate/reject matches
- Match metrics

##### Announcements
- System notifications
- Maintenance alerts
- Feature updates

##### API Metrics
- Rate limit tracking
- Usage statistics
- Error monitoring

#### Data Entry (/data_entry)
```python
@auth_required(['editor', 'admin'])
def data_entry_show():
    """Show data entry form."""
    state = get_data_entry_state()
    
    if not state.form_started:
        render_show_search()
    else:
        render_show_form(state.show_form)
```

##### Features
- Add new shows
- Edit existing shows
- Batch operations
- Data validation
- Auto-complete
- Network/Studio search

##### Form Sections
- Basic Info (title, date, network)
- Team Members
- Production Details
- TMDB Integration

#### Analytics Dashboard (/)
```python
@auth_required(['viewer', 'editor', 'admin'])
def main_show():
    """Main analytics dashboard."""
    state = get_dashboard_state()
    
    # Filters
    render_filters(state.filters)
    
    # Charts
    render_network_chart(state.filters)
    render_trend_chart(state.filters)
```

##### Network Analysis
- Show counts by network
- Network market share
- Parent company analysis

##### Trend Analysis
- Show volume over time
- Genre popularity
- Network activity

##### Market Insights
- Studio partnerships
- Network strategies
- Genre trends

### 5. Components

#### TMDB Components

##### TMDBMatchView
```python
def render_match_card(match: TMDBMatchState, on_validate=None):
    """Render a TMDB match card."""
    # Generate unique key for this match card
    card_key = f"tmdb_match_{match.our_show_id}_{match.tmdb_id}"
    
    # Handle UI state
    if f"{card_key}_expanded" not in st.session_state:
        st.session_state[f"{card_key}_expanded"] = match.expanded
    match.expanded = st.session_state[f"{card_key}_expanded"]
    
    with st.expander(match.name, expanded=match.expanded):
        # Show comparison
        col1, col2 = st.columns(2)
        with col1:
            render_our_show(match)
        with col2:
            render_tmdb_show(match)
            
        # Action buttons
        if st.button(f"Validate ({match.confidence}%)"):
            on_validate(match)
```

##### TMDBSearchView
```python
def render_tmdb_search(state: TMDBMatchingState):
    """Render TMDB search interface."""
    # Search input
    query = st.text_input("Search TMDB", key="tmdb_search")
    
    if query:
        with st.spinner("Searching TMDB..."):
            matches = match_service.search_and_match({
                'title': query
            })
            state.matches = matches
            update_admin_state(state)
```

#### Data Entry Components

##### ShowForm
```python
def render_show_form(form: ShowFormState):
    """Render show data entry form."""
    with st.form("show_form"):
        # Basic info
        form.title = st.text_input("Title", form.title)
        form.network = st.selectbox(
            "Network",
            options=get_networks(),
            format_func=lambda x: x['network']
        )
        
        # Validation
        if st.form_submit_button("Save"):
            try:
                validate_show_data(form)
                save_show(form)
                st.success("Show saved!")
            except ValidationError as e:
                st.error(str(e))
```

##### TeamForm
```python
def render_team_form(form: TeamFormState):
    """Render team member form."""
    with st.form("team_form"):
        form.name = st.text_input("Name")
        form.role = st.selectbox("Role", options=get_roles())
        
        if st.form_submit_button("Add Member"):
            add_team_member(form)
```

#### Analytics Components

##### NetworkChart
```python
def render_network_chart(filters: FilterState):
    """Render network analysis chart."""
    # Get data
    data = get_network_stats(filters)
    
    # Create chart
    chart = alt.Chart(data).mark_bar().encode(
        x='network_name',
        y='count'
    )
    
    st.altair_chart(chart)
```

##### TrendChart
```python
def render_trend_chart(filters: FilterState):
    """Render trend analysis chart."""
    # Get data
    data = get_trend_data(filters)
    
    # Create chart
    chart = alt.Chart(data).mark_line().encode(
        x='date',
        y='count',
        color='network_name'
    )
    
    st.altair_chart(chart)
```

## Best Practices

1. **State Management**
   - Use page-scoped state
   - Always use proper serialization
   - Clear state appropriately

2. **Error Handling**
   - Use section-specific error messages
   - Clear errors after display
   - Proper error state management

3. **UI State**
   - Persist component state properly
   - Use consistent key naming
   - Clear state on transitions

4. **Database Operations**
   - Use service role for admin operations
   - Handle rate limits
   - Proper error handling

5. **TMDB Integration**
   - Consider network data quality
   - Handle missing data gracefully
   - Track match confidence

## Common Tasks

1. **Adding a New Page**
   ```python
   # 1. Create state class
   @dataclass
   class MyPageState:
       field1: str = ""
   
   # 2. Create state functions
   def get_my_page_state():
       state = get_page_state("my_page")
       if "my_page" not in state:
           state["my_page"] = asdict(MyPageState())
       return MyPageState(**state["my_page"])
   
   # 3. Create page
   def my_page_show():
       state = get_my_page_state()
       # ... page logic
   ```

2. **Database Operations**
   ```python
   # Get data
   client = get_supabase_client()
   response = client.table('my_table').select('*').execute()
   
   # Update with service role
   admin_client = get_admin_client()
   admin_client.table('my_table').update(data).eq('id', id).execute()
   ```

3. **TMDB Integration**
   ```python
   # Search and match
   match_service = TMDBMatchService()
   matches = match_service.search_and_match(show_data)
   
   # Validate match
   success = match_service.validate_match(match)
   ```
