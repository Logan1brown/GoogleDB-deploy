# TV Series Database Dashboard

A Streamlit dashboard for TV series market analysis and insights.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key
```

## Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

## Authentication

The dashboard uses role-based access control:
- Admin: Full access to all features and user management
- Editor: Can view dashboard and edit data
- Viewer: Read-only access to dashboard
