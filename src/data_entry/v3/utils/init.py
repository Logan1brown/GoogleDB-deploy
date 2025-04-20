"""Initialize environment and configuration for the data entry app."""
import os
from pathlib import Path
from dotenv import load_dotenv

def init_environment():
    """Initialize environment variables."""
    # Find the project root (where .env is located)
    current_dir = Path(__file__).resolve().parent
    while current_dir.name != 'GoogleDB' and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    
    # Load environment variables
    env_path = current_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        raise RuntimeError(f"Could not find .env file in {current_dir}")
    
    # Verify required variables
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
