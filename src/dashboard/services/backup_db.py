"""Backup all tables from Supabase database"""
import json
from datetime import datetime
import os
from src.config.supabase_client import get_client  # Use centralized client

def backup_database():
    """Backup all tables to a JSON file"""
    # List of tables to backup
    tables = [
        'shows',
        'show_team',
        'tmdb_success_metrics',
        'status_types',
        'network_list',
        'studio_list',
        'source_types',
        'role_types',
        'genre_list',
        'subgenre_list',
        'order_types'
    ]
    
    backup = {}
    
    # Get Supabase client
    supabase = get_client(use_service_key=True)
    if not supabase:
        raise ValueError("Could not initialize Supabase client")
    
    # Backup each table
    for table_name in tables:
        print(f"Backing up {table_name}...")
        
        # Get all data from table
        result = supabase.table(table_name).select('*').execute()
        backup[table_name] = result.data
    
    # Save to backups folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'database_backup_{timestamp}.json'
    backup_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backups')
    backup_path = os.path.join(backup_dir, filename)
    
    with open(backup_path, 'w') as f:
        json.dump(backup, f, indent=2)
    
    print(f"\nBackup saved to {backup_path}")

if __name__ == '__main__':
    backup_database()
