"""Verify all required files and credentials are in place."""
import os
from pathlib import Path
from typing import List, Tuple

def check_file(path: Path, description: str) -> Tuple[bool, str]:
    """Check if a file exists and return status."""
    exists = path.exists()
    return exists, f"{'✓' if exists else '❌'} {description}: {path}"

def main():
    """Run all verification checks."""
    root = Path(__file__).parent.parent.parent
    required_files = [
        (root / 'config' / 'credentials.json', 'Google Sheets credentials'),
        (root / '.env', 'Environment variables'),
        (root / 'venv', 'Virtual environment')
    ]
    
    required_env = [
        'GOOGLE_SHEETS_CREDENTIALS_FILE',
        'GOOGLE_SHEETS_SPREADSHEET_ID',
        'SHOWS_SHEET_NAME',
        'TEAM_SHEET_NAME'
    ]
    
    print("\n=== Checking Required Files ===")
    all_files_present = True
    for path, desc in required_files:
        exists, message = check_file(path, desc)
        print(message)
        all_files_present = all_files_present and exists
    
    print("\n=== Checking Environment Variables ===")
    all_env_present = True
    for var in required_env:
        exists = var in os.environ
        print(f"{'✓' if exists else '❌'} {var}")
        all_env_present = all_env_present and exists
    
    if not all_files_present:
        print("\n❌ Missing required files!")
        print("Please follow setup instructions in docs/google_sheets_setup.md")
    
    if not all_env_present:
        print("\n❌ Missing environment variables!")
        print("Please check your .env file")
    
    if all_files_present and all_env_present:
        print("\n✅ All required files and variables present!")
        print("Ready to run connection tests")
        return 0
    return 1

if __name__ == "__main__":
    exit(main())
