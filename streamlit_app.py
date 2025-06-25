"""
TV Series Database Dashboard
Entry point for Streamlit Cloud deployment
"""
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the main function
from src.dashboard.app import main

if __name__ == "__main__":
    main()
