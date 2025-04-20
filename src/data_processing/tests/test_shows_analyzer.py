"""Test the ShowsAnalyzer with Supabase materialized views."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
env_path = project_root / '.env'
if env_path.exists():
    print(f"Loading environment from {env_path}")
    load_dotenv(env_path)
else:
    print("Warning: .env file not found!")
    print("Please create .env file with SUPABASE_URL and SUPABASE_KEY")
    sys.exit(1)

# Verify Supabase environment variables
required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
for var in required_vars:
    if not os.getenv(var):
        print(f"Error: {var} environment variable is required")
        sys.exit(1)

from src.data_processing.analyze_shows import ShowsAnalyzer

def test_shows_analyzer():
    try:
        # Initialize analyzer and fetch data
        analyzer = ShowsAnalyzer()
        shows_df, team_df, network_df = analyzer.fetch_data(force=True)
        
        # Test shows data
        print("\nShows DataFrame Info:")
        print("=====================")
        print(f"Number of shows: {len(shows_df)}")
        print(f"Columns: {shows_df.columns.tolist()}")
        print("\nSample of first 5 shows:")
        print(shows_df[['title', 'network_name', 'genre_name', 'status_name']].head())
        
        # Test team data
        print("\nTeam Data Info:")
        print("===============")
        if team_df is not None and len(team_df) > 0:
            print(f"Number of team entries: {len(team_df)}")
            print(f"Columns: {team_df.columns.tolist()}")
            print("\nSample of first 3 team entries:")
            print(team_df.head(3))
        else:
            print("No team data available")
        
        # Test network stats
        print("\nNetwork Stats Info:")
        print("==================")
        if network_df is not None and len(network_df) > 0:
            print(f"Number of networks: {len(network_df)}")
            print(f"Columns: {network_df.columns.tolist()}")
            print("\nSample of first 3 networks:")
            print(network_df[['network_name', 'total_shows', 'active_shows']].head(3))
        else:
            print("No network stats available")
        
        # Test basic stats
        print("\nBasic Statistics:")
        print("=================")
        stats = analyzer.generate_basic_stats()
        print(f"Total Shows: {stats['total_shows']}")
        print(f"Active Shows: {stats['active_shows']}")
        print(f"Average Episodes: {stats['avg_episodes']:.2f}")
        print("\nTop 3 Networks:")
        if 'networks' in stats:
            networks = [(name, data['total_shows'], data['active_shows']) 
                       for name, data in stats['networks'].items()]
            for network, total, active in sorted(networks, key=lambda x: x[1], reverse=True)[:3]:
                print(f"- {network}: {total} total shows ({active} active)")
        
        # Test array fields
        print("\nArray Field Examples:")
        print("===================")
        sample_show = shows_df.iloc[0]
        print(f"Show: {sample_show['title']}")
        print(f"Studios: {sample_show['studio_names']}")
        print(f"Subgenres: {sample_show['subgenre_names']}")
        
        # Test profile report generation
        print("\nGenerating Profile Report...")
        report_file = 'src/data_processing/tests/reports/test_profile_report.html'
        analyzer.generate_profile_report(output_file=report_file)
        print(f"Profile report generated: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"\nError testing shows analyzer: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == '__main__':
    print("\nTesting ShowsAnalyzer with Supabase...")
    success = test_shows_analyzer()
    print(f"\nTest {'succeeded' if success else 'failed'}")
