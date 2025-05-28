#!/usr/bin/env python3
"""Local script to collect RT data for a show"""

import sys
from sync_collector import RTCollector

def main():
    if len(sys.argv) != 2:
        print("Usage: python collect_show.py <show_id>")
        sys.exit(1)
        
    show_id = int(sys.argv[1])
    
    with RTCollector() as collector:
        result = collector.collect_show_data(show_id)
        print(result)

if __name__ == "__main__":
    main()
