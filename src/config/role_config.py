"""Role Configuration.

This module contains shared role-related constants used across different components
of the application, including market analysis and creative networks.
"""

# Standard role types used across the application
STANDARD_ROLES = {
    # Creative roles
    'Creator': 'Creative',
    'Writer': 'Creative',
    'Director': 'Creative',
    'Showrunner': 'Creative',
    'Co-Showrunner': 'Creative',
    'Writer/Executive Producer': 'Creative',
    'Creative Producer': 'Creative',
    
    # Production roles
    'Executive Producer': 'Production',
    'Producer': 'Production',
    'Co-Producer': 'Production',
    'Line Producer': 'Production',
    
    # Executive roles
    'Studio Executive': 'Executive',
    'Network Executive': 'Executive',
    'Development Executive': 'Executive',
    
    # Talent roles
    'Actor': 'Talent',
    'Host': 'Talent'
}

# Role categories for grouping and analysis
ROLE_CATEGORIES = {
    'Creative': 'Primary creative decision makers',
    'Production': 'Production management and oversight',
    'Executive': 'Studio and network leadership',
    'Talent': 'On-screen talent'
}
