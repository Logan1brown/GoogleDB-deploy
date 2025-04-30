"""Update show analysis tables with new descriptions."""

from supabase import create_client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY')
supabase = create_client(url, key)

# Show analysis data
SHOW_ANALYSES = {
    3806: {  # Dungeons & Dragons
        'description': {
            'time_setting': 'fantasy_era',  # Fantasy setting
            'location_setting': 'multiple_settings',  # Various D&D locations
            'tone': 'epic'  # Fantasy adventure
        },
        'plot_elements': [
            'action_adventure',  # Fantasy adventure
            'quest_journey'  # Typical D&D journey
        ],
        'thematic_elements': [
            'power',  # Fantasy power dynamics
            'duty'  # Quest obligations
        ]
    },
    3826: {  # Fledgling
        'description': {
            'time_setting': 'contemporary',  # Modern setting
            'location_setting': 'multiple_settings',  # Various locations
            'tone': 'dark'  # Horror/thriller
        },
        'plot_elements': [
            'mystery_reveal',  # Uncovering truth
            'supernatural'  # Vampire elements
        ],
        'thematic_elements': [
            'identity',  # Self-discovery
            'survival'  # Survival story
        ]
    },
    3874: {  # Invitation to a Bonfire
        'description': {
            'time_setting': 'mid_century',  # 1930s
            'location_setting': 'campus',  # Boarding school
            'tone': 'suspenseful'  # Psychological thriller
        },
        'plot_elements': [
            'romance',  # Love triangle
            'mystery_reveal'  # Psychological elements
        ],
        'thematic_elements': [
            'love',  # Romance
            'power'  # Power dynamics
        ]
    },
    3888: {  # King of the Hill
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'urban_suburban',  # Suburban setting
            'tone': 'humorous'  # Comedy series
        },
        'plot_elements': [
            'family_dynamics',  # Family relationships
            'social_commentary'  # Commentary on society
        ],
        'thematic_elements': [
            'family',  # Family bonds
            'change'  # Personal growth
        ]
    },
    3941: {  # On the Spectrum
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'urban_suburban',  # LA suburbs
            'tone': 'dramatic'  # Drama series
        },
        'plot_elements': [
            'coming_of_age',  # Growing up
            'social_commentary'  # Commentary on society
        ],
        'thematic_elements': [
            'identity',  # Self-discovery
            'community'  # Friendship and support
        ]
    },
    3943: {  # Once Upon a Time in Aztlan
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'urban_suburban',  # LA suburbs
            'tone': 'dramatic'  # Drama series
        },
        'plot_elements': [
            'family_dynamics',  # Family story
            'social_commentary'  # American dream commentary
        ],
        'thematic_elements': [
            'family',  # Family bonds
            'identity'  # Cultural identity
        ]
    },
    3958: {  # Peep Show
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'workplace',  # Office setting
            'tone': 'humorous'  # Comedy series
        },
        'plot_elements': [
            'conflict_power',  # Boss/assistant dynamic
            'social_commentary'  # Workplace commentary
        ],
        'thematic_elements': [
            'power',  # Power dynamics
            'ambition'  # Career goals
        ]
    },
    3981: {  # Sammy
        'description': {
            'time_setting': 'recent_past',  # Mob era
            'location_setting': 'urban_suburban',  # City setting
            'tone': 'dramatic'  # Drama series
        },
        'plot_elements': [
            'crime',  # Mob story
            'transformation'  # Character journey
        ],
        'thematic_elements': [
            'power',  # Mob power
            'morality'  # Moral choices
        ]
    },
    4007: {  # Supercrooks
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'multiple_settings',  # Various locations
            'tone': 'action_packed'  # Action series
        },
        'plot_elements': [
            'crime',  # Heist story
            'action_adventure'  # Action sequences
        ],
        'thematic_elements': [
            'ambition',  # Heist goals
            'loyalty'  # Team dynamics
        ]
    },
    4018: {  # That Black Ass Show
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'urban_suburban',  # City setting
            'tone': 'humorous'  # Comedy series
        },
        'plot_elements': [
            'coming_of_age',  # Young adult life
            'social_commentary'  # Social issues
        ],
        'thematic_elements': [
            'identity',  # Personal identity
            'community'  # Friendship bonds
        ]
    },
    4026: {  # The Batman: Gotham PD
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'major_city',  # Gotham City
            'tone': 'dark'  # Dark tone
        },
        'plot_elements': [
            'crime',  # Police drama
            'conspiracy'  # Corruption story
        ],
        'thematic_elements': [
            'justice',  # Law enforcement
            'morality'  # Moral choices
        ]
    },
    4041: {  # The Devil in the White City
        'description': {
            'time_setting': 'industrial',  # 1893
            'location_setting': 'major_city',  # Chicago
            'tone': 'dark'  # Dark historical drama
        },
        'plot_elements': [
            'crime',  # Serial killer story
            'social_commentary'  # Historical commentary
        ],
        'thematic_elements': [
            'ambition',  # Architect's vision
            'morality'  # Good vs evil
        ]
    },
    4044: {  # The Driver
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'major_city',  # Urban setting
            'tone': 'dramatic'  # Drama series
        },
        'plot_elements': [
            'crime',  # Crime elements
            'transformation'  # Character journey
        ],
        'thematic_elements': [
            'morality',  # Moral choices
            'survival'  # Survival story
        ]
    },
    4102: {  # The Trenches
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'multiple_settings',  # Various locations
            'tone': 'humorous'  # Comedy series
        },
        'plot_elements': [
            'supernatural',  # Monster elements
            'family_dynamics'  # Family story
        ],
        'thematic_elements': [
            'family',  # Family bonds
            'duty'  # Monster hunting duty
        ]
    },
    3746: {  # Avalon
        'description': {
            'time_setting': 'contemporary',  # Present day
            'location_setting': 'isolated',  # Catalina Island
            'tone': 'suspenseful'  # Detective series
        },
        'plot_elements': [
            'investigation',  # Detective work
            'mystery_reveal'  # Uncovering secrets
        ],
        'thematic_elements': [
            'justice'  # Detective work/crime solving
        ]
    },
    3754: {  # Beauty and the Beast: Little Town
        'description': {
            'time_setting': 'fantasy_era',  # Fairy tale setting
            'location_setting': 'small_town',  # The "little town"
            'tone': 'whimsical'  # Disney style
        },
        'plot_elements': [
            'transformation',  # Character development
            'quest_journey'  # Journey of discovery
        ],
        'thematic_elements': [
            'identity'  # Character backstory/development
        ]
    },
    3764: {  # Brother From Another Mother
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'rural',  # Texas ranch
            'tone': 'humorous'  # Comedy series
        },
        'plot_elements': [
            'family_dynamics'  # Living together
        ],
        'thematic_elements': [
            'loyalty',  # Testing friendship
            'family'  # Family relationships
        ]
    },
    3769: {  # Camp Friends
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'wilderness',  # Summer camp
            'tone': 'humorous'  # Comedy series
        },
        'plot_elements': [
            'coming_of_age',  # CIT summer
            'mystery_reveal'  # Mysterious new girl
        ],
        'thematic_elements': [
            'community',  # Camp friends
            'identity'  # Personal growth
        ]
    },
    3785: {  # Daddy Ball
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'urban_suburban',  # Long Island
            'tone': 'dramatic'  # Limited series
        },
        'plot_elements': [
            'conflict_power',  # Feud escalation
            'social_commentary'  # Commentary on parenting/sports
        ],
        'thematic_elements': [
            'ambition',  # Competitive drive
            'morality'  # Ethical choices
        ]
    },
    3791: {  # Dead Day
        'description': {
            'time_setting': 'contemporary',  # Modern day
            'location_setting': 'multiple_settings',  # Various locations
            'tone': 'dark'  # Supernatural drama
        },
        'plot_elements': [
            'supernatural',  # Dead returning
            'revenge'  # Unfinished business
        ],
        'thematic_elements': [
            'justice',  # Settling scores
            'redemption'  # Second chances
        ]
    },
    3796: {  # Demimonde
        'description': {
            'time_setting': 'near_future',  # Sci-fi setting
            'location_setting': 'fictional_world',  # Other world
            'tone': 'suspenseful'  # Sci-fi drama
        },
        'plot_elements': [
            'mystery_reveal',  # Unraveling conspiracy
            'conspiracy'  # Hidden plots
        ],
        'thematic_elements': [
            'family',  # Family reunion
            'survival'  # Overcoming challenges
        ]
    }
}

def get_type_id(table: str, code: str) -> int:
    """Get the ID for a type from its table using the code field."""
    result = supabase.table(table).select('id').eq('code', code).execute()
    if not result.data:
        raise ValueError(f"No {table} found with code: {code}")
    return result.data[0]['id']

def update_show_analysis(show_id: int, analysis: dict):
    """Update analysis tables for a show."""
    try:
        # 1. Update show_description_analysis
        desc = analysis['description']
        desc_data = {
            'show_id': show_id,
            'time_setting_id': get_type_id('time_setting_types', desc['time_setting']),
            'location_setting_id': get_type_id('location_setting_types', desc['location_setting']),
            'tone_id': get_type_id('tone_types', desc['tone'])
        }
        
        # Upsert description analysis
        supabase.table('show_description_analysis').upsert(desc_data).execute()
        
        # 2. Update show_plot_element_list
        # First delete existing entries
        supabase.table('show_plot_element_list').delete().eq('show_id', show_id).execute()
        
        # Then insert new ones
        plot_elements = []
        for element in analysis['plot_elements']:
            element_id = get_type_id('plot_element_types', element)
            plot_elements.append({
                'show_id': show_id,
                'plot_element_id': element_id
            })
        if plot_elements:
            supabase.table('show_plot_element_list').insert(plot_elements).execute()
        
        # 3. Update show_thematic_element_list
        # First delete existing entries
        supabase.table('show_thematic_element_list').delete().eq('show_id', show_id).execute()
        
        # Then insert new ones
        thematic_elements = []
        for element in analysis['thematic_elements']:
            element_id = get_type_id('thematic_element_types', element)
            thematic_elements.append({
                'show_id': show_id,
                'thematic_element_id': element_id
            })
        if thematic_elements:
            supabase.table('show_thematic_element_list').insert(thematic_elements).execute()
            
        print(f"Successfully updated analysis for show {show_id}")
            
    except Exception as e:
        print(f"Error updating show {show_id}: {str(e)}")
        raise

def main():
    """Update all show analyses."""
    for show_id, analysis in SHOW_ANALYSES.items():
        update_show_analysis(show_id, analysis)
        
if __name__ == "__main__":
    main()
