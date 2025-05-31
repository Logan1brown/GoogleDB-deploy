"""Test script for Show Optimizer components.

This script tests the basic functionality of the Show Optimizer components
without requiring integration into the full app.
"""

import pandas as pd
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.success_analysis import SuccessAnalyzer
from src.data_processing.show_optimizer.field_manager import FieldManager
from src.data_processing.show_optimizer.criteria_scorer import CriteriaScorer
from src.data_processing.show_optimizer.criteria_analyzer import CriteriaAnalyzer
from src.data_processing.show_optimizer.suggestion_analyzer import SuggestionAnalyzer
from src.data_processing.show_optimizer.show_optimizer import ShowOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_field_manager():
    """Test FieldManager functionality."""
    logger.info("Testing FieldManager...")
    
    # Initialize ShowsAnalyzer to get reference data
    shows_analyzer = ShowsAnalyzer()
    
    try:
        # Fetch comp data which includes reference data
        comp_df, reference_data = shows_analyzer.fetch_comp_data()
        logger.info(f"Fetched reference data with {len(reference_data)} tables")
        
        # Initialize FieldManager
        field_manager = FieldManager(reference_data)
        
        # Test getting field options
        logger.info("Testing get_options...")
        for field_name in field_manager.FIELD_CONFIGS.keys():
            options = field_manager.get_options(field_name)
            logger.info(f"Field {field_name}: {len(options)} options")
            if options:
                logger.info(f"Sample option: {options[0]}")
        
        # Test show matching with sample criteria
        logger.info("Testing match_shows...")
        shows_data = comp_df  # Use the comp_df as our shows data
        
        if not shows_data.empty:
            # Create sample criteria
            sample_criteria = {}
            
            # Add a genre if available
            if 'genre' in field_manager.FIELD_CONFIGS:
                genre_options = field_manager.get_options('genre')
                if genre_options:
                    sample_criteria['genre'] = genre_options[0].id
            
            # Add a network if available
            if 'network' in field_manager.FIELD_CONFIGS:
                network_options = field_manager.get_options('network')
                if network_options:
                    sample_criteria['network'] = network_options[0].id
            
            logger.info(f"Sample criteria: {sample_criteria}")
            
            # Match shows
            matched_shows, sample_size = field_manager.match_shows(sample_criteria, shows_data)
            logger.info(f"Matched {sample_size} shows")
            
            # Test confidence calculation
            confidence = field_manager.calculate_confidence(sample_criteria)
            logger.info(f"Confidence: {confidence}")
        else:
            logger.warning("Shows data is empty, skipping match_shows test")
    except Exception as e:
        logger.error(f"Error in test_field_manager: {e}", exc_info=True)

def test_criteria_scorer():
    """Test CriteriaScorer functionality."""
    logger.info("Testing CriteriaScorer...")
    
    try:
        # Initialize required components
        shows_analyzer = ShowsAnalyzer()
        success_analyzer = SuccessAnalyzer(shows_analyzer)
        
        # Initialize CriteriaScorer
        criteria_scorer = CriteriaScorer(shows_analyzer, success_analyzer)
        
        # Test fetching criteria data
        logger.info("Testing fetch_criteria_data...")
        criteria_data = criteria_scorer.fetch_criteria_data()
        if not criteria_data.empty:
            logger.info(f"Fetched {len(criteria_data)} rows of criteria data")
            logger.info(f"Sample columns: {list(criteria_data.columns)[:10]}")
        else:
            logger.warning("Criteria data is empty")
            return
        
        # Test network scoring with sample criteria
        logger.info("Testing calculate_network_scores...")
        field_manager = criteria_scorer.field_manager
        
        # Create sample criteria
        sample_criteria = {}
        
        # Add a genre if available
        if 'genre' in field_manager.FIELD_CONFIGS:
            genre_options = field_manager.get_options('genre')
            if genre_options:
                sample_criteria['genre'] = genre_options[0].id
        
        logger.info(f"Sample criteria: {sample_criteria}")
        
        # Calculate network scores
        if sample_criteria:
            network_matches = criteria_scorer.calculate_network_scores(str(sample_criteria))
            logger.info(f"Found {len(network_matches)} network matches")
            for i, match in enumerate(network_matches[:5]):  # Show top 5
                logger.info(f"Network {i+1}: {match.network_name} - Score: {match.success_probability:.2f} - Confidence: {match.confidence}")
        else:
            logger.warning("No sample criteria available, skipping network scoring test")
        
        # Test component scoring
        logger.info("Testing calculate_component_scores...")
        if sample_criteria:
            component_scores = criteria_scorer.calculate_component_scores(sample_criteria)
            logger.info(f"Found {len(component_scores)} component scores")
            for component, score in component_scores.items():
                logger.info(f"Component {component}: Score: {score.score:.2f} - Confidence: {score.confidence}")
        else:
            logger.warning("No sample criteria available, skipping component scoring test")
    except Exception as e:
        logger.error(f"Error in test_criteria_scorer: {e}", exc_info=True)

def test_criteria_analyzer():
    """Test CriteriaAnalyzer functionality."""
    logger.info("Testing CriteriaAnalyzer...")
    
    try:
        # Initialize required components
        shows_analyzer = ShowsAnalyzer()
        success_analyzer = SuccessAnalyzer(shows_analyzer)
        
        # Initialize CriteriaAnalyzer
        criteria_analyzer = CriteriaAnalyzer(shows_analyzer, success_analyzer)
        
        # Create sample criteria
        sample_criteria = {}
        
        # Add a genre if available
        field_manager = criteria_analyzer.criteria_scorer.field_manager
        if 'genre' in field_manager.FIELD_CONFIGS:
            genre_options = field_manager.get_options('genre')
            if genre_options:
                sample_criteria['genre'] = genre_options[0].id
        
        logger.info(f"Sample criteria: {sample_criteria}")
        
        if not sample_criteria:
            logger.warning("No sample criteria available, skipping tests")
            return
        
        # Test finding matching networks
        logger.info("Testing find_matching_networks...")
        network_tiers = criteria_analyzer.find_matching_networks(sample_criteria)
        logger.info(f"Found {len(network_tiers)} network tiers")
        for tier_name, networks in network_tiers.items():
            logger.info(f"Tier '{tier_name}': {len(networks)} networks")
            for i, network in enumerate(networks[:3]):  # Show top 3
                logger.info(f"  Network {i+1}: {network.network_name} - Score: {network.success_probability:.2f}")
        
        # Test analyzing components
        logger.info("Testing analyze_components...")
        component_scores = criteria_analyzer.analyze_components(sample_criteria)
        logger.info(f"Found {len(component_scores)} component scores")
        for component, score in component_scores.items():
            logger.info(f"Component {component}: Score: {score.score:.2f} - Confidence: {score.confidence}")
        
        # Test identifying success factors
        logger.info("Testing identify_success_factors...")
        success_factors = criteria_analyzer.identify_success_factors(sample_criteria, limit=3)
        logger.info(f"Found {len(success_factors)} success factors")
        for i, factor in enumerate(success_factors):
            logger.info(f"Factor {i+1}: {factor.criteria_type} '{factor.criteria_name}' - Impact: {factor.impact_score:.2f} - Confidence: {factor.confidence}")
        
        # Test ranking networks
        logger.info("Testing rank_networks_by_compatibility...")
        ranked_networks = criteria_analyzer.rank_networks_by_compatibility(sample_criteria, limit=5)
        logger.info(f"Found {len(ranked_networks)} ranked networks")
        for i, network in enumerate(ranked_networks):
            logger.info(f"Rank {i+1}: {network.network_name} - Compatibility: {network.compatibility_score:.2f}")
        
        # Test analyzing criteria confidence
        logger.info("Testing analyze_criteria_confidence...")
        confidence_levels = criteria_analyzer.analyze_criteria_confidence(sample_criteria)
        logger.info(f"Confidence levels: {confidence_levels}")
        
        # Test getting overall success rate
        logger.info("Testing get_overall_success_rate...")
        success_rate, confidence = criteria_analyzer.get_overall_success_rate(sample_criteria)
        logger.info(f"Overall success rate: {success_rate:.2f} - Confidence: {confidence}")
        
    except Exception as e:
        logger.error(f"Error in test_criteria_analyzer: {e}", exc_info=True)

def test_suggestion_analyzer():
    """Test SuggestionAnalyzer functionality."""
    logger.info("Testing SuggestionAnalyzer...")
    
    try:
        # Initialize required components
        shows_analyzer = ShowsAnalyzer()
        success_analyzer = SuccessAnalyzer(shows_analyzer)
        
        # Initialize SuggestionAnalyzer
        suggestion_analyzer = SuggestionAnalyzer(shows_analyzer, success_analyzer)
        
        # Create sample criteria
        sample_criteria = {}
        
        # Add a genre if available
        field_manager = suggestion_analyzer.criteria_analyzer.criteria_scorer.field_manager
        if 'genre' in field_manager.FIELD_CONFIGS:
            genre_options = field_manager.get_options('genre')
            if genre_options and len(genre_options) > 1:
                # Add two genres to potentially create a conflict
                sample_criteria['genre'] = [genre_options[0].id, genre_options[1].id]
            elif genre_options:
                sample_criteria['genre'] = genre_options[0].id
        
        logger.info(f"Sample criteria: {sample_criteria}")
        
        if not sample_criteria:
            logger.warning("No sample criteria available, skipping tests")
            return
        
        # Test analyzing show concept
        logger.info("Testing analyze_show_concept...")
        summary = suggestion_analyzer.analyze_show_concept(sample_criteria)
        
        logger.info(f"Overall success probability: {summary.overall_success_probability:.2f} - Confidence: {summary.confidence}")
        
        logger.info(f"Top networks: {len(summary.top_networks)}")
        for i, network in enumerate(summary.top_networks[:3]):  # Show top 3
            logger.info(f"Network {i+1}: {network.network_name} - Score: {network.success_probability:.2f}")
        
        logger.info(f"Component scores: {len(summary.component_scores)}")
        for component, score in summary.component_scores.items():
            logger.info(f"Component {component}: Score: {score.score:.2f} - Confidence: {score.confidence}")
        
        logger.info(f"Success factors: {len(summary.success_factors)}")
        for i, factor in enumerate(summary.success_factors):
            logger.info(f"Factor {i+1}: {factor.criteria_type} '{factor.criteria_name}' - Impact: {factor.impact_score:.2f}")
        
        logger.info(f"Recommendations: {len(summary.recommendations)}")
        for i, rec in enumerate(summary.recommendations):
            logger.info(f"Recommendation {i+1}: {rec.recommendation_type} {rec.criteria_type} - Impact: {rec.impact_score:.2f}")
            logger.info(f"  Explanation: {rec.explanation}")
        
    except Exception as e:
        logger.error(f"Error in test_suggestion_analyzer: {e}", exc_info=True)


def test_show_optimizer():
    """Test ShowOptimizer functionality."""
    logger.info("Testing ShowOptimizer...")
    
    try:
        # Initialize ShowOptimizer
        optimizer = ShowOptimizer()
        
        # Test initialization
        logger.info("Testing initialization...")
        initialized = optimizer.initialize()
        logger.info(f"Initialization successful: {initialized}")
        
        if not initialized:
            logger.warning("Initialization failed, skipping tests")
            return
        
        # Create sample criteria
        sample_criteria = {}
        
        # Add a genre if available
        if optimizer.field_manager and 'genre' in optimizer.field_manager.FIELD_CONFIGS:
            genre_options = optimizer.get_field_options('genre')
            if genre_options and len(genre_options) > 1:
                # Add two genres to potentially create a conflict
                sample_criteria['genre'] = [genre_options[0].id, genre_options[1].id]
            elif genre_options:
                sample_criteria['genre'] = genre_options[0].id
        
        logger.info(f"Sample criteria: {sample_criteria}")
        
        if not sample_criteria:
            logger.warning("No sample criteria available, skipping tests")
            return
        
        # Test validating criteria
        logger.info("Testing validate_criteria...")
        normalized_criteria, errors = optimizer.validate_criteria(sample_criteria)
        logger.info(f"Normalized criteria: {normalized_criteria}")
        logger.info(f"Validation errors: {errors}")
        
        # Test matching shows
        logger.info("Testing match_shows...")
        matching_shows, total_matches = optimizer.match_shows(normalized_criteria)
        logger.info(f"Found {total_matches} matching shows")
        
        # Test analyzing concept
        logger.info("Testing analyze_concept...")
        summary = optimizer.analyze_concept(normalized_criteria)
        if summary:
            logger.info(f"Overall success probability: {summary.overall_success_probability:.2f} - Confidence: {summary.confidence}")
            logger.info(f"Top networks: {len(summary.top_networks)}")
            logger.info(f"Component scores: {len(summary.component_scores)}")
            logger.info(f"Success factors: {len(summary.success_factors)}")
            logger.info(f"Recommendations: {len(summary.recommendations)}")
        else:
            logger.warning("No summary returned from analyze_concept")
        
        # Test getting network tiers
        logger.info("Testing get_network_tiers...")
        network_tiers = optimizer.get_network_tiers(normalized_criteria)
        logger.info(f"Found {len(network_tiers)} network tiers")
        
        # Test getting success factors
        logger.info("Testing get_success_factors...")
        success_factors = optimizer.get_success_factors(normalized_criteria)
        logger.info(f"Found {len(success_factors)} success factors")
        
        # Test getting recommendations
        logger.info("Testing get_recommendations...")
        recommendations = optimizer.get_recommendations(normalized_criteria)
        logger.info(f"Found {len(recommendations)} recommendations")
        
        # Test getting component scores
        logger.info("Testing get_component_scores...")
        component_scores = optimizer.get_component_scores(normalized_criteria)
        logger.info(f"Found {len(component_scores)} component scores")
        
        # Test getting overall success rate
        logger.info("Testing get_overall_success_rate...")
        success_rate, confidence = optimizer.get_overall_success_rate(normalized_criteria)
        logger.info(f"Overall success rate: {success_rate:.2f} - Confidence: {confidence}")
        
    except Exception as e:
        logger.error(f"Error in test_show_optimizer: {e}", exc_info=True)


def main():
    """Run all tests."""
    logger.info("Starting Show Optimizer component tests...")
    
    try:
        # Test FieldManager
        test_field_manager()
        
        # Test CriteriaScorer
        test_criteria_scorer()
        
        # Test CriteriaAnalyzer
        test_criteria_analyzer()
        
        # Test SuggestionAnalyzer
        test_suggestion_analyzer()
        
        # Test ShowOptimizer
        test_show_optimizer()
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)

if __name__ == "__main__":
    main()
