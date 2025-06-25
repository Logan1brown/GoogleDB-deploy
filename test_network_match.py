"""Test script for NetworkMatch class."""

from src.data_processing.show_optimizer.optimizer_data_contracts import NetworkMatch

def test_network_match():
    """Test NetworkMatch attribute access vs dictionary access."""
    # Create a NetworkMatch object
    network = NetworkMatch(
        network_id=1,
        network_name='Test Network',
        compatibility_score=0.85,
        success_probability=0.75,
        sample_size=100,
        confidence='high'
    )
    
    print("Testing attribute access:")
    print(f"Network ID: {network.network_id}")
    print(f"Network Name: {network.network_name}")
    print(f"Compatibility Score: {network.compatibility_score}")
    
    print("\nTesting dictionary-style access (should fail):")
    try:
        print(f"Network ID: {network['network_id']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return network

if __name__ == "__main__":
    test_network_match()
