#!/usr/bin/env python3
"""
Test script to verify the modular structure works correctly.
This script tests imports and basic functionality without running the full training.
"""

def test_imports():
    """Test that all modules can be imported successfully"""
    try:
        import config
        import data_loader
        import dataset
        import model
        import trainer
        import utils
        import main
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration values"""
    try:
        import config
        assert hasattr(config, 'device')
        assert hasattr(config, 'NUM_EPOCHS')
        assert hasattr(config, 'BATCH_SIZE')
        assert hasattr(config, 'HIDDEN_DIM')
        print("✓ Configuration values are properly defined")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_model_creation():
    """Test model creation without data"""
    try:
        import model
        import config
        
        # Create a simple model with dummy parameters
        dummy_model = model.RGCNModel(
            num_relations=10,
            hidden_dim=64,
            feature_num_classes={'occupation': 5, 'work_location': 3}
        )
        print("✓ Model creation successful")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing modular structure...")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The modular structure is ready.")
    else:
        print("✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
