"""
Test script to verify installation and dependencies for the Injury Prediction ML project.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Seaborn import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import joblib
        print("✓ Joblib imported successfully")
    except ImportError as e:
        print(f"✗ Joblib import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "Dataset.csv",
        "injury_prediction_ml.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} not found")
            all_files_exist = False
    
    return all_files_exist

def test_main_script():
    """Test if the main script can be imported."""
    print("\nTesting main script import...")
    
    try:
        from injury_prediction_ml import InjuryPredictionML
        print("✓ InjuryPredictionML class imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import InjuryPredictionML: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing main script: {e}")
        return False

def test_dataset_access():
    """Test if the dataset can be read."""
    print("\nTesting dataset access...")
    
    try:
        import pandas as pd
        df = pd.read_csv("Dataset.csv")
        print(f"✓ Dataset loaded successfully. Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return True
    except FileNotFoundError:
        print("✗ Dataset.csv not found")
        return False
    except Exception as e:
        print(f"✗ Error reading dataset: {e}")
        return False

def main():
    """Run all tests."""
    print("Injury Prediction ML - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Main Script Import", test_main_script),
        ("Dataset Access", test_dataset_access)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("You can now run: python injury_prediction_ml.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("Make sure to:")
        print("1. Install all dependencies: pip install -r requirements.txt")
        print("2. Ensure all files are in the correct directory")
        print("3. Check that Dataset.csv is present")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 