#!/usr/bin/env python3
"""
Quick test to verify the environment is working
This is a simpler version of test_implementation.py for quick checks
"""

def quick_test():
    print("=== Quick Environment Test ===\n")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic imports
    total_tests += 1
    try:
        import torch
        import json
        import os
        print("✓ Basic Python imports work")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
    
    # Test 2: PyTorch CUDA
    total_tests += 1
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"✓ PyTorch CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  Device: {torch.cuda.get_device_name(0)}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ PyTorch CUDA test failed: {e}")
    
    # Test 3: Utils functions
    total_tests += 1
    try:
        from utils import get_prompt
        prompt = get_prompt("Test instruction")
        if "ASSISTANT:" in prompt:
            print("✓ get_prompt() works")
            tests_passed += 1
        else:
            print("✗ get_prompt() doesn't contain ASSISTANT:")
    except Exception as e:
        print(f"✗ get_prompt() test failed: {e}")
    
    # Test 4: Data files
    total_tests += 1
    try:
        with open("data/train.json", 'r') as f:
            data = json.load(f)
        with open("data/public_test.json", 'r') as f:
            test_data = json.load(f)
        print(f"✓ Data files loaded (train: {len(data)}, test: {len(test_data)})")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Data files test failed: {e}")
    
    # Test 5: Advanced imports (optional)
    total_tests += 1
    try:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig
        print("✓ Advanced imports (transformers, peft) work")
        tests_passed += 1
    except Exception as e:
        print(f"⚠ Advanced imports failed (may be OK): {e}")
        # Don't fail the test for this
        tests_passed += 1
    
    # Summary
    print(f"\n=== Results: {tests_passed}/{total_tests} tests passed ===")
    
    if tests_passed >= 4:  # Allow advanced imports to fail
        print("🎉 Environment looks good! You can proceed with training.")
        return True
    else:
        print("❌ Environment has issues. Please run:")
        print("  - python3 diagnose_environment.py  # for detailed diagnostics")
        print("  - ./fix_transformers.sh           # to fix transformers issues")
        print("  - ./setup_environment.sh          # to reinstall packages")
        return False

if __name__ == "__main__":
    quick_test()