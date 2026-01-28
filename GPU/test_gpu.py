"""
GPU Test Script for String Art Generator
Tests if GPU is available and working correctly.
"""

import sys

def test_gpu():
    """Test GPU availability and basic operations."""
    print("=" * 60)
    print("GPU Test for String Art Generator")
    print("=" * 60)
    print()
    
    # Test 1: Check if PyTorch is installed
    print("1. Checking PyTorch installation...")
    try:
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("   ✗ PyTorch is NOT installed")
        print("   Install with: pip install torch")
        return False
    
    # Test 2: Check CUDA availability
    print("\n2. Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"   ✓ CUDA is available!")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ Number of GPUs: {torch.cuda.device_count()}")
        
        # Get GPU information
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     Name: {torch.cuda.get_device_name(i)}")
            print(f"     Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("   ✗ CUDA is NOT available")
        print("   This could mean:")
        print("     - No NVIDIA GPU detected")
        print("     - CUDA drivers not installed")
        print("     - PyTorch installed without CUDA support")
        print("   The code will use CPU instead.")
    
    # Test 3: Test basic GPU operations
    print("\n3. Testing GPU operations...")
    if cuda_available:
        try:
            device = torch.device('cuda')
            print(f"   Using device: {device}")
            
            # Create a test tensor on GPU
            print("   Creating test tensor on GPU...")
            test_tensor = torch.randn(1000, 1000, device=device)
            print(f"   ✓ Tensor created on GPU: {test_tensor.device}")
            
            # Perform some operations
            print("   Performing matrix multiplication...")
            result = torch.matmul(test_tensor, test_tensor)
            print(f"   ✓ Matrix multiplication successful")
            print(f"   Result shape: {result.shape}")
            
            # Test memory operations
            print("   Testing memory operations...")
            test_image = torch.zeros(500, 500, device=device)
            test_image[100:200, 100:200] = 1.0
            print(f"   ✓ Memory operations successful")
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize()
            print("   ✓ GPU operations completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Error during GPU operations: {e}")
            print("   GPU may not be working correctly")
            return False
    else:
        print("   Skipping GPU operations test (CUDA not available)")
        print("   Testing CPU fallback...")
        try:
            device = torch.device('cpu')
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.matmul(test_tensor, test_tensor)
            print("   ✓ CPU operations working correctly")
            return True
        except Exception as e:
            print(f"   ✗ Error during CPU operations: {e}")
            return False
    
    # Test 4: Test String Art Generator GPU compatibility
    print("\n4. Testing String Art Generator GPU compatibility...")
    try:
        from string_art import StringArtGenerator, TORCH_AVAILABLE
        
        if TORCH_AVAILABLE:
            print("   ✓ String Art Generator can use PyTorch")
            if cuda_available:
                print("   ✓ GPU acceleration will be enabled")
            else:
                print("   ⚠ GPU acceleration will be disabled (CPU mode)")
        else:
            print("   ✗ String Art Generator cannot use PyTorch")
            return False
            
    except ImportError as e:
        print(f"   ✗ Cannot import StringArtGenerator: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("GPU Test Complete!")
    print("=" * 60)
    
    if cuda_available:
        print("\n✓ Your system is ready for GPU-accelerated string art generation!")
        print("  The String Art Generator will automatically use your GPU.")
    else:
        print("\n⚠ GPU is not available. String art generation will use CPU.")
        print("  This will work but may be slower.")
        print("\n  To enable GPU:")
        print("  1. Ensure you have an NVIDIA GPU")
        print("  2. Install CUDA drivers from NVIDIA")
        print("  3. Install PyTorch with CUDA:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    return True


if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)
