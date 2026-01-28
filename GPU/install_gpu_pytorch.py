"""
Helper script to install the correct PyTorch version for your GPU.
Automatically detects CUDA version and installs compatible PyTorch.
"""

import subprocess
import sys
import re

def get_cuda_version_from_nvidia_smi():
    """Try to get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Look for CUDA Version in output
            match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    return None

def check_pytorch_cuda():
    """Check if PyTorch with CUDA is already installed."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"✓ PyTorch with CUDA {cuda_version} is already installed")
            return True, cuda_version
        else:
            print("⚠ PyTorch is installed but CUDA is not available")
            return False, None
    except ImportError:
        return False, None

def install_pytorch_for_cuda(cuda_version):
    """Install PyTorch for a specific CUDA version."""
    cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
    
    # Map CUDA versions to PyTorch installation commands
    install_commands = {
        '12.4': 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124',
        '12.1': 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121',
        '11.8': 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118',
    }
    
    # Find closest matching CUDA version
    if cuda_major_minor in install_commands:
        command = install_commands[cuda_major_minor]
    elif cuda_major_minor.startswith('12.'):
        # Default to CUDA 12.1 for any 12.x
        command = install_commands['12.1']
        print(f"  Note: Using CUDA 12.1 build (your CUDA is {cuda_major_minor})")
    elif cuda_major_minor.startswith('11.'):
        # Default to CUDA 11.8 for any 11.x
        command = install_commands['11.8']
        print(f"  Note: Using CUDA 11.8 build (your CUDA is {cuda_major_minor})")
    else:
        print(f"  ⚠ Unknown CUDA version {cuda_major_minor}")
        print(f"  Installing PyTorch for CUDA 11.8 (most compatible)")
        command = install_commands['11.8']
    
    print(f"\nInstalling PyTorch with CUDA support...")
    print(f"Command: {command}\n")
    
    try:
        # Split command and install
        parts = command.split()
        subprocess.run(parts, check=True)
        print("\n✓ PyTorch installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        print("\nYou may need to:")
        print("  1. Check your internet connection")
        print("  2. Upgrade pip: pip install --upgrade pip")
        print("  3. Install manually from: https://pytorch.org/get-started/locally/")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def main():
    """Main installation function."""
    print("=" * 60)
    print("PyTorch GPU Installation Helper")
    print("=" * 60)
    print()
    
    # Check if already installed
    installed, cuda_version = check_pytorch_cuda()
    if installed:
        response = input("\nPyTorch with CUDA is already installed. Reinstall? (y/N): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Try to detect CUDA version
    print("Detecting CUDA version...")
    cuda_version = get_cuda_version_from_nvidia_smi()
    
    if cuda_version:
        print(f"✓ Detected CUDA version: {cuda_version}")
        install_pytorch_for_cuda(cuda_version)
    else:
        print("⚠ Could not automatically detect CUDA version")
        print("\nOptions:")
        print("  1. Check CUDA version manually: nvidia-smi")
        print("  2. Install PyTorch for common CUDA versions:")
        print()
        print("  For CUDA 11.8 (most common):")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("  For CUDA 12.1:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("  For CUDA 12.4:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("  Or install CPU-only version:")
        print("    pip install torch torchvision")
        
        response = input("\nAttempt automatic installation for CUDA 11.8? (Y/n): ")
        if response.lower() != 'n':
            # Try CUDA 11.8 as default
            install_pytorch_for_cuda('11.8')
    
    # Verify installation
    print("\n" + "=" * 60)
    print("Verifying installation...")
    print("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ Success! PyTorch {torch.__version__} with CUDA {torch.version.cuda} is ready")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ PyTorch installed but CUDA not available")
            print("  This may be a CPU-only installation.")
    except ImportError:
        print("✗ PyTorch installation failed or incomplete")
        sys.exit(1)
    
    print("\nRun 'python test_gpu.py' to test GPU functionality.")

if __name__ == "__main__":
    main()
