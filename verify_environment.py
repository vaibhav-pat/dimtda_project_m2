# File: verify_m2_environment.py
# A definitive script to verify the stability and correctness of a PyTorch
# environment on Apple Silicon (M1/M2/M3).
# Updated to expect torchvision version 0.14.1.

import sys
import importlib
import platform

# --- Configuration: Define the exact versions we expect for this project ---
EXPECTED_VERSIONS = {
    "python": "3.10.14",
    "torch": "1.13.1",
    "torchvision": "0.14.1",  # <-- Corrected to the version available for M2/arm64
    "numpy": "1.24.4",
    "transformers": "4.33.2",
    "sacrebleu": "2.3.1",
    "jieba": "0.42.1",
    "zss": "1.2.0"
}

# --- Main Verification Logic ---

print("-" * 80)
print("--- Starting Apple Silicon (M2) Environment Verification ---")
print(f"Platform: {platform.system()} | Architecture: {platform.machine()} (Expected: arm64)")
print("-" * 80)

all_ok = True

def check_version(pkg, expected, actual):
    """Compares versions and prints a formatted status."""
    global all_ok
    if expected in actual:
        print(f"✅ {pkg:<12} | Version: {actual} (Correctly Pinned)")
    else:
        print(f"❌ {pkg:<12} | WRONG VERSION: {actual} (Expected ~{expected})")
        all_ok = False

# 1. Check Python and Libraries
print("--- Python Package Verification ---")
check_version("Python", EXPECTED_VERSIONS["python"], sys.version)

libs_to_check = [
    'torch', 'torchvision', 'numpy', 'transformers',
    'sacrebleu', 'jieba', 'zss'
]

for lib_name in libs_to_check:
    try:
        lib = importlib.import_module(lib_name)
        check_version(lib_name.capitalize(), EXPECTED_VERSIONS[lib_name], lib.__version__)
    except ImportError:
        print(f"❌ {lib_name.capitalize():<12} | NOT FOUND - Please install.")
        all_ok = False
    except Exception as e:
        print(f"❌ {lib_name.capitalize():<12} | FAILED with an unexpected error: {e}")
        all_ok = False

# 2. Hardware Backend Verification (MPS for Apple Silicon)
print("-" * 80)
print("--- Hardware Backend Verification (MPS for M2) ---")
try:
    import torch
    
    is_mps_built = torch.backends.mps.is_built()
    print(f"MPS Built?       | {'✅ Yes' if is_mps_built else '❌ No - Your PyTorch build is incorrect.'}")
    if not is_mps_built: all_ok = False

    is_mps_available = torch.backends.mps.is_available()
    print(f"MPS Available?   | {'✅ Yes - Acceleration is ready.' if is_mps_available else '❌ No - Check macOS version (must be 12.3+).'}")
    if not is_mps_available: all_ok = False

    if is_mps_available:
        # Final proof: move a tensor to the GPU to confirm runtime operation
        try:
            tensor_mps = torch.rand(2, 2).to("mps")
            print(f"MPS Operation    | ✅ Successfully moved a tensor to the MPS device.")
        except Exception as e:
            print(f"MPS Operation    | ❌ FAILED during test operation: {e}")
            all_ok = False

except Exception as e:
    print(f"❌ An error occurred during the PyTorch hardware check: {e}")
    all_ok = False

# 3. Final Summary
print("-" * 80)
if all_ok:
    print("✅ SUCCESS: Your M2 environment is stable, correctly pinned, and ready for development.")
else:
    print("❌ FAILED: Environment has critical errors. Please review the output above.")
print("-" * 80)