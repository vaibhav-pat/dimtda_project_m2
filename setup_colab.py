#!/usr/bin/env python3
"""
Setup script for Google Colab environment
This script prepares the environment and downloads necessary models
"""

import os
import sys
import subprocess
import requests
import zipfile
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    return result

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    run_command("pip install -r requirements.txt")

def setup_environment():
    """Setup environment variables and paths"""
    print("🔧 Setting up environment...")
    
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Create necessary directories
    directories = [
        "pretrained_models",
        "data/DoTA_dataset",
        "output",
        "pretrain_output",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_models():
    """Download necessary pre-trained models"""
    print("📥 Downloading pre-trained models...")
    
    # Model URLs (you'll need to replace these with actual URLs)
    models = {
        "dit-base": "https://huggingface.co/microsoft/dit-base",
        "nougat-small": "https://huggingface.co/facebook/nougat-small",
        "mbart-large-50": "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt"
    }
    
    for model_name, model_url in models.items():
        model_path = f"pretrained_models/{model_name}"
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            # Note: In actual implementation, you'd use huggingface_hub to download
            # For now, we'll create a placeholder
            os.makedirs(model_path, exist_ok=True)
            print(f"Created placeholder for {model_name}")

def create_sample_data():
    """Create sample data structure for testing"""
    print("📊 Creating sample data structure...")
    
    # Create sample split file
    sample_split = {
        "train_name_list": ["sample_001", "sample_002"],
        "valid_name_list": ["sample_003"],
        "test_name_list": ["sample_004"]
    }
    
    import json
    with open("data/DoTA_dataset/generated_split_200_50_50.json", "w") as f:
        json.dump(sample_split, f, indent=2)
    
    # Create sample directories
    os.makedirs("data/DoTA_dataset/imgs", exist_ok=True)
    os.makedirs("data/DoTA_dataset/en_mmd", exist_ok=True)
    os.makedirs("data/DoTA_dataset/zh_mmd", exist_ok=True)
    
    print("Sample data structure created")

def verify_setup():
    """Verify that the setup is working correctly"""
    print("✅ Verifying setup...")
    
    # Check if required packages are installed
    try:
        import torch
        import transformers
        import jieba
        import PIL
        print("✓ All required packages are installed")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA is not available, will use CPU")
    
    # Check if directories exist
    required_dirs = ["pretrained_models", "data/DoTA_dataset", "output"]
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            return False
    
    print("🎉 Setup verification completed successfully!")
    return True

def main():
    """Main setup function"""
    print("🚀 Starting DIMTDA Colab Setup...")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Setup environment
    setup_environment()
    
    # Download models (placeholder for now)
    download_models()
    
    # Create sample data
    create_sample_data()
    
    # Verify setup
    if verify_setup():
        print("\n🎉 Setup completed successfully!")
        print("You can now run the training scripts in Colab.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
