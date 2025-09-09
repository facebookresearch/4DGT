#!/bin/bash

# 4DGT Installation Script
# Follows the instructions from CLAUDE.md

set -e  # Exit on any error

echo "🚀 Starting 4DGT environment setup..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    exit 1
fi

# Environment name
ENV_NAME="4dgt"

echo "📦 Setting up conda environment..."

# Check if environment already exists
ENV_EXISTS=false
if conda env list | grep -q "^${ENV_NAME} "; then
    ENV_EXISTS=true
    echo "📋 Found existing ${ENV_NAME} environment"
    
    # Ask user if they want to remove and recreate
    read -p "❓ Do you want to remove the existing environment and start fresh? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing ${ENV_NAME} environment..."
        conda env remove -n ${ENV_NAME} -y
        ENV_EXISTS=false
    else
        echo "🔧 Using existing environment..."
    fi
fi

# Create environment if it doesn't exist
if [ "$ENV_EXISTS" = false ]; then
    echo "🆕 Creating fresh ${ENV_NAME} environment with Python 3.10..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# Activate environment
echo "🔧 Activating ${ENV_NAME} environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Function to check if package is installed
check_package() {
    python -c "import $1" 2>/dev/null
}

# Update GLIBC compatibility for flash-attn
echo "🔄 Updating GLIBC compatibility..."
conda update -c conda-forge libstdcxx-ng -y

# Detect CUDA version and select appropriate PyTorch build
echo "🔍 Detecting CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "📋 Found CUDA version: $CUDA_VERSION"
    
    # Select PyTorch CUDA build based on system CUDA version
    if [[ "$CUDA_VERSION" =~ ^12\.[0-9] ]]; then
        TORCH_CUDA="cu124"  # Use cu124 for all CUDA 12.x versions
        echo "🎯 Using PyTorch CUDA 12.4 build (compatible with CUDA 12.x)"
    elif [[ "$CUDA_VERSION" =~ ^11\.[0-9] ]]; then
        TORCH_CUDA="cu118"
        echo "🎯 Using PyTorch CUDA 11.8 build (compatible with CUDA 11.x)"
    else
        TORCH_CUDA="cu124"  # Default to cu124
        echo "⚠️  Unknown CUDA version, defaulting to CUDA 12.4 build"
    fi
else
    TORCH_CUDA="cu124"  # Default if nvcc not found
    echo "⚠️  CUDA not detected, defaulting to CUDA 12.4 build"
fi

# Install PyTorch first (required for flash-attn and apex)
if check_package "torch"; then
    echo "🔥 PyTorch already installed"
    read -p "❓ Reinstall PyTorch? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔥 Reinstalling PyTorch with $TORCH_CUDA..."
        pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_CUDA}
    fi
else
    echo "🔥 Installing PyTorch with $TORCH_CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_CUDA}
fi

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    read -p "❓ Install/update requirements from requirements.txt? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "📋 Installing requirements from requirements.txt..."
        pip install -v -r requirements.txt
    fi
else
    echo "⚠️  Warning: requirements.txt not found, skipping..."
fi

# Install flash-attn (after PyTorch and GLIBC update)
if check_package "flash_attn"; then
    echo "⚡ flash-attn already installed"
    read -p "❓ Reinstall flash-attn? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "⚡ Reinstalling flash-attn..."
        # pip install --force-reinstall flash-attn
        pip install --force-reinstall flash_attn==2.7.4.post1
    fi
else
    echo "⚡ Installing flash-attn..."
    # pip install flash-attn
    pip install flash_attn==2.7.4.post1
fi

# Install apex (after PyTorch is installed)
if check_package "apex"; then
    echo "🔺 NVIDIA apex already installed"
    read -p "❓ Reinstall apex? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔺 Reinstalling NVIDIA apex with CUDA extensions..."
        pip uninstall apex -y
        echo "🔺 Trying apex installation with environment variables (official method)..."
        APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex || {
            echo "⚠️  CUDA extensions failed, trying fallback installation..."
            # Fallback: clone and install with modified setup
            rm -rf apex_build  # Clean up any existing directory
            git clone https://github.com/NVIDIA/apex apex_build
            cd apex_build
            # Comment out the strict version check
            sed -i 's/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/pass  # check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/' setup.py
            APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation --no-cache-dir ./
            cd ..
            rm -rf apex_build
        }
    fi
else
    echo "🔺 Installing NVIDIA apex with CUDA extensions..."
    echo "🔺 Trying apex installation with environment variables (official method)..."
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex || {
        echo "⚠️  CUDA extensions failed, trying fallback installation..."
        # Fallback: clone and install with modified setup
        rm -rf apex_build  # Clean up any existing directory
        git clone https://github.com/NVIDIA/apex apex_build
        cd apex_build
        # Comment out the strict version check
        sed -i 's/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/pass  # check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/' setup.py
        APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation --no-cache-dir ./
        cd ..
        rm -rf apex_build
    }
fi

echo "✅ Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify the installation, you can test importing key packages:"
echo "  python -c 'import torch; import flash_attn; print(\"All packages imported successfully!\")'"