# 4DGT Installation Guide

## Quick Installation

Use the automated installation script:

```bash
bash scripts/install.sh
```

This script will interactively guide you through the setup process with options to skip already installed components.

## Manual Installation

### Environment Setup

**IMPORTANT**: All development and testing for this project must be done within the `4dgt` conda environment.

#### Activate Environment
```bash
conda activate 4dgt
```

#### Manual Installation Commands
Always run these commands within the activated `4dgt` environment:

```bash
# Create fresh environment (if needed to avoid GraalPy issues)
conda env remove -n 4dgt
conda create -n 4dgt python=3.10
conda activate 4dgt

# Update GLIBC compatibility for flash-attn
conda update -c conda-forge libstdcxx-ng

# Install PyTorch first (required for flash-attn and apex)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements
pip install -v -r requirements.txt

# Install flash-attn (after PyTorch and GLIBC update)
pip install flash-attn

# Install NVIDIA Apex with CUDA extensions (after PyTorch is installed)
# Required for FusedLayerNorm optimization used in the transformer blocks

# Method 1: Direct installation (official method)
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex

# Method 2: If Method 1 fails due to CUDA version mismatch, use fallback installation
git clone https://github.com/NVIDIA/apex
cd apex
# Disable strict CUDA version check (safe for minor version differences like 12.4 vs 12.9)
sed -i 's/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/pass  # check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/' setup.py
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
cd ..
rm -rf apex
```

### CUDA Compatibility
- Project uses PyTorch with CUDA 12.4 builds
- Compatible with CUDA 12.x versions (including 12.8)
- Ensure CUDA toolkit >= 12.0 is installed on the system

### Common Issues

#### NVIDIA Apex Installation Issues
Apex is **required** for `FusedLayerNorm`, which provides significant performance improvements for the transformer blocks in 4DGT.

**Why Apex is needed:**
- 4DGT uses `FusedLayerNorm` from apex for optimized layer normalization
- Without apex, you'll get `RuntimeError: FusedLayerNorm not available. Please install apex.`
- The CUDA extensions provide 20-30% speedup over standard PyTorch LayerNorm

**Common apex installation problems:**

1. **CUDA Version Mismatch** (most common):
   ```
   RuntimeError: Cuda extensions are being compiled with a version of Cuda that does not match the version used to compile Pytorch binaries.
   ```
   **Solution:** Use Method 2 above (fallback installation with disabled version check)

2. **Missing PyTorch during build**:
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **Solution:** Always use `--no-build-isolation` flag and install PyTorch first

3. **FusedLayerNorm runtime error**:
   ```
   ImportError: No module named 'fused_layer_norm_cuda'
   ```
   **Solution:** Apex was installed without CUDA extensions. Reinstall with `APEX_CUDA_EXT=1`

#### Other Installation Issues
- **flash-attn installation fails**: Install PyTorch first, update libstdcxx-ng
- **CUDA version mismatch**: Use cu124 PyTorch builds for CUDA 12.x compatibility  
- **GLIBC version errors**: Run `conda update -c conda-forge libstdcxx-ng` before installing flash-attn
- **GraalPy compatibility issues**: Recreate environment with standard Python 3.10

#### Flash-attn Memory Issues
If flash-attn installation runs out of memory:
```bash
# Limit compilation jobs to reduce memory usage
MAX_JOBS=1 FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-binary flash-attn --no-build-isolation --verbose
```

#### Verification Commands
After installation, verify everything works:
```bash
conda activate 4dgt
# Test basic imports
python -c "import torch; import flash_attn; import apex; print('✅ All packages imported successfully!')"

# Test FusedLayerNorm specifically  
python -c "from apex.normalization import FusedLayerNorm; norm = FusedLayerNorm(512); print('✅ FusedLayerNorm available!')"
```

## Verification

After installation, verify everything is working:

```bash
conda activate 4dgt
python -c "import torch; import flash_attn; import apex; print('✅ All packages imported successfully!')"
```