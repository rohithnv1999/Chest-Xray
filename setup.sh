#!/bin/bash

# Chest X-Ray AI Diagnosis System - Setup Script
# For MacBook (Apple Silicon or Intel)
# Student ID: 15700249

set -e  # Exit on error

echo "=========================================="
echo "  Chest X-Ray AI Diagnosis Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"
echo ""

# Create project structure
echo "Creating project structure..."
mkdir -p data/{raw,processed,splits}
mkdir -p models/checkpoints
mkdir -p src
mkdir -p notebooks
mkdir -p streamlit_app/{assets,components}
mkdir -p results/{figures,metrics,gradcam_outputs}
mkdir -p tests
echo "✓ Directory structure created"
echo ""

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created and activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ Pip upgraded"
echo ""

# Install PyTorch (Apple Silicon optimized)
echo "Installing PyTorch..."
if [[ $(uname -m) == 'arm64' ]]; then
    echo "Detected Apple Silicon - Installing MPS-optimized PyTorch"
    pip install torch torchvision torchaudio
else
    echo "Detected Intel Mac - Installing standard PyTorch"
    pip install torch torchvision torchaudio
fi
echo "✓ PyTorch installed"
echo ""

# Install other dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create __init__.py files
echo "Creating __init__.py files..."
touch src/__init__.py
touch streamlit_app/__init__.py
touch tests/__init__.py
echo "✓ Package structure initialized"
echo ""

# Download dataset instructions
echo "=========================================="
echo "  DATASET DOWNLOAD INSTRUCTIONS"
echo "=========================================="
echo ""
echo "You need to download the NIH ChestX-ray14 dataset:"
echo ""
echo "1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC"
echo "2. Download the following files:"
echo "   - Data_Entry_2017.csv (metadata file)"
echo "   - images_001.tar.gz through images_012.tar.gz"
echo ""
echo "3. Extract all tar.gz files to: data/raw/images/"
echo "   Command: tar -xzf images_001.tar.gz -C data/raw/images/"
echo ""
echo "4. Place Data_Entry_2017.csv in: data/raw/"
echo ""
echo "Estimated total size: ~42GB"
echo ""
echo "=========================================="
echo ""

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data
data/raw/
data/processed/
*.tar.gz
*.zip

# Models
models/checkpoints/
models/*.pth
*.h5
*.pkl

# Results
results/
runs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
.env.local
EOF
echo "✓ .gitignore created"
echo ""

# Test imports
echo "Testing imports..."
python3 << 'PYEOF'
import torch
import torchvision
import numpy
import pandas
import cv2
import PIL
import sklearn
import matplotlib
import seaborn
import streamlit

print("✓ All imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, 'mps'):
    print(f"MPS available: {torch.backends.mps.is_available()}")
PYEOF
echo ""

echo "=========================================="
echo "  SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download the NIH ChestX-ray14 dataset (see instructions above)"
echo "2. Run: python src/data_preprocessing.py  (to prepare data)"
echo "3. Run: python src/train.py  (to train model)"
echo "4. Run: python src/evaluate.py  (to evaluate model)"
echo "5. Run: streamlit run streamlit_app/app.py  (to launch interface)"
echo ""
echo "For detailed instructions, see README.md"
echo ""