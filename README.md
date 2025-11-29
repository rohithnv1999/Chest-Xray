# 🫁 Deep Learning for Automated Multi-Label Chest X-Ray Diagnosis

**Student ID:** 15700249  
**Module:** 7150CEM Data Science Project  
**University:** Coventry University

## 📋 Project Overview

This project implements an intelligent clinical decision support system for automated multi-label classification of chest X-ray pathologies using deep learning. The system combines DenseNet-121 with attention mechanisms and Grad-CAM explainability to detect 14 thoracic conditions with clinical-grade accuracy.

### Key Features

- ✅ **Multi-Label Classification**: Simultaneously detects 14 thoracic pathologies
- ✅ **Attention Mechanisms**: Squeeze-and-Excitation + Spatial Attention
- ✅ **Visual Explainability**: Grad-CAM heatmaps for interpretability
- ✅ **Clinical Interface**: Interactive Streamlit web application
- ✅ **High Performance**: <10 second inference, >0.85 AUROC target

### Detected Pathologies

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Infiltration
9. Mass
10. Nodule
11. Pleural Thickening
12. Pneumonia
13. Pneumothorax
14. No Finding

---

## 🚀 Quick Start Guide for MacBook

### Prerequisites

- macOS 11.0 or later
- Python 3.8+ installed
- Minimum 50GB free disk space (for dataset)
- 8GB+ RAM recommended
- VS Code installed

### Step 1: Initial Setup

```bash
# Open Terminal (⌘ + Space, type "Terminal")

# Navigate to your desired location
cd ~/Documents

# Clone or create project directory
mkdir chest-xray-diagnosis
cd chest-xray-diagnosis

# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh
```

### Step 2: Download Dataset

The NIH ChestX-ray14 dataset is required for training.

**Option A: Manual Download**

1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Download files:
   - `Data_Entry_2017.csv`
   - `images_001.tar.gz` through `images_012.tar.gz`

3. Extract images:
```bash
cd data/raw
mkdir images

# Extract all tar files
for i in {001..012}; do
    tar -xzf images_${i}.tar.gz -C images/
    echo "Extracted images_${i}.tar.gz"
done

# Move CSV file
mv Data_Entry_2017.csv data/raw/
```

**Option B: Using wget (if available)**

```bash
cd data/raw

# Download metadata
wget https://nihcc.app.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.csv -O Data_Entry_2017.csv

# Note: Image files are too large for direct wget
# Use Manual Download for image files
```

---

## 💻 Complete Workflow Commands

### 1. Activate Environment

**Every time you start working, activate the virtual environment:**

```bash
cd chest-xray-diagnosis
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 2. Data Preprocessing

```bash
# Run data preprocessing to create train/val/test splits
python src/data_preprocessing.py
```

**Expected Output:**
- Creates stratified splits (70% train, 15% val, 15% test)
- Saves splits to `data/splits/`
- Computes class weights for weighted loss
- Takes ~5-10 minutes

### 3. Model Training

```bash
# Start training (this will take 48-60 hours on GPU, longer on CPU)
python src/train.py

# To run in background and save output:
nohup python src/train.py > training.log 2>&1 &

# Monitor training progress:
tail -f training.log

# Or use TensorBoard:
tensorboard --logdir=runs
# Then open http://localhost:6006 in browser
```

**Training Options:**

```bash
# Resume training from checkpoint
python src/train.py --resume models/checkpoints/checkpoint_epoch_50.pth

# Train with specific GPU (if multiple available)
CUDA_VISIBLE_DEVICES=0 python src/train.py
```

**Expected Training Time:**
- **With NVIDIA GPU**: 48-60 hours
- **With Apple Silicon (MPS)**: 60-80 hours
- **CPU only**: 200+ hours (not recommended)

### 4. Model Evaluation

```bash
# Evaluate trained model on test set
python src/evaluate.py

# Evaluate with subgroup analysis
python src/evaluate.py --subgroup-analysis

# Generate comprehensive report
python src/evaluate.py --full-report
```

**Generated Outputs:**
- Test metrics (AUROC, sensitivity, specificity)
- ROC curves for all classes
- Confusion matrices
- Subgroup analysis (age, gender)
- Performance report (JSON + TXT)

### 5. Generate Grad-CAM Visualizations

```bash
# Generate Grad-CAM heatmaps for 50 samples
python src/gradcam.py

# Generate for specific number of samples
python src/gradcam.py --num-samples 100

# Generate for specific class
python src/gradcam.py --class-name Pneumonia
```

**Output Location:** `results/gradcam_outputs/`

### 6. Launch Clinical Interface

```bash
# Start Streamlit application
streamlit run streamlit_app/app.py

# Application will open in your browser automatically
# If not, navigate to: http://localhost:8501
```

**Interface Features:**
- Upload chest X-ray images
- Real-time AI diagnosis
- Interactive Grad-CAM visualizations
- Confidence scores
- Clinical report generation
- Export results

---

## 📁 Project Structure

```
chest-xray-diagnosis/
│
├── data/
│   ├── raw/                          # Original NIH dataset
│   │   ├── images/                   # All chest X-ray images
│   │   └── Data_Entry_2017.csv       # Metadata file
│   ├── processed/                    # Preprocessed data (auto-generated)
│   └── splits/                       # Train/val/test splits (auto-generated)
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── class_weights.json
│
├── models/
│   ├── checkpoints/                  # Training checkpoints
│   │   └── checkpoint_epoch_*.pth
│   └── best_model.pth                # Best performing model
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration settings
│   ├── data_preprocessing.py         # Data loading & preprocessing
│   ├── model.py                      # DenseNet-121 architecture
│   ├── train.py                      # Training pipeline
│   ├── evaluate.py                   # Evaluation metrics
│   ├── gradcam.py                    # Grad-CAM implementation
│   └── utils.py                      # Helper functions
│
├── streamlit_app/
│   ├── app.py                        # Main Streamlit interface
│   ├── components.py                 # UI components
│   └── assets/                       # Images, CSS, etc.
│
├── results/
│   ├── figures/                      # Generated plots
│   ├── metrics/                      # Performance metrics
│   └── gradcam_outputs/              # Grad-CAM visualizations
│
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── tests/                            # Unit tests
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Setup script
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
└── training.log                      # Training logs (generated)
```

---

## ⚙️ Configuration

Edit `src/config.py` to modify:

### Key Parameters

```python
# Model Architecture
BACKBONE = 'densenet121'
IMAGE_SIZE = 512
NUM_CLASSES = 14

# Training
BATCH_SIZE = 32           # Reduce if out of memory
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.4

# Performance Targets
TARGET_AUROC = 0.85
TARGET_SENSITIVITY = 0.90
TARGET_SPECIFICITY = 0.85
```

### Memory Optimization

If you encounter memory issues:

```python
# In config.py, reduce:
BATCH_SIZE = 16            # Instead of 32
NUM_WORKERS = 2            # Instead of 4
USE_MIXED_PRECISION = True # Enable if supported
```

---

## 🔍 Monitoring Training

### Using TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# Open in browser
open http://localhost:6006
```

**Metrics Available:**
- Training loss (per batch)
- Validation loss (per epoch)
- Validation AUROC (per epoch)
- Learning rate schedule
- Per-class AUROC

### Training Progress Logs

```bash
# Real-time monitoring
tail -f training.log

# Search for specific epoch
grep "Epoch 50" training.log

# Check best AUROC
grep "best AUROC" training.log
```

---

## 📊 Expected Results

Based on proposal targets:

| Metric | Target | Expected Range |
|--------|--------|----------------|
| Mean AUROC | >0.85 | 0.84-0.87 |
| Sensitivity (Critical) | >0.90 | 0.88-0.92 |
| Specificity | >0.85 | 0.83-0.88 |
| Inference Time | <10s | 6-8s (CPU) |
| F1 Score | - | 0.75-0.82 |

### Per-Class Performance (Expected)

Top performing classes:
- Cardiomegaly: AUROC ~0.90
- Effusion: AUROC ~0.88
- Mass: AUROC ~0.87

Challenging classes:
- Infiltration: AUROC ~0.70
- Nodule: AUROC ~0.75

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**

```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # or even 8

# Reduce number of workers
NUM_WORKERS = 2
```

**2. Dataset Not Found**

```bash
# Check data directory structure
ls -la data/raw/images/
ls -la data/raw/Data_Entry_2017.csv

# If missing, re-download dataset
```

**3. Model Loading Error**

```bash
# Check if model file exists
ls -la models/best_model.pth

# If missing, complete training first
python src/train.py
```

**4. Streamlit Won't Start**

```bash
# Reinstall Streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear

# Run with verbose logging
streamlit run streamlit_app/app.py --logger.level=debug
```

**5. Import Errors**

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Performance Optimization

**For Apple Silicon Macs:**

```python
# In config.py, use MPS backend
import torch
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

**For Intel Macs with AMD GPU:**

```bash
# Use CPU (AMD GPUs not supported by PyTorch)
# Expect longer training times
```

---

## 📝 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test model architecture
python src/model.py

# Test data preprocessing
python src/data_preprocessing.py

# Test Grad-CAM
python src/gradcam.py --test-mode
```

---

## 📈 Creating Jupyter Notebooks for Analysis

```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter
jupyter notebook

# Create new notebook in notebooks/ folder
```

**Useful Notebooks:**
- Data exploration and visualization
- Model performance analysis
- Error analysis
- Grad-CAM quality assessment

---

## 🔬 Advanced Usage

### Custom Training Script

```python
# train_custom.py
from src.config import Config
from src.model import create_model
from src.data_preprocessing import create_data_loaders
from src.train import Trainer

config = Config()
config.NUM_EPOCHS = 50  # Custom epochs
config.LEARNING_RATE = 5e-5  # Custom LR

train_loader, val_loader, _ = create_data_loaders(config)
trainer = Trainer(config)
trainer.train(train_loader, val_loader)
```

### Ensemble Models

```python
# Load multiple models for ensemble
models = []
for i in range(1, 6):
    model = create_model(config)
    load_checkpoint(f'models/checkpoint_fold_{i}.pth', model)
    models.append(model)

# Average predictions
predictions = torch.stack([m(x) for m in models]).mean(dim=0)
```

### Custom Evaluation Metrics

```python
# Add to evaluate.py
from sklearn.metrics import matthews_corrcoef

def compute_mcc(labels, predictions):
    binary_preds = (predictions >= 0.5).astype(int)
    return matthews_corrcoef(labels.ravel(), binary_preds.ravel())
```

---

## 📚 References & Resources

### Dataset
- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Wang et al. (2017): ChestX-ray8 paper

### Architecture
- DenseNet: Huang et al. (2017)
- Squeeze-and-Excitation: Hu et al. (2018)
- Grad-CAM: Selvaraju et al. (2017)

### Documentation
- PyTorch: https://pytorch.org/docs/
- Streamlit: https://docs.streamlit.io/
- scikit-learn: https://scikit-learn.org/

---

## 📧 Support

For issues or questions:
1. Check Troubleshooting section above
2. Review training logs
3. Check TensorBoard metrics
4. Contact supervisor via module page

---

## 📄 License & Ethics

This project is for educational purposes only.
- Ethics approval obtained: See proposal document
- Patient data handled according to NHS guidelines
- AI system intended for decision support, not replacement of clinical judgment

---

## ✅ Project Checklist

Before final submission, ensure:

- [ ] Dataset downloaded and preprocessed
- [ ] Model trained for minimum 50 epochs
- [ ] Best AUROC >0.83 achieved
- [ ] Test set evaluation complete
- [ ] Grad-CAM visualizations generated (50+ samples)
- [ ] Streamlit interface functional
- [ ] All plots saved in results/figures/
- [ ] Metrics reports generated
- [ ] Training logs preserved
- [ ] Code commented and documented
- [ ] README.md complete
- [ ] Supervisor meetings documented in appendices

---

**Good luck with your project! 🎓**

Last Updated: 2025
Student ID: 15700249