"""
Configuration file for Chest X-Ray Diagnosis System
Student ID: 15700249
"""

import torch
from pathlib import Path

class Config:
    """Configuration class for all hyperparameters and settings"""
    
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    SPLITS_DIR = DATA_DIR / 'splits'
    MODELS_DIR = PROJECT_ROOT / 'models'
    CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    FIGURES_DIR = RESULTS_DIR / 'figures'
    METRICS_DIR = RESULTS_DIR / 'metrics'
    GRADCAM_DIR = RESULTS_DIR / 'gradcam_outputs'
    
    # Dataset Configuration
    DATASET_NAME = 'NIH_ChestXray14'
    IMAGE_SIZE = 512
    NUM_CLASSES = 14
    CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration',
        'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
        'Pneumothorax', 'No_Finding'
    ]
    
    # Data Split Ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Data Augmentation Parameters
    ROTATION_RANGE = 15  # degrees
    BRIGHTNESS_RANGE = 0.2  # ±20%
    HORIZONTAL_FLIP_PROB = 0.5
    GAUSSIAN_NOISE_STD = 0.02
    
    # Training Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 1e-4  # L2 regularization
    DROPOUT_RATE = 0.4
    
    # Optimizer Parameters (Adam)
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8
    
    # Learning Rate Scheduler
    SCHEDULER_TYPE = 'cosine'  # cosine annealing
    T_MAX = NUM_EPOCHS
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_DELTA = 0.001
    
    # Model Architecture
    BACKBONE = 'densenet121'
    PRETRAINED = True
    FEATURE_DIM = 1024
    ATTENTION_TYPE = 'squeeze_excitation'  # or 'spatial'
    
    # Loss Function
    LOSS_FUNCTION = 'weighted_bce'  # weighted binary cross-entropy
    POS_WEIGHT_MULTIPLIER = 2.0  # Adjust for class imbalance
    
    # Mixed Precision Training
    USE_MIXED_PRECISION = True
    
    # Multi-GPU Training
    USE_MULTI_GPU = torch.cuda.device_count() > 1
    
    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0  # For DataLoader
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # Evaluation Metrics Thresholds
    SENSITIVITY_THRESHOLD = 0.90
    SPECIFICITY_THRESHOLD = 0.85
    INFERENCE_TIME_THRESHOLD = 10  # seconds
    
    # Grad-CAM Configuration
    GRADCAM_LAYER = 'densenet.features.denseblock4'  # Last dense block
    GRADCAM_COLORMAP = 'jet'
    GRADCAM_ALPHA = 0.4  # Overlay transparency
    
    # Clinical Decision Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60
    LOW_CONFIDENCE_THRESHOLD = 0.30
    
    # Logging and Checkpointing
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_CHECKPOINT_EVERY = 5  # Save every N epochs
    TENSORBOARD_LOG_DIR = PROJECT_ROOT / 'runs'
    
    # Random Seeds for Reproducibility
    RANDOM_SEED = 42
    
    # Performance Requirements (as per proposal)
    TARGET_AUROC = 0.85
    TARGET_SENSITIVITY_CRITICAL = 0.90  # For pneumothorax, large effusions
    TARGET_SPECIFICITY = 0.85
    
    # Subgroup Analysis Categories
    AGE_QUARTILES = [0, 40, 55, 70, 120]  # Age groups for analysis
    SEX_CATEGORIES = ['M', 'F']
    PROJECTION_TYPES = ['PA', 'AP']
    
    # Clinical Metadata
    CRITICAL_FINDINGS = ['Pneumothorax', 'Effusion']  # High priority
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist"""
        dirs = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.SPLITS_DIR, cls.MODELS_DIR, cls.CHECKPOINTS_DIR,
            cls.RESULTS_DIR, cls.FIGURES_DIR, cls.METRICS_DIR,
            cls.GRADCAM_DIR, cls.TENSORBOARD_LOG_DIR
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        print("✓ All directories created successfully")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CHEST X-RAY DIAGNOSIS SYSTEM - CONFIGURATION")
        print("="*60)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print(f"Mixed Precision: {cls.USE_MIXED_PRECISION}")
        print(f"Multi-GPU: {cls.USE_MULTI_GPU}")
        print("="*60 + "\n")

if __name__ == "__main__":
    # Test configuration
    Config.create_directories()
    Config.print_config()