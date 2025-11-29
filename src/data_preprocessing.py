"""
Data Preprocessing Pipeline - FIXED VERSION
"""

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config

class ChestXrayDataset(Dataset):
    """PyTorch Dataset for Chest X-rays"""
    
    def __init__(self, dataframe, image_dir, config, mode='train'):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.config = config
        self.mode = mode
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.RandomRotation(self.config.ROTATION_RANGE),
                transforms.RandomHorizontalFlip(p=self.config.HORIZONTAL_FLIP_PROB),
                transforms.ColorJitter(brightness=self.config.BRIGHTNESS_RANGE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = self.image_dir / img_name
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply CLAHE
            img_array = np.array(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            img_array = clahe.apply(img_gray)
            image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB))
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return blank image instead of crashing
            image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), color='black')
        
        image = self.transform(image)
        
        labels = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)
        for i, label in enumerate(self.config.CLASS_NAMES):
            labels[i] = self.df.iloc[idx][label]
        
        metadata = {
            'image_name': img_name,
            'patient_id': self.df.iloc[idx]['Patient ID'],
            'age': self.df.iloc[idx].get('Patient Age', -1),
            'gender': self.df.iloc[idx].get('Patient Gender', 'Unknown')
        }
        
        return image, labels, metadata


def create_stratified_split(config):
    """Create train/val/test splits - FIXED VERSION"""
    print("\nCreating stratified data splits...")
    
    # Try filtered CSV first
    csv_filtered = config.RAW_DATA_DIR / 'Data_Entry_2017_filtered.csv'
    csv_original = config.RAW_DATA_DIR / 'Data_Entry_2017.csv'
    
    csv_path = csv_filtered if csv_filtered.exists() else csv_original
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    print(f"Loading: {csv_path.name}")
    labels_df = pd.read_csv(csv_path)
    
    # Process labels
    labels_df['Finding_Labels'] = labels_df['Finding Labels'].apply(
        lambda x: x.split('|') if isinstance(x, str) else []
    )
    
    # Create binary labels
    for label in config.CLASS_NAMES:
        labels_df[label] = labels_df['Finding_Labels'].apply(
            lambda x: 1 if label.replace('_', ' ') in ' '.join(x) else 0
        )
    
    # Patient-level split
    patient_groups = labels_df.groupby('Patient ID').agg({
        'Image Index': 'count',
        **{label: 'max' for label in config.CLASS_NAMES}
    }).reset_index()
    
    train_patients, temp_patients = train_test_split(
        patient_groups['Patient ID'],
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=patient_groups['No_Finding']
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED
    )
    
    train_df = labels_df[labels_df['Patient ID'].isin(train_patients)]
    val_df = labels_df[labels_df['Patient ID'].isin(val_patients)]
    test_df = labels_df[labels_df['Patient ID'].isin(test_patients)]
    
    print(f"✓ Train set: {len(train_df)} images from {len(train_patients)} patients")
    print(f"✓ Validation set: {len(val_df)} images from {len(val_patients)} patients")
    print(f"✓ Test set: {len(test_df)} images from {len(test_patients)} patients")
    
    # Save splits
    train_df.to_csv(config.SPLITS_DIR / 'train.csv', index=False)
    val_df.to_csv(config.SPLITS_DIR / 'val.csv', index=False)
    test_df.to_csv(config.SPLITS_DIR / 'test.csv', index=False)
    print("✓ Splits saved")
    
    # Compute class weights
    print("\nComputing class weights...")
    class_counts = {}
    total_samples = len(train_df)
    
    for label in config.CLASS_NAMES:
        pos_count = train_df[label].sum()
        neg_count = total_samples - pos_count
        weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        class_counts[label] = {
            'positive': int(pos_count),
            'negative': int(neg_count),
            'weight': float(weight)
        }
    
    with open(config.SPLITS_DIR / 'class_weights.json', 'w') as f:
        json.dump(class_counts, f, indent=2)
    
    print("✓ Class weights computed")
    return train_df, val_df, test_df


def create_data_loaders(config):
    """Create data loaders - FIXED VERSION"""
    
    # Check if splits exist
    if not (config.SPLITS_DIR / 'train.csv').exists():
        print("Splits not found. Creating now...")
        create_stratified_split(config)
    
    train_df = pd.read_csv(config.SPLITS_DIR / 'train.csv')
    val_df = pd.read_csv(config.SPLITS_DIR / 'val.csv')
    test_df = pd.read_csv(config.SPLITS_DIR / 'test.csv')
    
    train_dataset = ChestXrayDataset(train_df, config.RAW_DATA_DIR / 'images', config, mode='train')
    val_dataset = ChestXrayDataset(val_df, config.RAW_DATA_DIR / 'images', config, mode='val')
    test_dataset = ChestXrayDataset(test_df, config.RAW_DATA_DIR / 'images', config, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                             num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                           num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    
    print(f"\n✓ Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = Config()
    config.create_directories()
    
    # Create splits
    create_stratified_split(config)
    
    print("\n✓ Data preprocessing complete!")
    print("Next: python src/train.py")