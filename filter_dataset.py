"""
Filter NIH Dataset CSV to only include available images
Run this BEFORE data_preprocessing.py
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
RAW_DATA_DIR = Path('data/raw')
IMAGE_DIR = RAW_DATA_DIR / 'images'
CSV_FILE = RAW_DATA_DIR / 'Data_Entry_2017.csv'
OUTPUT_CSV = RAW_DATA_DIR / 'Data_Entry_2017_filtered.csv'

print("="*60)
print("FILTERING DATASET TO MATCH AVAILABLE IMAGES")
print("="*60)

# Read CSV
print("\n1. Loading CSV file...")
df = pd.read_csv(CSV_FILE)
print(f"   Original CSV entries: {len(df)}")

# Get available images
print("\n2. Scanning available images...")
available_images = set([f.name for f in IMAGE_DIR.glob('*.png')])
print(f"   Found {len(available_images)} images in folder")

# Filter dataframe
print("\n3. Filtering CSV to match available images...")
df_filtered = df[df['Image Index'].isin(available_images)].copy()
print(f"   Filtered CSV entries: {len(df_filtered)}")
print(f"   Removed: {len(df) - len(df_filtered)} entries")

# Check class distribution
print("\n4. Class distribution in filtered dataset:")
df_filtered['Finding_Labels'] = df_filtered['Finding Labels'].apply(
    lambda x: x.split('|') if isinstance(x, str) else []
)

class_names = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration',
    'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
    'Pneumothorax', 'No Finding'
]

for label in class_names:
    count = df_filtered['Finding_Labels'].apply(
        lambda x: 1 if label.replace('_', ' ') in ' '.join(x) else 0
    ).sum()
    percentage = (count / len(df_filtered)) * 100
    print(f"   {label:<20}: {count:>6} ({percentage:>5.2f}%)")

# Save filtered CSV
print(f"\n5. Saving filtered CSV to: {OUTPUT_CSV}")
df_filtered.drop('Finding_Labels', axis=1).to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*60)
print("✅ FILTERING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Run: python src/data_preprocessing.py")
print("2. This will now use the filtered CSV automatically")
print("="*60)