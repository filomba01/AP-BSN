#!/usr/bin/env python3
"""
Split prepared satellite image patches into train/validation/test directories.
"""

import os
import shutil
import random
import argparse
from pathlib import Path

def split_satellite_dataset(source_dir="dataset/prep/SatelliteImages_s512_o128", 
                          train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                          random_seed=42):
    """
    Split satellite image patches into train/val/test directories.
    
    Args:
        source_dir: Path to source prepared dataset
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1) 
        test_ratio: Fraction for testing (default: 0.1)
        random_seed: Random seed for reproducible splits (default: 42)
    """
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    source_path = Path(source_dir)
    clean_dir = source_path / "CL"
    
    if not clean_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {clean_dir}")
    
    # Get all image files and shuffle them
    image_files = [f for f in os.listdir(clean_dir) if f.lower().endswith('.png')]
    if len(image_files) == 0:
        raise ValueError(f"No PNG files found in {clean_dir}")
    
    print(f"Found {len(image_files)} images in {clean_dir}")
    
    # Set random seed for reproducible splits
    random.seed(random_seed)
    random.shuffle(image_files)
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val  # Ensure all images are used
    
    print(f"Splitting: {n_train} train, {n_val} val, {n_test} test")
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Create target directories
    # We now follow the directory convention expected by `prep_SatelliteImages`:
    #   dataset/prep/SatelliteImages_s512_o128/<split>/CL/*.png
    # where <split> is one of {train, val, test}

    base_dir = source_path  # keep splits inside the *same* prepared folder
    splits = {
        'train': (train_files, base_dir / "train" / "CL"),
        'val':   (val_files,   base_dir / "val" / "CL"),
        'test':  (test_files,  base_dir / "test" / "CL")
    }
    
    # Copy files to target directories
    for split_name, (files, target_dir) in splits.items():
        print(f"\nCreating {split_name} split...")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for i, filename in enumerate(files):
            source_file = clean_dir / filename
            target_file = target_dir / filename
            
            shutil.copy2(source_file, target_file)
            
            if (i + 1) % 100 == 0 or (i + 1) == len(files):
                print(f"  Copied {i + 1}/{len(files)} files to {target_dir}")
    
    print(f"\n✅ Dataset split complete!")
    print(f"📁 Train: {len(train_files)} images -> {splits['train'][1]}")
    print(f"📁 Val:   {len(val_files)} images -> {splits['val'][1]}")
    print(f"📁 Test:  {len(test_files)} images -> {splits['test'][1]}")
    
    # Show directory structure
    print(f"\n📂 New directory structure:")
    for split_name, (_, target_dir) in splits.items():
        print(f"   {target_dir}")

def main():
    parser = argparse.ArgumentParser(description='Split satellite dataset into train/val/test')
    parser.add_argument('--source', default='dataset/prep/SatelliteImages_s512_o128',
                       help='Source directory with prepared patches')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    try:
        split_satellite_dataset(
            source_dir=args.source,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 