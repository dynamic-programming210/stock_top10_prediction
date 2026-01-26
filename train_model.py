#!/usr/bin/env python
"""
Training script for Stock Top-10 Predictor
Run this after collecting sufficient data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_bars import load_existing_bars
from features.build_features import build_and_save_features
from models.train import train_full_pipeline
import json

def main():
    print("=" * 60)
    print("Stock Top-10 Predictor - Model Training")
    print("=" * 60)
    
    # Step 1: Load bars
    print("\n[1/3] Loading OHLCV data...")
    bars = load_existing_bars()
    
    if bars.empty:
        print("ERROR: No bars data found!")
        print("Run 'python app/update_daily.py --setup' first to fetch data.")
        return
    
    print(f"  Loaded {len(bars)} bars for {bars['symbol'].nunique()} symbols")
    print(f"  Date range: {bars['date'].min()} to {bars['date'].max()}")
    
    # Step 2: Build features
    print("\n[2/3] Building features...")
    features = build_and_save_features(bars)
    
    if features.empty:
        print("ERROR: Could not build features!")
        return
    
    print(f"  Built {len(features)} feature rows for {features['symbol'].nunique()} symbols")
    
    # Step 3: Train model
    print("\n[3/3] Training models...")
    try:
        ranker, regressor, metrics, model_dir = train_full_pipeline(features)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nModel saved to: {model_dir}")
        print("\nMetrics Summary:")
        print(f"  Training samples: {metrics.get('train_samples', 'N/A')}")
        print(f"  Validation samples: {metrics.get('val_samples', 'N/A')}")
        
        if 'regressor' in metrics:
            print(f"  Regressor Val RMSE: {metrics['regressor'].get('val_rmse', 'N/A'):.4f}")
            print(f"  Regressor Val Corr: {metrics['regressor'].get('val_corr', 'N/A'):.4f}")
        
        print("\nNext steps:")
        print("  1. Run 'python app/update_daily.py' to generate predictions")
        print("  2. Run 'streamlit run app/web.py' to view the web app")
        
    except Exception as e:
        print(f"ERROR: Training failed - {e}")
        raise

if __name__ == "__main__":
    main()
