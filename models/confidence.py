"""
F3: Confidence Intervals for Predictions
Computes prediction uncertainty using model variance and historical accuracy
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR
from utils import get_logger

logger = get_logger(__name__)

# Performance history for calibration
PERFORMANCE_DIR = OUTPUTS_DIR / "performance"


def compute_prediction_std(
    predictions_df: pd.DataFrame,
    model_predictions: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Compute prediction standard deviation based on multiple factors:
    1. Historical prediction error for similar stocks
    2. Model ensemble disagreement (if available)
    3. Stock-specific volatility
    
    Returns DataFrame with confidence intervals added
    """
    df = predictions_df.copy()
    
    # Base uncertainty from predicted return magnitude
    # Higher predicted returns typically have higher uncertainty
    base_std = df['pred_ret_5'].abs() * 0.5 + 0.02  # Min 2% uncertainty
    
    # Adjust by volatility if available
    if 'vol_10_z' in df.columns:
        # High volatility stocks have more uncertainty
        vol_factor = 1 + df['vol_10_z'].clip(-2, 2) * 0.1
        base_std = base_std * vol_factor
    elif 'vol_10' in df.columns:
        vol_factor = 1 + (df['vol_10'] / df['vol_10'].mean() - 1) * 0.2
        base_std = base_std * vol_factor.clip(0.8, 1.5)
    
    # Adjust by rank score confidence
    if 'rank_score' in df.columns:
        # Higher rank scores = more confident = tighter intervals
        rank_normalized = (df['rank_score'] - df['rank_score'].min()) / \
                         (df['rank_score'].max() - df['rank_score'].min() + 1e-6)
        confidence_factor = 1.2 - rank_normalized * 0.4  # 0.8 to 1.2
        base_std = base_std * confidence_factor
    
    df['pred_std'] = base_std.clip(0.01, 0.15)  # Cap at 1-15%
    
    return df


def add_confidence_intervals(
    df: pd.DataFrame,
    confidence_level: float = 0.90
) -> pd.DataFrame:
    """
    Add confidence intervals to predictions
    
    Args:
        df: DataFrame with pred_ret_5 column
        confidence_level: Confidence level (default 90%)
    
    Returns:
        DataFrame with pred_lower, pred_upper, pred_range columns
    """
    df = df.copy()
    
    # Compute prediction std if not present
    if 'pred_std' not in df.columns:
        df = compute_prediction_std(df)
    
    # Z-score for confidence level (90% = 1.645, 95% = 1.96)
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Compute intervals
    df['pred_lower'] = df['pred_ret_5'] - z_score * df['pred_std']
    df['pred_upper'] = df['pred_ret_5'] + z_score * df['pred_std']
    df['pred_range'] = df['pred_upper'] - df['pred_lower']
    
    # Confidence score (inverse of relative uncertainty)
    df['confidence_score'] = 1 - (df['pred_std'] / (df['pred_ret_5'].abs() + 0.01)).clip(0, 1)
    
    # Price intervals
    if 'close' in df.columns:
        df['price_lower'] = df['close'] * (1 + df['pred_lower'])
        df['price_upper'] = df['close'] * (1 + df['pred_upper'])
    
    return df


def calibrate_from_history(lookback_days: int = 90) -> Dict[str, float]:
    """
    Calibrate uncertainty estimates from historical prediction errors
    
    Returns dict with calibration parameters
    """
    perf_file = PERFORMANCE_DIR / "performance_history.parquet"
    
    if not perf_file.exists():
        logger.warning("No performance history for calibration")
        return {'scale_factor': 1.0, 'base_error': 0.03}
    
    try:
        df = pd.read_parquet(perf_file)
        
        if df.empty or 'actual_ret_5' not in df.columns:
            return {'scale_factor': 1.0, 'base_error': 0.03}
        
        # Compute historical prediction errors
        df = df.dropna(subset=['pred_ret_5', 'actual_ret_5'])
        
        if len(df) < 10:
            return {'scale_factor': 1.0, 'base_error': 0.03}
        
        errors = df['pred_ret_5'] - df['actual_ret_5']
        
        calibration = {
            'scale_factor': errors.std() / 0.03,  # Scale relative to default
            'base_error': errors.std(),
            'mean_error': errors.mean(),
            'rmse': np.sqrt((errors ** 2).mean()),
            'n_samples': len(df)
        }
        
        logger.info(f"Calibration from {len(df)} samples: RMSE={calibration['rmse']:.4f}")
        
        return calibration
        
    except Exception as e:
        logger.warning(f"Calibration failed: {e}")
        return {'scale_factor': 1.0, 'base_error': 0.03}


def format_confidence_interval(row: pd.Series) -> str:
    """Format confidence interval for display"""
    if 'pred_lower' not in row or 'pred_upper' not in row:
        return "N/A"
    
    lower = row['pred_lower'] * 100
    upper = row['pred_upper'] * 100
    
    return f"{lower:+.1f}% to {upper:+.1f}%"


def get_confidence_label(score: float) -> str:
    """Get human-readable confidence label"""
    if score >= 0.8:
        return "🟢 High"
    elif score >= 0.6:
        return "🟡 Medium"
    elif score >= 0.4:
        return "🟠 Low"
    else:
        return "🔴 Very Low"


def compute_portfolio_confidence(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute aggregate confidence metrics for the portfolio
    """
    if df.empty or 'confidence_score' not in df.columns:
        return {}
    
    return {
        'avg_confidence': df['confidence_score'].mean(),
        'min_confidence': df['confidence_score'].min(),
        'high_confidence_count': (df['confidence_score'] >= 0.7).sum(),
        'avg_pred_range': df['pred_range'].mean() if 'pred_range' in df.columns else None
    }


# CLI test
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'close': [185.0, 380.0, 140.0, 175.0, 850.0],
        'pred_ret_5': [0.03, 0.025, 0.02, 0.035, 0.04],
        'rank_score': [0.85, 0.82, 0.78, 0.75, 0.90],
        'vol_10': [0.015, 0.018, 0.02, 0.022, 0.025]
    })
    
    print("📊 Testing Confidence Intervals\n")
    
    df = add_confidence_intervals(sample_data, confidence_level=0.90)
    
    print("Predictions with 90% Confidence Intervals:")
    print("-" * 70)
    
    for _, row in df.iterrows():
        ci_str = format_confidence_interval(row)
        conf_label = get_confidence_label(row['confidence_score'])
        print(f"{row['symbol']:6s} | Pred: {row['pred_ret_5']*100:+.1f}% | "
              f"CI: {ci_str:20s} | Confidence: {conf_label}")
    
    print("\n" + "-" * 70)
    portfolio = compute_portfolio_confidence(df)
    print(f"Portfolio Avg Confidence: {portfolio['avg_confidence']:.2f}")
    print(f"High Confidence Stocks: {portfolio['high_confidence_count']}/{len(df)}")
