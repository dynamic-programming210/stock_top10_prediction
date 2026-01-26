"""
Feature engineering for stock prediction
Computes technical features and cross-sectional z-scores
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, FEAT_Z_FILE, MIN_BARS_FOR_FEATURES
from utils import get_logger, safe_divide

logger = get_logger(__name__)


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute return features for each symbol
    Assumes df is sorted by date and contains single symbol
    """
    close = df['close']
    
    df = df.copy()
    
    # Returns over different horizons
    df['ret_1'] = close.pct_change(1)
    df['ret_3'] = close.pct_change(3)
    df['ret_5'] = close.pct_change(5)
    df['ret_10'] = close.pct_change(10)
    
    return df


def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute range/volatility features
    """
    df = df.copy()
    
    # Daily range as percentage of close
    daily_range = (df['high'] - df['low']) / df['close']
    
    # Average range over windows
    df['range_5'] = daily_range.rolling(5).mean()
    df['range_10'] = daily_range.rolling(10).mean()
    
    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility features (rolling std of returns)
    """
    df = df.copy()
    
    # Daily returns
    daily_ret = df['close'].pct_change()
    
    # Rolling volatility
    df['vol_5'] = daily_ret.rolling(5).std()
    df['vol_10'] = daily_ret.rolling(10).std()
    
    return df


def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute candlestick pattern features
    """
    df = df.copy()
    
    high = df['high']
    low = df['low']
    open_ = df['open']
    close = df['close']
    
    # Body size as percentage of range
    body = close - open_
    range_ = high - low
    df['body_pct'] = body / range_.replace(0, np.nan)
    
    # Close position within daily range (0=low, 1=high)
    df['close_pos'] = (close - low) / range_.replace(0, np.nan)
    
    # Wick sizes as percentage of range
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    
    df['upper_wick_pct'] = upper_wick / range_.replace(0, np.nan)
    df['lower_wick_pct'] = lower_wick / range_.replace(0, np.nan)
    
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume/liquidity features
    """
    df = df.copy()
    
    volume = df['volume']
    close = df['close']
    
    # Volume ratio (current vs 10-day average)
    vol_ma10 = volume.rolling(10).mean()
    df['vol_ratio'] = volume / vol_ma10.replace(0, np.nan)
    
    # Volume change over 5 days
    df['vol_chg_5'] = volume.pct_change(5)
    
    # Dollar volume (20-day average)
    dollar_volume = close * volume
    df['dv_20'] = dollar_volume.rolling(20).mean()
    
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    B1: Compute RSI (Relative Strength Index)
    RSI measures momentum on a 0-100 scale
    """
    df = df.copy()
    
    # Calculate price changes
    delta = df['close'].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Calculate average gains and losses (using EMA for smoother results)
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    return df


def compute_macd(
    df: pd.DataFrame, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> pd.DataFrame:
    """
    B1: Compute MACD (Moving Average Convergence Divergence)
    MACD shows relationship between two EMAs
    """
    df = df.copy()
    close = df['close']
    
    # Calculate EMAs
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line = Fast EMA - Slow EMA
    df['macd_line'] = ema_fast - ema_slow
    
    # Signal line = EMA of MACD line
    df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
    
    # MACD histogram = MACD line - Signal line
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    
    return df


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    B1: Compute Bollinger Bands
    Shows price position relative to volatility bands
    """
    df = df.copy()
    close = df['close']
    
    # Middle band (SMA)
    middle = close.rolling(period).mean()
    
    # Standard deviation
    std = close.rolling(period).std()
    
    # Upper and lower bands
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    # BB Position: where price is within the bands (-1 to 1, 0 = middle)
    # Values > 1 means above upper band, < -1 means below lower band
    band_width = upper - lower
    df['bb_position'] = (close - middle) / (band_width / 2).replace(0, np.nan)
    
    # BB Width: normalized band width (volatility indicator)
    df['bb_width'] = band_width / middle.replace(0, np.nan)
    
    return df


def compute_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Compute forward return target
    """
    df = df.copy()
    
    # Forward return (shifted negative means future values)
    df[f'fwd_ret_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1
    
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for a single symbol's data
    """
    df = compute_returns(df)
    df = compute_range_features(df)
    df = compute_volatility_features(df)
    df = compute_candlestick_features(df)
    df = compute_volume_features(df)
    # B1: Technical indicators
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_target(df)
    
    return df


def build_feature_table(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature table for all symbols
    
    Args:
        bars_df: DataFrame with columns [date, symbol, open, high, low, close, volume]
        
    Returns:
        DataFrame with features computed per symbol
    """
    logger.info(f"Building features for {bars_df['symbol'].nunique()} symbols...")
    
    all_features = []
    
    for symbol, group in bars_df.groupby('symbol'):
        group = group.sort_values('date').copy()
        
        if len(group) < MIN_BARS_FOR_FEATURES:
            logger.debug(f"Skipping {symbol}: only {len(group)} bars")
            continue
            
        features = compute_all_features(group)
        all_features.append(features)
    
    if not all_features:
        logger.warning("No features computed!")
        return pd.DataFrame()
    
    result = pd.concat(all_features, ignore_index=True)
    logger.info(f"Computed features for {result['symbol'].nunique()} symbols, {len(result)} rows")
    
    return result


def cross_sectional_zscore(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Apply cross-sectional z-score normalization by date
    
    For each date, each feature is normalized to have mean=0, std=1
    across all symbols on that date.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
        
    df = df.copy()
    
    logger.info(f"Applying cross-sectional z-score for {len(feature_cols)} features...")
    
    # Z-score each feature within each date
    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"Feature column {col} not found, skipping z-score")
            continue
            
        # Group by date and compute z-score
        def zscore(x):
            std = x.std()
            if std == 0 or pd.isna(std):
                return x - x.mean()
            return (x - x.mean()) / std
        
        df[f'{col}_z'] = df.groupby('date')[col].transform(zscore)
    
    return df


def build_and_save_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features, apply z-scoring, and save to parquet
    """
    # Compute raw features
    features_df = build_feature_table(bars_df)
    
    if features_df.empty:
        return features_df
    
    # Apply z-scoring
    features_z = cross_sectional_zscore(features_df)
    
    # Save
    features_z.to_parquet(FEAT_Z_FILE, index=False)
    logger.info(f"Saved features to {FEAT_Z_FILE}")
    
    return features_z


def load_features() -> pd.DataFrame:
    """Load features from parquet file"""
    if not FEAT_Z_FILE.exists():
        logger.warning(f"Feature file not found: {FEAT_Z_FILE}")
        return pd.DataFrame()
    
    df = pd.read_parquet(FEAT_Z_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_feature_coverage_by_date(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Compute feature coverage statistics by date
    
    Returns DataFrame with:
    - date
    - total_symbols: number of symbols with data on that date
    - valid_symbols: number with all features non-null
    - coverage_rate: valid/total
    """
    if feature_cols is None:
        feature_cols = [f'{c}_z' for c in FEATURE_COLS]
    
    # Check which rows have all features valid
    df = df.copy()
    valid_features = [c for c in feature_cols if c in df.columns]
    
    if not valid_features:
        logger.warning("No z-scored features found")
        return pd.DataFrame()
    
    df['has_all_features'] = df[valid_features].notna().all(axis=1)
    
    # Aggregate by date
    coverage = df.groupby('date').agg(
        total_symbols=('symbol', 'count'),
        valid_symbols=('has_all_features', 'sum')
    ).reset_index()
    
    coverage['coverage_rate'] = coverage['valid_symbols'] / coverage['total_symbols']
    
    return coverage


def select_asof_date(
    df: pd.DataFrame, 
    min_coverage_rate: float = 0.6,
    feature_cols: List[str] = None
) -> Optional[pd.Timestamp]:
    """
    Select the best asof_date for inference
    
    Returns the latest date with coverage >= min_coverage_rate,
    or the date with max coverage if none meet the threshold.
    """
    coverage = get_feature_coverage_by_date(df, feature_cols)
    
    if coverage.empty:
        return None
    
    # Sort by date descending
    coverage = coverage.sort_values('date', ascending=False)
    
    # Find latest date meeting coverage threshold
    meeting_threshold = coverage[coverage['coverage_rate'] >= min_coverage_rate]
    
    if not meeting_threshold.empty:
        asof_date = meeting_threshold.iloc[0]['date']
        logger.info(f"Selected asof_date: {asof_date} (coverage: {meeting_threshold.iloc[0]['coverage_rate']:.1%})")
        return pd.Timestamp(asof_date)
    
    # Fall back to max coverage date
    best_idx = coverage['coverage_rate'].idxmax()
    best_row = coverage.loc[best_idx]
    logger.warning(f"No date meets {min_coverage_rate:.0%} coverage, using {best_row['date']} ({best_row['coverage_rate']:.1%})")
    return pd.Timestamp(best_row['date'])


if __name__ == "__main__":
    # Test feature computation
    from data.fetch_bars import load_existing_bars
    
    bars = load_existing_bars()
    if not bars.empty:
        print(f"Loaded {len(bars)} bars for {bars['symbol'].nunique()} symbols")
        
        features = build_and_save_features(bars)
        print(f"\nFeature table: {features.shape}")
        print(f"Columns: {features.columns.tolist()}")
        
        # Check coverage
        coverage = get_feature_coverage_by_date(features)
        print(f"\nCoverage by date (last 5 days):")
        print(coverage.tail())
        
        # Select asof date
        asof = select_asof_date(features)
        print(f"\nSelected asof_date: {asof}")
    else:
        print("No bars data found. Run data fetching first.")
