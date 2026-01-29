"""
Configuration settings for Stock Top-10 Predictor
"""
import os
from pathlib import Path

# ============ PATHS ============
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_DIR = MODELS_DIR  # Alias for monitoring.py compatibility
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
APP_DIR = PROJECT_ROOT / "app"
FEATURES_FILE = DATA_DIR / "feat_z.parquet"  # Alias for monitoring.py

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, APP_DIR]:
    d.mkdir(exist_ok=True)

# ============ API KEYS ============
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "HOCJEXBA2YA9A56M")

# ============ DATA SETTINGS ============
# Alpha Vantage free tier: 25 requests/day, 5 requests/minute
AV_REQUESTS_PER_MINUTE = 5
AV_REQUESTS_PER_DAY = 25
SYMBOLS_PER_RUN = 20  # How many symbols to update per daily run

# Historical data settings
HISTORY_YEARS = 3  # Years of historical data to fetch
MIN_BARS_FOR_FEATURES = 30  # Minimum bars needed to compute features

# ============ FEATURE SETTINGS ============
FEATURE_COLS = [
    # Original features
    'ret_1', 'ret_3', 'ret_5', 'ret_10',
    'range_5', 'range_10',
    'vol_5', 'vol_10',
    'body_pct', 'close_pos', 'upper_wick_pct', 'lower_wick_pct',
    'vol_ratio', 'vol_chg_5', 'dv_20',
    # B1: Technical indicators
    'rsi_14',                           # RSI (14-period)
    'macd_line', 'macd_signal', 'macd_hist',  # MACD
    'bb_position', 'bb_width',          # Bollinger Bands
    # B2-B5: Additional technical indicators
    'sma_cross',                        # B2: SMA crossover signal
    'ema_cross',                        # B2: EMA crossover signal
    'atr_14', 'atr_pct',                # B3: ATR
    'obv_slope',                        # B4: OBV trend
    'stoch_k', 'stoch_d',               # B5: Stochastic Oscillator
    # B6: Momentum and mean-reversion signals
    'roc_5', 'roc_10', 'roc_20',        # Rate of Change
    'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',  # Price vs MAs
    'price_zscore_20',                  # Price z-score (mean reversion)
    'dist_from_52w_high', 'dist_from_52w_low',  # Distance from extremes
    'momentum_composite',               # Combined momentum score
    'mean_reversion_signal',            # Mean reversion opportunity
    # Task 3: News sentiment features
    'news_sentiment_avg',               # Average news sentiment (-1 to 1)
    'news_sentiment_std',               # Sentiment volatility
    'news_count',                       # Number of recent articles
    'news_positive_ratio',              # % positive articles
    'news_negative_ratio',              # % negative articles
]

# A4: Sector mapping for classification
SECTOR_MAP = {
    'Information Technology': 0,
    'Financials': 1,
    'Health Care': 2,
    'Consumer Discretionary': 3,
    'Consumer Staples': 4,
    'Communication Services': 5,
    'Industrials': 6,
    'Energy': 7,
    'Utilities': 8,
    'Real Estate': 9,
    'Materials': 10,
}

# ============ MODEL SETTINGS ============
CURRENT_MODEL_VERSION = "v001"
CAND_K = 50  # Top K candidates from ranker for regression
TOP_N = 10   # Final top-N picks

# Coverage settings
MIN_COVERAGE_RATE = 0.6  # At least 60% of universe must have features

# ============ LIGHTGBM PARAMS ============
RANKER_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_iterations': 200,
    'early_stopping_rounds': 20,
    'seed': 42
}

REGRESSOR_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_iterations': 200,
    'early_stopping_rounds': 20,
    'seed': 42
}

# ============ FILE NAMES ============
UNIVERSE_SYMBOLS_FILE = DATA_DIR / "universe_symbols.txt"
UNIVERSE_META_FILE = DATA_DIR / "universe_meta.parquet"
UPDATE_QUEUE_FILE = DATA_DIR / "update_queue.json"
BARS_FILE = DATA_DIR / "bars.parquet"
FEAT_Z_FILE = DATA_DIR / "feat_z.parquet"

TOP10_LATEST_FILE = OUTPUTS_DIR / "top10_latest.parquet"
TOP10_HISTORY_FILE = OUTPUTS_DIR / "top10_history.parquet"
QUALITY_REPORT_FILE = OUTPUTS_DIR / "quality_report.json"

def get_model_dir(version: str = None) -> Path:
    """Get model directory for a specific version"""
    version = version or CURRENT_MODEL_VERSION
    model_dir = MODELS_DIR / version
    model_dir.mkdir(exist_ok=True)
    return model_dir
