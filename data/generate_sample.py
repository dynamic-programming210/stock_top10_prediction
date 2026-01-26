"""
Generate sample data for testing when Alpha Vantage is rate limited
This creates realistic synthetic OHLCV data for development/testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, BARS_FILE
from data.fetch_universe import load_universe_symbols
from utils import get_logger

logger = get_logger(__name__)


def generate_random_walk_prices(
    initial_price: float,
    n_days: int,
    volatility: float = 0.02,
    drift: float = 0.0001
) -> np.ndarray:
    """Generate random walk prices"""
    returns = np.random.normal(drift, volatility, n_days)
    prices = initial_price * np.cumprod(1 + returns)
    return prices


def generate_ohlcv(
    symbol: str,
    start_date: datetime,
    n_days: int = 252,
    initial_price: float = None
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for a symbol
    """
    if initial_price is None:
        initial_price = np.random.uniform(20, 500)
    
    # Generate close prices with random walk
    volatility = np.random.uniform(0.015, 0.035)  # Daily vol 1.5-3.5%
    close_prices = generate_random_walk_prices(initial_price, n_days, volatility)
    
    # Generate OHLV from close
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    
    # Generate realistic intraday ranges
    daily_ranges = np.random.uniform(0.005, 0.03, n_days)  # 0.5-3% daily range
    
    high = close_prices * (1 + daily_ranges * np.random.uniform(0.3, 0.7, n_days))
    low = close_prices * (1 - daily_ranges * np.random.uniform(0.3, 0.7, n_days))
    
    # Open is somewhere between previous close and current day range
    open_prices = np.zeros(n_days)
    open_prices[0] = initial_price
    for i in range(1, n_days):
        gap = np.random.normal(0, 0.005)  # Small overnight gap
        open_prices[i] = close_prices[i-1] * (1 + gap)
        # Ensure open is within high/low
        open_prices[i] = np.clip(open_prices[i], low[i], high[i])
    
    # Volume with some patterns
    base_volume = np.random.uniform(1e6, 50e6)
    volume = base_volume * np.random.lognormal(0, 0.5, n_days)
    
    # Add volume spikes on big moves
    big_moves = np.abs(np.diff(close_prices, prepend=close_prices[0]) / close_prices) > 0.02
    volume[big_moves] *= np.random.uniform(1.5, 3.0, big_moves.sum())
    
    df = pd.DataFrame({
        'date': dates,
        'symbol': symbol,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume.astype(int)
    })
    
    return df


def generate_sample_data(
    n_symbols: int = 100,
    n_days: int = 252,
    start_date: datetime = None
) -> pd.DataFrame:
    """
    Generate sample data for multiple symbols
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=n_days * 1.5)
    
    symbols = load_universe_symbols()[:n_symbols]
    
    logger.info(f"Generating sample data for {len(symbols)} symbols, {n_days} days each...")
    
    all_data = []
    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i+1}/{len(symbols)}")
        
        df = generate_ohlcv(symbol, start_date, n_days)
        all_data.append(df)
    
    result = pd.concat(all_data, ignore_index=True)
    
    # Remove weekends that might have slipped in
    result = result[result['date'].dt.dayofweek < 5]
    
    logger.info(f"Generated {len(result)} total bars")
    
    return result


def save_sample_data(df: pd.DataFrame):
    """Save generated data to parquet"""
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(BARS_FILE, index=False)
    logger.info(f"Saved sample data to {BARS_FILE}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample stock data")
    parser.add_argument('--symbols', type=int, default=100, help='Number of symbols')
    parser.add_argument('--days', type=int, default=252, help='Number of trading days')
    
    args = parser.parse_args()
    
    df = generate_sample_data(n_symbols=args.symbols, n_days=args.days)
    save_sample_data(df)
    
    print(f"\nGenerated data summary:")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total bars: {len(df)}")
