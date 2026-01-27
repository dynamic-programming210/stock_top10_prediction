"""
Pytest configuration and shared fixtures
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return project root path"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """
    Create comprehensive OHLCV sample data for testing
    This data is used across multiple test modules
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
    
    # Generate realistic price walk
    returns = np.random.randn(100) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'open': prices * (1 + np.random.randn(100) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'close': prices,
        'adj_close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # Ensure high >= low
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    return df


@pytest.fixture(scope="session")
def multi_symbol_data():
    """Create multi-symbol OHLCV data"""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    dates = pd.date_range(start='2024-01-01', periods=60, freq='B')
    
    np.random.seed(42)
    data = []
    
    for symbol in symbols:
        base_price = np.random.uniform(100, 500)
        returns = np.random.randn(60) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'symbol': symbol,
                'open': prices[i] * (1 + np.random.randn() * 0.005),
                'high': prices[i] * (1 + abs(np.random.randn()) * 0.01),
                'low': prices[i] * (1 - abs(np.random.randn()) * 0.01),
                'close': prices[i],
                'adj_close': prices[i],
                'volume': np.random.randint(1000000, 50000000)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_feature_cols():
    """Standard feature column names"""
    return [
        'ret_1', 'ret_5', 'ret_20',
        'vol_5', 'vol_20',
        'volume_ratio',
        'body_pct', 'upper_shadow', 'lower_shadow',
        'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_pct'
    ]


@pytest.fixture
def mock_z_feature_cols(mock_feature_cols):
    """Z-scored feature column names"""
    return [f'{c}_z' for c in mock_feature_cols]
