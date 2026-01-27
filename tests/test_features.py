"""
Unit tests for feature computation functions
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestReturnFeatures:
    """Tests for return calculation functions"""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='B')
        np.random.seed(42)
        
        # Generate realistic price data
        prices = 100 + np.cumsum(np.random.randn(30) * 2)
        
        return pd.DataFrame({
            'date': dates,
            'symbol': 'TEST',
            'open': prices * (1 + np.random.randn(30) * 0.01),
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'adj_close': prices,
            'volume': np.random.randint(1000000, 5000000, 30)
        })
    
    def test_compute_returns(self, sample_ohlcv):
        """Test return calculations"""
        from features.build_features import compute_returns
        
        result = compute_returns(sample_ohlcv.copy())
        
        assert 'ret_1' in result.columns
        assert 'ret_5' in result.columns
        assert 'ret_20' in result.columns
        
        # Check first values are NaN (no prior data)
        assert pd.isna(result['ret_1'].iloc[0])
        
        # Check calculation for ret_1
        expected_ret_1 = (result['adj_close'].iloc[1] - result['adj_close'].iloc[0]) / result['adj_close'].iloc[0]
        assert np.isclose(result['ret_1'].iloc[1], expected_ret_1, rtol=1e-5)
    
    def test_compute_volatility(self, sample_ohlcv):
        """Test volatility calculations"""
        from features.build_features import compute_volatility
        
        # First compute returns
        df = sample_ohlcv.copy()
        df['ret_1'] = df['adj_close'].pct_change()
        
        result = compute_volatility(df)
        
        assert 'vol_5' in result.columns
        assert 'vol_20' in result.columns
        
        # Volatility should be positive
        valid_vol = result['vol_20'].dropna()
        assert (valid_vol >= 0).all()
    
    def test_compute_forward_returns(self, sample_ohlcv):
        """Test forward return calculations (targets)"""
        from features.build_features import compute_forward_returns
        
        result = compute_forward_returns(sample_ohlcv.copy())
        
        assert 'fwd_ret_5' in result.columns
        
        # Last 5 values should be NaN (no future data)
        assert result['fwd_ret_5'].tail(5).isna().all()


class TestTechnicalIndicators:
    """Tests for technical indicator calculations"""
    
    @pytest.fixture
    def sample_ohlcv_extended(self):
        """Create extended sample OHLCV for indicator calculations"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.randn(50) * 2)
        
        return pd.DataFrame({
            'date': dates,
            'symbol': 'TEST',
            'open': prices * (1 + np.random.randn(50) * 0.01),
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'adj_close': prices,
            'volume': np.random.randint(1000000, 5000000, 50)
        })
    
    def test_compute_rsi(self, sample_ohlcv_extended):
        """Test RSI calculation"""
        from features.build_features import compute_rsi
        
        result = compute_rsi(sample_ohlcv_extended.copy())
        
        assert 'rsi_14' in result.columns
        
        # RSI should be between 0 and 100
        valid_rsi = result['rsi_14'].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_compute_macd(self, sample_ohlcv_extended):
        """Test MACD calculation"""
        from features.build_features import compute_macd
        
        result = compute_macd(sample_ohlcv_extended.copy())
        
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns
        
        # MACD histogram should be difference of MACD and signal
        valid_idx = result['macd_hist'].dropna().index
        np.testing.assert_array_almost_equal(
            result.loc[valid_idx, 'macd_hist'],
            result.loc[valid_idx, 'macd'] - result.loc[valid_idx, 'macd_signal'],
            decimal=10
        )
    
    def test_compute_bollinger(self, sample_ohlcv_extended):
        """Test Bollinger Bands calculation"""
        from features.build_features import compute_bollinger
        
        result = compute_bollinger(sample_ohlcv_extended.copy())
        
        assert 'bb_upper' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_pct' in result.columns
        
        # Upper band should be > middle > lower
        valid_idx = result['bb_upper'].dropna().index
        sma = result.loc[valid_idx, 'adj_close'].rolling(20).mean()
        
        # bb_pct should be between 0 and 1 (mostly)
        valid_pct = result['bb_pct'].dropna()
        assert valid_pct.median() > 0
        assert valid_pct.median() < 1
    
    def test_compute_stochastic(self, sample_ohlcv_extended):
        """Test Stochastic Oscillator calculation"""
        from features.build_features import compute_stochastic
        
        result = compute_stochastic(sample_ohlcv_extended.copy())
        
        assert 'stoch_k' in result.columns
        assert 'stoch_d' in result.columns
        
        # Stochastic should be between 0 and 100
        valid_k = result['stoch_k'].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
    
    def test_compute_atr(self, sample_ohlcv_extended):
        """Test ATR calculation"""
        from features.build_features import compute_atr
        
        result = compute_atr(sample_ohlcv_extended.copy())
        
        assert 'atr_14' in result.columns
        
        # ATR should be positive
        valid_atr = result['atr_14'].dropna()
        assert (valid_atr > 0).all()
    
    def test_compute_obv(self, sample_ohlcv_extended):
        """Test OBV calculation"""
        from features.build_features import compute_obv
        
        result = compute_obv(sample_ohlcv_extended.copy())
        
        assert 'obv' in result.columns
        assert 'obv_ma' in result.columns
        
        # OBV should exist for all rows
        assert result['obv'].isna().sum() == 0


class TestFeatureNormalization:
    """Tests for feature normalization"""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data"""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='B')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'ret_1': np.random.randn() * 0.02,
                    'vol_5': np.random.uniform(0.01, 0.05),
                    'rsi_14': np.random.uniform(30, 70)
                })
        
        return pd.DataFrame(data)
    
    def test_z_score_normalization(self, sample_features):
        """Test cross-sectional z-score normalization"""
        from features.build_features import z_score_normalize
        
        result = z_score_normalize(sample_features.copy(), ['ret_1', 'vol_5', 'rsi_14'])
        
        assert 'ret_1_z' in result.columns
        assert 'vol_5_z' in result.columns
        assert 'rsi_14_z' in result.columns
        
        # Z-scores within each date should have mean ~0 and std ~1
        for date in result['date'].unique():
            date_data = result[result['date'] == date]
            for col in ['ret_1_z', 'vol_5_z', 'rsi_14_z']:
                mean = date_data[col].mean()
                std = date_data[col].std()
                assert np.isclose(mean, 0, atol=1e-10)
                assert np.isclose(std, 1, atol=1e-10) or len(date_data) == 1


class TestFeaturePipeline:
    """Integration tests for full feature pipeline"""
    
    def test_full_feature_build(self):
        """Test building features for a symbol"""
        from features.build_features import build_features_for_symbol
        
        # Create minimal test data
        dates = pd.date_range(start='2024-01-01', periods=60, freq='B')
        np.random.seed(42)
        prices = 150 + np.cumsum(np.random.randn(60) * 3)
        
        ohlcv = pd.DataFrame({
            'date': dates,
            'symbol': 'AAPL',
            'open': prices * (1 + np.random.randn(60) * 0.01),
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'adj_close': prices,
            'volume': np.random.randint(10000000, 50000000, 60)
        })
        
        result = build_features_for_symbol(ohlcv)
        
        # Should have all expected feature columns
        expected_features = ['ret_1', 'ret_5', 'vol_5', 'rsi_14', 'macd', 'bb_pct']
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"
        
        # Should have forward return target
        assert 'fwd_ret_5' in result.columns


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
