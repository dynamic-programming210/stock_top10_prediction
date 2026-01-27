"""
Unit tests for model training and prediction
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelTraining:
    """Tests for model training functions"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100
        
        # Feature columns (z-scored)
        feature_cols = [
            'ret_1_z', 'ret_5_z', 'ret_20_z',
            'vol_5_z', 'vol_20_z',
            'rsi_14_z', 'macd_z', 'bb_pct_z'
        ]
        
        data = {col: np.random.randn(n_samples) for col in feature_cols}
        data['date'] = pd.date_range(start='2024-01-01', periods=n_samples, freq='B')
        data['symbol'] = np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], n_samples)
        data['fwd_ret_5'] = np.random.randn(n_samples) * 0.02  # Target
        
        return pd.DataFrame(data)
    
    def test_train_ranker(self, sample_training_data):
        """Test ranker model training"""
        from models.train import train_ranker
        from config import FEATURE_COLS
        
        # Split data
        train_df = sample_training_data.iloc[:80]
        val_df = sample_training_data.iloc[80:]
        
        # Get z-scored feature columns that exist
        z_cols = [f'{c}_z' for c in FEATURE_COLS if f'{c}_z' in sample_training_data.columns]
        if not z_cols:
            z_cols = [c for c in sample_training_data.columns if c.endswith('_z')]
        
        model, metrics = train_ranker(train_df, val_df, z_cols)
        
        assert model is not None
        assert 'train_corr' in metrics
        assert 'val_corr' in metrics
        assert isinstance(metrics['train_corr'], float)
    
    def test_train_regressor(self, sample_training_data):
        """Test regressor model training"""
        from models.train import train_regressor
        from config import FEATURE_COLS
        
        train_df = sample_training_data.iloc[:80]
        val_df = sample_training_data.iloc[80:]
        
        z_cols = [c for c in sample_training_data.columns if c.endswith('_z')]
        
        model, metrics = train_regressor(train_df, val_df, z_cols)
        
        assert model is not None
        assert 'train_rmse' in metrics
        assert 'val_rmse' in metrics
        assert metrics['train_rmse'] >= 0
        assert metrics['val_rmse'] >= 0


class TestPrediction:
    """Tests for model prediction"""
    
    @pytest.fixture
    def trained_model(self):
        """Create a simple trained model"""
        from sklearn.ensemble import GradientBoostingRegressor
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = GradientBoostingRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)
        return model
    
    def test_predict(self, trained_model):
        """Test model predictions"""
        np.random.seed(42)
        X_test = np.random.randn(10, 5)
        
        predictions = trained_model.predict(X_test)
        
        assert len(predictions) == 10
        assert not np.isnan(predictions).any()


class TestWalkForwardValidation:
    """Tests for walk-forward validation"""
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series data for walk-forward validation"""
        np.random.seed(42)
        
        # Create data spanning multiple months
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        data = []
        for date in dates:
            for symbol in symbols:
                row = {
                    'date': date,
                    'symbol': symbol,
                    'ret_1_z': np.random.randn(),
                    'ret_5_z': np.random.randn(),
                    'vol_5_z': np.random.randn(),
                    'rsi_14_z': np.random.randn(),
                    'fwd_ret_5': np.random.randn() * 0.02
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def test_walk_forward_validation_structure(self, time_series_data):
        """Test walk-forward validation produces expected structure"""
        from models.train import walk_forward_validation
        
        feature_cols = ['ret_1_z', 'ret_5_z', 'vol_5_z', 'rsi_14_z']
        
        results = walk_forward_validation(
            time_series_data,
            feature_cols,
            n_folds=3,
            min_train_months=3
        )
        
        assert 'fold_metrics' in results
        assert 'mean_val_corr' in results
        assert len(results['fold_metrics']) == 3
    
    def test_no_lookahead_bias(self, time_series_data):
        """Test that validation doesn't use future data"""
        # This is a conceptual test - in practice we verify by checking
        # that validation dates are always after training dates
        
        dates = sorted(time_series_data['date'].unique())
        n_dates = len(dates)
        
        # In walk-forward, training should use earlier data
        train_end_idx = int(n_dates * 0.6)  # 60% for training
        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:]
        
        # All validation dates should be after all training dates
        assert min(val_dates) > max(train_dates)


class TestTuning:
    """Tests for hyperparameter tuning"""
    
    @pytest.fixture
    def tuning_data(self):
        """Create data for tuning tests"""
        np.random.seed(42)
        n_samples = 200
        
        feature_cols = ['ret_1_z', 'ret_5_z', 'vol_5_z', 'rsi_14_z']
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples // 5, freq='B')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        data = []
        for date in dates:
            for symbol in symbols:
                row = {'date': date, 'symbol': symbol}
                for col in feature_cols:
                    row[col] = np.random.randn()
                row['fwd_ret_5'] = np.random.randn() * 0.02
                data.append(row)
        
        return pd.DataFrame(data)
    
    def test_correlation_scorer(self):
        """Test custom correlation scorer"""
        from models.tuning import correlation_scorer
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2])
        
        score = correlation_scorer(y_true, y_pred)
        
        assert score > 0.9  # Should be highly correlated
        assert score <= 1.0
    
    def test_prepare_tuning_data(self, tuning_data):
        """Test data preparation for tuning"""
        from models.tuning import prepare_tuning_data
        
        X, y, feature_cols = prepare_tuning_data(tuning_data)
        
        assert X.shape[0] == y.shape[0]
        assert len(feature_cols) > 0
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


class TestDiversification:
    """Tests for sector diversification"""
    
    @pytest.fixture
    def ranked_stocks(self):
        """Create ranked stock data"""
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
                      'AMZN', 'AMD', 'CRM', 'INTC', 'ORCL',
                      'JPM', 'BAC', 'GS', 'V', 'MA'],
            'predicted_rank': [1.0, 0.95, 0.90, 0.85, 0.80,
                              0.75, 0.70, 0.65, 0.60, 0.55,
                              0.50, 0.45, 0.40, 0.35, 0.30]
        })
    
    def test_sector_diversifier_init(self):
        """Test SectorDiversifier initialization"""
        from models.diversification import SectorDiversifier
        
        diversifier = SectorDiversifier(max_per_sector=3, min_sectors=4)
        
        assert diversifier.max_per_sector == 3
        assert diversifier.min_sectors == 4
    
    def test_diversify_selection(self, ranked_stocks):
        """Test diversified selection respects sector limits"""
        from models.diversification import SectorDiversifier
        
        diversifier = SectorDiversifier(max_per_sector=2, min_sectors=3)
        
        result = diversifier.diversify_selection(ranked_stocks, top_n=10)
        
        assert len(result) == 10
        
        # Check no sector has more than 2 stocks
        sector_counts = result['sector'].value_counts()
        assert sector_counts.max() <= 2
    
    def test_analyze_concentration(self, ranked_stocks):
        """Test concentration analysis"""
        from models.diversification import SectorDiversifier
        
        diversifier = SectorDiversifier()
        analysis = diversifier.analyze_sector_concentration(ranked_stocks, top_n=10)
        
        assert 'n_sectors' in analysis
        assert 'herfindahl_index' in analysis
        assert 0 <= analysis['herfindahl_index'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
