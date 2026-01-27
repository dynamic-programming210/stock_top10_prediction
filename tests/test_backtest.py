"""
Unit tests for backtesting functionality
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBacktester:
    """Tests for Backtester class"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for backtesting"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                   'NVDA', 'JPM', 'V', 'JNJ', 'UNH']
        
        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'predicted_rank': np.random.uniform(0, 1),
                    'fwd_ret_5': np.random.randn() * 0.02
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for backtesting"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                   'NVDA', 'JPM', 'V', 'JNJ', 'UNH']
        
        np.random.seed(42)
        data = []
        prices = {s: 100 + np.cumsum(np.random.randn(50) * 2) for s in symbols}
        
        for i, date in enumerate(dates):
            for symbol in symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'adj_close': prices[symbol][i]
                })
        
        return pd.DataFrame(data)
    
    def test_backtester_init(self):
        """Test Backtester initialization"""
        from models.backtest import Backtester
        
        bt = Backtester(
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        assert bt.initial_capital == 100000
        assert bt.transaction_cost == 0.001
        assert bt.slippage == 0.0005
    
    def test_run_backtest(self, sample_predictions, sample_price_data):
        """Test running a backtest"""
        from models.backtest import Backtester
        
        bt = Backtester(initial_capital=100000)
        
        results = bt.run_backtest(
            predictions=sample_predictions,
            prices=sample_price_data,
            top_n=5,
            rebalance_freq=5
        )
        
        assert 'portfolio_values' in results
        assert 'returns' in results
        assert 'metrics' in results
        
        # Portfolio values should be a list
        assert len(results['portfolio_values']) > 0
    
    def test_backtest_metrics(self, sample_predictions, sample_price_data):
        """Test backtest metrics calculation"""
        from models.backtest import Backtester
        
        bt = Backtester(initial_capital=100000)
        results = bt.run_backtest(
            predictions=sample_predictions,
            prices=sample_price_data,
            top_n=5,
            rebalance_freq=5
        )
        
        metrics = results['metrics']
        
        # Check key metrics exist
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics


class TestBacktestMetrics:
    """Tests for individual metric calculations"""
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        from models.backtest import calculate_sharpe_ratio
        
        # Create returns with known characteristics
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005  # ~12% annual return
        
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        # Sharpe should be reasonable for slightly positive returns
        assert -3 < sharpe < 3
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        from models.backtest import calculate_max_drawdown
        
        # Create portfolio values with a clear drawdown
        values = [100, 110, 105, 95, 90, 100, 110, 105]
        
        max_dd = calculate_max_drawdown(values)
        
        assert max_dd > 0  # Drawdown is positive
        assert max_dd <= 1  # Can't exceed 100%
        
        # Max drawdown should be from 110 to 90 = 18.18%
        expected_dd = (110 - 90) / 110
        assert np.isclose(max_dd, expected_dd, rtol=0.01)
    
    def test_calculate_win_rate(self):
        """Test win rate calculation"""
        from models.backtest import calculate_win_rate
        
        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.005]
        # 4 positive, 2 negative = 66.67% win rate
        
        win_rate = calculate_win_rate(returns)
        
        assert np.isclose(win_rate, 4/6, rtol=0.01)


class TestBenchmarkComparison:
    """Tests for benchmark comparison functionality"""
    
    @pytest.fixture
    def portfolio_returns(self):
        """Sample portfolio returns"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        return pd.Series(np.random.randn(50) * 0.01, index=dates)
    
    @pytest.fixture
    def benchmark_returns(self):
        """Sample benchmark returns"""
        np.random.seed(43)
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        return pd.Series(np.random.randn(50) * 0.008, index=dates)
    
    def test_calculate_alpha_beta(self, portfolio_returns, benchmark_returns):
        """Test alpha and beta calculation"""
        from models.backtest import calculate_alpha_beta
        
        alpha, beta = calculate_alpha_beta(portfolio_returns, benchmark_returns)
        
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        # Beta should be reasonable
        assert -2 < beta < 2
    
    def test_calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        """Test information ratio calculation"""
        from models.backtest import calculate_information_ratio
        
        ir = calculate_information_ratio(portfolio_returns, benchmark_returns)
        
        assert isinstance(ir, float)
        # IR should be finite
        assert np.isfinite(ir)


class TestTransactionCosts:
    """Tests for transaction cost modeling"""
    
    def test_apply_transaction_costs(self):
        """Test transaction cost application"""
        from models.backtest import apply_transaction_costs
        
        trade_value = 10000  # $10,000 trade
        cost_rate = 0.001  # 0.1% cost
        
        net_value = apply_transaction_costs(trade_value, cost_rate)
        
        expected = trade_value * (1 - cost_rate)
        assert np.isclose(net_value, expected)
    
    def test_apply_slippage(self):
        """Test slippage application"""
        from models.backtest import apply_slippage
        
        price = 100
        slippage_rate = 0.0005  # 5 bps
        
        # For a buy, slippage increases cost
        buy_price = apply_slippage(price, slippage_rate, is_buy=True)
        assert buy_price > price
        
        # For a sell, slippage decreases proceeds
        sell_price = apply_slippage(price, slippage_rate, is_buy=False)
        assert sell_price < price


class TestBacktestReport:
    """Tests for backtest report generation"""
    
    @pytest.fixture
    def sample_backtest_results(self):
        """Create sample backtest results"""
        return {
            'portfolio_values': [100000, 101000, 102000, 101500, 103000],
            'returns': [0.0, 0.01, 0.0099, -0.0049, 0.0148],
            'metrics': {
                'total_return': 0.03,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.01,
                'win_rate': 0.75,
                'n_trades': 20
            },
            'trades': []
        }
    
    def test_generate_report(self, sample_backtest_results):
        """Test report generation"""
        from models.backtest import generate_backtest_report
        
        report = generate_backtest_report(sample_backtest_results)
        
        assert isinstance(report, str)
        assert 'Total Return' in report
        assert 'Sharpe Ratio' in report
        assert 'Max Drawdown' in report


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
