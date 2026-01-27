"""
D5: Backtesting Framework for Stock Top-10 Predictions
Simulates historical portfolio performance based on model predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURE_COLS, CURRENT_MODEL_VERSION, CAND_K, TOP_N,
    get_model_dir, OUTPUTS_DIR
)
from utils import get_logger

logger = get_logger(__name__)

# Backtest output files
BACKTEST_RESULTS_FILE = OUTPUTS_DIR / "backtest_results.json"
BACKTEST_TRADES_FILE = OUTPUTS_DIR / "backtest_trades.parquet"


class Backtester:
    """
    Backtesting engine for stock prediction strategies
    
    Simulates equal-weight portfolio of top-N predictions,
    rebalanced at specified frequency.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        top_n: int = TOP_N,
        rebalance_days: int = 5,
        transaction_cost: float = 0.001,  # 0.1% per trade (10 bps)
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        """
        Args:
            initial_capital: Starting portfolio value
            top_n: Number of stocks to hold
            rebalance_days: Days between rebalancing (5 = weekly)
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Slippage as fraction of trade value
        """
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # State
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {shares, entry_price}
        
        # History
        self.value_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.rebalance_dates: List[pd.Timestamp] = []
    
    def reset(self):
        """Reset backtester to initial state"""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.value_history = []
        self.trade_history = []
        self.rebalance_dates = []
    
    def _get_position_value(self, prices: Dict[str, float]) -> float:
        """Calculate total value of current positions"""
        total = 0.0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                total += pos['shares'] * prices[symbol]
        return total
    
    def _execute_trade(
        self,
        symbol: str,
        shares: float,
        price: float,
        date: pd.Timestamp,
        side: str
    ):
        """Execute a trade with costs"""
        trade_value = abs(shares * price)
        costs = trade_value * (self.transaction_cost + self.slippage)
        
        if side == 'buy':
            total_cost = trade_value + costs
            if total_cost > self.cash:
                # Adjust shares to fit available cash
                shares = (self.cash - costs) / price
                trade_value = shares * price
                costs = trade_value * (self.transaction_cost + self.slippage)
                total_cost = trade_value + costs
            
            self.cash -= total_cost
            
            if symbol in self.positions:
                # Add to existing position
                old_shares = self.positions[symbol]['shares']
                old_cost = self.positions[symbol]['entry_price'] * old_shares
                new_cost = price * shares
                total_shares = old_shares + shares
                avg_price = (old_cost + new_cost) / total_shares
                self.positions[symbol] = {
                    'shares': total_shares,
                    'entry_price': avg_price
                }
            else:
                self.positions[symbol] = {
                    'shares': shares,
                    'entry_price': price
                }
        
        elif side == 'sell':
            if symbol not in self.positions:
                return
            
            pos = self.positions[symbol]
            shares = min(shares, pos['shares'])
            trade_value = shares * price
            costs = trade_value * (self.transaction_cost + self.slippage)
            
            self.cash += trade_value - costs
            
            remaining_shares = pos['shares'] - shares
            if remaining_shares <= 0:
                del self.positions[symbol]
            else:
                self.positions[symbol]['shares'] = remaining_shares
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'price': price,
            'value': trade_value,
            'costs': costs
        })
    
    def rebalance(
        self,
        date: pd.Timestamp,
        target_symbols: List[str],
        prices: Dict[str, float]
    ):
        """
        Rebalance portfolio to equal-weight target symbols
        
        Args:
            date: Current date
            target_symbols: Symbols to hold after rebalancing
            prices: Current prices for all symbols
        """
        # Calculate current portfolio value
        position_value = self._get_position_value(prices)
        total_value = self.cash + position_value
        
        # Target allocation per symbol
        target_value_per_symbol = total_value / len(target_symbols) if target_symbols else 0
        
        # Sell positions not in target
        for symbol in list(self.positions.keys()):
            if symbol not in target_symbols:
                if symbol in prices:
                    self._execute_trade(
                        symbol,
                        self.positions[symbol]['shares'],
                        prices[symbol],
                        date,
                        'sell'
                    )
        
        # Adjust existing positions and buy new ones
        for symbol in target_symbols:
            if symbol not in prices:
                continue
            
            price = prices[symbol]
            target_shares = target_value_per_symbol / price
            
            current_shares = self.positions.get(symbol, {}).get('shares', 0)
            diff_shares = target_shares - current_shares
            
            if abs(diff_shares * price) < 100:  # Skip tiny adjustments
                continue
            
            if diff_shares > 0:
                self._execute_trade(symbol, diff_shares, price, date, 'buy')
            elif diff_shares < 0:
                self._execute_trade(symbol, -diff_shares, price, date, 'sell')
        
        self.rebalance_dates.append(date)
    
    def update_value(self, date: pd.Timestamp, prices: Dict[str, float]):
        """Record portfolio value for a given date"""
        position_value = self._get_position_value(prices)
        total_value = self.cash + position_value
        
        self.portfolio_value = total_value
        self.value_history.append({
            'date': date,
            'portfolio_value': total_value,
            'cash': self.cash,
            'position_value': position_value,
            'n_positions': len(self.positions)
        })
    
    def run(
        self,
        features_df: pd.DataFrame,
        bars_df: pd.DataFrame,
        ranker,
        regressor,
        feature_cols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """
        Run backtest over historical data
        
        Args:
            features_df: Feature dataframe with z-scores
            bars_df: Raw OHLCV data for prices
            ranker: Trained ranking model
            regressor: Trained regression model
            feature_cols: Feature columns used by models
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            
        Returns:
            Dict with backtest results and metrics
        """
        self.reset()
        
        # Get valid feature columns
        valid_features = [c for c in feature_cols if c in features_df.columns]
        
        # Filter date range
        dates = sorted(features_df['date'].unique())
        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]
        
        if len(dates) < 2:
            logger.error("Not enough dates for backtesting")
            return {}
        
        logger.info(f"Running backtest from {dates[0].date()} to {dates[-1].date()}")
        logger.info(f"Total trading days: {len(dates)}")
        
        last_rebalance = None
        
        for i, date in enumerate(dates):
            # Get prices for this date
            date_bars = bars_df[bars_df['date'] == date]
            if date_bars.empty:
                continue
            
            prices = dict(zip(date_bars['symbol'], date_bars['close']))
            
            # Check if we should rebalance
            should_rebalance = (
                last_rebalance is None or
                (date - last_rebalance).days >= self.rebalance_days
            )
            
            if should_rebalance:
                # Get predictions for this date
                date_features = features_df[features_df['date'] == date].copy()
                date_features = date_features.dropna(subset=valid_features)
                
                if len(date_features) >= self.top_n:
                    # Predict ranking scores
                    X = date_features[valid_features].values
                    date_features['rank_score'] = ranker.predict(X)
                    
                    # Get top candidates
                    top_cands = date_features.nlargest(CAND_K, 'rank_score')
                    
                    # Predict returns for candidates
                    if regressor is not None:
                        X_cands = top_cands[valid_features].values
                        top_cands['pred_ret'] = regressor.predict(X_cands)
                        top_n = top_cands.nlargest(self.top_n, 'pred_ret')
                    else:
                        top_n = top_cands.nlargest(self.top_n, 'rank_score')
                    
                    target_symbols = top_n['symbol'].tolist()
                    
                    # Rebalance
                    self.rebalance(date, target_symbols, prices)
                    last_rebalance = date
            
            # Update portfolio value
            self.update_value(date, prices)
            
            if (i + 1) % 50 == 0:
                logger.debug(f"Processed {i + 1}/{len(dates)} days, "
                           f"Portfolio: ${self.portfolio_value:,.2f}")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"\nBacktest Complete!")
        logger.info(f"Final Portfolio Value: ${self.portfolio_value:,.2f}")
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        return {
            'metrics': metrics,
            'value_history': self.value_history,
            'trade_count': len(self.trade_history),
            'rebalance_count': len(self.rebalance_dates)
        }
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if len(self.value_history) < 2:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.value_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # Basic metrics
        total_return = df['portfolio_value'].iloc[-1] / self.initial_capital - 1
        
        # Annualized return
        n_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        years = n_days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        daily_vol = df['daily_return'].std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annualized_return / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        df['cummax'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        # Win rate (days with positive returns)
        positive_days = (df['daily_return'] > 0).sum()
        total_days = df['daily_return'].notna().sum()
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        # Calculate trade statistics
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            total_costs = trades_df['costs'].sum()
            avg_trade_size = trades_df['value'].mean()
        else:
            total_costs = 0
            avg_trade_size = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': df['portfolio_value'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trading_costs': total_costs,
            'avg_trade_size': avg_trade_size,
            'n_trades': len(self.trade_history),
            'n_rebalances': len(self.rebalance_dates),
            'backtest_days': len(df)
        }
    
    def get_value_df(self) -> pd.DataFrame:
        """Get portfolio value history as DataFrame"""
        return pd.DataFrame(self.value_history)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        return pd.DataFrame(self.trade_history)


def run_backtest(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    model_version: str = None,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 100000.0,
    rebalance_days: int = 5
) -> Dict:
    """
    Run backtest with saved model
    
    Args:
        features_df: Feature dataframe
        bars_df: OHLCV data
        model_version: Model version to use
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        rebalance_days: Days between rebalancing
        
    Returns:
        Backtest results dict
    """
    from models.train import load_model, get_z_feature_cols
    
    # Load model
    ranker, regressor, schema = load_model(model_version)
    feature_cols = schema.get('feature_cols', get_z_feature_cols())
    
    # Run backtest
    backtester = Backtester(
        initial_capital=initial_capital,
        rebalance_days=rebalance_days
    )
    
    results = backtester.run(
        features_df=features_df,
        bars_df=bars_df,
        ranker=ranker,
        regressor=regressor,
        feature_cols=feature_cols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Save results
    if results:
        with open(BACKTEST_RESULTS_FILE, 'w') as f:
            json.dump(results['metrics'], f, indent=2, default=str)
        logger.info(f"Saved backtest results to {BACKTEST_RESULTS_FILE}")
        
        trades_df = backtester.get_trades_df()
        if not trades_df.empty:
            trades_df.to_parquet(BACKTEST_TRADES_FILE, index=False)
            logger.info(f"Saved trades to {BACKTEST_TRADES_FILE}")
    
    return results


def compare_to_benchmark(
    backtest_results: Dict,
    bars_df: pd.DataFrame,
    benchmark_symbol: str = 'SPY'
) -> Dict:
    """
    Compare backtest results to a benchmark (e.g., SPY)
    
    Args:
        backtest_results: Results from run_backtest
        bars_df: OHLCV data including benchmark
        benchmark_symbol: Symbol to use as benchmark
        
    Returns:
        Comparison metrics
    """
    if 'value_history' not in backtest_results:
        return {}
    
    # Get portfolio history
    portfolio_df = pd.DataFrame(backtest_results['value_history'])
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    
    # Get benchmark data
    bench_df = bars_df[bars_df['symbol'] == benchmark_symbol].copy()
    bench_df['date'] = pd.to_datetime(bench_df['date'])
    
    if bench_df.empty:
        logger.warning(f"Benchmark {benchmark_symbol} not found in data")
        return {}
    
    # Merge on dates
    merged = portfolio_df.merge(bench_df[['date', 'close']], on='date', how='inner')
    
    if len(merged) < 2:
        return {}
    
    # Normalize to starting values
    merged['portfolio_norm'] = merged['portfolio_value'] / merged['portfolio_value'].iloc[0]
    merged['benchmark_norm'] = merged['close'] / merged['close'].iloc[0]
    
    # Calculate returns
    portfolio_return = merged['portfolio_norm'].iloc[-1] - 1
    benchmark_return = merged['benchmark_norm'].iloc[-1] - 1
    
    # Alpha (excess return)
    alpha = portfolio_return - benchmark_return
    
    # Calculate beta
    portfolio_daily = merged['portfolio_value'].pct_change().dropna()
    benchmark_daily = merged['close'].pct_change().dropna()
    
    if len(portfolio_daily) > 0 and len(benchmark_daily) > 0:
        covariance = np.cov(portfolio_daily, benchmark_daily)[0, 1]
        benchmark_variance = benchmark_daily.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        # Information ratio
        tracking_error = (portfolio_daily - benchmark_daily).std() * np.sqrt(252)
        annualized_alpha = alpha * (252 / len(merged))  # Rough annualization
        info_ratio = annualized_alpha / tracking_error if tracking_error > 0 else 0
        
        # Benchmark Sharpe ratio
        bench_annual_return = (1 + benchmark_return) ** (252 / len(merged)) - 1
        bench_volatility = benchmark_daily.std() * np.sqrt(252)
        bench_sharpe = bench_annual_return / bench_volatility if bench_volatility > 0 else 0
    else:
        beta = 1
        info_ratio = 0
        bench_sharpe = 0
    
    return {
        'benchmark': benchmark_symbol,
        'portfolio_return': portfolio_return,
        'benchmark_return': benchmark_return,
        'alpha': alpha,
        'beta': beta,
        'information_ratio': info_ratio,
        'benchmark_sharpe': bench_sharpe
    }


def fetch_benchmark_data(benchmark_symbol: str = 'SPY') -> pd.DataFrame:
    """
    D6: Fetch benchmark data (SPY) using yfinance
    
    Args:
        benchmark_symbol: Benchmark ticker (default: SPY)
        
    Returns:
        DataFrame with benchmark OHLCV data
    """
    try:
        from data.fetch_yfinance import YFinanceClient
        
        client = YFinanceClient()
        df = client.fetch_daily_bars(benchmark_symbol)
        
        if df is not None and not df.empty:
            logger.info(f"Fetched {len(df)} bars for benchmark {benchmark_symbol}")
            return df
        else:
            logger.warning(f"Could not fetch benchmark {benchmark_symbol}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching benchmark: {e}")
        return pd.DataFrame()


def run_backtest_with_benchmark(
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    model_version: str = None,
    benchmark_symbol: str = 'SPY',
    **kwargs
) -> Dict:
    """
    D6: Run backtest and automatically compare to benchmark
    
    Fetches benchmark data if not present in bars_df
    """
    # Check if benchmark is in bars
    if benchmark_symbol not in bars_df['symbol'].unique():
        logger.info(f"Fetching benchmark {benchmark_symbol} data...")
        bench_df = fetch_benchmark_data(benchmark_symbol)
        if not bench_df.empty:
            bars_df = pd.concat([bars_df, bench_df], ignore_index=True)
    
    # Run backtest
    results = run_backtest(
        features_df=features_df,
        bars_df=bars_df,
        model_version=model_version,
        **kwargs
    )
    
    # Add benchmark comparison
    if results:
        comparison = compare_to_benchmark(results, bars_df, benchmark_symbol)
        results['benchmark_comparison'] = comparison
    
    return results


def generate_backtest_report(results: Dict, output_path: Path = None) -> str:
    """
    D6: Generate a formatted backtest report with benchmark comparison
    
    Returns:
        Markdown-formatted report string
    """
    if not results or 'metrics' not in results:
        return "No backtest results available"
    
    metrics = results['metrics']
    comparison = results.get('benchmark_comparison', {})
    
    report = []
    report.append("# Backtest Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    # Portfolio Performance
    report.append("## Portfolio Performance\n")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Initial Capital | ${metrics['initial_capital']:,.2f} |")
    report.append(f"| Final Value | ${metrics['final_value']:,.2f} |")
    report.append(f"| Total Return | {metrics['total_return']:.2%} |")
    report.append(f"| Annualized Return | {metrics['annualized_return']:.2%} |")
    report.append(f"| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} |")
    report.append(f"| Max Drawdown | {metrics['max_drawdown']:.2%} |")
    report.append(f"| Win Rate | {metrics['win_rate']:.2%} |")
    report.append(f"| Total Trades | {metrics['n_trades']} |")
    report.append(f"| Trading Costs | ${metrics['total_trading_costs']:,.2f} |")
    report.append("")
    
    # Benchmark Comparison
    if comparison:
        bench = comparison.get('benchmark', 'SPY')
        report.append(f"## vs {bench} Benchmark\n")
        report.append(f"| Metric | Portfolio | {bench} | Difference |")
        report.append(f"|--------|-----------|------|------------|")
        
        port_ret = comparison.get('portfolio_return', 0)
        bench_ret = comparison.get('benchmark_return', 0)
        alpha = comparison.get('alpha', 0)
        
        report.append(f"| Total Return | {port_ret:.2%} | {bench_ret:.2%} | {alpha:+.2%} |")
        report.append(f"| Beta | {comparison.get('beta', 1):.2f} | 1.00 | - |")
        report.append(f"| Information Ratio | {comparison.get('information_ratio', 0):.2f} | - | - |")
        report.append("")
        
        # Performance Summary
        if alpha > 0:
            report.append(f"✅ **Outperformed** {bench} by {alpha:.2%}")
        else:
            report.append(f"❌ **Underperformed** {bench} by {abs(alpha):.2%}")
    
    report_text = "\n".join(report)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved backtest report to {output_path}")
    
    return report_text


# =============================================================================
# Helper functions for testing and external use
# =============================================================================

def calculate_sharpe_ratio(returns, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Annualized Sharpe ratio
    """
    returns = np.array(returns)
    if len(returns) < 2:
        return 0.0
    
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    return (mean_excess / std_excess) * np.sqrt(252)


def calculate_max_drawdown(portfolio_values) -> float:
    """
    Calculate maximum drawdown from portfolio values
    
    Args:
        portfolio_values: List or array of portfolio values over time
        
    Returns:
        Maximum drawdown as a positive fraction (0 to 1)
    """
    values = np.array(portfolio_values)
    if len(values) < 2:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate drawdown at each point
    drawdowns = (running_max - values) / running_max
    
    return float(np.max(drawdowns))


def calculate_win_rate(returns) -> float:
    """
    Calculate win rate (fraction of positive returns)
    
    Args:
        returns: Array of returns
        
    Returns:
        Win rate as fraction (0 to 1)
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    
    positive = np.sum(returns > 0)
    return positive / len(returns)


def calculate_alpha_beta(portfolio_returns, benchmark_returns) -> Tuple[float, float]:
    """
    Calculate alpha and beta vs benchmark using linear regression
    
    Args:
        portfolio_returns: Series or array of portfolio returns
        benchmark_returns: Series or array of benchmark returns
        
    Returns:
        (alpha, beta) tuple, annualized
    """
    port_ret = np.array(portfolio_returns)
    bench_ret = np.array(benchmark_returns)
    
    # Align lengths
    min_len = min(len(port_ret), len(bench_ret))
    port_ret = port_ret[:min_len]
    bench_ret = bench_ret[:min_len]
    
    if len(port_ret) < 2:
        return 0.0, 1.0
    
    # Calculate beta (slope) and alpha (intercept)
    cov = np.cov(port_ret, bench_ret)[0, 1]
    var = np.var(bench_ret, ddof=1)
    
    if var == 0:
        return 0.0, 1.0
    
    beta = cov / var
    alpha = np.mean(port_ret) - beta * np.mean(bench_ret)
    
    # Annualize alpha
    alpha_annual = alpha * 252
    
    return float(alpha_annual), float(beta)


def calculate_information_ratio(portfolio_returns, benchmark_returns) -> float:
    """
    Calculate information ratio (active return / tracking error)
    
    Args:
        portfolio_returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Annualized information ratio
    """
    port_ret = np.array(portfolio_returns)
    bench_ret = np.array(benchmark_returns)
    
    min_len = min(len(port_ret), len(bench_ret))
    port_ret = port_ret[:min_len]
    bench_ret = bench_ret[:min_len]
    
    if len(port_ret) < 2:
        return 0.0
    
    # Active returns
    active_returns = port_ret - bench_ret
    
    mean_active = np.mean(active_returns)
    std_active = np.std(active_returns, ddof=1)
    
    if std_active == 0:
        return 0.0
    
    # Annualize
    return (mean_active / std_active) * np.sqrt(252)


def apply_transaction_costs(trade_value: float, cost_rate: float) -> float:
    """
    Apply transaction costs to a trade value
    
    Args:
        trade_value: Gross trade value
        cost_rate: Cost as fraction of trade value
        
    Returns:
        Net value after costs
    """
    return trade_value * (1 - cost_rate)


def apply_slippage(price: float, slippage_rate: float, is_buy: bool) -> float:
    """
    Apply slippage to a price
    
    Args:
        price: Original price
        slippage_rate: Slippage as fraction
        is_buy: True for buy orders, False for sell
        
    Returns:
        Price after slippage
    """
    if is_buy:
        # Buy at higher price (worse for buyer)
        return price * (1 + slippage_rate)
    else:
        # Sell at lower price (worse for seller)
        return price * (1 - slippage_rate)


if __name__ == "__main__":
    # Test backtest
    from features import load_features
    from data.fetch_bars import load_existing_bars
    
    features = load_features()
    bars = load_existing_bars()
    
    if not features.empty and not bars.empty:
        print("Running backtest...")
        
        results = run_backtest(
            features_df=features,
            bars_df=bars,
            initial_capital=100000,
            rebalance_days=5
        )
        
        if results:
            print("\nBacktest Metrics:")
            for key, value in results['metrics'].items():
                if isinstance(value, float):
                    if 'return' in key or 'rate' in key or 'drawdown' in key:
                        print(f"  {key}: {value:.2%}")
                    elif 'ratio' in key:
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value:,.2f}")
                else:
                    print(f"  {key}: {value}")
            
            # Compare to benchmark
            comparison = compare_to_benchmark(results, bars)
            if comparison:
                print(f"\nVs {comparison['benchmark']}:")
                print(f"  Portfolio Return: {comparison['portfolio_return']:.2%}")
                print(f"  Benchmark Return: {comparison['benchmark_return']:.2%}")
                print(f"  Alpha: {comparison['alpha']:.2%}")
                print(f"  Beta: {comparison['beta']:.2f}")
    else:
        print("Missing features or bars data. Run data fetch and feature building first.")
