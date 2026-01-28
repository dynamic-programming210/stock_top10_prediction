"""
D2: Performance Tracking
Tracks and analyzes prediction accuracy over time
Computes hit rates, actual vs predicted returns, and portfolio performance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR, TOP10_HISTORY_FILE
from data.fetch_bars import load_existing_bars
from utils import get_logger

logger = get_logger(__name__)

PERFORMANCE_DIR = OUTPUTS_DIR / "performance"
PERFORMANCE_DIR.mkdir(exist_ok=True)


class PerformanceTracker:
    """Tracks prediction accuracy and portfolio performance over time"""
    
    def __init__(self, horizon_days: int = 5):
        self.horizon_days = horizon_days
        self.predictions_df = None
        self.bars_df = None
        self.performance_df = None
        
    def load_data(self) -> 'PerformanceTracker':
        """Load predictions and price data"""
        if TOP10_HISTORY_FILE.exists():
            self.predictions_df = pd.read_parquet(TOP10_HISTORY_FILE)
            self.predictions_df['date'] = pd.to_datetime(self.predictions_df['date'])
            logger.info(f"Loaded {len(self.predictions_df)} historical predictions")
        else:
            logger.warning("No prediction history found")
            self.predictions_df = pd.DataFrame()
        
        self.bars_df = load_existing_bars()
        if not self.bars_df.empty:
            self.bars_df['date'] = pd.to_datetime(self.bars_df['date'])
            logger.info(f"Loaded price data for {self.bars_df['symbol'].nunique()} symbols")
        
        return self
    
    def compute_actual_returns(self) -> pd.DataFrame:
        """Compute actual forward returns for all predictions"""
        if self.predictions_df.empty or self.bars_df.empty:
            logger.warning("Missing data for computing returns")
            return pd.DataFrame()
        
        predictions = self.predictions_df.copy()
        bars = self.bars_df[['date', 'symbol', 'close']].copy()
        
        results = []
        
        for _, pred in predictions.iterrows():
            pred_date = pred['date']
            symbol = pred['symbol']
            
            symbol_bars = bars[bars['symbol'] == symbol].sort_values('date')
            pred_bar = symbol_bars[symbol_bars['date'] == pred_date]
            
            if pred_bar.empty:
                continue
            
            future_bars = symbol_bars[symbol_bars['date'] > pred_date].head(self.horizon_days)
            
            if future_bars.empty:
                continue
            
            actual_price = future_bars.iloc[-1]['close']
            actual_date = future_bars.iloc[-1]['date']
            pred_price = pred_bar.iloc[0]['close']
            
            actual_ret = (actual_price - pred_price) / pred_price
            pred_ret = pred.get('pred_ret_5', 0)
            
            results.append({
                'pred_date': pred_date,
                'actual_date': actual_date,
                'symbol': symbol,
                'pred_price': float(pred_price),
                'actual_price': float(actual_price),
                'pred_ret_5': float(pred_ret),
                'actual_ret_5': float(actual_ret),
                'rank_score': pred.get('rank_score', 0),
                'direction_hit': (pred_ret > 0) == (actual_ret > 0),
                'positive_return': actual_ret > 0,
                'beat_benchmark': actual_ret > 0.002
            })
        
        self.performance_df = pd.DataFrame(results)
        logger.info(f"Computed actual returns for {len(results)} predictions")
        
        return self.performance_df
    
    def compute_metrics(self) -> Dict:
        """Compute comprehensive performance metrics"""
        if self.performance_df is None or self.performance_df.empty:
            self.compute_actual_returns()
        
        if self.performance_df.empty:
            return {}
        
        df = self.performance_df
        
        metrics = {
            'period': {
                'start': str(df['pred_date'].min().date()),
                'end': str(df['pred_date'].max().date()),
                'n_prediction_days': df['pred_date'].nunique(),
                'total_predictions': len(df)
            },
            'accuracy': {
                'direction_accuracy': float(df['direction_hit'].mean()),
                'positive_return_rate': float(df['positive_return'].mean()),
                'beat_benchmark_rate': float(df['beat_benchmark'].mean())
            },
            'returns': {
                'mean_predicted': float(df['pred_ret_5'].mean()),
                'mean_actual': float(df['actual_ret_5'].mean()),
                'std_actual': float(df['actual_ret_5'].std()),
                'median_actual': float(df['actual_ret_5'].median()),
                'correlation': float(df['pred_ret_5'].corr(df['actual_ret_5'])),
                'prediction_error_mae': float((df['pred_ret_5'] - df['actual_ret_5']).abs().mean())
            },
            'risk': {
                'max_drawdown_single': float(df['actual_ret_5'].min()),
                'worst_prediction': float(df.loc[df['actual_ret_5'].idxmin(), 'actual_ret_5']) if len(df) > 0 else 0,
                'best_prediction': float(df.loc[df['actual_ret_5'].idxmax(), 'actual_ret_5']) if len(df) > 0 else 0,
                'sharpe_proxy': float(df['actual_ret_5'].mean() / df['actual_ret_5'].std()) if df['actual_ret_5'].std() > 0 else 0
            }
        }
        
        daily_perf = df.groupby('pred_date').agg({
            'actual_ret_5': 'mean',
            'direction_hit': 'mean',
            'positive_return': 'mean'
        }).reset_index()
        
        metrics['daily_stats'] = {
            'avg_daily_return': float(daily_perf['actual_ret_5'].mean()),
            'winning_days': int((daily_perf['actual_ret_5'] > 0).sum()),
            'losing_days': int((daily_perf['actual_ret_5'] < 0).sum()),
            'win_rate_by_day': float((daily_perf['actual_ret_5'] > 0).mean())
        }
        
        return metrics
    
    def compute_rolling_metrics(self, window: int = 20) -> pd.DataFrame:
        """Compute rolling performance metrics"""
        if self.performance_df is None or self.performance_df.empty:
            self.compute_actual_returns()
        
        if self.performance_df.empty:
            return pd.DataFrame()
        
        daily = self.performance_df.groupby('pred_date').agg({
            'actual_ret_5': ['mean', 'sum', 'count'],
            'direction_hit': 'mean',
            'positive_return': 'mean'
        })
        daily.columns = ['avg_return', 'total_return', 'n_picks', 'direction_accuracy', 'positive_rate']
        daily = daily.reset_index().sort_values('pred_date')
        
        daily['rolling_avg_return'] = daily['avg_return'].rolling(window, min_periods=1).mean()
        daily['rolling_accuracy'] = daily['direction_accuracy'].rolling(window, min_periods=1).mean()
        daily['cumulative_return'] = (1 + daily['avg_return']).cumprod() - 1
        
        return daily
    
    def generate_report(self, save: bool = True) -> Dict:
        """Generate comprehensive performance report"""
        self.load_data()
        self.compute_actual_returns()
        metrics = self.compute_metrics()
        rolling = self.compute_rolling_metrics()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'metrics': metrics,
            'rolling_summary': {
                'final_cumulative_return': float(rolling['cumulative_return'].iloc[-1]) if not rolling.empty else 0,
                'avg_rolling_return': float(rolling['rolling_avg_return'].mean()) if not rolling.empty else 0,
                'avg_rolling_accuracy': float(rolling['rolling_accuracy'].mean()) if not rolling.empty else 0
            }
        }
        
        if not self.performance_df.empty:
            recent = self.performance_df.tail(50)
            report['recent_performance'] = {
                'last_50_predictions': {
                    'avg_return': float(recent['actual_ret_5'].mean()),
                    'direction_accuracy': float(recent['direction_hit'].mean()),
                    'positive_rate': float(recent['positive_return'].mean())
                }
            }
            
            best = self.performance_df.nlargest(5, 'actual_ret_5')[['pred_date', 'symbol', 'actual_ret_5', 'pred_ret_5']]
            worst = self.performance_df.nsmallest(5, 'actual_ret_5')[['pred_date', 'symbol', 'actual_ret_5', 'pred_ret_5']]
            
            report['notable_predictions'] = {
                'best_performers': best.to_dict('records'),
                'worst_performers': worst.to_dict('records')
            }
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = PERFORMANCE_DIR / f"performance_report_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report saved to {report_path}")
            
            if not rolling.empty:
                rolling_path = PERFORMANCE_DIR / f"rolling_performance_{timestamp}.csv"
                rolling.to_csv(rolling_path, index=False)
        
        return report
    
    def get_symbol_performance(self, symbol: str = None) -> pd.DataFrame:
        """Get performance breakdown by symbol"""
        if self.performance_df is None:
            self.compute_actual_returns()
        
        if self.performance_df.empty:
            return pd.DataFrame()
        
        if symbol:
            return self.performance_df[self.performance_df['symbol'] == symbol]
        
        symbol_perf = self.performance_df.groupby('symbol').agg({
            'actual_ret_5': ['mean', 'std', 'count'],
            'direction_hit': 'mean',
            'positive_return': 'mean'
        })
        symbol_perf.columns = ['avg_return', 'return_std', 'n_picks', 'direction_accuracy', 'positive_rate']
        symbol_perf = symbol_perf.reset_index().sort_values('avg_return', ascending=False)
        
        return symbol_perf
    
    def print_summary(self, report: Dict = None):
        """Print performance summary to console"""
        if report is None:
            report = self.generate_report(save=False)
        
        metrics = report.get('metrics', {})
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TRACKING SUMMARY")
        print("=" * 60)
        
        period = metrics.get('period', {})
        print(f"\n📅 Period: {period.get('start', 'N/A')} to {period.get('end', 'N/A')}")
        print(f"   Prediction Days: {period.get('n_prediction_days', 0)}")
        print(f"   Total Predictions: {period.get('total_predictions', 0)}")
        
        accuracy = metrics.get('accuracy', {})
        print(f"\n🎯 Accuracy:")
        print(f"   Direction Accuracy: {accuracy.get('direction_accuracy', 0):.1%}")
        print(f"   Positive Return Rate: {accuracy.get('positive_return_rate', 0):.1%}")
        print(f"   Beat Benchmark Rate: {accuracy.get('beat_benchmark_rate', 0):.1%}")
        
        returns = metrics.get('returns', {})
        print(f"\n💰 Returns:")
        print(f"   Mean Predicted: {returns.get('mean_predicted', 0)*100:.2f}%")
        print(f"   Mean Actual: {returns.get('mean_actual', 0)*100:.2f}%")
        print(f"   Prediction Correlation: {returns.get('correlation', 0):.4f}")
        print(f"   MAE: {returns.get('prediction_error_mae', 0)*100:.2f}%")
        
        risk = metrics.get('risk', {})
        print(f"\n⚠️ Risk:")
        print(f"   Best Single Pick: {risk.get('best_prediction', 0)*100:.2f}%")
        print(f"   Worst Single Pick: {risk.get('worst_prediction', 0)*100:.2f}%")
        print(f"   Sharpe Proxy: {risk.get('sharpe_proxy', 0):.3f}")
        
        daily = metrics.get('daily_stats', {})
        print(f"\n📈 Daily Stats:")
        print(f"   Winning Days: {daily.get('winning_days', 0)}")
        print(f"   Losing Days: {daily.get('losing_days', 0)}")
        print(f"   Win Rate by Day: {daily.get('win_rate_by_day', 0):.1%}")
        
        rolling = report.get('rolling_summary', {})
        print(f"\n📊 Cumulative:")
        print(f"   Total Return: {rolling.get('final_cumulative_return', 0)*100:.2f}%")
        
        print("\n" + "=" * 60)


def track_performance(save_report: bool = True, print_summary: bool = True) -> Dict:
    """Main function to track and report performance"""
    tracker = PerformanceTracker()
    report = tracker.generate_report(save=save_report)
    
    if print_summary:
        tracker.print_summary(report)
    
    return report


def get_latest_performance() -> Dict:
    """Get the most recent performance metrics"""
    tracker = PerformanceTracker()
    tracker.load_data()
    tracker.compute_actual_returns()
    return tracker.compute_metrics()


def export_performance_to_csv(output_path: str = None) -> str:
    """Export detailed performance data to CSV"""
    tracker = PerformanceTracker()
    tracker.load_data()
    tracker.compute_actual_returns()
    
    if tracker.performance_df.empty:
        logger.warning("No performance data to export")
        return None
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = PERFORMANCE_DIR / f"performance_detail_{timestamp}.csv"
    
    tracker.performance_df.to_csv(output_path, index=False)
    logger.info(f"Performance data exported to {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Tracking")
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    parser.add_argument('--no-save', action='store_true', help='Skip saving report')
    
    args = parser.parse_args()
    
    print("\n🔍 Running Performance Tracking...\n")
    
    report = track_performance(save_report=not args.no_save, print_summary=True)
    
    if args.export:
        path = export_performance_to_csv()
        if path:
            print(f"\n📁 Exported to: {path}")
