"""
D1: Export utilities for stock predictions
Exports Top-10 predictions, backtest results to CSV/Excel formats
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR, TOP10_LATEST_FILE, TOP10_HISTORY_FILE
from utils import get_logger

logger = get_logger(__name__)

# Export directory
EXPORTS_DIR = OUTPUTS_DIR / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)


def export_top10_to_csv(
    output_path: str = None,
    include_history: bool = False,
    history_days: int = 30
) -> str:
    """
    Export Top-10 predictions to CSV
    
    Args:
        output_path: Custom output path (default: auto-generated)
        include_history: Include historical predictions
        history_days: Number of days of history to include
        
    Returns:
        Path to exported file
    """
    if not TOP10_LATEST_FILE.exists():
        logger.error("No Top-10 predictions found")
        return None
    
    latest_df = pd.read_parquet(TOP10_LATEST_FILE)
    latest_df['date'] = pd.to_datetime(latest_df['date'])
    
    if include_history and TOP10_HISTORY_FILE.exists():
        history_df = pd.read_parquet(TOP10_HISTORY_FILE)
        history_df['date'] = pd.to_datetime(history_df['date'])
        
        cutoff = datetime.now() - pd.Timedelta(days=history_days)
        history_df = history_df[history_df['date'] >= cutoff]
        
        combined = pd.concat([latest_df, history_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'symbol'], keep='first')
        combined = combined.sort_values(['date', 'pred_ret_5'], ascending=[False, False])
        export_df = combined
    else:
        export_df = latest_df
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"top10_{timestamp}.csv"
        output_path = EXPORTS_DIR / filename
    
    # Select and rename columns for readability
    columns_map = {
        'date': 'Date',
        'symbol': 'Symbol',
        'close': 'Close Price',
        'rank_score': 'Rank Score',
        'pred_ret_5': 'Predicted 5-Day Return',
        'pred_price_5d': 'Predicted Price (5-Day)',
        'reason_human': 'Analysis',
        'sector': 'Sector'
    }
    
    available_cols = [c for c in columns_map.keys() if c in export_df.columns]
    export_df = export_df[available_cols].copy()
    export_df = export_df.rename(columns={c: columns_map[c] for c in available_cols})
    
    # Format numeric columns
    if 'Predicted 5-Day Return' in export_df.columns:
        export_df['Predicted 5-Day Return'] = (export_df['Predicted 5-Day Return'] * 100).round(2)
        export_df = export_df.rename(columns={'Predicted 5-Day Return': 'Predicted Return (%)'})
    
    if 'Close Price' in export_df.columns:
        export_df['Close Price'] = export_df['Close Price'].round(2)
    
    if 'Predicted Price (5-Day)' in export_df.columns:
        export_df['Predicted Price (5-Day)'] = export_df['Predicted Price (5-Day)'].round(2)
    
    if 'Rank Score' in export_df.columns:
        export_df['Rank Score'] = export_df['Rank Score'].round(4)
    
    export_df.to_csv(output_path, index=False)
    logger.info(f"Exported Top-10 to CSV: {output_path}")
    
    return str(output_path)


def export_top10_to_excel(
    output_path: str = None,
    include_history: bool = True,
    include_summary: bool = True
) -> str:
    """
    Export Top-10 predictions to Excel with multiple sheets
    """
    try:
        import openpyxl
    except ImportError:
        logger.error("openpyxl not installed. Run: pip install openpyxl")
        return None
    
    if not TOP10_LATEST_FILE.exists():
        logger.error("No Top-10 predictions found")
        return None
    
    latest_df = pd.read_parquet(TOP10_LATEST_FILE)
    latest_df['date'] = pd.to_datetime(latest_df['date'])
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"stock_predictions_{timestamp}.xlsx"
        output_path = EXPORTS_DIR / filename
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Latest Top-10
        latest_export = latest_df.copy()
        if 'pred_ret_5' in latest_export.columns:
            latest_export['pred_ret_5_pct'] = (latest_export['pred_ret_5'] * 100).round(2)
        latest_export.to_excel(writer, sheet_name='Latest Top-10', index=False)
        
        # Sheet 2: Historical predictions
        if include_history and TOP10_HISTORY_FILE.exists():
            history_df = pd.read_parquet(TOP10_HISTORY_FILE)
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df = history_df.sort_values(['date', 'pred_ret_5'], ascending=[False, False])
            
            if 'pred_ret_5' in history_df.columns:
                history_df['pred_ret_5_pct'] = (history_df['pred_ret_5'] * 100).round(2)
            
            history_df.to_excel(writer, sheet_name='Historical Predictions', index=False)
        
        # Sheet 3: Summary statistics
        if include_summary:
            summary_data = generate_summary_stats(latest_df)
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 4: Sector breakdown
        if 'sector' in latest_df.columns:
            sector_counts = latest_df['sector'].value_counts().reset_index()
            sector_counts.columns = ['Sector', 'Count']
            sector_counts.to_excel(writer, sheet_name='Sector Breakdown', index=False)
    
    logger.info(f"Exported to Excel: {output_path}")
    return str(output_path)


def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """Generate summary statistics for predictions"""
    summary = {
        'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Prediction Date': str(df['date'].max()) if 'date' in df.columns else 'N/A',
        'Number of Picks': len(df),
    }
    
    if 'pred_ret_5' in df.columns:
        summary['Avg Predicted Return (%)'] = round(df['pred_ret_5'].mean() * 100, 2)
        summary['Min Predicted Return (%)'] = round(df['pred_ret_5'].min() * 100, 2)
        summary['Max Predicted Return (%)'] = round(df['pred_ret_5'].max() * 100, 2)
    
    if 'rank_score' in df.columns:
        summary['Avg Rank Score'] = round(df['rank_score'].mean(), 4)
    
    if 'sector' in df.columns:
        summary['Unique Sectors'] = df['sector'].nunique()
    
    return summary


def export_backtest_results(results: Dict, output_format: str = 'excel') -> str:
    """Export backtest results to CSV or Excel"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    portfolio_values = results.get('value_history', results.get('portfolio_values', []))
    if isinstance(portfolio_values, list) and len(portfolio_values) > 0:
        if isinstance(portfolio_values[0], dict):
            values_df = pd.DataFrame(portfolio_values)
        else:
            values_df = pd.DataFrame({'value': portfolio_values})
    else:
        values_df = pd.DataFrame()
    
    metrics = results.get('metrics', {})
    metrics_df = pd.DataFrame([metrics])
    
    if output_format == 'csv':
        values_path = EXPORTS_DIR / f"backtest_values_{timestamp}.csv"
        metrics_path = EXPORTS_DIR / f"backtest_metrics_{timestamp}.csv"
        
        if not values_df.empty:
            values_df.to_csv(values_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)
        
        logger.info(f"Exported backtest CSVs to {EXPORTS_DIR}")
        return str(metrics_path)
    
    else:
        output_path = EXPORTS_DIR / f"backtest_report_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            if not values_df.empty:
                values_df.to_excel(writer, sheet_name='Portfolio Values', index=False)
            
            trades = results.get('trade_history', results.get('trades', []))
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
        
        logger.info(f"Exported backtest to Excel: {output_path}")
        return str(output_path)


def export_feature_importance(model, feature_names: List[str], output_path: str = None) -> str:
    """Export feature importance from trained model"""
    if not hasattr(model, 'feature_importances_'):
        logger.error("Model does not have feature_importances_ attribute")
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    importance_df['Importance (%)'] = (importance_df['Importance'] / importance_df['Importance'].sum() * 100).round(2)
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = EXPORTS_DIR / f"feature_importance_{timestamp}.csv"
    
    importance_df[['Rank', 'Feature', 'Importance', 'Importance (%)']].to_csv(output_path, index=False)
    logger.info(f"Exported feature importance to: {output_path}")
    
    return str(output_path)


def quick_export(format: str = 'csv') -> str:
    """Quick export of latest Top-10 predictions"""
    if format == 'csv':
        return export_top10_to_csv()
    else:
        return export_top10_to_excel()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Stock Predictions")
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'excel'],
                        help='Export format (default: csv)')
    parser.add_argument('--history', action='store_true', help='Include historical predictions')
    parser.add_argument('--days', type=int, default=30, help='Days of history (default: 30)')
    parser.add_argument('--output', type=str, help='Custom output path')
    
    args = parser.parse_args()
    
    if args.format == 'csv':
        path = export_top10_to_csv(output_path=args.output, include_history=args.history, history_days=args.days)
    else:
        path = export_top10_to_excel(output_path=args.output, include_history=args.history)
    
    if path:
        print(f"\n✅ Export successful: {path}")
    else:
        print("\n❌ Export failed")
