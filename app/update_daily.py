"""
Daily update pipeline for stock predictions
Orchestrates data fetching, feature building, and top-10 generation
A1: Updated to support yfinance (default) or Alpha Vantage
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TOP10_LATEST_FILE, TOP10_HISTORY_FILE, QUALITY_REPORT_FILE,
    OUTPUTS_DIR, MIN_COVERAGE_RATE, CURRENT_MODEL_VERSION,
    SYMBOLS_PER_RUN
)
from data.fetch_universe import load_universe_symbols
# A1: Import both data sources
from data.fetch_bars import load_existing_bars, save_bars
from features.build_features import (
    build_and_save_features, load_features, select_asof_date,
    get_feature_coverage_by_date
)
from models.train import load_model, generate_top10, get_z_feature_cols
from utils import get_logger

logger = get_logger(__name__)


def ensure_output_dir():
    """Ensure output directory exists"""
    OUTPUTS_DIR.mkdir(exist_ok=True)


def load_top10_history() -> pd.DataFrame:
    """Load historical top-10 predictions"""
    if TOP10_HISTORY_FILE.exists():
        df = pd.read_parquet(TOP10_HISTORY_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()


def save_top10_latest(df: pd.DataFrame):
    """Save latest top-10"""
    ensure_output_dir()
    df.to_parquet(TOP10_LATEST_FILE, index=False)
    logger.info(f"Saved latest top-10 to {TOP10_LATEST_FILE}")


def save_top10_history(df: pd.DataFrame, new_top10: pd.DataFrame):
    """
    Append new top-10 to history, de-duplicating by date+symbol
    """
    ensure_output_dir()
    
    if df.empty:
        combined = new_top10
    else:
        combined = pd.concat([df, new_top10], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'symbol'], keep='last')
        combined = combined.sort_values(['date', 'pred_ret_5'], ascending=[False, False])
    
    combined.to_parquet(TOP10_HISTORY_FILE, index=False)
    logger.info(f"Saved top-10 history ({len(combined)} records) to {TOP10_HISTORY_FILE}")
    
    return combined


def generate_quality_report(
    bars_df: pd.DataFrame,
    features_df: pd.DataFrame,
    top10_df: pd.DataFrame,
    update_results: Dict,
    asof_date: Optional[pd.Timestamp]
) -> Dict:
    """
    Generate quality/health report for monitoring
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'asof_date': str(asof_date) if asof_date else None,
        'data': {
            'total_bars': len(bars_df) if not bars_df.empty else 0,
            'unique_symbols': bars_df['symbol'].nunique() if not bars_df.empty else 0,
            'date_range': {
                'start': str(bars_df['date'].min()) if not bars_df.empty else None,
                'end': str(bars_df['date'].max()) if not bars_df.empty else None
            }
        },
        'features': {
            'total_rows': len(features_df) if not features_df.empty else 0,
            'unique_symbols': features_df['symbol'].nunique() if not features_df.empty else 0
        },
        'top10': {
            'generated': len(top10_df) > 0,
            'count': len(top10_df)
        },
        'update_results': update_results
    }
    
    # Add coverage stats
    if not features_df.empty:
        coverage = get_feature_coverage_by_date(features_df)
        if not coverage.empty and asof_date:
            asof_coverage = coverage[coverage['date'] == asof_date]
            if not asof_coverage.empty:
                report['coverage'] = {
                    'asof_date_symbols': int(asof_coverage.iloc[0]['valid_symbols']),
                    'asof_date_rate': float(asof_coverage.iloc[0]['coverage_rate'])
                }
    
    return report


def save_quality_report(report: Dict):
    """Save quality report to JSON"""
    ensure_output_dir()
    with open(QUALITY_REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved quality report to {QUALITY_REPORT_FILE}")


def run_daily_update(
    batch_size: int = SYMBOLS_PER_RUN,
    model_version: str = None,
    skip_data_update: bool = False,
    force_retrain: bool = False,
    use_yfinance: bool = True  # A1: Default to yfinance
) -> Dict:
    """
    Run the daily update pipeline
    
    Steps:
    1. Incremental data update (if not skipped)
    2. Rebuild features
    3. Select asof_date by coverage
    4. Generate top-10 predictions
    5. Save outputs and quality report
    
    Args:
        batch_size: Number of symbols to update
        model_version: Model version to use
        skip_data_update: Skip fetching new data
        force_retrain: Force model retraining
        use_yfinance: A1: Use yfinance (True) or Alpha Vantage (False)
        
    Returns:
        Summary dict of the update
    """
    model_version = model_version or CURRENT_MODEL_VERSION
    
    logger.info("=" * 50)
    logger.info("Starting daily update pipeline")
    logger.info("=" * 50)
    
    result = {
        'status': 'unknown',
        'timestamp': datetime.now().isoformat(),
        'steps': {}
    }
    
    try:
        # Step 1: Load universe
        logger.info("\n[Step 1] Loading universe...")
        symbols = load_universe_symbols()
        result['steps']['universe'] = {'symbols': len(symbols)}
        logger.info(f"Loaded {len(symbols)} symbols")
        
        # Step 2: Incremental data update
        if not skip_data_update:
            logger.info(f"\n[Step 2] Fetching data...")
            
            if use_yfinance:
                # A1: Use yfinance (unlimited, fast batch downloads)
                from data.fetch_yfinance import fetch_all_universe, update_incremental
                logger.info("Using yfinance (batch download)...")
                bars_df = fetch_all_universe(symbols)
                update_results = {
                    'source': 'yfinance',
                    'symbols_fetched': bars_df['symbol'].nunique() if not bars_df.empty else 0,
                    'bars_fetched': len(bars_df)
                }
            else:
                # Original Alpha Vantage (rate limited)
                from data.fetch_bars import incremental_update
                logger.info(f"Using Alpha Vantage (batch_size={batch_size})...")
                update_results = incremental_update(symbols, batch_size)
            
            result['steps']['data_update'] = update_results
            logger.info(f"Update results: {update_results}")
        else:
            logger.info("\n[Step 2] Skipping data update")
            update_results = {'skipped': True}
            result['steps']['data_update'] = update_results
        
        # Step 3: Load bars and build features
        logger.info("\n[Step 3] Building features...")
        bars_df = load_existing_bars()
        
        if bars_df.empty:
            logger.error("No bars data found!")
            result['status'] = 'no_data'
            return result
        
        features_df = build_and_save_features(bars_df)
        result['steps']['features'] = {
            'rows': len(features_df),
            'symbols': features_df['symbol'].nunique() if not features_df.empty else 0
        }
        
        if features_df.empty:
            logger.error("No features computed!")
            result['status'] = 'no_features'
            return result
        
        # Step 4: Select asof_date
        logger.info("\n[Step 4] Selecting asof_date...")
        feature_cols = get_z_feature_cols()
        asof_date = select_asof_date(features_df, MIN_COVERAGE_RATE, feature_cols)
        
        if asof_date is None:
            logger.error("Could not determine asof_date!")
            result['status'] = 'no_asof_date'
            return result
        
        result['steps']['asof_date'] = str(asof_date)
        
        # Step 5: Load model
        logger.info(f"\n[Step 5] Loading model ({model_version})...")
        try:
            ranker, regressor, schema = load_model(model_version)
            feature_cols = schema.get('feature_cols', feature_cols)
            result['steps']['model'] = {'version': model_version, 'loaded': True}
        except FileNotFoundError:
            logger.warning(f"Model {model_version} not found. Need to train first.")
            result['status'] = 'model_not_found'
            result['steps']['model'] = {'version': model_version, 'loaded': False}
            
            # Generate quality report anyway
            report = generate_quality_report(
                bars_df, features_df, pd.DataFrame(), update_results, asof_date
            )
            save_quality_report(report)
            
            return result
        
        # Step 6: Generate top-10
        logger.info(f"\n[Step 6] Generating top-10 for {asof_date}...")
        top10 = generate_top10(
            features_df, ranker, regressor, feature_cols, asof_date
        )
        
        if top10.empty:
            logger.warning("No top-10 generated!")
            result['steps']['top10'] = {'count': 0}
        else:
            result['steps']['top10'] = {'count': len(top10)}
            
            # Save outputs
            logger.info("\n[Step 7] Saving outputs...")
            save_top10_latest(top10)
            
            history = load_top10_history()
            save_top10_history(history, top10)
            
            logger.info("\nTop-10 predictions:")
            print(top10[['symbol', 'close', 'pred_ret_5', 'pred_price_5d', 'reason_human']].to_string())
        
        # Step 7: Quality report
        logger.info("\n[Step 8] Generating quality report...")
        report = generate_quality_report(
            bars_df, features_df, top10, update_results, asof_date
        )
        save_quality_report(report)
        
        result['status'] = 'success'
        result['quality_report'] = report
        
        logger.info("\n" + "=" * 50)
        logger.info("Daily update complete!")
        logger.info("=" * 50)
        
        return result
        
    except Exception as e:
        logger.error(f"Daily update failed: {e}", exc_info=True)
        result['status'] = 'error'
        result['error'] = str(e)
        return result


def run_initial_setup(max_symbols: int = None):
    """
    Run initial setup to bootstrap the system
    Fetches data for a subset of symbols (due to API limits)
    """
    logger.info("Running initial setup...")
    
    # 1. Get universe
    from data.fetch_universe import update_universe
    df = update_universe()
    symbols = df['symbol'].tolist()
    
    # 2. Fetch initial data (limited by API)
    logger.info(f"\nNote: Alpha Vantage free tier limits apply.")
    logger.info(f"Will fetch data for {max_symbols or SYMBOLS_PER_RUN} symbols per run.")
    logger.info(f"Run the update multiple times to build up data.")
    
    # 3. Run update
    result = run_daily_update(batch_size=max_symbols or SYMBOLS_PER_RUN)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Top-10 Daily Update")
    parser.add_argument('--setup', action='store_true', help='Run initial setup')
    parser.add_argument('--batch-size', type=int, default=SYMBOLS_PER_RUN,
                        help=f'Number of symbols to update (default: {SYMBOLS_PER_RUN})')
    parser.add_argument('--skip-data', action='store_true', 
                        help='Skip data update (use existing data)')
    parser.add_argument('--model-version', type=str, default=CURRENT_MODEL_VERSION,
                        help=f'Model version to use (default: {CURRENT_MODEL_VERSION})')
    # A1: Add data source option
    parser.add_argument('--use-alpha-vantage', action='store_true',
                        help='Use Alpha Vantage instead of yfinance (default: yfinance)')
    # D5: Add backtest option
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after update')
    # C8: Add walk-forward validation option
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run walk-forward validation')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for walk-forward validation (default: 5)')
    
    args = parser.parse_args()
    
    if args.setup:
        result = run_initial_setup(args.batch_size)
    else:
        result = run_daily_update(
            batch_size=args.batch_size,
            model_version=args.model_version,
            skip_data_update=args.skip_data,
            use_yfinance=not args.use_alpha_vantage  # A1: Default to yfinance
        )
    
    # C8: Walk-forward validation
    if args.walk_forward:
        from models.train import walk_forward_validation
        features = load_features()
        if not features.empty:
            print("\nRunning walk-forward validation...")
            wf_results = walk_forward_validation(features, n_splits=args.n_folds)
            print(f"\nWalk-Forward Results saved.")
    
    # D5: Run backtest
    if args.backtest:
        from models.backtest import run_backtest
        features = load_features()
        bars = load_existing_bars()
        if not features.empty and not bars.empty:
            print("\nRunning backtest...")
            bt_results = run_backtest(features, bars)
            print(f"\nBacktest complete. Results saved to outputs/")
    
    print(f"\nResult: {result['status']}")
