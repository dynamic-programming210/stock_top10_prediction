"""
A2: Fundamental Data Fetcher
Fetches fundamental financial data using yfinance for enhanced predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yfinance as yf
except ImportError:
    yf = None

from config import DATA_DIR
from utils import get_logger

logger = get_logger(__name__)

# Output files
FUNDAMENTALS_FILE = DATA_DIR / "fundamentals.parquet"


def fetch_fundamental_data(symbols: List[str], progress: bool = True) -> pd.DataFrame:
    """
    Fetch fundamental data for a list of symbols using yfinance
    
    Fundamental metrics include:
    - P/E ratio (trailing and forward)
    - P/B ratio (price to book)
    - P/S ratio (price to sales)
    - EPS (earnings per share)
    - Dividend yield
    - Market cap
    - Beta
    - ROE, ROA
    - Debt to equity
    - Revenue growth
    - Profit margins
    """
    if yf is None:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()
    
    logger.info(f"Fetching fundamental data for {len(symbols)} symbols...")
    
    data = []
    failed = []
    
    for i, symbol in enumerate(symbols):
        if progress and (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(symbols)} symbols")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                failed.append(symbol)
                continue
            
            row = {
                'symbol': symbol,
                'fetch_date': datetime.now().date(),
                
                # Valuation ratios
                'pe_trailing': info.get('trailingPE'),
                'pe_forward': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                
                # Earnings & Dividends
                'eps_trailing': info.get('trailingEps'),
                'eps_forward': info.get('forwardEps'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                
                # Size & Risk
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'beta': info.get('beta'),
                
                # Profitability
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                
                # Financial Health
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                
                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                
                # Analyst Estimates
                'target_mean_price': info.get('targetMeanPrice'),
                'target_high_price': info.get('targetHighPrice'),
                'target_low_price': info.get('targetLowPrice'),
                'recommendation_mean': info.get('recommendationMean'),
                'n_analyst_opinions': info.get('numberOfAnalystOpinions'),
                
                # Sector info (backup)
                'sector': info.get('sector'),
                'industry': info.get('industry'),
            }
            
            data.append(row)
            
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            failed.append(symbol)
    
    if failed:
        logger.warning(f"Failed to fetch fundamentals for {len(failed)} symbols")
    
    if not data:
        logger.error("No fundamental data fetched")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    logger.info(f"Fetched fundamentals for {len(df)} symbols")
    
    return df


def save_fundamentals(df: pd.DataFrame):
    """Save fundamentals to parquet file"""
    df.to_parquet(FUNDAMENTALS_FILE, index=False)
    logger.info(f"Saved fundamentals to {FUNDAMENTALS_FILE}")


def load_fundamentals() -> pd.DataFrame:
    """Load fundamentals from parquet file"""
    if not FUNDAMENTALS_FILE.exists():
        logger.warning(f"Fundamentals file not found: {FUNDAMENTALS_FILE}")
        return pd.DataFrame()
    
    df = pd.read_parquet(FUNDAMENTALS_FILE)
    return df


def compute_fundamental_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived fundamental features for ML models
    """
    df = fund_df.copy()
    
    # Value scores (lower P/E, P/B = more value)
    if 'pe_trailing' in df.columns:
        df['value_pe'] = 1 / df['pe_trailing'].clip(lower=1)
    if 'pb_ratio' in df.columns:
        df['value_pb'] = 1 / df['pb_ratio'].clip(lower=0.1)
    if 'ps_ratio' in df.columns:
        df['value_ps'] = 1 / df['ps_ratio'].clip(lower=0.1)
    
    # Growth composite
    growth_cols = ['revenue_growth', 'earnings_growth']
    available_growth = [c for c in growth_cols if c in df.columns]
    if available_growth:
        df['growth_composite'] = df[available_growth].mean(axis=1)
    
    # Quality composite (ROE + ROA + margins)
    quality_cols = ['roe', 'roa', 'profit_margin']
    available_quality = [c for c in quality_cols if c in df.columns]
    if available_quality:
        df['quality_composite'] = df[available_quality].mean(axis=1)
    
    # Financial health score
    if 'current_ratio' in df.columns and 'debt_to_equity' in df.columns:
        df['financial_health'] = (
            df['current_ratio'].clip(upper=5) / 5 - 
            df['debt_to_equity'].clip(upper=300) / 300
        )
    
    # Analyst sentiment (lower recommendation = buy)
    if 'recommendation_mean' in df.columns:
        df['analyst_sentiment'] = 5 - df['recommendation_mean'].clip(1, 5)
    
    # Log market cap (for size factor)
    if 'market_cap' in df.columns:
        df['log_market_cap'] = np.log10(df['market_cap'].clip(lower=1e6))
    
    return df


def merge_fundamentals_with_features(
    features_df: pd.DataFrame,
    fund_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge fundamental data with daily feature data"""
    if fund_df.empty:
        return features_df
    
    fund_features = compute_fundamental_features(fund_df)
    
    fund_cols = [
        'symbol', 'pe_trailing', 'pe_forward', 'peg_ratio', 'pb_ratio', 'ps_ratio',
        'dividend_yield', 'beta', 'profit_margin', 'roe', 'roa',
        'debt_to_equity', 'revenue_growth', 'earnings_growth',
        'recommendation_mean', 'log_market_cap',
        'value_pe', 'value_pb', 'value_ps',
        'growth_composite', 'quality_composite', 'financial_health',
        'analyst_sentiment'
    ]
    
    available_cols = ['symbol'] + [c for c in fund_cols[1:] if c in fund_features.columns]
    fund_to_merge = fund_features[available_cols].copy()
    
    merged = features_df.merge(fund_to_merge, on='symbol', how='left')
    logger.info(f"Merged fundamentals: {len(available_cols) - 1} fundamental features added")
    
    return merged


def update_fundamentals(symbols: List[str] = None) -> pd.DataFrame:
    """Update fundamental data for the universe"""
    if symbols is None:
        from data.fetch_universe import load_universe_symbols
        symbols = load_universe_symbols()
    
    df = fetch_fundamental_data(symbols)
    
    if not df.empty:
        save_fundamentals(df)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Fundamental Data")
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to fetch')
    parser.add_argument('--sample', type=int, default=0, help='Fetch sample N symbols')
    
    args = parser.parse_args()
    
    if args.symbols:
        symbols = args.symbols
    elif args.sample > 0:
        from data.fetch_universe import load_universe_symbols
        all_symbols = load_universe_symbols()
        symbols = all_symbols[:args.sample]
    else:
        from data.fetch_universe import load_universe_symbols
        symbols = load_universe_symbols()
    
    print(f"Fetching fundamentals for {len(symbols)} symbols...")
    df = fetch_fundamental_data(symbols)
    
    if not df.empty:
        save_fundamentals(df)
        print(f"\nFundamentals saved: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
