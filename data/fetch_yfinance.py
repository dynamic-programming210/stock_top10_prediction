"""
A1: Fetch OHLCV data from Yahoo Finance using yfinance
No API key required, unlimited free data
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import sys
import time
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BARS_FILE, DATA_DIR, HISTORY_YEARS
from utils import get_logger

logger = get_logger(__name__)


class YFinanceClient:
    """
    Yahoo Finance client using yfinance library
    No rate limits, batch downloads supported
    """
    
    def __init__(self):
        self.failed_symbols: List[str] = []
        
    def fetch_daily_bars(
        self, 
        symbol: str, 
        start_date: str = None,
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a single symbol
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD), defaults to HISTORY_YEARS ago
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with columns [date, symbol, open, high, low, close, volume]
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start = datetime.now() - timedelta(days=HISTORY_YEARS * 365)
            start_date = start.strftime('%Y-%m-%d')
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                self.failed_symbols.append(symbol)
                return None
            
            # Standardize column names
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            
            # Rename 'date' column if needed (sometimes it's index name)
            if 'date' not in df.columns and df.index.name == 'Date':
                df = df.reset_index()
                df.columns = df.columns.str.lower()
            
            # Ensure date column exists
            if 'date' not in df.columns:
                if 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'date'})
                else:
                    # Use index
                    df = df.reset_index()
                    df = df.rename(columns={df.columns[0]: 'date'})
            
            # Select and rename columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['symbol'] = symbol
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['date'] = pd.to_datetime(df['date'])
            
            # Reorder columns
            df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.debug(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            self.failed_symbols.append(symbol)
            return None
    
    def fetch_batch(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data for multiple symbols using yfinance batch download
        Much faster than individual requests
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            progress: Show download progress bar
            
        Returns:
            DataFrame with all symbols' data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start = datetime.now() - timedelta(days=HISTORY_YEARS * 365)
            start_date = start.strftime('%Y-%m-%d')
        
        logger.info(f"Batch downloading {len(symbols)} symbols from {start_date} to {end_date}")
        
        try:
            # Download all at once - much faster
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=progress,
                threads=True
            )
            
            if data.empty:
                logger.warning("No data returned from batch download")
                return pd.DataFrame()
            
            # Handle multi-level columns for multiple symbols
            all_dfs = []
            
            if isinstance(data.columns, pd.MultiIndex):
                # Multiple symbols - columns are (Price, Symbol)
                for symbol in symbols:
                    try:
                        symbol_data = data.xs(symbol, axis=1, level=1)
                        if symbol_data.empty:
                            self.failed_symbols.append(symbol)
                            continue
                        
                        df = symbol_data.reset_index()
                        df.columns = df.columns.str.lower()
                        df['symbol'] = symbol
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        df['date'] = pd.to_datetime(df['date'])
                        df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                        df = df.dropna(subset=['close'])
                        all_dfs.append(df)
                    except Exception as e:
                        logger.debug(f"Error processing {symbol}: {e}")
                        self.failed_symbols.append(symbol)
            else:
                # Single symbol
                df = data.reset_index()
                df.columns = df.columns.str.lower()
                df['symbol'] = symbols[0]
                df['date'] = pd.to_datetime(df['date']).dt.date
                df['date'] = pd.to_datetime(df['date'])
                df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                df = df.dropna(subset=['close'])
                all_dfs.append(df)
            
            if not all_dfs:
                return pd.DataFrame()
            
            result = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Downloaded {len(result)} bars for {result['symbol'].nunique()} symbols")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch download error: {e}")
            # Fall back to individual downloads
            logger.info("Falling back to individual downloads...")
            return self.fetch_individual(symbols, start_date, end_date)
    
    def fetch_individual(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch symbols one by one (slower but more reliable)
        """
        all_dfs = []
        
        for i, symbol in enumerate(symbols):
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(symbols)} symbols")
            
            df = self.fetch_daily_bars(symbol, start_date, end_date)
            if df is not None:
                all_dfs.append(df)
            
            # Small delay to be nice to Yahoo
            time.sleep(0.1)
        
        if not all_dfs:
            return pd.DataFrame()
        
        return pd.concat(all_dfs, ignore_index=True)


def fetch_all_universe(symbols: List[str] = None) -> pd.DataFrame:
    """
    Fetch data for entire universe
    
    Args:
        symbols: List of symbols. If None, loads from universe file.
        
    Returns:
        DataFrame with all OHLCV data
    """
    from data.fetch_universe import load_universe_symbols
    
    if symbols is None:
        symbols = load_universe_symbols()
    
    if not symbols:
        logger.error("No symbols to fetch")
        return pd.DataFrame()
    
    client = YFinanceClient()
    df = client.fetch_batch(symbols)
    
    if not df.empty:
        df.to_parquet(BARS_FILE, index=False)
        logger.info(f"Saved {len(df)} bars to {BARS_FILE}")
    
    if client.failed_symbols:
        logger.warning(f"Failed symbols ({len(client.failed_symbols)}): {client.failed_symbols[:10]}...")
    
    return df


def update_incremental(symbols: List[str] = None, days_back: int = 30) -> pd.DataFrame:
    """
    Incrementally update existing data with recent bars
    
    Args:
        symbols: Symbols to update (defaults to universe)
        days_back: How many days back to fetch for update
        
    Returns:
        Updated DataFrame
    """
    from data.fetch_universe import load_universe_symbols
    
    if symbols is None:
        symbols = load_universe_symbols()
    
    # Load existing data
    existing_df = load_existing_bars()
    
    # Fetch recent data
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    client = YFinanceClient()
    new_df = client.fetch_batch(symbols, start_date=start_date)
    
    if new_df.empty:
        logger.warning("No new data fetched")
        return existing_df
    
    if existing_df.empty:
        result = new_df
    else:
        # Remove overlapping data from existing
        max_new_date = new_df['date'].min()
        existing_df = existing_df[existing_df['date'] < max_new_date]
        
        # Combine
        result = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates
        result = result.drop_duplicates(subset=['date', 'symbol'], keep='last')
    
    # Sort and save
    result = result.sort_values(['symbol', 'date'])
    result.to_parquet(BARS_FILE, index=False)
    logger.info(f"Updated to {len(result)} total bars")
    
    return result


def load_existing_bars() -> pd.DataFrame:
    """Load existing bars from parquet file"""
    if not BARS_FILE.exists():
        logger.info("No existing bars file found")
        return pd.DataFrame()
    
    df = pd.read_parquet(BARS_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df


if __name__ == "__main__":
    # Test yfinance fetch
    print("Testing yfinance data fetch...")
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    client = YFinanceClient()
    df = client.fetch_batch(test_symbols)
    
    print(f"\nFetched {len(df)} bars for {df['symbol'].nunique()} symbols")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSample data:")
    print(df.head(10))
