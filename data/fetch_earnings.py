"""
G3: Earnings Calendar Integration
Fetches upcoming earnings dates for stocks
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from utils import get_logger

logger = get_logger(__name__)

# Cache file
EARNINGS_CACHE_FILE = DATA_DIR / "earnings_calendar.parquet"
CACHE_EXPIRY_HOURS = 12


def fetch_earnings_date(symbol: str) -> Optional[datetime]:
    """
    Fetch next earnings date for a symbol
    
    Returns datetime of next earnings or None if not available
    """
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        
        if calendar is None or calendar.empty:
            return None
        
        # calendar is a DataFrame with earnings date info
        if 'Earnings Date' in calendar.index:
            earnings_date = calendar.loc['Earnings Date']
            if isinstance(earnings_date, pd.Series):
                earnings_date = earnings_date.iloc[0]
            if pd.notna(earnings_date):
                if isinstance(earnings_date, str):
                    return pd.to_datetime(earnings_date)
                return earnings_date
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to fetch earnings for {symbol}: {e}")
        return None


def fetch_earnings_batch(
    symbols: List[str],
    delay: float = 0.05,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Fetch earnings dates for multiple symbols
    
    Returns DataFrame with symbol and next_earnings_date
    """
    results = []
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        if show_progress and (i + 1) % 50 == 0:
            logger.info(f"Fetching earnings: {i + 1}/{total}")
        
        earnings_date = fetch_earnings_date(symbol)
        
        results.append({
            'symbol': symbol,
            'next_earnings_date': earnings_date,
            'fetch_time': datetime.now()
        })
        
        if delay > 0:
            time.sleep(delay)
    
    df = pd.DataFrame(results)
    return df


def load_earnings_cache() -> pd.DataFrame:
    """Load cached earnings data if fresh"""
    if EARNINGS_CACHE_FILE.exists():
        df = pd.read_parquet(EARNINGS_CACHE_FILE)
        
        # Check if cache is fresh
        if 'fetch_time' in df.columns and not df.empty:
            latest_fetch = pd.to_datetime(df['fetch_time']).max()
            age_hours = (datetime.now() - latest_fetch).total_seconds() / 3600
            
            if age_hours < CACHE_EXPIRY_HOURS:
                return df
    
    return pd.DataFrame()


def save_earnings_cache(df: pd.DataFrame):
    """Save earnings data to cache"""
    df.to_parquet(EARNINGS_CACHE_FILE, index=False)
    logger.info(f"Saved earnings cache for {len(df)} symbols")


def get_earnings_for_symbols(symbols: List[str], force_refresh: bool = False) -> pd.DataFrame:
    """
    Get earnings dates for symbols, using cache when possible
    """
    cached = load_earnings_cache()
    
    if not force_refresh and not cached.empty:
        # Use cached data for symbols we have
        cached_symbols = set(cached['symbol'].tolist())
        missing = [s for s in symbols if s not in cached_symbols]
        
        if not missing:
            return cached[cached['symbol'].isin(symbols)]
        
        # Fetch only missing symbols
        if missing:
            new_data = fetch_earnings_batch(missing)
            df = pd.concat([cached, new_data], ignore_index=True)
            save_earnings_cache(df)
            return df[df['symbol'].isin(symbols)]
    
    # Fetch all
    df = fetch_earnings_batch(symbols)
    save_earnings_cache(df)
    return df


def get_upcoming_earnings(symbols: List[str], days_ahead: int = 14) -> pd.DataFrame:
    """
    Get symbols with earnings in the next N days
    
    Returns DataFrame with symbol, next_earnings_date, days_until_earnings
    """
    df = get_earnings_for_symbols(symbols)
    
    if df.empty:
        return pd.DataFrame()
    
    today = datetime.now().date()
    cutoff = today + timedelta(days=days_ahead)
    
    # Filter to upcoming earnings
    df['next_earnings_date'] = pd.to_datetime(df['next_earnings_date'])
    df = df.dropna(subset=['next_earnings_date'])
    
    df['earnings_date'] = df['next_earnings_date'].dt.date
    df = df[(df['earnings_date'] >= today) & (df['earnings_date'] <= cutoff)]
    
    # Calculate days until earnings
    df['days_until_earnings'] = df['next_earnings_date'].apply(
        lambda x: (x.date() - today).days if pd.notna(x) else None
    )
    
    df = df.sort_values('next_earnings_date')
    
    return df[['symbol', 'next_earnings_date', 'days_until_earnings']]


def flag_earnings_risk(symbols: List[str], days_threshold: int = 7) -> Dict[str, int]:
    """
    Return dict of symbol -> days_until_earnings for stocks with imminent earnings
    
    Useful for flagging risk in predictions
    """
    upcoming = get_upcoming_earnings(symbols, days_ahead=days_threshold)
    
    if upcoming.empty:
        return {}
    
    return dict(zip(upcoming['symbol'], upcoming['days_until_earnings']))


# CLI
if __name__ == "__main__":
    import argparse
    from data.fetch_universe import load_universe_symbols
    
    parser = argparse.ArgumentParser(description="Fetch earnings calendar")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=14, help='Days ahead to check')
    parser.add_argument('--refresh', action='store_true', help='Force refresh cache')
    
    args = parser.parse_args()
    
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = load_universe_symbols()[:50]  # Sample for testing
    
    print(f"\n📅 Fetching earnings calendar for {len(symbols)} symbols...")
    
    upcoming = get_upcoming_earnings(symbols, days_ahead=args.days)
    
    if upcoming.empty:
        print(f"No earnings in the next {args.days} days")
    else:
        print(f"\n🎯 Stocks with earnings in next {args.days} days:\n")
        for _, row in upcoming.iterrows():
            days = int(row['days_until_earnings'])
            emoji = "🔴" if days <= 3 else "🟡" if days <= 7 else "🟢"
            print(f"  {emoji} {row['symbol']}: {row['next_earnings_date'].strftime('%Y-%m-%d')} ({days} days)")
