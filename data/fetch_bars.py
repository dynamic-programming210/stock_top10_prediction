"""
Fetch OHLCV data from Alpha Vantage with robust rate limiting
"""
import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ALPHA_VANTAGE_API_KEY, AV_REQUESTS_PER_MINUTE, 
    BARS_FILE, UPDATE_QUEUE_FILE, DATA_DIR, SYMBOLS_PER_RUN
)
from utils import get_logger, retry_with_backoff

logger = get_logger(__name__)

class AlphaVantageClient:
    """
    Alpha Vantage API client with rate limiting and throttle detection
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        self.request_times: List[float] = []
        self.daily_request_count = 0
        self.throttled = False
        
    def _wait_for_rate_limit(self):
        """
        Ensure we don't exceed rate limits (5 requests/minute for free tier)
        Add minimum 12-second delay between requests for free tier safety
        """
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= AV_REQUESTS_PER_MINUTE:
            # Wait until the oldest request is more than 60 seconds old
            sleep_time = 60 - (now - self.request_times[0]) + 1
            if sleep_time > 0:
                logger.info(f"Rate limit: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        elif self.request_times:
            # For free tier, wait at least 12 seconds between requests
            time_since_last = now - self.request_times[-1]
            if time_since_last < 12:
                sleep_time = 12 - time_since_last + 0.5
                logger.info(f"Spacing requests: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        self.request_times.append(time.time())
        self.daily_request_count += 1
    
    def _check_throttle_response(self, data: dict) -> bool:
        """
        Check if response indicates throttling
        Alpha Vantage returns these messages when rate limited
        """
        if "Note" in data:
            logger.warning(f"API Note: {data['Note']}")
            return True
        if "Information" in data:
            logger.warning(f"API Information: {data['Information']}")
            if "rate limit" in data["Information"].lower() or "premium" in data["Information"].lower():
                return True
        return False
    
    def fetch_daily_bars(self, symbol: str, outputsize: str = "full") -> Tuple[Optional[pd.DataFrame], str]:
        """
        Fetch daily OHLCV data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            outputsize: "compact" (100 days) or "full" (20+ years)
            
        Returns:
            Tuple of (DataFrame or None, status message)
        """
        if self.throttled:
            return None, "throttled"
            
        self._wait_for_rate_limit()
        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": "json"
        }
        
        try:
            logger.info(f"Fetching {symbol}...")
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for throttling
            if self._check_throttle_response(data):
                self.throttled = True
                return None, "throttled"
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"API error for {symbol}: {data['Error Message']}")
                return None, f"error: {data['Error Message']}"
            
            # Parse time series data
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                logger.warning(f"No data for {symbol}")
                return None, "no_data"
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype({
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            })
            
            df['symbol'] = symbol
            df.index.name = 'date'
            df = df.reset_index()
            
            logger.info(f"Got {len(df)} bars for {symbol}")
            return df, "success"
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for {symbol}")
            return None, "timeout"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {e}")
            return None, f"request_error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            return None, f"error: {str(e)}"


class UpdateQueue:
    """
    Manages the rotation queue for incremental updates
    """
    
    def __init__(self, queue_file: Path = UPDATE_QUEUE_FILE):
        self.queue_file = queue_file
        self.state = self._load_state()
        
    def _load_state(self) -> dict:
        """Load queue state from file"""
        if self.queue_file.exists():
            with open(self.queue_file, 'r') as f:
                return json.load(f)
        return {
            "queue": [],
            "last_updated": None,
            "stats": {
                "success_count": 0,
                "fail_count": 0,
                "last_run_results": {}
            }
        }
    
    def _save_state(self):
        """Save queue state to file"""
        self.queue_file.parent.mkdir(exist_ok=True)
        with open(self.queue_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
            
    def initialize(self, symbols: List[str]):
        """Initialize queue with all symbols"""
        self.state["queue"] = symbols.copy()
        self._save_state()
        logger.info(f"Initialized queue with {len(symbols)} symbols")
        
    def get_next_batch(self, batch_size: int = SYMBOLS_PER_RUN) -> List[str]:
        """Get next batch of symbols to update"""
        if not self.state["queue"]:
            return []
        
        batch = self.state["queue"][:batch_size]
        return batch
    
    def mark_completed(self, symbol: str, status: str):
        """Mark a symbol as completed and move to end of queue"""
        if symbol in self.state["queue"]:
            self.state["queue"].remove(symbol)
            self.state["queue"].append(symbol)  # Move to end
            
        # Update stats
        if status == "success":
            self.state["stats"]["success_count"] += 1
        else:
            self.state["stats"]["fail_count"] += 1
            
        self.state["stats"]["last_run_results"][symbol] = status
        self.state["last_updated"] = datetime.now().isoformat()
        self._save_state()
        
    def get_stats(self) -> dict:
        """Get queue statistics"""
        return self.state["stats"]


def load_existing_bars() -> pd.DataFrame:
    """Load existing bars from parquet file"""
    if BARS_FILE.exists():
        df = pd.read_parquet(BARS_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

def save_bars(df: pd.DataFrame):
    """Save bars to parquet file"""
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(BARS_FILE, index=False)
    logger.info(f"Saved {len(df)} bars to {BARS_FILE}")

def merge_bars(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new bars with existing, keeping latest values for duplicates
    """
    if existing.empty:
        return new
    if new.empty:
        return existing
        
    combined = pd.concat([existing, new], ignore_index=True)
    # Drop duplicates, keeping last (new data takes precedence)
    combined = combined.drop_duplicates(subset=['date', 'symbol'], keep='last')
    combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)
    return combined

def fetch_initial_data(symbols: List[str], max_symbols: int = None) -> pd.DataFrame:
    """
    Fetch initial historical data for multiple symbols
    Used for first-time setup
    """
    client = AlphaVantageClient()
    all_bars = []
    
    symbols_to_fetch = symbols[:max_symbols] if max_symbols else symbols
    
    for i, symbol in enumerate(symbols_to_fetch):
        logger.info(f"Progress: {i+1}/{len(symbols_to_fetch)}")
        
        df, status = client.fetch_daily_bars(symbol, outputsize="full")
        
        if df is not None:
            all_bars.append(df)
        
        if client.throttled:
            logger.warning("API throttled, stopping early")
            break
    
    if all_bars:
        result = pd.concat(all_bars, ignore_index=True)
        logger.info(f"Fetched {len(result)} total bars for {len(all_bars)} symbols")
        return result
    
    return pd.DataFrame()

def incremental_update(symbols: List[str], batch_size: int = SYMBOLS_PER_RUN) -> Dict:
    """
    Perform incremental update using rotation queue
    Returns statistics about the update
    """
    queue = UpdateQueue()
    
    # Initialize queue if empty
    if not queue.state["queue"]:
        queue.initialize(symbols)
    
    # Get next batch
    batch = queue.get_next_batch(batch_size)
    if not batch:
        return {"status": "empty_queue", "updated": 0}
    
    client = AlphaVantageClient()
    existing_bars = load_existing_bars()
    new_bars = []
    results = {"success": 0, "failed": 0, "throttled": False}
    
    for symbol in batch:
        df, status = client.fetch_daily_bars(symbol, outputsize="compact")
        
        if status == "throttled":
            results["throttled"] = True
            logger.warning("Throttled, ending update early")
            break
            
        if df is not None:
            new_bars.append(df)
            results["success"] += 1
        else:
            results["failed"] += 1
            
        queue.mark_completed(symbol, status)
    
    # Merge and save
    if new_bars:
        new_df = pd.concat(new_bars, ignore_index=True)
        merged = merge_bars(existing_bars, new_df)
        save_bars(merged)
        results["total_bars"] = len(merged)
        results["unique_symbols"] = merged['symbol'].nunique()
    
    return results


if __name__ == "__main__":
    # Test fetching a single symbol
    from fetch_universe import load_universe_symbols
    
    symbols = load_universe_symbols()
    print(f"Loaded {len(symbols)} symbols")
    
    # Test incremental update
    results = incremental_update(symbols, batch_size=5)
    print(f"\nUpdate results: {results}")
