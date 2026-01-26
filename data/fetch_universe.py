"""
Fetch and maintain S&P 500 universe list
Uses Wikipedia as the source for current S&P 500 constituents
"""
import pandas as pd
import requests
import io
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import UNIVERSE_SYMBOLS_FILE, UNIVERSE_META_FILE, DATA_DIR
from utils import get_logger

logger = get_logger(__name__)

# Fallback S&P 500 list (top ~100 by market cap + diversified selection)
FALLBACK_SP500 = [
    # Technology
    ('AAPL', 'Apple Inc.', 'Information Technology'),
    ('MSFT', 'Microsoft Corporation', 'Information Technology'),
    ('GOOGL', 'Alphabet Inc. Class A', 'Communication Services'),
    ('GOOG', 'Alphabet Inc. Class C', 'Communication Services'),
    ('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary'),
    ('NVDA', 'NVIDIA Corporation', 'Information Technology'),
    ('META', 'Meta Platforms Inc.', 'Communication Services'),
    ('TSLA', 'Tesla Inc.', 'Consumer Discretionary'),
    ('AVGO', 'Broadcom Inc.', 'Information Technology'),
    ('ORCL', 'Oracle Corporation', 'Information Technology'),
    ('CRM', 'Salesforce Inc.', 'Information Technology'),
    ('AMD', 'Advanced Micro Devices Inc.', 'Information Technology'),
    ('ADBE', 'Adobe Inc.', 'Information Technology'),
    ('CSCO', 'Cisco Systems Inc.', 'Information Technology'),
    ('INTC', 'Intel Corporation', 'Information Technology'),
    ('IBM', 'International Business Machines', 'Information Technology'),
    ('QCOM', 'Qualcomm Inc.', 'Information Technology'),
    ('TXN', 'Texas Instruments Inc.', 'Information Technology'),
    ('INTU', 'Intuit Inc.', 'Information Technology'),
    ('NOW', 'ServiceNow Inc.', 'Information Technology'),
    ('AMAT', 'Applied Materials Inc.', 'Information Technology'),
    ('MU', 'Micron Technology Inc.', 'Information Technology'),
    ('LRCX', 'Lam Research Corporation', 'Information Technology'),
    ('ADI', 'Analog Devices Inc.', 'Information Technology'),
    ('KLAC', 'KLA Corporation', 'Information Technology'),
    # Financials
    ('BRK-B', 'Berkshire Hathaway Inc. Class B', 'Financials'),
    ('JPM', 'JPMorgan Chase & Co.', 'Financials'),
    ('V', 'Visa Inc.', 'Financials'),
    ('MA', 'Mastercard Inc.', 'Financials'),
    ('BAC', 'Bank of America Corp.', 'Financials'),
    ('WFC', 'Wells Fargo & Company', 'Financials'),
    ('GS', 'Goldman Sachs Group Inc.', 'Financials'),
    ('MS', 'Morgan Stanley', 'Financials'),
    ('C', 'Citigroup Inc.', 'Financials'),
    ('AXP', 'American Express Company', 'Financials'),
    ('BLK', 'BlackRock Inc.', 'Financials'),
    ('SCHW', 'Charles Schwab Corporation', 'Financials'),
    ('SPGI', 'S&P Global Inc.', 'Financials'),
    ('CME', 'CME Group Inc.', 'Financials'),
    ('PGR', 'Progressive Corporation', 'Financials'),
    # Healthcare
    ('UNH', 'UnitedHealth Group Inc.', 'Health Care'),
    ('JNJ', 'Johnson & Johnson', 'Health Care'),
    ('LLY', 'Eli Lilly and Company', 'Health Care'),
    ('PFE', 'Pfizer Inc.', 'Health Care'),
    ('ABBV', 'AbbVie Inc.', 'Health Care'),
    ('MRK', 'Merck & Co. Inc.', 'Health Care'),
    ('TMO', 'Thermo Fisher Scientific Inc.', 'Health Care'),
    ('ABT', 'Abbott Laboratories', 'Health Care'),
    ('DHR', 'Danaher Corporation', 'Health Care'),
    ('BMY', 'Bristol-Myers Squibb Company', 'Health Care'),
    ('AMGN', 'Amgen Inc.', 'Health Care'),
    ('GILD', 'Gilead Sciences Inc.', 'Health Care'),
    ('ISRG', 'Intuitive Surgical Inc.', 'Health Care'),
    ('VRTX', 'Vertex Pharmaceuticals Inc.', 'Health Care'),
    ('REGN', 'Regeneron Pharmaceuticals Inc.', 'Health Care'),
    # Consumer
    ('WMT', 'Walmart Inc.', 'Consumer Staples'),
    ('PG', 'Procter & Gamble Company', 'Consumer Staples'),
    ('KO', 'Coca-Cola Company', 'Consumer Staples'),
    ('PEP', 'PepsiCo Inc.', 'Consumer Staples'),
    ('COST', 'Costco Wholesale Corporation', 'Consumer Staples'),
    ('HD', 'Home Depot Inc.', 'Consumer Discretionary'),
    ('MCD', 'McDonald\'s Corporation', 'Consumer Discretionary'),
    ('NKE', 'Nike Inc.', 'Consumer Discretionary'),
    ('LOW', 'Lowe\'s Companies Inc.', 'Consumer Discretionary'),
    ('SBUX', 'Starbucks Corporation', 'Consumer Discretionary'),
    ('TGT', 'Target Corporation', 'Consumer Discretionary'),
    ('TJX', 'TJX Companies Inc.', 'Consumer Discretionary'),
    # Industrials
    ('CAT', 'Caterpillar Inc.', 'Industrials'),
    ('GE', 'General Electric Company', 'Industrials'),
    ('HON', 'Honeywell International Inc.', 'Industrials'),
    ('UPS', 'United Parcel Service Inc.', 'Industrials'),
    ('RTX', 'RTX Corporation', 'Industrials'),
    ('BA', 'Boeing Company', 'Industrials'),
    ('DE', 'Deere & Company', 'Industrials'),
    ('LMT', 'Lockheed Martin Corporation', 'Industrials'),
    ('UNP', 'Union Pacific Corporation', 'Industrials'),
    ('MMM', 'Minnesota Mining and Manufacturing', 'Industrials'),
    # Energy
    ('XOM', 'Exxon Mobil Corporation', 'Energy'),
    ('CVX', 'Chevron Corporation', 'Energy'),
    ('COP', 'ConocoPhillips', 'Energy'),
    ('SLB', 'Schlumberger Limited', 'Energy'),
    ('EOG', 'EOG Resources Inc.', 'Energy'),
    # Utilities & Real Estate
    ('NEE', 'NextEra Energy Inc.', 'Utilities'),
    ('DUK', 'Duke Energy Corporation', 'Utilities'),
    ('SO', 'Southern Company', 'Utilities'),
    ('AMT', 'American Tower Corporation', 'Real Estate'),
    ('PLD', 'Prologis Inc.', 'Real Estate'),
    ('CCI', 'Crown Castle Inc.', 'Real Estate'),
    # Communication Services
    ('NFLX', 'Netflix Inc.', 'Communication Services'),
    ('DIS', 'Walt Disney Company', 'Communication Services'),
    ('CMCSA', 'Comcast Corporation', 'Communication Services'),
    ('VZ', 'Verizon Communications Inc.', 'Communication Services'),
    ('T', 'AT&T Inc.', 'Communication Services'),
    # Materials
    ('LIN', 'Linde plc', 'Materials'),
    ('APD', 'Air Products and Chemicals Inc.', 'Materials'),
    ('SHW', 'Sherwin-Williams Company', 'Materials'),
]


def fetch_sp500_from_wikipedia() -> pd.DataFrame:
    """
    Fetch S&P 500 constituents from Wikipedia
    Returns DataFrame with symbol, name, sector columns
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    logger.info(f"Fetching S&P 500 list from Wikipedia...")
    
    try:
        # Use requests with proper headers to avoid 403
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        
        # Standardize column names
        df = df.rename(columns={
            'Symbol': 'symbol',
            'Security': 'name',
            'GICS Sector': 'sector',
            'GICS Sub-Industry': 'sub_industry',
            'Headquarters Location': 'headquarters',
            'Date added': 'date_added',
            'CIK': 'cik',
            'Founded': 'founded'
        })
        
        # Clean symbol column (some have dots that need to be replaced with dashes for Alpha Vantage)
        df['symbol'] = df['symbol'].str.replace('.', '-', regex=False)
        
        # Select relevant columns
        cols = ['symbol', 'name', 'sector']
        if 'sub_industry' in df.columns:
            cols.append('sub_industry')
            
        df = df[cols].copy()
        
        logger.info(f"Found {len(df)} S&P 500 constituents")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to fetch from Wikipedia: {e}")
        logger.info("Using fallback S&P 500 list...")
        return get_fallback_sp500()


def get_fallback_sp500() -> pd.DataFrame:
    """Return fallback list of major S&P 500 stocks"""
    df = pd.DataFrame(FALLBACK_SP500, columns=['symbol', 'name', 'sector'])
    logger.info(f"Using fallback list with {len(df)} symbols")
    return df

def save_universe(df: pd.DataFrame) -> None:
    """
    Save universe to both text file and parquet
    """
    DATA_DIR.mkdir(exist_ok=True)
    
    # Save symbols to text file (one per line)
    symbols = df['symbol'].tolist()
    with open(UNIVERSE_SYMBOLS_FILE, 'w') as f:
        f.write('\n'.join(symbols))
    logger.info(f"Saved {len(symbols)} symbols to {UNIVERSE_SYMBOLS_FILE}")
    
    # Save metadata to parquet
    df.to_parquet(UNIVERSE_META_FILE, index=False)
    logger.info(f"Saved universe metadata to {UNIVERSE_META_FILE}")

def load_universe_symbols() -> list:
    """
    Load universe symbols from text file
    """
    if not UNIVERSE_SYMBOLS_FILE.exists():
        logger.warning("Universe file not found, fetching fresh data...")
        df = fetch_sp500_from_wikipedia()
        save_universe(df)
        
    with open(UNIVERSE_SYMBOLS_FILE, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    return symbols

def load_universe_meta() -> pd.DataFrame:
    """
    Load universe metadata from parquet
    """
    if not UNIVERSE_META_FILE.exists():
        logger.warning("Universe metadata not found, fetching fresh data...")
        df = fetch_sp500_from_wikipedia()
        save_universe(df)
        return df
        
    return pd.read_parquet(UNIVERSE_META_FILE)

def update_universe() -> pd.DataFrame:
    """
    Update universe list (call periodically to catch index changes)
    """
    df = fetch_sp500_from_wikipedia()
    save_universe(df)
    return df

if __name__ == "__main__":
    # Run this script to initialize/update the universe
    df = update_universe()
    print(f"\nS&P 500 Universe:")
    print(f"Total symbols: {len(df)}")
    print(f"\nSector breakdown:")
    print(df['sector'].value_counts())
    print(f"\nFirst 10 symbols:")
    print(df.head(10))
