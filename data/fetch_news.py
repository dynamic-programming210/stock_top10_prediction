"""
Task 3: News Sentiment Analysis
Fetches news from Yahoo Finance and computes sentiment features
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from config import DATA_DIR
from utils import get_logger

logger = get_logger(__name__)

# Output file
NEWS_SENTIMENT_FILE = DATA_DIR / "news_sentiment.parquet"
NEWS_CACHE_FILE = DATA_DIR / "news_cache.parquet"

# Settings
MAX_NEWS_AGE_DAYS = 7  # Only consider news from last N days
MIN_NEWS_FOR_SENTIMENT = 1  # Minimum articles needed


def ensure_vader_downloaded():
    """Download VADER lexicon if not present"""
    if not VADER_AVAILABLE:
        logger.error("NLTK not installed. Run: pip install nltk")
        return False
    
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logger.info("Downloading VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)
    
    return True


def get_sentiment_analyzer():
    """Get VADER sentiment analyzer"""
    if not ensure_vader_downloaded():
        return None
    return SentimentIntensityAnalyzer()


def fetch_news_for_symbol(symbol: str, max_articles: int = 10) -> List[Dict]:
    """
    Fetch recent news for a symbol from Yahoo Finance
    
    Returns list of dicts with: title, publisher, link, publish_time, sentiment
    """
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return []
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=MAX_NEWS_AGE_DAYS)
        
        for item in news[:max_articles]:
            # Handle new yfinance API structure (nested under 'content')
            content = item.get('content', item)  # Fall back to item itself if no 'content' key
            
            # Parse publish time
            pub_time = None
            pub_date_str = content.get('pubDate') or item.get('providerPublishTime')
            if pub_date_str:
                if isinstance(pub_date_str, str):
                    # ISO format: '2026-01-28T23:43:09Z'
                    try:
                        pub_time = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        pub_time = pub_time.replace(tzinfo=None)  # Remove timezone for comparison
                    except:
                        pass
                elif isinstance(pub_date_str, (int, float)):
                    # Unix timestamp
                    pub_time = datetime.fromtimestamp(pub_date_str)
            
            # Skip old articles
            if pub_time and pub_time < cutoff_date:
                continue
            
            # Extract title
            title = content.get('title', '') or item.get('title', '')
            
            # Extract publisher
            provider = content.get('provider', {})
            publisher = provider.get('displayName', '') if isinstance(provider, dict) else ''
            
            # Extract link
            canonical = content.get('canonicalUrl', {})
            link = canonical.get('url', '') if isinstance(canonical, dict) else item.get('link', '')
            
            article = {
                'symbol': symbol,
                'title': title,
                'publisher': publisher,
                'link': link,
                'publish_time': pub_time,
                'type': content.get('contentType', 'STORY')
            }
            articles.append(article)
        
        return articles
        
    except Exception as e:
        logger.debug(f"Failed to fetch news for {symbol}: {e}")
        return []


def analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> Dict[str, float]:
    """
    Analyze sentiment of text using VADER
    
    Returns dict with: compound, pos, neg, neu scores
    """
    if not text or not analyzer:
        return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
    
    scores = analyzer.polarity_scores(text)
    return scores


def compute_sentiment_features(articles: List[Dict], analyzer: SentimentIntensityAnalyzer) -> Dict[str, float]:
    """
    Compute aggregated sentiment features from list of articles
    
    Returns dict with sentiment features
    """
    if not articles or len(articles) < MIN_NEWS_FOR_SENTIMENT:
        return {
            'news_sentiment_avg': 0.0,
            'news_sentiment_std': 0.0,
            'news_count': 0,
            'news_positive_ratio': 0.0,
            'news_negative_ratio': 0.0,
            'news_neutral_ratio': 1.0
        }
    
    sentiments = []
    for article in articles:
        title = article.get('title', '')
        if title:
            scores = analyze_sentiment(title, analyzer)
            sentiments.append(scores['compound'])
    
    if not sentiments:
        return {
            'news_sentiment_avg': 0.0,
            'news_sentiment_std': 0.0,
            'news_count': 0,
            'news_positive_ratio': 0.0,
            'news_negative_ratio': 0.0,
            'news_neutral_ratio': 1.0
        }
    
    sentiments = np.array(sentiments)
    
    # Classify sentiments
    positive = (sentiments > 0.05).sum()
    negative = (sentiments < -0.05).sum()
    neutral = len(sentiments) - positive - negative
    
    features = {
        'news_sentiment_avg': float(np.mean(sentiments)),
        'news_sentiment_std': float(np.std(sentiments)) if len(sentiments) > 1 else 0.0,
        'news_count': len(sentiments),
        'news_positive_ratio': positive / len(sentiments),
        'news_negative_ratio': negative / len(sentiments),
        'news_neutral_ratio': neutral / len(sentiments)
    }
    
    return features


def fetch_news_sentiment_batch(
    symbols: List[str],
    max_articles_per_symbol: int = 10,
    delay_between_requests: float = 0.1,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Fetch news and compute sentiment for multiple symbols
    
    Returns DataFrame with sentiment features per symbol
    """
    analyzer = get_sentiment_analyzer()
    if analyzer is None:
        logger.error("Sentiment analyzer not available")
        return pd.DataFrame()
    
    results = []
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        if show_progress and (i + 1) % 50 == 0:
            logger.info(f"Processing news: {i + 1}/{total} symbols")
        
        # Fetch news
        articles = fetch_news_for_symbol(symbol, max_articles_per_symbol)
        
        # Compute sentiment features
        features = compute_sentiment_features(articles, analyzer)
        features['symbol'] = symbol
        features['fetch_date'] = datetime.now().date()
        
        results.append(features)
        
        # Small delay to avoid rate limiting
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['symbol', 'fetch_date', 'news_count', 'news_sentiment_avg', 
            'news_sentiment_std', 'news_positive_ratio', 'news_negative_ratio', 
            'news_neutral_ratio']
    df = df[[c for c in cols if c in df.columns]]
    
    return df


def save_news_sentiment(df: pd.DataFrame, path: Path = NEWS_SENTIMENT_FILE):
    """Save news sentiment data"""
    df.to_parquet(path, index=False)
    logger.info(f"Saved news sentiment for {len(df)} symbols to {path}")


def load_news_sentiment(path: Path = NEWS_SENTIMENT_FILE) -> pd.DataFrame:
    """Load news sentiment data"""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def update_news_sentiment(
    symbols: List[str],
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Update news sentiment data
    
    - Loads existing data
    - Refreshes symbols older than 1 day or missing
    - Returns merged DataFrame
    """
    existing = load_news_sentiment()
    today = datetime.now().date()
    
    if not force_refresh and not existing.empty:
        # Find symbols that need refresh
        existing['fetch_date'] = pd.to_datetime(existing['fetch_date']).dt.date
        fresh = existing[existing['fetch_date'] == today]
        fresh_symbols = set(fresh['symbol'].tolist())
        symbols_to_fetch = [s for s in symbols if s not in fresh_symbols]
    else:
        symbols_to_fetch = symbols
    
    if not symbols_to_fetch:
        logger.info("All symbols have fresh news sentiment data")
        return existing
    
    logger.info(f"Fetching news sentiment for {len(symbols_to_fetch)} symbols...")
    new_data = fetch_news_sentiment_batch(symbols_to_fetch)
    
    if existing.empty:
        result = new_data
    else:
        # Remove old data for symbols we just fetched
        existing = existing[~existing['symbol'].isin(symbols_to_fetch)]
        result = pd.concat([existing, new_data], ignore_index=True)
    
    # Save
    save_news_sentiment(result)
    
    return result


def get_news_features_for_date(asof_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Get news sentiment features ready to merge with other features
    
    Returns DataFrame with symbol as index
    """
    df = load_news_sentiment()
    
    if df.empty:
        return pd.DataFrame()
    
    # Use latest data for each symbol
    df = df.sort_values('fetch_date', ascending=False)
    df = df.drop_duplicates(subset=['symbol'], keep='first')
    
    # Select feature columns
    feature_cols = [
        'symbol', 'news_sentiment_avg', 'news_sentiment_std',
        'news_count', 'news_positive_ratio', 'news_negative_ratio'
    ]
    df = df[[c for c in feature_cols if c in df.columns]]
    
    return df


def print_sentiment_summary(df: pd.DataFrame):
    """Print summary of sentiment data"""
    if df.empty:
        print("No sentiment data available")
        return
    
    print("\n📰 News Sentiment Summary")
    print("=" * 50)
    print(f"Total symbols: {len(df)}")
    print(f"Symbols with news: {(df['news_count'] > 0).sum()}")
    print(f"Avg articles per symbol: {df['news_count'].mean():.1f}")
    print(f"\nSentiment distribution:")
    print(f"  Positive avg: {(df['news_sentiment_avg'] > 0.05).sum()} symbols")
    print(f"  Negative avg: {(df['news_sentiment_avg'] < -0.05).sum()} symbols")
    print(f"  Neutral avg: {((df['news_sentiment_avg'] >= -0.05) & (df['news_sentiment_avg'] <= 0.05)).sum()} symbols")
    
    # Top positive
    top_pos = df.nlargest(5, 'news_sentiment_avg')[['symbol', 'news_sentiment_avg', 'news_count']]
    print(f"\n🟢 Most positive sentiment:")
    for _, row in top_pos.iterrows():
        print(f"  {row['symbol']}: {row['news_sentiment_avg']:.3f} ({int(row['news_count'])} articles)")
    
    # Top negative
    top_neg = df.nsmallest(5, 'news_sentiment_avg')[['symbol', 'news_sentiment_avg', 'news_count']]
    print(f"\n🔴 Most negative sentiment:")
    for _, row in top_neg.iterrows():
        print(f"  {row['symbol']}: {row['news_sentiment_avg']:.3f} ({int(row['news_count'])} articles)")


# ============ CLI ============

if __name__ == "__main__":
    import argparse
    from data.fetch_universe import load_universe_symbols
    
    parser = argparse.ArgumentParser(description="Fetch and analyze news sentiment")
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (default: all)')
    parser.add_argument('--refresh', action='store_true', help='Force refresh all')
    parser.add_argument('--test', type=str, help='Test single symbol')
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode: single symbol
        print(f"\n🔍 Testing news fetch for {args.test}...")
        analyzer = get_sentiment_analyzer()
        articles = fetch_news_for_symbol(args.test)
        
        if not articles:
            print(f"No recent news found for {args.test}")
        else:
            print(f"\nFound {len(articles)} articles:")
            for art in articles:
                sentiment = analyze_sentiment(art['title'], analyzer)
                emoji = "🟢" if sentiment['compound'] > 0.05 else "🔴" if sentiment['compound'] < -0.05 else "⚪"
                print(f"\n{emoji} [{sentiment['compound']:.2f}] {art['title'][:80]}...")
                print(f"   Publisher: {art['publisher']}, Date: {art['publish_time']}")
            
            features = compute_sentiment_features(articles, analyzer)
            print(f"\n📊 Aggregated features:")
            for k, v in features.items():
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        # Full update
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        else:
            symbols = load_universe_symbols()
        
        df = update_news_sentiment(symbols, force_refresh=args.refresh)
        print_sentiment_summary(df)
