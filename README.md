# S&P 500 Top-10 Stock Predictor

A machine learning system that predicts the top 10 S&P 500 stocks most likely to outperform over the next 5 trading days.

## Features

- 📊 **Multiple Data Sources**: yfinance (unlimited, free) or Alpha Vantage
- 🧮 **Technical Features**: Returns, volatility, volume, RSI, MACD, Bollinger Bands
- 🤖 **Two-Stage ML Model**: GradientBoosting Ranker + Regressor for accurate predictions
- 📈 **Web Dashboard**: Streamlit app to view latest and historical predictions
- 🔄 **Walk-Forward Validation**: Proper time-series cross-validation to avoid look-ahead bias
- 📉 **Backtesting Framework**: Simulate historical portfolio performance with realistic costs
- 📁 **Model Versioning**: Support for multiple model versions (v001, v002, ...)

## Quick Start

### 1. Install Dependencies

```bash
cd stock
pip install -r requirements.txt
```

### 2. Fetch Data & Train Model

```bash
# Fetch data using yfinance (unlimited, recommended)
python app/update_daily.py --setup

# Or use Alpha Vantage (rate limited)
python app/update_daily.py --setup --use-alpha-vantage --batch-size 20
```

### 3. Run Daily Updates

```bash
# Normal daily update with yfinance
python app/update_daily.py

# Skip data fetch (use existing data)
python app/update_daily.py --skip-data

# Include backtest
python app/update_daily.py --backtest

# Run walk-forward validation
python app/update_daily.py --walk-forward --n-folds 5
```

### 4. Launch Web App

```bash
streamlit run app/web.py
```

Open http://localhost:8501 in your browser.

## Technical Indicators (B1)

The system computes 21 features including:

| Category | Features |
|----------|----------|
| Returns | ret_1, ret_3, ret_5, ret_10 |
| Volatility | vol_5, vol_10, range_5, range_10 |
| Volume | vol_ratio, vol_chg_5, dv_20 |
| Candlestick | body_pct, close_pos, upper_wick_pct, lower_wick_pct |
| **RSI** | rsi_14 (14-period Relative Strength Index) |
| **MACD** | macd_line, macd_signal, macd_hist |
| **Bollinger** | bb_position, bb_width |

## Walk-Forward Validation (C8)

Proper time-series cross-validation that prevents look-ahead bias:

```python
from models.train import walk_forward_validation
from features.build_features import load_features

features = load_features()
results = walk_forward_validation(features, n_splits=5, min_train_days=126)
print(f"Mean Correlation: {results['aggregate']['mean_correlation']:.4f}")
print(f"Top-10 Hit Rate: {results['aggregate']['mean_hit_rate']:.2%}")
```

## Backtesting (D5)

Simulate portfolio performance with transaction costs:

```python
from models.backtest import run_backtest
from features.build_features import load_features
from data.fetch_bars import load_existing_bars

results = run_backtest(
    features_df=load_features(),
    bars_df=load_existing_bars(),
    initial_capital=100000,
    rebalance_days=5  # Weekly rebalancing
)
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
```

## Project Structure

```
stock/
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── data/
│   ├── fetch_universe.py  # S&P 500 list fetcher
│   ├── fetch_yfinance.py  # yfinance data fetcher (A1)
│   ├── fetch_bars.py      # Alpha Vantage data fetcher
│   └── *.parquet          # Data files
├── features/
│   └── build_features.py  # Feature engineering (B1: RSI/MACD/BB)
├── models/
│   ├── train.py           # Training & walk-forward validation (C8)
│   ├── backtest.py        # Backtesting framework (D5)
│   └── v001/              # Model artifacts
├── app/
│   ├── update_daily.py    # Daily update pipeline
│   └── web.py             # Streamlit web app
└── outputs/
    ├── top10_*.parquet    # Predictions
    ├── backtest_*.json    # Backtest results
    └── quality_report.json
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SYMBOLS_PER_RUN` | 20 | Symbols to update per run |
| `MIN_COVERAGE_RATE` | 0.6 | Minimum symbol coverage for predictions |
| `CAND_K` | 50 | Candidates from ranker |
| `TOP_N` | 10 | Final top picks |
| `CURRENT_MODEL_VERSION` | v001 | Active model version |

## Feature Engineering

### Raw Features (per symbol)
- Returns: `ret_1`, `ret_3`, `ret_5`, `ret_10`
- Range: `range_5`, `range_10`
- Volatility: `vol_5`, `vol_10`
- Candlestick: `body_pct`, `close_pos`, `upper_wick_pct`, `lower_wick_pct`
- Volume: `vol_ratio`, `vol_chg_5`, `dv_20`

### Cross-Sectional Z-Scoring
All features are normalized across stocks within each date to compare relative performance.

## Model Architecture

### Stage 1: Ranking Model
- LightGBM Ranker (`lambdarank` objective)
- Sorts stocks by likelihood to outperform
- Outputs `rank_score`

### Stage 2: Regression Model
- LightGBM Regressor
- Estimates 5-day forward return (`pred_ret_5`)
- Used for final sorting and price prediction

## API Rate Limits

Alpha Vantage free tier limits:
- 25 requests/day
- 5 requests/minute

The system handles this by:
1. Rotation queue: Cycles through symbols
2. Throttle detection: Stops on API limits
3. Incremental updates: Progress over multiple runs

## Daily Automation

### Using cron (Linux/Mac)
```bash
# Run at 6 PM EST on weekdays
0 18 * * 1-5 cd /path/to/stock && python app/update_daily.py >> logs/cron.log 2>&1
```

### Using launchd (Mac)
Create `~/Library/LaunchAgents/com.stock.update.plist` and load it.

## Disclaimer

⚠️ **This tool is for educational and research purposes only.** It does not constitute financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT License
