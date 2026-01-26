# S&P 500 Top-10 Stock Predictor

A machine learning system that predicts the top 10 S&P 500 stocks most likely to outperform over the next 5 trading days.

## Features

- 📊 **Daily OHLCV Data**: Collects data from Alpha Vantage (free tier compatible)
- 🧮 **Technical Features**: Computes returns, volatility, volume signals, candlestick patterns
- 🤖 **Two-Stage ML Model**: LightGBM Ranker + Regressor for accurate predictions
- 📈 **Web Dashboard**: Streamlit app to view latest and historical predictions
- 🔄 **Incremental Updates**: Rate-limit aware updates that progress through the universe
- 📁 **Model Versioning**: Support for multiple model versions (v001, v002, ...)

## Quick Start

### 1. Install Dependencies

```bash
cd stock
pip install -r requirements.txt
```

### 2. Initial Setup (Fetch Universe & Data)

Due to Alpha Vantage API limits (25 requests/day on free tier), data is fetched incrementally:

```bash
# Fetch S&P 500 universe and start data collection
python app/update_daily.py --setup --batch-size 20
```

Run this multiple times over several days to build up historical data.

### 3. Train the Model

Once you have enough data (at least 50 symbols with 30+ days of history):

```bash
python -c "
from data.fetch_bars import load_existing_bars
from features.build_features import build_and_save_features
from models.train import train_full_pipeline

bars = load_existing_bars()
features = build_and_save_features(bars)
ranker, regressor, metrics, model_dir = train_full_pipeline(features)
print(f'Model saved to: {model_dir}')
"
```

### 4. Run Daily Updates

```bash
# Normal daily update
python app/update_daily.py

# Skip data fetch (use existing data)
python app/update_daily.py --skip-data
```

### 5. Launch Web App

```bash
streamlit run app/web.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
stock/
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── data/
│   ├── fetch_universe.py  # S&P 500 list fetcher
│   ├── fetch_bars.py      # Alpha Vantage data fetcher
│   ├── universe_symbols.txt
│   ├── universe_meta.parquet
│   ├── bars.parquet
│   └── feat_z.parquet
├── features/
│   └── build_features.py  # Feature engineering
├── models/
│   ├── train.py           # Model training & inference
│   └── v001/
│       ├── ranker.pkl
│       ├── reg.pkl
│       ├── schema.json
│       └── metrics.json
├── app/
│   ├── update_daily.py    # Daily update pipeline
│   └── web.py             # Streamlit web app
└── outputs/
    ├── top10_latest.parquet
    ├── top10_history.parquet
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
