# PROMPT: Build a US Stock Top-10 Predictor (S&P 500)

You are an engineering assistant. Help me build a stock prediction software end-to-end.  
Goal: **predict the Top-10 S&P 500 stocks most likely to outperform over the next 5 trading days**, using daily OHLCV data, and publish results on a website.

## Context / Constraints
- My computer: **Mac**
- Primary data source: **Alpha Vantage** (I already have an API key "HOCJEXBA2YA9A56M" and note it's a key for free API in Alpha Vantage not a premium API in Alpha Vantage, so there's a daily limit of request)
- Data frequency: **daily (EOD)**
- Must support **model versioning** (v001, v002, …) and iterative upgrades
- Must support **daily automated updates** (update data + features + Top-10; retraining optional)
- Must be robust to **API rate limits/throttling** and partial data failures

---

## Product Requirements (What the software must do)

### Inputs
- Universe: **S&P 500 constituents**
- Data per symbol: daily **Open, High, Low, Close, Volume**
- Historical data storage: local files (Parquet)

### Outputs (Daily)
Generate a daily Top-10 table for an `asof_date`:
- `date` (asof_date)
- `symbol`
- `close`
- `rank_score` (ranking model score)
- `pred_ret_5` (predicted 5-trading-day forward return)
- `pred_price_5d = close * (1 + pred_ret_5)`
- `evidence_short` (compact evidence string)
- `reason_human` (human-friendly explanation)

Also maintain:
- `top10_latest.parquet` (latest)
- `top10_history.parquet` (append-only, de-duplicated by date+symbol)

### Web App
- A **Streamlit** web UI:
  - shows latest Top-10
  - lets the user select a date to view historical Top-10
  - displays key columns (symbol/close/rank_score/pred_ret_5/pred_price_5d/reason_human)
  - Top-10 sorted by `pred_ret_5` descending while retaining `rank_score`

---

## Modeling Requirements (MVP → Iteration)

### Feature Engineering (per symbol, daily)
Compute features using OHLCV and rolling windows, then **cross-sectional z-score by date**:
- Returns: `ret_1`, `ret_3`, `ret_5`, `ret_10`
- Range: `range_5`, `range_10` = avg((high-low)/close)
- Volatility: `vol_5`, `vol_10` = rolling std of daily returns
- Candlestick: `body_pct`, `close_pos`, `upper_wick_pct`, `lower_wick_pct`
- Volume/liquidity: `vol_ratio` (vol/10d avg), `vol_chg_5`, `dv_20` (20d avg close*volume)

### Targets
- `fwd_ret_5`: forward 5-trading-day return (future close / current close - 1)

### Models (Two-stage)
1) **Ranking model**: LightGBM Ranker to sort stocks within each date by future performance  
2) **Regression model**: LightGBM Regressor to estimate `fwd_ret_5` (used for final Top-10 sorting + predicted price)

Inference:
- Predict `rank_score` for all eligible stocks on `asof_date`
- Take top `CAND_K` by `rank_score` (e.g., 50)
- Sort candidates by `pred_ret_5` to pick final Top-10
- If regressor missing, fall back to ranking score

Explainability:
- Prefer “no-SHAP mode” for portability
- Generate `reason_human` from standardized feature signals (mean-reversion vs momentum vs volume/strength, plus risk note)

---

## Data Pipeline & Storage (Local Project Layout)

### Directories
- `data/`
  - `universe_symbols.txt` (one symbol per line)
  - `universe_meta.parquet` (symbol/name/sector)
  - `update_queue.json` (rotation queue for incremental updates)
  - `bars.parquet` (historical OHLCV for universe)
  - `feat_z.parquet` (feature table after z-scoring)
- `models/<version>/`
  - `schema.json` (feature_cols, thresholds)
  - `ranker.pkl`
  - `reg.pkl` (optional)
  - `metrics.json` (offline evaluation)
- `outputs/`
  - `top10_latest.parquet`
  - `top10_history.parquet`
  - `quality_report.json` (coverage + health metrics)

### Incremental Updates (Critical)
Because Alpha Vantage free tier is rate-limited:
- Update only **N symbols per run** (rotation queue)
- Detect throttle responses and stop early
- Track fail reasons and success counts
- Never crash when some symbols fail

### Coverage-aware `asof_date`
To avoid ranking on a date with too few symbols:
- compute feature coverage per date
- choose the **latest date** meeting `min_coverage_rate`
- else fall back to max-coverage date

---

## Milestones & Step-by-step Deliverables

### Milestone A — Local MVP (Train + Top-10 + UI)
1) Build S&P 500 universe list → `data/universe_symbols.txt` + `data/universe_meta.parquet`
2) Download historical daily bars (Alpha Vantage) into `data/bars.parquet`
3) Build features + z-scored table → `data/feat_z.parquet`
4) Train `ranker.pkl` + `reg.pkl`, save under `models/v001/` with `schema.json` and `metrics.json`
5) Implement `app/update_daily.py`:
   - rotation queue + incremental fetch
   - update bars + features
   - select `asof_date` by coverage
   - generate `outputs/top10_latest.parquet` and `outputs/top10_history.parquet`
   - write `outputs/quality_report.json`
6) Implement Streamlit `app/web.py`:
   - view latest Top-10
   - choose date to view historical Top-10

### Milestone B — Production hardening (Daily automation)
1) Add robust Alpha Vantage throttle detection (Note/Information) + backoff/early-stop
2) Maintain queue so each run progresses through universe
3) Add “self-test” on new data: coverage stats + sanity checks
4) Add logging and clear error messages; never crash on partial failures

### Milestone C — Model iteration (v002+)
1) Walk-forward evaluation and baselines
2) Feature improvements and leakage checks
3) Add new model versions (v002, v003…) and switch via config

### Milestone D — Public website launch
1) Deploy Streamlit (or split FastAPI + Streamlit) to a public host
2) Schedule daily update job (cron/GitHub Actions/server scheduler)
3) Add monitoring/alerts using `quality_report.json`

---

## Success Criteria
- Every trading day, the system produces a Top-10 list and stores history
- The website can show latest and historical Top-10 by date
- The pipeline is stable under rate limits and partial failures
- Models are versioned and replaceable without breaking the product
