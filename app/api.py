"""
E5: FastAPI REST API for Stock Top-10 Predictor
Provides programmatic access to predictions, history, and system status
"""
import json
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

from config import (
    TOP10_LATEST_FILE, TOP10_HISTORY_FILE, QUALITY_REPORT_FILE,
    UNIVERSE_META_FILE, CURRENT_MODEL_VERSION, OUTPUTS_DIR
)

# ============ Pydantic Models ============

class StockPrediction(BaseModel):
    """Single stock prediction"""
    symbol: str = Field(..., description="Stock ticker symbol")
    close: float = Field(..., description="Current closing price")
    pred_ret_5: float = Field(..., description="Predicted 5-day return")
    pred_price_5d: float = Field(..., description="Predicted price in 5 days")
    rank_score: float = Field(..., description="Model ranking score")
    reason_human: Optional[str] = Field(None, description="Human-readable explanation")
    evidence_short: Optional[str] = Field(None, description="Short evidence string")
    sector: Optional[str] = Field(None, description="Stock sector")


class Top10Response(BaseModel):
    """Top-10 predictions response"""
    asof_date: str = Field(..., description="Prediction date")
    model_version: str = Field(..., description="Model version used")
    predictions: List[StockPrediction] = Field(..., description="List of top-10 predictions")
    generated_at: str = Field(..., description="API response timestamp")


class HistoryResponse(BaseModel):
    """Historical predictions response"""
    start_date: str
    end_date: str
    total_records: int
    dates: List[str]
    predictions: Dict[str, List[StockPrediction]]


class SymbolHistoryResponse(BaseModel):
    """Symbol prediction history response"""
    symbol: str
    appearances: int
    history: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """System health status"""
    status: str = Field(..., description="Overall system status")
    model_version: str
    last_update: Optional[str] = None
    data_coverage: Optional[float] = None
    total_symbols: Optional[int] = None
    checks: Dict[str, Any] = Field(default_factory=dict)


class PerformanceResponse(BaseModel):
    """Performance metrics response"""
    direction_accuracy: Optional[float] = None
    positive_return_rate: Optional[float] = None
    correlation: Optional[float] = None
    mae: Optional[float] = None
    total_predictions_tracked: Optional[int] = None


class UniverseResponse(BaseModel):
    """Universe metadata response"""
    total_symbols: int
    sectors: Dict[str, int]
    symbols: Optional[List[Dict[str, str]]] = None


# ============ FastAPI App ============

app = FastAPI(
    title="S&P 500 Top-10 Stock Predictor API",
    description="""
    RESTful API for accessing stock predictions.
    
    ## Features
    - Get latest top-10 predictions
    - Access historical predictions by date
    - Track symbol performance history
    - Monitor system health
    - View universe metadata
    
    ## Model Information
    Uses a two-stage prediction approach:
    1. **Ranking Model**: LightGBM ranker to sort stocks
    2. **Regression Model**: LightGBM regressor for return prediction
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Data Loading Functions ============

def load_latest_top10() -> pd.DataFrame:
    """Load latest top-10 predictions"""
    if TOP10_LATEST_FILE.exists():
        return pd.read_parquet(TOP10_LATEST_FILE)
    return pd.DataFrame()


def load_history() -> pd.DataFrame:
    """Load historical predictions"""
    if TOP10_HISTORY_FILE.exists():
        return pd.read_parquet(TOP10_HISTORY_FILE)
    return pd.DataFrame()


def load_quality_report() -> Dict:
    """Load quality report"""
    if QUALITY_REPORT_FILE.exists():
        with open(QUALITY_REPORT_FILE) as f:
            return json.load(f)
    return {}


def load_universe_meta() -> pd.DataFrame:
    """Load universe metadata"""
    if UNIVERSE_META_FILE.exists():
        return pd.read_parquet(UNIVERSE_META_FILE)
    return pd.DataFrame()


def load_performance_metrics() -> Dict:
    """Load performance tracking data"""
    perf_file = OUTPUTS_DIR / "performance" / "performance_history.parquet"
    if perf_file.exists():
        df = pd.read_parquet(perf_file)
        if not df.empty and 'actual_ret_5' in df.columns:
            valid = df.dropna(subset=['actual_ret_5'])
            if not valid.empty:
                direction_hit = ((valid['pred_ret_5'] > 0) == (valid['actual_ret_5'] > 0)).mean()
                positive_rate = (valid['actual_ret_5'] > 0).mean()
                corr = valid['pred_ret_5'].corr(valid['actual_ret_5'])
                mae = (valid['pred_ret_5'] - valid['actual_ret_5']).abs().mean()
                return {
                    'direction_accuracy': float(direction_hit),
                    'positive_return_rate': float(positive_rate),
                    'correlation': float(corr) if pd.notna(corr) else None,
                    'mae': float(mae),
                    'total_predictions_tracked': len(valid)
                }
    return {}


def df_to_predictions(df: pd.DataFrame, universe_meta: pd.DataFrame = None) -> List[StockPrediction]:
    """Convert DataFrame rows to StockPrediction objects"""
    predictions = []
    
    # Get sector mapping
    sector_map = {}
    if universe_meta is not None and not universe_meta.empty:
        sector_map = dict(zip(universe_meta['symbol'], universe_meta['sector']))
    
    for _, row in df.iterrows():
        pred = StockPrediction(
            symbol=row['symbol'],
            close=float(row['close']),
            pred_ret_5=float(row['pred_ret_5']),
            pred_price_5d=float(row['pred_price_5d']),
            rank_score=float(row['rank_score']),
            reason_human=row.get('reason_human'),
            evidence_short=row.get('evidence_short'),
            sector=sector_map.get(row['symbol'])
        )
        predictions.append(pred)
    
    return predictions


# ============ API Endpoints ============

@app.get("/", tags=["Root"])
async def root():
    """API root - welcome message and links"""
    return {
        "message": "S&P 500 Top-10 Stock Predictor API",
        "version": "1.0.0",
        "model_version": CURRENT_MODEL_VERSION,
        "endpoints": {
            "docs": "/docs",
            "latest": "/api/v1/top10/latest",
            "history": "/api/v1/top10/history",
            "health": "/api/v1/health",
            "universe": "/api/v1/universe"
        }
    }


@app.get("/api/v1/top10/latest", response_model=Top10Response, tags=["Predictions"])
async def get_latest_top10():
    """
    Get the latest top-10 stock predictions.
    
    Returns the most recent predictions with:
    - Stock symbols and current prices
    - Predicted 5-day returns and prices
    - Model ranking scores
    - Human-readable explanations
    """
    df = load_latest_top10()
    
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No predictions available. Run the update pipeline first."
        )
    
    universe_meta = load_universe_meta()
    predictions = df_to_predictions(df, universe_meta)
    
    asof_date = df['date'].iloc[0]
    if isinstance(asof_date, pd.Timestamp):
        asof_date = asof_date.strftime('%Y-%m-%d')
    
    return Top10Response(
        asof_date=str(asof_date),
        model_version=CURRENT_MODEL_VERSION,
        predictions=predictions,
        generated_at=datetime.now().isoformat()
    )


@app.get("/api/v1/top10/history", tags=["Predictions"])
async def get_history(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(30, ge=1, le=365, description="Max number of dates to return")
):
    """
    Get historical top-10 predictions.
    
    Filter by date range or get the most recent predictions.
    """
    df = load_history()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No historical data available")
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply date filters
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data for specified date range")
    
    # Get unique dates and limit
    dates = sorted(df['date'].unique(), reverse=True)[:limit]
    
    universe_meta = load_universe_meta()
    predictions_by_date = {}
    
    for date_val in dates:
        date_df = df[df['date'] == date_val].copy()
        date_str = pd.Timestamp(date_val).strftime('%Y-%m-%d')
        predictions_by_date[date_str] = df_to_predictions(date_df, universe_meta)
    
    return {
        "start_date": min(predictions_by_date.keys()),
        "end_date": max(predictions_by_date.keys()),
        "total_records": len(df),
        "dates": sorted(predictions_by_date.keys(), reverse=True),
        "predictions": predictions_by_date,
        "generated_at": datetime.now().isoformat()
    }


@app.get("/api/v1/top10/date/{date}", response_model=Top10Response, tags=["Predictions"])
async def get_predictions_by_date(date: str):
    """
    Get top-10 predictions for a specific date.
    
    Date format: YYYY-MM-DD
    """
    df = load_history()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No historical data available")
    
    df['date'] = pd.to_datetime(df['date'])
    target_date = pd.to_datetime(date)
    
    date_df = df[df['date'] == target_date]
    
    if date_df.empty:
        # Find nearest available date
        available_dates = sorted(df['date'].unique())
        raise HTTPException(
            status_code=404,
            detail=f"No predictions for {date}. Available dates: {[d.strftime('%Y-%m-%d') for d in available_dates[-5:]]}"
        )
    
    universe_meta = load_universe_meta()
    predictions = df_to_predictions(date_df, universe_meta)
    
    return Top10Response(
        asof_date=date,
        model_version=CURRENT_MODEL_VERSION,
        predictions=predictions,
        generated_at=datetime.now().isoformat()
    )


@app.get("/api/v1/symbol/{symbol}", response_model=SymbolHistoryResponse, tags=["Symbols"])
async def get_symbol_history(
    symbol: str,
    limit: int = Query(30, ge=1, le=100, description="Max appearances to return")
):
    """
    Get prediction history for a specific symbol.
    
    Shows all dates when the symbol appeared in top-10.
    """
    symbol = symbol.upper()
    df = load_history()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No historical data available")
    
    symbol_df = df[df['symbol'] == symbol].copy()
    
    if symbol_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {symbol} has never appeared in top-10 predictions"
        )
    
    symbol_df['date'] = pd.to_datetime(symbol_df['date'])
    symbol_df = symbol_df.sort_values('date', ascending=False).head(limit)
    
    history = []
    for _, row in symbol_df.iterrows():
        history.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "close": float(row['close']),
            "pred_ret_5": float(row['pred_ret_5']),
            "pred_price_5d": float(row['pred_price_5d']),
            "rank_score": float(row['rank_score'])
        })
    
    return SymbolHistoryResponse(
        symbol=symbol,
        appearances=len(df[df['symbol'] == symbol]),
        history=history
    )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Get system health status.
    
    Checks:
    - Data availability
    - Model status
    - Last update time
    - Coverage metrics
    """
    quality_report = load_quality_report()
    latest_df = load_latest_top10()
    
    checks = {
        "latest_predictions": not latest_df.empty,
        "history_available": load_history().shape[0] > 0,
        "quality_report": bool(quality_report),
        "universe_loaded": not load_universe_meta().empty
    }
    
    all_ok = all(checks.values())
    
    response = HealthResponse(
        status="healthy" if all_ok else "degraded",
        model_version=CURRENT_MODEL_VERSION,
        checks=checks
    )
    
    if quality_report:
        response.last_update = quality_report.get('generated_at')
        response.total_symbols = quality_report.get('data', {}).get('unique_symbols')
        coverage = quality_report.get('coverage', {})
        response.data_coverage = coverage.get('asof_date_rate')
    
    return response


@app.get("/api/v1/performance", response_model=PerformanceResponse, tags=["System"])
async def get_performance():
    """
    Get prediction performance metrics.
    
    Requires performance tracking to be enabled (--track-performance).
    """
    metrics = load_performance_metrics()
    
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="No performance data available. Run with --track-performance flag."
        )
    
    return PerformanceResponse(**metrics)


@app.get("/api/v1/universe", response_model=UniverseResponse, tags=["Universe"])
async def get_universe(
    include_symbols: bool = Query(False, description="Include full symbol list")
):
    """
    Get S&P 500 universe metadata.
    
    Returns sector breakdown and optionally full symbol list.
    """
    df = load_universe_meta()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Universe metadata not available")
    
    sectors = df['sector'].value_counts().to_dict()
    
    response = UniverseResponse(
        total_symbols=len(df),
        sectors=sectors
    )
    
    if include_symbols:
        response.symbols = df[['symbol', 'name', 'sector']].to_dict('records')
    
    return response


@app.get("/api/v1/universe/sector/{sector}", tags=["Universe"])
async def get_sector_symbols(sector: str):
    """
    Get all symbols in a specific sector.
    """
    df = load_universe_meta()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Universe metadata not available")
    
    # Case-insensitive sector matching
    sector_df = df[df['sector'].str.lower() == sector.lower()]
    
    if sector_df.empty:
        available_sectors = df['sector'].unique().tolist()
        raise HTTPException(
            status_code=404,
            detail=f"Sector '{sector}' not found. Available: {available_sectors}"
        )
    
    return {
        "sector": sector_df['sector'].iloc[0],
        "count": len(sector_df),
        "symbols": sector_df[['symbol', 'name']].to_dict('records')
    }


@app.get("/api/v1/dates", tags=["Predictions"])
async def get_available_dates(
    limit: int = Query(30, ge=1, le=365, description="Max dates to return")
):
    """
    Get list of dates with available predictions.
    """
    df = load_history()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No historical data available")
    
    df['date'] = pd.to_datetime(df['date'])
    dates = sorted(df['date'].unique(), reverse=True)[:limit]
    
    return {
        "total_dates": len(df['date'].unique()),
        "dates": [pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates]
    }


# ============ Error Handlers ============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
