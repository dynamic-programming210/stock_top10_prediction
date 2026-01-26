"""
Streamlit Web App for Stock Top-10 Predictor
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TOP10_LATEST_FILE, TOP10_HISTORY_FILE, QUALITY_REPORT_FILE,
    UNIVERSE_META_FILE, CURRENT_MODEL_VERSION
)

# Page config
st.set_page_config(
    page_title="S&P 500 Top-10 Predictor",
    page_icon="📈",
    layout="wide"
)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_latest_top10():
    """Load latest top-10 predictions"""
    if TOP10_LATEST_FILE.exists():
        df = pd.read_parquet(TOP10_LATEST_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_history():
    """Load historical top-10 predictions"""
    if TOP10_HISTORY_FILE.exists():
        df = pd.read_parquet(TOP10_HISTORY_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_universe_meta():
    """Load universe metadata"""
    if UNIVERSE_META_FILE.exists():
        return pd.read_parquet(UNIVERSE_META_FILE)
    return pd.DataFrame()


def load_quality_report():
    """Load quality report"""
    if QUALITY_REPORT_FILE.exists():
        with open(QUALITY_REPORT_FILE, 'r') as f:
            return json.load(f)
    return {}


def format_percent(val):
    """Format value as percentage"""
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:.2f}%"


def format_price(val):
    """Format value as price"""
    if pd.isna(val):
        return "N/A"
    return f"${val:.2f}"


def main():
    # Title
    st.title("📈 S&P 500 Top-10 Stock Predictor")
    st.markdown("*Predicting the top 10 stocks most likely to outperform over the next 5 trading days*")
    
    # Load data
    latest_df = load_latest_top10()
    history_df = load_history()
    quality_report = load_quality_report()
    universe_meta = load_universe_meta()
    
    # Sidebar
    st.sidebar.header("📊 Data Status")
    
    if quality_report:
        asof_date = quality_report.get('asof_date', 'Unknown')
        st.sidebar.metric("As-of Date", asof_date)
        
        if 'data' in quality_report:
            st.sidebar.metric("Total Symbols", quality_report['data'].get('unique_symbols', 'N/A'))
        
        if 'coverage' in quality_report:
            coverage = quality_report['coverage']
            st.sidebar.metric(
                "Coverage Rate", 
                f"{coverage.get('asof_date_rate', 0) * 100:.1f}%"
            )
        
        generated_at = quality_report.get('generated_at', '')
        if generated_at:
            st.sidebar.caption(f"Last updated: {generated_at[:19]}")
    else:
        st.sidebar.warning("No quality report found")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model Version: {CURRENT_MODEL_VERSION}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🏆 Latest Top-10", "📅 Historical", "ℹ️ About"])
    
    with tab1:
        st.header("Latest Top-10 Predictions")
        
        if latest_df.empty:
            st.warning("No predictions available yet. Run the update pipeline first.")
            st.code("python app/update_daily.py --setup", language="bash")
        else:
            # Display date
            pred_date = latest_df['date'].iloc[0]
            st.subheader(f"Predictions for: {pred_date.strftime('%Y-%m-%d')}")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            avg_pred_ret = latest_df['pred_ret_5'].mean()
            max_pred_ret = latest_df['pred_ret_5'].max()
            min_pred_ret = latest_df['pred_ret_5'].min()
            
            col1.metric("Avg Predicted Return", format_percent(avg_pred_ret))
            col2.metric("Max Predicted Return", format_percent(max_pred_ret))
            col3.metric("Min Predicted Return", format_percent(min_pred_ret))
            col4.metric("# Predictions", len(latest_df))
            
            st.markdown("---")
            
            # Main table
            display_cols = ['symbol', 'close', 'pred_ret_5', 'pred_price_5d', 'rank_score', 'reason_human']
            display_df = latest_df[[c for c in display_cols if c in latest_df.columns]].copy()
            
            # Format columns
            if 'close' in display_df.columns:
                display_df['close'] = display_df['close'].apply(format_price)
            if 'pred_price_5d' in display_df.columns:
                display_df['pred_price_5d'] = display_df['pred_price_5d'].apply(format_price)
            if 'pred_ret_5' in display_df.columns:
                display_df['pred_ret_5'] = display_df['pred_ret_5'].apply(format_percent)
            if 'rank_score' in display_df.columns:
                display_df['rank_score'] = display_df['rank_score'].round(3)
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'symbol': 'Symbol',
                'close': 'Current Price',
                'pred_ret_5': 'Pred. 5-Day Return',
                'pred_price_5d': 'Pred. Price (5d)',
                'rank_score': 'Rank Score',
                'reason_human': 'Reason'
            })
            
            # Display with highlighting
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=450
            )
            
            # Additional info
            st.markdown("---")
            
            # Expand for sector breakdown
            if not universe_meta.empty and 'symbol' in latest_df.columns:
                with st.expander("📊 Sector Breakdown"):
                    merged = latest_df.merge(universe_meta[['symbol', 'sector']], on='symbol', how='left')
                    sector_counts = merged['sector'].value_counts()
                    st.bar_chart(sector_counts)
    
    with tab2:
        st.header("Historical Top-10 Predictions")
        
        if history_df.empty:
            st.warning("No historical data available yet.")
        else:
            # Date selector
            available_dates = history_df['date'].dt.date.unique()
            available_dates = sorted(available_dates, reverse=True)
            
            selected_date = st.selectbox(
                "Select Date",
                options=available_dates,
                format_func=lambda x: x.strftime('%Y-%m-%d')
            )
            
            if selected_date:
                # Filter to selected date
                date_df = history_df[history_df['date'].dt.date == selected_date].copy()
                
                if date_df.empty:
                    st.warning(f"No data for {selected_date}")
                else:
                    st.subheader(f"Top-10 for {selected_date.strftime('%Y-%m-%d')}")
                    
                    # Display table
                    display_cols = ['symbol', 'close', 'pred_ret_5', 'pred_price_5d', 'rank_score', 'reason_human']
                    display_df = date_df[[c for c in display_cols if c in date_df.columns]].copy()
                    
                    if 'close' in display_df.columns:
                        display_df['close'] = display_df['close'].apply(format_price)
                    if 'pred_price_5d' in display_df.columns:
                        display_df['pred_price_5d'] = display_df['pred_price_5d'].apply(format_price)
                    if 'pred_ret_5' in display_df.columns:
                        display_df['pred_ret_5'] = display_df['pred_ret_5'].apply(format_percent)
                    if 'rank_score' in display_df.columns:
                        display_df['rank_score'] = display_df['rank_score'].round(3)
                    
                    display_df = display_df.rename(columns={
                        'symbol': 'Symbol',
                        'close': 'Price (at pred)',
                        'pred_ret_5': 'Pred. 5-Day Return',
                        'pred_price_5d': 'Pred. Price (5d)',
                        'rank_score': 'Rank Score',
                        'reason_human': 'Reason'
                    })
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Summary stats
            st.markdown("---")
            st.subheader("📈 History Summary")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Dates", len(available_dates))
            col2.metric("Date Range", f"{min(available_dates)} to {max(available_dates)}")
            col3.metric("Unique Symbols", history_df['symbol'].nunique())
            
            # Most frequent symbols
            with st.expander("🔥 Most Frequently Selected Symbols"):
                freq = history_df['symbol'].value_counts().head(20)
                st.bar_chart(freq)
    
    with tab3:
        st.header("About This App")
        
        st.markdown("""
        ### 📈 S&P 500 Top-10 Stock Predictor
        
        This application predicts the top 10 S&P 500 stocks most likely to outperform 
        over the next 5 trading days using machine learning.
        
        #### How It Works
        
        1. **Data Collection**: Daily OHLCV (Open, High, Low, Close, Volume) data is 
           collected from Alpha Vantage for all S&P 500 constituents.
        
        2. **Feature Engineering**: Technical features are computed including:
           - Returns over multiple horizons (1, 3, 5, 10 days)
           - Volatility measures
           - Volume/liquidity signals
           - Candlestick patterns
        
        3. **Cross-Sectional Z-Scoring**: Features are normalized within each date 
           to compare stocks against each other.
        
        4. **Two-Stage Prediction**:
           - **Ranking Model**: LightGBM ranker sorts stocks by likelihood to outperform
           - **Regression Model**: Estimates actual 5-day forward returns
        
        5. **Top-10 Selection**: Final picks are sorted by predicted return.
        
        #### Columns Explained
        
        | Column | Description |
        |--------|-------------|
        | Symbol | Stock ticker |
        | Current Price | Latest closing price |
        | Pred. 5-Day Return | Predicted return over next 5 trading days |
        | Pred. Price (5d) | Expected price in 5 days |
        | Rank Score | Model's ranking score (higher = more likely to outperform) |
        | Reason | Human-readable explanation for selection |
        
        #### ⚠️ Disclaimer
        
        This tool is for educational and research purposes only. It does not constitute 
        financial advice. Past performance does not guarantee future results. Always do 
        your own research before making investment decisions.
        
        #### Technical Details
        
        - **Data Source**: Alpha Vantage (free tier)
        - **Update Frequency**: Daily
        - **Model**: LightGBM (Ranker + Regressor)
        - **Universe**: S&P 500 constituents
        """)
        
        # System status
        st.markdown("---")
        st.subheader("🔧 System Status")
        
        if quality_report:
            with st.expander("Quality Report Details"):
                st.json(quality_report)


if __name__ == "__main__":
    main()
