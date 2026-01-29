"""
Streamlit Web App for Stock Top-10 Predictor
Enhanced with F4 Charts, G3 Earnings Calendar, F3 Confidence Intervals
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TOP10_LATEST_FILE, TOP10_HISTORY_FILE, QUALITY_REPORT_FILE,
    UNIVERSE_META_FILE, CURRENT_MODEL_VERSION, OUTPUTS_DIR
)

# Page config
st.set_page_config(
    page_title="S&P 500 Top-10 Predictor",
    page_icon="📈",
    layout="wide"
)


# ============ Data Loading Functions ============

@st.cache_data(ttl=300)
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


@st.cache_data(ttl=3600)
def load_universe_meta():
    """Load universe metadata"""
    if UNIVERSE_META_FILE.exists():
        return pd.read_parquet(UNIVERSE_META_FILE)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_performance_history():
    """Load performance tracking history"""
    perf_file = OUTPUTS_DIR / "performance" / "performance_history.parquet"
    if perf_file.exists():
        return pd.read_parquet(perf_file)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_earnings_data(symbols: list):
    """Get earnings calendar data for symbols"""
    try:
        from data.fetch_earnings import flag_earnings_risk
        earnings_risk = flag_earnings_risk(symbols, days_threshold=14)
        return earnings_risk
    except Exception as e:
        return {}


def add_confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Add confidence intervals to predictions"""
    try:
        from models.confidence import add_confidence_intervals as compute_ci
        return compute_ci(df, confidence_level=0.90)
    except Exception as e:
        # Fallback simple calculation
        df = df.copy()
        df['pred_std'] = df['pred_ret_5'].abs() * 0.4 + 0.02
        df['pred_lower'] = df['pred_ret_5'] - 1.645 * df['pred_std']
        df['pred_upper'] = df['pred_ret_5'] + 1.645 * df['pred_std']
        df['confidence_score'] = 0.7
        return df


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


def get_confidence_color(score):
    """Get emoji color based on confidence score"""
    if score >= 0.7:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    else:
        return "🔴"


# ============ Chart Functions (F4) ============

def render_predictions_chart(df: pd.DataFrame):
    """Render bar chart of predicted returns with confidence intervals"""
    import plotly.graph_objects as go
    
    df_plot = df.sort_values('pred_ret_5', ascending=True).copy()
    
    colors = ['#00CC96' if x > 0 else '#EF553B' for x in df_plot['pred_ret_5']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_plot['symbol'],
        x=df_plot['pred_ret_5'] * 100,
        orientation='h',
        marker_color=colors,
        name='Predicted Return',
        text=[f"{x*100:.1f}%" for x in df_plot['pred_ret_5']],
        textposition='outside'
    ))
    
    if 'pred_lower' in df_plot.columns and 'pred_upper' in df_plot.columns:
        error_minus = (df_plot['pred_ret_5'] - df_plot['pred_lower']) * 100
        error_plus = (df_plot['pred_upper'] - df_plot['pred_ret_5']) * 100
        
        fig.add_trace(go.Scatter(
            y=df_plot['symbol'],
            x=df_plot['pred_ret_5'] * 100,
            error_x=dict(
                type='data',
                symmetric=False,
                array=error_plus.tolist(),
                arrayminus=error_minus.tolist(),
                color='rgba(0,0,0,0.3)',
                thickness=2
            ),
            mode='markers',
            marker=dict(size=1, color='rgba(0,0,0,0)'),
            name='90% CI',
            showlegend=True
        ))
    
    fig.update_layout(
        title='Predicted 5-Day Returns with 90% Confidence Intervals',
        xaxis_title='Predicted Return (%)',
        yaxis_title='Symbol',
        height=400,
        showlegend=True,
        xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1)
    )
    
    return fig


def render_sector_pie_chart(df: pd.DataFrame, universe_meta: pd.DataFrame):
    """Render sector breakdown pie chart"""
    import plotly.express as px
    
    merged = df.merge(universe_meta[['symbol', 'sector']], on='symbol', how='left')
    sector_counts = merged['sector'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'Count']
    
    fig = px.pie(sector_counts, values='Count', names='Sector', title='Sector Distribution', hole=0.4)
    fig.update_layout(height=350)
    
    return fig


def render_performance_chart(perf_df: pd.DataFrame):
    """Render historical performance chart"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if perf_df.empty or 'pred_date' not in perf_df.columns:
        return None
    
    perf_df = perf_df.copy()
    perf_df['pred_date'] = pd.to_datetime(perf_df['pred_date'])
    daily = perf_df.groupby('pred_date').agg({
        'actual_ret_5': 'mean',
        'pred_ret_5': 'mean'
    }).reset_index().dropna()
    
    if daily.empty:
        return None
    
    daily['cum_actual'] = (1 + daily['actual_ret_5']).cumprod() - 1
    daily['cum_pred'] = (1 + daily['pred_ret_5']).cumprod() - 1
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Cumulative Returns', 'Daily Returns'),
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4])
    
    fig.add_trace(go.Scatter(
        x=daily['pred_date'], y=daily['cum_actual'] * 100,
        name='Actual', line=dict(color='#00CC96', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=daily['pred_date'], y=daily['cum_pred'] * 100,
        name='Predicted', line=dict(color='#636EFA', width=2, dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=daily['pred_date'], y=daily['actual_ret_5'] * 100,
        name='Daily Actual', marker_color='#00CC96', opacity=0.7
    ), row=2, col=1)
    
    fig.update_layout(
        height=500,
        title='Model Performance Over Time',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    fig.update_yaxes(title_text='Return (%)', row=1, col=1)
    fig.update_yaxes(title_text='Return (%)', row=2, col=1)
    
    return fig


def render_symbol_history_chart(history_df: pd.DataFrame, symbol: str):
    """Render history chart for a specific symbol"""
    import plotly.express as px
    
    symbol_data = history_df[history_df['symbol'] == symbol].copy()
    
    if symbol_data.empty:
        return None
    
    symbol_data = symbol_data.sort_values('date')
    
    fig = px.line(
        symbol_data, x='date', y='pred_ret_5',
        title=f'{symbol} - Historical Predictions',
        markers=True
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Predicted Return',
        yaxis_tickformat='.1%',
        height=300
    )
    
    return fig


def main():
    # Title
    st.title("📈 S&P 500 Top-10 Stock Predictor")
    st.markdown("*Predicting the top 10 stocks most likely to outperform over the next 5 trading days*")
    
    # Load data
    latest_df = load_latest_top10()
    history_df = load_history()
    quality_report = load_quality_report()
    universe_meta = load_universe_meta()
    perf_df = load_performance_history()
    
    # Add confidence intervals to latest predictions
    if not latest_df.empty:
        latest_df = add_confidence_intervals(latest_df)
        
        # Get earnings data
        symbols = latest_df['symbol'].tolist()
        earnings_risk = get_earnings_data(symbols)
        if earnings_risk:
            latest_df['earnings_days'] = latest_df['symbol'].map(earnings_risk)
    
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
    
    # Main content tabs - enhanced with new tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Latest Top-10", 
        "📊 Charts & Analysis",
        "📅 Historical", 
        "📈 Performance",
        "ℹ️ About"
    ])
    
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
            avg_confidence = latest_df['confidence_score'].mean() if 'confidence_score' in latest_df.columns else 0.7
            
            col1.metric("Avg Predicted Return", format_percent(avg_pred_ret))
            col2.metric("Max Predicted Return", format_percent(max_pred_ret))
            col3.metric("Avg Confidence", f"{avg_confidence:.0%}")
            col4.metric("# Predictions", len(latest_df))
            
            st.markdown("---")
            
            # G3: Earnings warnings
            if 'earnings_days' in latest_df.columns:
                earnings_soon = latest_df[latest_df['earnings_days'].notna()].copy()
                if not earnings_soon.empty:
                    st.warning(f"⚠️ **Earnings Alert**: {len(earnings_soon)} stocks have earnings within 14 days!")
                    with st.expander("View Earnings Calendar"):
                        for _, row in earnings_soon.iterrows():
                            days = int(row['earnings_days'])
                            emoji = "🔴" if days <= 3 else "🟡" if days <= 7 else "🟢"
                            st.write(f"{emoji} **{row['symbol']}**: Earnings in {days} days")
            
            # Main table with confidence intervals
            display_cols = ['symbol', 'close', 'pred_ret_5', 'pred_lower', 'pred_upper', 
                          'confidence_score', 'pred_price_5d', 'reason_human']
            display_df = latest_df[[c for c in display_cols if c in latest_df.columns]].copy()
            
            # Format columns
            if 'close' in display_df.columns:
                display_df['close'] = display_df['close'].apply(format_price)
            if 'pred_price_5d' in display_df.columns:
                display_df['pred_price_5d'] = display_df['pred_price_5d'].apply(format_price)
            if 'pred_ret_5' in display_df.columns:
                display_df['pred_ret_5'] = display_df['pred_ret_5'].apply(format_percent)
            if 'pred_lower' in display_df.columns:
                display_df['pred_lower'] = display_df['pred_lower'].apply(format_percent)
            if 'pred_upper' in display_df.columns:
                display_df['pred_upper'] = display_df['pred_upper'].apply(format_percent)
            if 'confidence_score' in display_df.columns:
                display_df['confidence_score'] = display_df['confidence_score'].apply(
                    lambda x: f"{get_confidence_color(x)} {x:.0%}"
                )
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'symbol': 'Symbol',
                'close': 'Price',
                'pred_ret_5': 'Prediction',
                'pred_lower': 'Low (90%)',
                'pred_upper': 'High (90%)',
                'confidence_score': 'Confidence',
                'pred_price_5d': 'Target Price',
                'reason_human': 'Reason'
            })
            
            # Display with highlighting
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=450
            )
    
    # ========== TAB 2: Charts & Analysis (F4) ==========
    with tab2:
        st.header("📊 Charts & Analysis")
        
        if latest_df.empty:
            st.warning("No data available for charts")
        else:
            # Predictions chart with CI
            st.subheader("Predicted Returns with Confidence Intervals")
            fig_pred = render_predictions_chart(latest_df)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sector breakdown
                if not universe_meta.empty:
                    st.subheader("Sector Distribution")
                    fig_sector = render_sector_pie_chart(latest_df, universe_meta)
                    st.plotly_chart(fig_sector, use_container_width=True)
            
            with col2:
                # Confidence distribution
                st.subheader("Confidence Distribution")
                if 'confidence_score' in latest_df.columns:
                    import plotly.express as px
                    fig_conf = px.histogram(
                        latest_df, x='confidence_score', nbins=10,
                        title='Prediction Confidence Distribution'
                    )
                    fig_conf.update_layout(
                        xaxis_title='Confidence Score',
                        yaxis_title='Count',
                        height=350
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            # Symbol deep dive
            st.markdown("---")
            st.subheader("🔍 Symbol Deep Dive")
            
            selected_symbol = st.selectbox(
                "Select a symbol to analyze",
                options=latest_df['symbol'].tolist()
            )
            
            if selected_symbol:
                symbol_row = latest_df[latest_df['symbol'] == selected_symbol].iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", format_price(symbol_row['close']))
                col2.metric("Predicted Return", format_percent(symbol_row['pred_ret_5']))
                col3.metric("Target Price", format_price(symbol_row.get('pred_price_5d', symbol_row['close'] * (1 + symbol_row['pred_ret_5']))))
                
                if 'confidence_score' in symbol_row:
                    col4.metric("Confidence", f"{symbol_row['confidence_score']:.0%}")
                
                # F3: Confidence interval display
                if 'pred_lower' in symbol_row and 'pred_upper' in symbol_row:
                    st.info(f"📊 **90% Confidence Interval**: {format_percent(symbol_row['pred_lower'])} to {format_percent(symbol_row['pred_upper'])}")
                
                # G3: Earnings warning
                if 'earnings_days' in symbol_row and pd.notna(symbol_row['earnings_days']):
                    days = int(symbol_row['earnings_days'])
                    if days <= 7:
                        st.warning(f"⚠️ **Earnings Alert**: {selected_symbol} reports earnings in {days} days!")
                    else:
                        st.info(f"📅 Earnings in {days} days")
                
                # Historical predictions for this symbol
                if not history_df.empty:
                    fig_hist = render_symbol_history_chart(history_df, selected_symbol)
                    if fig_hist:
                        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
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
    
    # ========== TAB 4: Performance ==========
    with tab4:
        st.header("📈 Model Performance")
        
        if perf_df.empty:
            st.warning("No performance data available. Run with --track-performance flag.")
            st.code("python app/update_daily.py --track-performance", language="bash")
        else:
            valid_perf = perf_df.dropna(subset=['actual_ret_5', 'pred_ret_5'])
            
            if not valid_perf.empty:
                direction_correct = ((valid_perf['pred_ret_5'] > 0) == (valid_perf['actual_ret_5'] > 0)).mean()
                positive_rate = (valid_perf['actual_ret_5'] > 0).mean()
                avg_return = valid_perf['actual_ret_5'].mean()
                correlation = valid_perf['pred_ret_5'].corr(valid_perf['actual_ret_5'])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Direction Accuracy", f"{direction_correct:.1%}")
                col2.metric("Positive Return Rate", f"{positive_rate:.1%}")
                col3.metric("Avg Actual Return", f"{avg_return:.2%}")
                col4.metric("Pred/Actual Correlation", f"{correlation:.3f}")
                
                st.markdown("---")
                
                fig_perf = render_performance_chart(perf_df)
                if fig_perf:
                    st.plotly_chart(fig_perf, use_container_width=True)
                
                with st.expander("📊 Detailed Statistics"):
                    mae = (valid_perf['pred_ret_5'] - valid_perf['actual_ret_5']).abs().mean()
                    rmse = np.sqrt(((valid_perf['pred_ret_5'] - valid_perf['actual_ret_5']) ** 2).mean())
                    
                    st.write(f"- **Total Predictions Tracked**: {len(valid_perf)}")
                    st.write(f"- **Mean Absolute Error (MAE)**: {mae:.4f}")
                    st.write(f"- **Root Mean Square Error (RMSE)**: {rmse:.4f}")
                    st.write(f"- **Best Actual Return**: {valid_perf['actual_ret_5'].max():.2%}")
                    st.write(f"- **Worst Actual Return**: {valid_perf['actual_ret_5'].min():.2%}")
            else:
                st.info("Waiting for actual returns to be recorded (5 trading days after predictions)")
    
    # ========== TAB 5: About ==========
    with tab5:
        st.header("About This App")
        
        st.markdown("""
        ### 📈 S&P 500 Top-10 Stock Predictor
        
        This application predicts the top 10 S&P 500 stocks most likely to outperform 
        over the next 5 trading days using machine learning.
        
        #### Features
        
        - **🏆 Daily Predictions**: Top 10 stocks with predicted returns
        - **📊 Confidence Intervals**: 90% confidence ranges for each prediction
        - **📅 Earnings Calendar**: Warnings for stocks with upcoming earnings
        - **📈 Performance Tracking**: Historical accuracy metrics
        - **🔍 Deep Analysis**: Charts and detailed symbol analysis
        
        #### How It Works
        
        1. **Data Collection**: Daily OHLCV data from yfinance + Yahoo Finance news
        2. **Feature Engineering**: 40+ technical indicators and sentiment features
        3. **Two-Stage Prediction**: LightGBM ranker + regressor ensemble
        4. **Confidence Estimation**: Uncertainty quantification for each prediction
        5. **Risk Alerts**: Earnings calendar integration
        
        #### Columns Explained
        
        | Column | Description |
        |--------|-------------|
        | Symbol | Stock ticker |
        | Price | Latest closing price |
        | Prediction | Predicted 5-day return |
        | Low/High (90%) | 90% confidence interval bounds |
        | Confidence | Model's confidence in the prediction |
        | Target Price | Expected price in 5 days |
        
        #### ⚠️ Disclaimer
        
        This tool is for educational and research purposes only. It does not constitute 
        financial advice. Always do your own research before making investment decisions.
        """)
        
        st.markdown("---")
        st.subheader("🔧 System Status")
        
        if quality_report:
            with st.expander("Quality Report Details"):
                st.json(quality_report)


if __name__ == "__main__":
    main()
