"""
Model training and inference for stock ranking/prediction
Two-stage model: Gradient Boosting Ranker + Regressor using scikit-learn
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pickle
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURE_COLS, CURRENT_MODEL_VERSION, CAND_K, TOP_N, get_model_dir
)
from utils import get_logger

logger = get_logger(__name__)

# Model parameters
RANKER_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': 42
}

REGRESSOR_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': 42
}


def get_z_feature_cols() -> List[str]:
    """Get z-scored feature column names"""
    return [f'{c}_z' for c in FEATURE_COLS]


def prepare_training_data(
    df: pd.DataFrame,
    train_end_date: str = None,
    min_samples_per_date: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training data with train/validation split
    
    Args:
        df: Feature dataframe with targets
        train_end_date: End date for training (rest is validation)
        min_samples_per_date: Minimum symbols per date to include
        
    Returns:
        (train_df, val_df)
    """
    feature_cols = get_z_feature_cols()
    target_col = 'fwd_ret_5'
    
    # Filter to rows with valid features and target
    valid_features = [c for c in feature_cols if c in df.columns]
    required_cols = valid_features + [target_col, 'date', 'symbol']
    
    df = df[required_cols].dropna()
    
    # Filter dates with enough samples
    date_counts = df.groupby('date').size()
    valid_dates = date_counts[date_counts >= min_samples_per_date].index
    df = df[df['date'].isin(valid_dates)].copy()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Split train/validation
    if train_end_date:
        train_mask = df['date'] <= pd.Timestamp(train_end_date)
    else:
        # Use last 20% of dates for validation
        unique_dates = df['date'].unique()
        split_idx = int(len(unique_dates) * 0.8)
        split_date = unique_dates[split_idx]
        train_mask = df['date'] < split_date
    
    train_df = df[train_mask].copy()
    val_df = df[~train_mask].copy()
    
    logger.info(f"Training data: {len(train_df)} samples, {train_df['date'].nunique()} dates")
    logger.info(f"Validation data: {len(val_df)} samples, {val_df['date'].nunique()} dates")
    
    return train_df, val_df


def create_ranking_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Create group sizes for LightGBM ranker
    Each date is a separate group
    """
    return df.groupby('date').size().values


def create_ranking_labels(df: pd.DataFrame, target_col: str = 'fwd_ret_5') -> np.ndarray:
    """
    Create ranking labels from forward returns
    Labels are relative ranks within each date (higher = better)
    """
    # Rank within each date (higher return = higher rank)
    labels = df.groupby('date')[target_col].rank(method='dense').values
    return labels.astype(int)


def train_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str] = None,
    params: dict = None
) -> Tuple[GradientBoostingRegressor, dict]:
    """
    Train ranking model using GradientBoostingRegressor
    (predicts ranking score based on future returns)
    
    Returns:
        (model, metrics_dict)
    """
    if feature_cols is None:
        feature_cols = get_z_feature_cols()
    if params is None:
        params = RANKER_PARAMS.copy()
    
    valid_features = [c for c in feature_cols if c in train_df.columns]
    
    # Prepare training data - use forward returns as target for ranking
    X_train = train_df[valid_features].values
    y_train = train_df['fwd_ret_5'].values  # Use actual returns as ranking target
    
    # Prepare validation data
    X_val = val_df[valid_features].values
    y_val = val_df['fwd_ret_5'].values
    
    # Train model
    logger.info("Training ranker...")
    
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    # Compute metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Compute ranking metrics (correlation with actual returns)
    train_corr = np.corrcoef(train_pred, y_train)[0, 1]
    val_corr = np.corrcoef(val_pred, y_val)[0, 1]
    
    metrics = {
        'n_estimators': params.get('n_estimators', 100),
        'train_corr': float(train_corr),
        'val_corr': float(val_corr),
        'feature_importance': dict(zip(valid_features, model.feature_importances_.tolist()))
    }
    
    logger.info(f"Ranker training complete. Train Corr: {train_corr:.4f}, Val Corr: {val_corr:.4f}")
    
    return model, metrics


def train_regressor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str] = None,
    params: dict = None
) -> Tuple[GradientBoostingRegressor, dict]:
    """
    Train regression model to predict forward returns
    
    Returns:
        (model, metrics_dict)
    """
    if feature_cols is None:
        feature_cols = get_z_feature_cols()
    if params is None:
        params = REGRESSOR_PARAMS.copy()
    
    valid_features = [c for c in feature_cols if c in train_df.columns]
    target_col = 'fwd_ret_5'
    
    # Prepare data
    X_train = train_df[valid_features].values
    y_train = train_df[target_col].values
    
    X_val = val_df[valid_features].values
    y_val = val_df[target_col].values
    
    # Train model
    logger.info("Training regressor...")
    
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    
    # Compute metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    metrics = {
        'n_estimators': params.get('n_estimators', 100),
        'train_rmse': float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
        'val_rmse': float(np.sqrt(np.mean((val_pred - y_val) ** 2))),
        'train_corr': float(np.corrcoef(train_pred, y_train)[0, 1]),
        'val_corr': float(np.corrcoef(val_pred, y_val)[0, 1]),
        'feature_importance': dict(zip(valid_features, model.feature_importances_.tolist()))
    }
    
    logger.info(f"Regressor training complete. Val RMSE: {metrics['val_rmse']:.4f}, Val Corr: {metrics['val_corr']:.4f}")
    
    return model, metrics


def save_model(
    ranker: GradientBoostingRegressor,
    regressor: GradientBoostingRegressor,
    metrics: dict,
    feature_cols: List[str],
    version: str = None
) -> Path:
    """
    Save model artifacts to versioned directory
    """
    version = version or CURRENT_MODEL_VERSION
    model_dir = get_model_dir(version)
    
    # Save ranker
    ranker_path = model_dir / 'ranker.pkl'
    with open(ranker_path, 'wb') as f:
        pickle.dump(ranker, f)
    logger.info(f"Saved ranker to {ranker_path}")
    
    # Save regressor
    if regressor is not None:
        reg_path = model_dir / 'reg.pkl'
        with open(reg_path, 'wb') as f:
            pickle.dump(regressor, f)
        logger.info(f"Saved regressor to {reg_path}")
    
    # Save schema
    schema = {
        'version': version,
        'feature_cols': feature_cols,
        'created_at': datetime.now().isoformat(),
        'cand_k': CAND_K,
        'top_n': TOP_N
    }
    schema_path = model_dir / 'schema.json'
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Saved schema to {schema_path}")
    
    # Save metrics
    metrics_path = model_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return model_dir


def load_model(version: str = None) -> Tuple[GradientBoostingRegressor, Optional[GradientBoostingRegressor], dict]:
    """
    Load model artifacts
    
    Returns:
        (ranker, regressor_or_None, schema)
    """
    version = version or CURRENT_MODEL_VERSION
    model_dir = get_model_dir(version)
    
    # Load ranker
    ranker_path = model_dir / 'ranker.pkl'
    if not ranker_path.exists():
        raise FileNotFoundError(f"Ranker not found: {ranker_path}")
    
    with open(ranker_path, 'rb') as f:
        ranker = pickle.load(f)
    
    # Load regressor (optional)
    reg_path = model_dir / 'reg.pkl'
    regressor = None
    if reg_path.exists():
        with open(reg_path, 'rb') as f:
            regressor = pickle.load(f)
    
    # Load schema
    schema_path = model_dir / 'schema.json'
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    logger.info(f"Loaded model version {version}")
    
    return ranker, regressor, schema


def predict_rankings(
    ranker: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Predict ranking scores for all samples
    """
    valid_features = [c for c in feature_cols if c in df.columns]
    X = df[valid_features].values
    
    df = df.copy()
    df['rank_score'] = ranker.predict(X)
    
    return df


def predict_returns(
    regressor: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Predict forward returns
    """
    if regressor is None:
        df = df.copy()
        df['pred_ret_5'] = df['rank_score'] / df['rank_score'].max() * 0.1  # Proxy
        return df
    
    valid_features = [c for c in feature_cols if c in df.columns]
    X = df[valid_features].values
    
    df = df.copy()
    df['pred_ret_5'] = regressor.predict(X)
    
    return df


def generate_explanation(row: pd.Series, feature_cols: List[str]) -> Tuple[str, str]:
    """
    Generate human-readable explanation without SHAP
    Uses z-scored features to determine signal type
    """
    signals = []
    risk_notes = []
    
    # Momentum signals
    if 'ret_5_z' in row and pd.notna(row['ret_5_z']):
        if row['ret_5_z'] > 1.0:
            signals.append("strong momentum")
        elif row['ret_5_z'] < -1.0:
            signals.append("mean-reversion candidate")
    
    # Volume signals
    if 'vol_ratio_z' in row and pd.notna(row['vol_ratio_z']):
        if row['vol_ratio_z'] > 1.5:
            signals.append("volume surge")
        elif row['vol_ratio_z'] < -1.0:
            signals.append("low volume")
    
    # Volatility risk
    if 'vol_10_z' in row and pd.notna(row['vol_10_z']):
        if row['vol_10_z'] > 1.5:
            risk_notes.append("high volatility")
    
    # Candlestick patterns
    if 'close_pos_z' in row and pd.notna(row['close_pos_z']):
        if row['close_pos_z'] > 1.0:
            signals.append("bullish close")
        elif row['close_pos_z'] < -1.0:
            signals.append("bearish close")
    
    # Build evidence string
    evidence_parts = []
    for col in ['ret_5_z', 'vol_ratio_z', 'vol_10_z']:
        if col in row and pd.notna(row[col]):
            evidence_parts.append(f"{col.replace('_z', '')}={row[col]:.2f}")
    evidence_short = "; ".join(evidence_parts[:4])
    
    # Build human explanation
    if signals:
        reason = "Selected due to " + ", ".join(signals[:3])
    else:
        reason = "Selected by model ranking"
    
    if risk_notes:
        reason += f". Note: {', '.join(risk_notes)}"
    
    return evidence_short, reason


def generate_top10(
    df: pd.DataFrame,
    ranker,
    regressor,
    feature_cols: List[str],
    asof_date: pd.Timestamp,
    cand_k: int = CAND_K,
    top_n: int = TOP_N
) -> pd.DataFrame:
    """
    Generate top-10 predictions for a given date
    
    Args:
        df: Feature dataframe
        ranker: Ranking model
        regressor: Regression model (optional)
        feature_cols: Z-scored feature columns
        asof_date: Date to generate predictions for
        cand_k: Number of candidates from ranker
        top_n: Final number of picks
        
    Returns:
        DataFrame with top-10 predictions
    """
    # Filter to asof_date
    date_df = df[df['date'] == asof_date].copy()
    
    if date_df.empty:
        logger.warning(f"No data for date {asof_date}")
        return pd.DataFrame()
    
    # Filter to rows with valid features
    valid_features = [c for c in feature_cols if c in date_df.columns]
    date_df = date_df.dropna(subset=valid_features)
    
    logger.info(f"Generating top-10 for {asof_date} with {len(date_df)} eligible symbols")
    
    # Step 1: Ranking scores
    date_df = predict_rankings(ranker, date_df, feature_cols)
    
    # Step 2: Take top candidates by rank score
    candidates = date_df.nlargest(cand_k, 'rank_score')
    
    # Step 3: Predict returns
    candidates = predict_returns(regressor, candidates, feature_cols)
    
    # Step 4: Final top-N by predicted return
    top10 = candidates.nlargest(top_n, 'pred_ret_5').copy()
    
    # Add computed columns
    top10['pred_price_5d'] = top10['close'] * (1 + top10['pred_ret_5'])
    
    # Generate explanations
    explanations = top10.apply(lambda row: generate_explanation(row, feature_cols), axis=1)
    top10['evidence_short'] = explanations.apply(lambda x: x[0])
    top10['reason_human'] = explanations.apply(lambda x: x[1])
    
    # Select output columns
    output_cols = [
        'date', 'symbol', 'close', 'rank_score', 'pred_ret_5',
        'pred_price_5d', 'evidence_short', 'reason_human'
    ]
    
    # Add any available z-features for reference
    for col in valid_features[:5]:
        if col in top10.columns:
            output_cols.append(col)
    
    top10 = top10[[c for c in output_cols if c in top10.columns]]
    top10 = top10.sort_values('pred_ret_5', ascending=False).reset_index(drop=True)
    
    return top10


def train_full_pipeline(
    features_df: pd.DataFrame,
    version: str = None
) -> Tuple[GradientBoostingRegressor, GradientBoostingRegressor, dict, Path]:
    """
    Complete training pipeline
    
    Returns:
        (ranker, regressor, metrics, model_dir)
    """
    version = version or CURRENT_MODEL_VERSION
    feature_cols = get_z_feature_cols()
    
    # Prepare data
    train_df, val_df = prepare_training_data(features_df)
    
    if train_df.empty or val_df.empty:
        raise ValueError("Insufficient data for training")
    
    # Train models
    ranker, ranker_metrics = train_ranker(train_df, val_df, feature_cols)
    regressor, reg_metrics = train_regressor(train_df, val_df, feature_cols)
    
    # Combine metrics
    metrics = {
        'ranker': ranker_metrics,
        'regressor': reg_metrics,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'train_dates': train_df['date'].nunique(),
        'val_dates': val_df['date'].nunique()
    }
    
    # Save
    model_dir = save_model(ranker, regressor, metrics, feature_cols, version)
    
    return ranker, regressor, metrics, model_dir


if __name__ == "__main__":
    # Test training
    from features import load_features
    
    features = load_features()
    
    if not features.empty:
        print(f"Loaded features: {features.shape}")
        
        ranker, regressor, metrics, model_dir = train_full_pipeline(features)
        
        print(f"\nTraining complete!")
        print(f"Model saved to: {model_dir}")
        print(f"\nMetrics:")
        print(json.dumps(metrics, indent=2, default=str))
    else:
        print("No features found. Run feature building first.")
