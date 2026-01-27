"""
C1: Hyperparameter tuning for stock prediction models
Uses scikit-learn's GridSearchCV and RandomizedSearchCV with time-series splits
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, get_model_dir, OUTPUTS_DIR
from utils import get_logger

logger = get_logger(__name__)

# Hyperparameter search spaces
RANKER_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Smaller grid for quick tuning
RANKER_PARAM_GRID_SMALL = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.03, 0.05],
    'subsample': [0.8],
}

# Random search distributions (wider exploration)
RANKER_PARAM_DISTRIBUTIONS = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}


def correlation_scorer(y_true, y_pred):
    """
    Custom scorer: Pearson correlation between predictions and actual returns
    Higher correlation = better ranking ability
    """
    if len(y_true) < 2:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def get_z_feature_cols() -> List[str]:
    """Get z-scored feature column names"""
    return [f'{c}_z' for c in FEATURE_COLS]


def prepare_tuning_data(
    df: pd.DataFrame,
    min_samples_per_date: int = 30
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for hyperparameter tuning
    
    Returns:
        (X, y, feature_cols)
    """
    feature_cols = get_z_feature_cols()
    target_col = 'fwd_ret_5'
    
    # Filter to valid rows
    valid_features = [c for c in feature_cols if c in df.columns]
    required_cols = valid_features + [target_col, 'date', 'symbol']
    df = df[required_cols].dropna()
    
    # Filter dates with enough samples
    date_counts = df.groupby('date').size()
    valid_dates = date_counts[date_counts >= min_samples_per_date].index
    df = df[df['date'].isin(valid_dates)].copy()
    
    # Sort by date (important for time series split)
    df = df.sort_values('date')
    
    X = df[valid_features].values
    y = df[target_col].values
    
    return X, y, valid_features


def tune_with_grid_search(
    df: pd.DataFrame,
    param_grid: Dict = None,
    n_splits: int = 3,
    n_jobs: int = -1,
    verbose: int = 1
) -> Dict:
    """
    Tune hyperparameters using GridSearchCV with TimeSeriesSplit
    
    Args:
        df: Feature dataframe
        param_grid: Parameter grid (default: RANKER_PARAM_GRID_SMALL)
        n_splits: Number of time series splits
        n_jobs: Parallel jobs (-1 = all cores)
        verbose: Verbosity level
        
    Returns:
        Dict with best params, scores, and CV results
    """
    if param_grid is None:
        param_grid = RANKER_PARAM_GRID_SMALL
    
    X, y, feature_cols = prepare_tuning_data(df)
    
    logger.info(f"Starting GridSearchCV with {n_splits} time-series splits...")
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Parameter grid: {param_grid}")
    
    # Count total combinations
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    logger.info(f"Total parameter combinations: {n_combinations}")
    
    # Create time series cross-validator
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Create scorer
    scorer = make_scorer(correlation_scorer)
    
    # Create base model
    base_model = GradientBoostingRegressor(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    start_time = datetime.now()
    grid_search.fit(X, y)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"GridSearchCV complete in {elapsed:.1f}s")
    logger.info(f"Best score: {grid_search.best_score_:.4f}")
    logger.info(f"Best params: {grid_search.best_params_}")
    
    # Compile results
    results = {
        'method': 'GridSearchCV',
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'elapsed_seconds': elapsed,
        'n_splits': n_splits,
        'n_combinations': n_combinations,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
            'mean_train_score': grid_search.cv_results_['mean_train_score'].tolist(),
            'params': [str(p) for p in grid_search.cv_results_['params']]
        }
    }
    
    return results


def tune_with_random_search(
    df: pd.DataFrame,
    param_distributions: Dict = None,
    n_iter: int = 50,
    n_splits: int = 3,
    n_jobs: int = -1,
    verbose: int = 1
) -> Dict:
    """
    Tune hyperparameters using RandomizedSearchCV with TimeSeriesSplit
    
    Args:
        df: Feature dataframe
        param_distributions: Parameter distributions
        n_iter: Number of random combinations to try
        n_splits: Number of time series splits
        n_jobs: Parallel jobs
        verbose: Verbosity level
        
    Returns:
        Dict with best params, scores, and CV results
    """
    if param_distributions is None:
        param_distributions = RANKER_PARAM_DISTRIBUTIONS
    
    X, y, feature_cols = prepare_tuning_data(df)
    
    logger.info(f"Starting RandomizedSearchCV with {n_iter} iterations, {n_splits} splits...")
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create time series cross-validator
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Create scorer
    scorer = make_scorer(correlation_scorer)
    
    # Create base model
    base_model = GradientBoostingRegressor(random_state=42)
    
    # Random search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        random_state=42
    )
    
    start_time = datetime.now()
    random_search.fit(X, y)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"RandomizedSearchCV complete in {elapsed:.1f}s")
    logger.info(f"Best score: {random_search.best_score_:.4f}")
    logger.info(f"Best params: {random_search.best_params_}")
    
    # Compile results
    results = {
        'method': 'RandomizedSearchCV',
        'best_params': random_search.best_params_,
        'best_score': float(random_search.best_score_),
        'elapsed_seconds': elapsed,
        'n_splits': n_splits,
        'n_iter': n_iter,
        'cv_results': {
            'mean_test_score': random_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': random_search.cv_results_['std_test_score'].tolist(),
            'mean_train_score': random_search.cv_results_['mean_train_score'].tolist(),
            'params': [str(p) for p in random_search.cv_results_['params']]
        }
    }
    
    return results


def auto_tune(
    df: pd.DataFrame,
    method: str = 'random',
    quick: bool = True
) -> Dict:
    """
    Automatic hyperparameter tuning with sensible defaults
    
    Args:
        df: Feature dataframe
        method: 'grid' or 'random'
        quick: If True, use smaller search space for faster results
        
    Returns:
        Tuning results dict
    """
    logger.info(f"Auto-tuning with method={method}, quick={quick}")
    
    if method == 'grid':
        param_grid = RANKER_PARAM_GRID_SMALL if quick else RANKER_PARAM_GRID
        results = tune_with_grid_search(df, param_grid=param_grid)
    else:
        n_iter = 20 if quick else 50
        results = tune_with_random_search(df, n_iter=n_iter)
    
    # Save results
    output_file = OUTPUTS_DIR / f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved tuning results to {output_file}")
    
    return results


def apply_best_params(best_params: Dict) -> None:
    """
    Apply best parameters to the model configuration
    Updates RANKER_PARAMS and REGRESSOR_PARAMS in train.py
    
    Note: This prints the params to be manually copied to train.py
    """
    logger.info("Best parameters to apply:")
    logger.info("=" * 40)
    
    # Format for train.py
    print("\n# C1: Tuned parameters - copy to models/train.py")
    print("RANKER_PARAMS = {")
    for key, value in best_params.items():
        if isinstance(value, str):
            print(f"    '{key}': '{value}',")
        else:
            print(f"    '{key}': {value},")
    print("    'random_state': 42")
    print("}")
    print("\nREGRESSOR_PARAMS = RANKER_PARAMS.copy()  # Use same params for regressor")


def compare_default_vs_tuned(
    df: pd.DataFrame,
    tuned_params: Dict,
    n_splits: int = 3
) -> Dict:
    """
    Compare default parameters vs tuned parameters using walk-forward validation
    
    Returns:
        Comparison results
    """
    from models.train import RANKER_PARAMS, walk_forward_validation
    
    logger.info("Comparing default vs tuned parameters...")
    
    # Validate with default params
    logger.info("\n[Default Parameters]")
    # We'll run a simple comparison using cross-validation
    X, y, feature_cols = prepare_tuning_data(df)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Default model
    default_scores = []
    for train_idx, test_idx in tscv.split(X):
        model = GradientBoostingRegressor(**RANKER_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        score = correlation_scorer(y[test_idx], pred)
        default_scores.append(score)
    
    # Tuned model
    tuned_scores = []
    tuned_params_full = tuned_params.copy()
    tuned_params_full['random_state'] = 42
    
    for train_idx, test_idx in tscv.split(X):
        model = GradientBoostingRegressor(**tuned_params_full)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        score = correlation_scorer(y[test_idx], pred)
        tuned_scores.append(score)
    
    comparison = {
        'default': {
            'params': RANKER_PARAMS,
            'mean_score': float(np.mean(default_scores)),
            'std_score': float(np.std(default_scores)),
            'fold_scores': default_scores
        },
        'tuned': {
            'params': tuned_params,
            'mean_score': float(np.mean(tuned_scores)),
            'std_score': float(np.std(tuned_scores)),
            'fold_scores': tuned_scores
        },
        'improvement': float(np.mean(tuned_scores) - np.mean(default_scores))
    }
    
    logger.info(f"\nDefault: {comparison['default']['mean_score']:.4f} ± {comparison['default']['std_score']:.4f}")
    logger.info(f"Tuned:   {comparison['tuned']['mean_score']:.4f} ± {comparison['tuned']['std_score']:.4f}")
    logger.info(f"Improvement: {comparison['improvement']:+.4f}")
    
    return comparison


if __name__ == "__main__":
    import argparse
    from features.build_features import load_features
    
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument('--method', type=str, default='random', choices=['grid', 'random'],
                        help='Tuning method (default: random)')
    parser.add_argument('--quick', action='store_true', help='Use smaller search space')
    parser.add_argument('--n-iter', type=int, default=30, help='Iterations for random search')
    parser.add_argument('--compare', action='store_true', help='Compare default vs tuned')
    
    args = parser.parse_args()
    
    # Load features
    features = load_features()
    
    if features.empty:
        print("No features found. Run feature building first.")
        exit(1)
    
    print(f"Loaded features: {features.shape}")
    
    # Run tuning
    if args.method == 'grid':
        results = tune_with_grid_search(
            features, 
            param_grid=RANKER_PARAM_GRID_SMALL if args.quick else RANKER_PARAM_GRID
        )
    else:
        results = tune_with_random_search(features, n_iter=args.n_iter)
    
    print(f"\nBest Score: {results['best_score']:.4f}")
    print(f"Best Params: {results['best_params']}")
    
    # Print params to copy
    apply_best_params(results['best_params'])
    
    # Compare if requested
    if args.compare:
        comparison = compare_default_vs_tuned(features, results['best_params'])
        print(f"\nImprovement: {comparison['improvement']:+.4f}")
