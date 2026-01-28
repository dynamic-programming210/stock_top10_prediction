"""
C2: Ensemble Models
Combines multiple models to improve prediction robustness and reduce variance
Supports: Voting, Stacking, and Weighted Average ensembles
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
import pickle
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CURRENT_MODEL_VERSION, get_model_dir
from models.train import get_z_feature_cols, prepare_training_data
from utils import get_logger

logger = get_logger(__name__)


# Base model configurations
BASE_MODELS = {
    'gradient_boosting': {
        'class': GradientBoostingRegressor,
        'params': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'random_state': 42
        }
    },
    'random_forest': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_leaf': 5,
            'random_state': 43
        }
    },
    'adaboost': {
        'class': AdaBoostRegressor,
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 44
        }
    },
    'ridge': {
        'class': Ridge,
        'params': {
            'alpha': 1.0
        }
    },
    'elastic_net': {
        'class': ElasticNet,
        'params': {
            'alpha': 0.1,
            'l1_ratio': 0.5,
            'random_state': 45
        }
    }
}


class EnsembleRanker:
    """
    Ensemble model for ranking stocks
    Combines multiple base models using different strategies
    """
    
    def __init__(
        self,
        strategy: str = 'weighted_average',
        base_model_names: List[str] = None,
        weights: List[float] = None
    ):
        """
        Args:
            strategy: 'voting', 'stacking', 'weighted_average'
            base_model_names: List of base model names from BASE_MODELS
            weights: Weights for weighted_average strategy
        """
        self.strategy = strategy
        self.base_model_names = base_model_names or ['gradient_boosting', 'random_forest', 'ridge']
        self.weights = weights
        self.models = {}
        self.meta_model = None
        self.fitted = False
        self.feature_importances_ = None
        
    def _create_base_models(self) -> List[Tuple[str, object]]:
        """Create instances of base models"""
        estimators = []
        for name in self.base_model_names:
            if name not in BASE_MODELS:
                logger.warning(f"Unknown model: {name}, skipping")
                continue
            config = BASE_MODELS[name]
            model = config['class'](**config['params'])
            estimators.append((name, model))
        return estimators
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleRanker':
        """Fit the ensemble model"""
        logger.info(f"Training ensemble ({self.strategy}) with {len(self.base_model_names)} models...")
        
        if self.strategy == 'voting':
            self._fit_voting(X, y)
        elif self.strategy == 'stacking':
            self._fit_stacking(X, y)
        elif self.strategy == 'weighted_average':
            self._fit_weighted_average(X, y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.fitted = True
        logger.info("Ensemble training complete")
        return self
    
    def _fit_voting(self, X: np.ndarray, y: np.ndarray):
        """Fit using VotingRegressor"""
        estimators = self._create_base_models()
        self.ensemble = VotingRegressor(estimators=estimators)
        self.ensemble.fit(X, y)
        self._compute_feature_importances()
    
    def _fit_stacking(self, X: np.ndarray, y: np.ndarray):
        """Fit using StackingRegressor with Ridge meta-learner"""
        estimators = self._create_base_models()
        self.meta_model = Ridge(alpha=1.0)
        
        self.ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=self.meta_model,
            cv=5,
            passthrough=False
        )
        self.ensemble.fit(X, y)
        self._compute_feature_importances()
    
    def _fit_weighted_average(self, X: np.ndarray, y: np.ndarray):
        """Fit individual models and use weighted average for prediction"""
        estimators = self._create_base_models()
        
        for name, model in estimators:
            logger.info(f"  Training {name}...")
            model.fit(X, y)
            self.models[name] = model
        
        if self.weights is None:
            self.weights = self._learn_weights(X, y)
        
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()
        
        logger.info(f"  Model weights: {dict(zip(self.base_model_names, self.weights))}")
        self._compute_feature_importances()
    
    def _learn_weights(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> np.ndarray:
        """Learn optimal weights via cross-validation"""
        from sklearn.model_selection import KFold
        from scipy.optimize import minimize
        
        n_models = len(self.models)
        fold_predictions = np.zeros((len(y), n_models))
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for model_idx, (name, model) in enumerate(self.models.items()):
                config = BASE_MODELS[name]
                cv_model = config['class'](**config['params'])
                cv_model.fit(X_train, y_train)
                fold_predictions[val_idx, model_idx] = cv_model.predict(X_val)
        
        def objective(w):
            w = np.abs(w)
            w = w / w.sum()
            combined = (fold_predictions * w).sum(axis=1)
            return np.mean((y - combined) ** 2)
        
        w0 = np.ones(n_models) / n_models
        result = minimize(objective, w0, method='Nelder-Mead')
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        return optimal_weights
    
    def _compute_feature_importances(self):
        """Compute weighted average of feature importances"""
        importances = []
        weights_for_importance = []
        
        if self.strategy == 'weighted_average':
            for (name, model), weight in zip(self.models.items(), self.weights):
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
                    weights_for_importance.append(weight)
        else:
            estimators = self.ensemble.estimators_ if hasattr(self.ensemble, 'estimators_') else []
            for est in estimators:
                if hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
                    weights_for_importance.append(1.0)
        
        if importances:
            weights_for_importance = np.array(weights_for_importance)
            weights_for_importance = weights_for_importance / weights_for_importance.sum()
            self.feature_importances_ = np.average(np.array(importances), axis=0, weights=weights_for_importance)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.strategy == 'weighted_average':
            predictions = np.zeros((len(X), len(self.models)))
            for idx, (name, model) in enumerate(self.models.items()):
                predictions[:, idx] = model.predict(X)
            return (predictions * self.weights).sum(axis=1)
        else:
            return self.ensemble.predict(X)
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each base model"""
        if self.strategy == 'weighted_average':
            return {name: model.predict(X) for name, model in self.models.items()}
        else:
            results = {}
            for name, est in zip(self.base_model_names, self.ensemble.estimators_):
                results[name] = est.predict(X)
            return results


def train_ensemble_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str] = None,
    strategy: str = 'weighted_average',
    base_models: List[str] = None
) -> Tuple[EnsembleRanker, dict]:
    """Train an ensemble ranker"""
    if feature_cols is None:
        feature_cols = get_z_feature_cols()
    
    valid_features = [c for c in feature_cols if c in train_df.columns]
    
    X_train = train_df[valid_features].values
    y_train = train_df['fwd_ret_5'].values
    X_val = val_df[valid_features].values
    y_val = val_df['fwd_ret_5'].values
    
    ensemble = EnsembleRanker(strategy=strategy, base_model_names=base_models)
    ensemble.fit(X_train, y_train)
    
    train_pred = ensemble.predict(X_train)
    val_pred = ensemble.predict(X_val)
    
    train_corr = np.corrcoef(train_pred, y_train)[0, 1]
    val_corr = np.corrcoef(val_pred, y_val)[0, 1]
    
    metrics = {
        'strategy': strategy,
        'base_models': ensemble.base_model_names,
        'weights': ensemble.weights.tolist() if ensemble.weights is not None else None,
        'train_corr': float(train_corr),
        'val_corr': float(val_corr),
        'train_rmse': float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
        'val_rmse': float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    }
    
    if strategy == 'weighted_average':
        individual_preds = ensemble.get_individual_predictions(X_val)
        metrics['individual_val_corr'] = {
            name: float(np.corrcoef(pred, y_val)[0, 1])
            for name, pred in individual_preds.items()
        }
    
    logger.info(f"Ensemble training complete. Val Corr: {val_corr:.4f}")
    return ensemble, metrics


def save_ensemble(ensemble: EnsembleRanker, metrics: dict, feature_cols: List[str], version: str = None) -> Path:
    """Save ensemble model"""
    version = version or CURRENT_MODEL_VERSION
    model_dir = get_model_dir(version)
    
    ensemble_path = model_dir / 'ensemble.pkl'
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble, f)
    logger.info(f"Saved ensemble to {ensemble_path}")
    
    metrics_path = model_dir / 'ensemble_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return model_dir


def load_ensemble(version: str = None) -> Tuple[EnsembleRanker, dict]:
    """Load ensemble model"""
    version = version or CURRENT_MODEL_VERSION
    model_dir = get_model_dir(version)
    
    ensemble_path = model_dir / 'ensemble.pkl'
    if not ensemble_path.exists():
        raise FileNotFoundError(f"Ensemble not found: {ensemble_path}")
    
    with open(ensemble_path, 'rb') as f:
        ensemble = pickle.load(f)
    
    metrics = {}
    metrics_path = model_dir / 'ensemble_metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    return ensemble, metrics


def compare_models(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """Compare individual models and ensemble strategies"""
    if feature_cols is None:
        feature_cols = get_z_feature_cols()
    
    valid_features = [c for c in feature_cols if c in train_df.columns]
    
    X_train = train_df[valid_features].values
    y_train = train_df['fwd_ret_5'].values
    X_val = val_df[valid_features].values
    y_val = val_df['fwd_ret_5'].values
    
    results = []
    
    logger.info("Comparing individual models...")
    for name, config in BASE_MODELS.items():
        model = config['class'](**config['params'])
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_corr = np.corrcoef(val_pred, y_val)[0, 1]
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        
        results.append({'model': name, 'type': 'individual', 'val_corr': val_corr, 'val_rmse': val_rmse})
        logger.info(f"  {name}: corr={val_corr:.4f}, rmse={val_rmse:.4f}")
    
    logger.info("\nComparing ensemble strategies...")
    for strategy in ['voting', 'stacking', 'weighted_average']:
        try:
            ensemble = EnsembleRanker(strategy=strategy, base_model_names=['gradient_boosting', 'random_forest', 'ridge'])
            ensemble.fit(X_train, y_train)
            
            val_pred = ensemble.predict(X_val)
            val_corr = np.corrcoef(val_pred, y_val)[0, 1]
            val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
            
            results.append({'model': f'ensemble_{strategy}', 'type': 'ensemble', 'val_corr': val_corr, 'val_rmse': val_rmse})
            logger.info(f"  {strategy}: corr={val_corr:.4f}, rmse={val_rmse:.4f}")
        except Exception as e:
            logger.warning(f"  {strategy} failed: {e}")
    
    return pd.DataFrame(results).sort_values('val_corr', ascending=False)


if __name__ == "__main__":
    from features.build_features import load_features
    
    features = load_features()
    
    if not features.empty:
        print("Comparing models and ensembles...\n")
        train_df, val_df = prepare_training_data(features)
        comparison = compare_models(train_df, val_df)
        print("\n" + "=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)
        print(comparison.to_string(index=False))
        
        print("\n\nTraining weighted_average ensemble...")
        ensemble, metrics = train_ensemble_ranker(train_df, val_df, strategy='weighted_average')
        print(f"\nEnsemble metrics:")
        print(json.dumps(metrics, indent=2))
        
        save_ensemble(ensemble, metrics, get_z_feature_cols())
        print("\nEnsemble saved!")
    else:
        print("No features found. Run feature building first.")
