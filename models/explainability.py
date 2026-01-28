"""
C7: SHAP Explainability
Provides model interpretability using SHAP (SHapley Additive exPlanations)
Generates feature importance explanations for individual predictions and overall model behavior
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import warnings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR, get_model_dir, CURRENT_MODEL_VERSION
from models.train import get_z_feature_cols
from utils import get_logger

logger = get_logger(__name__)

EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"
EXPLANATIONS_DIR.mkdir(exist_ok=True)


def check_shap_available() -> bool:
    """Check if SHAP is installed"""
    try:
        import shap
        return True
    except ImportError:
        return False


class SHAPExplainer:
    """SHAP-based model explainability for local and global explanations"""
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or []
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
    def initialize_explainer(self, background_data: np.ndarray, max_background: int = 100) -> 'SHAPExplainer':
        """Initialize SHAP explainer with background data"""
        if not check_shap_available():
            logger.warning("SHAP not installed. Install with: pip install shap")
            return self
        
        import shap
        
        if len(background_data) > max_background:
            indices = np.random.choice(len(background_data), max_background, replace=False)
            background_data = background_data[indices]
        
        self.background_data = background_data
        model_type = type(self.model).__name__
        
        try:
            if model_type in ['GradientBoostingRegressor', 'GradientBoostingClassifier',
                             'RandomForestRegressor', 'RandomForestClassifier']:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info(f"Initialized TreeExplainer for {model_type}")
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, background_data)
                logger.info(f"Initialized KernelExplainer for {model_type}")
        except Exception as e:
            logger.warning(f"Failed to create TreeExplainer, falling back to Kernel: {e}")
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
        
        return self
    
    def explain_predictions(self, X: np.ndarray, max_samples: int = 1000) -> np.ndarray:
        """Compute SHAP values for predictions"""
        if self.explainer is None:
            logger.error("Explainer not initialized. Call initialize_explainer() first.")
            return None
        
        if len(X) > max_samples:
            logger.info(f"Subsampling {max_samples} of {len(X)} samples for SHAP")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.shap_values = self.explainer.shap_values(X)
        
        logger.info("SHAP values computed")
        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from SHAP values"""
        if self.shap_values is None:
            logger.error("No SHAP values. Call explain_predictions() first.")
            return pd.DataFrame()
        
        importance = np.abs(self.shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def explain_single_prediction(self, X_single: np.ndarray, top_n: int = 5) -> Dict:
        """Explain a single prediction"""
        if self.explainer is None:
            return self._fallback_explanation(X_single, top_n)
        
        X_single = X_single.reshape(1, -1) if X_single.ndim == 1 else X_single
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_vals = self.explainer.shap_values(X_single)[0]
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        prediction = self.model.predict(X_single)[0]
        
        contributions = list(zip(self.feature_names, shap_vals))
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        
        positive_drivers = [(f, v) for f, v in contributions_sorted if v > 0][:top_n]
        negative_drivers = [(f, v) for f, v in contributions_sorted if v < 0][:top_n]
        
        return {
            'prediction': float(prediction),
            'base_value': float(base_value),
            'top_positive_features': [{'feature': f, 'contribution': float(v)} for f, v in positive_drivers],
            'top_negative_features': [{'feature': f, 'contribution': float(v)} for f, v in negative_drivers],
            'all_contributions': {f: float(v) for f, v in contributions}
        }
    
    def _fallback_explanation(self, X_single: np.ndarray, top_n: int) -> Dict:
        """Fallback explanation without SHAP"""
        X_single = X_single.reshape(1, -1) if X_single.ndim == 1 else X_single
        prediction = self.model.predict(X_single)[0]
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            contributions = list(zip(self.feature_names, importance))
            contributions_sorted = sorted(contributions, key=lambda x: x[1], reverse=True)
            return {
                'prediction': float(prediction),
                'method': 'feature_importance_fallback',
                'top_features': [{'feature': f, 'importance': float(v)} for f, v in contributions_sorted[:top_n]]
            }
        
        return {'prediction': float(prediction), 'method': 'no_explanation'}
    
    def generate_human_explanation(self, X_single: np.ndarray, symbol: str = None) -> str:
        """Generate human-readable explanation"""
        explanation = self.explain_single_prediction(X_single)
        
        pred = explanation['prediction']
        direction = "bullish" if pred > 0 else "bearish"
        
        lines = []
        if symbol:
            lines.append(f"**{symbol}** - Predicted return: {pred*100:.2f}% ({direction})")
        else:
            lines.append(f"Predicted return: {pred*100:.2f}% ({direction})")
        
        lines.append("\n**Key Drivers:**")
        
        if 'top_positive_features' in explanation:
            for item in explanation['top_positive_features'][:3]:
                feature = item['feature'].replace('_z', '').replace('_', ' ')
                contrib = item['contribution'] * 100
                lines.append(f"  + {feature}: +{contrib:.2f}%")
        
        if 'top_negative_features' in explanation:
            for item in explanation['top_negative_features'][:3]:
                feature = item['feature'].replace('_z', '').replace('_', ' ')
                contrib = item['contribution'] * 100
                lines.append(f"  - {feature}: {contrib:.2f}%")
        
        return "\n".join(lines)


def generate_shap_report(model, features_df: pd.DataFrame, feature_cols: List[str], 
                         asof_date: pd.Timestamp = None, save_path: str = None) -> Dict:
    """Generate comprehensive SHAP analysis report"""
    valid_features = [c for c in feature_cols if c in features_df.columns]
    
    if asof_date is not None:
        df = features_df[features_df['date'] == asof_date]
    else:
        df = features_df
    
    df = df.dropna(subset=valid_features)
    
    if df.empty:
        logger.warning("No valid data for SHAP analysis")
        return {}
    
    X = df[valid_features].values
    
    explainer = SHAPExplainer(model, valid_features)
    explainer.initialize_explainer(X)
    
    shap_values = explainer.explain_predictions(X)
    
    if shap_values is None:
        return {}
    
    importance_df = explainer.get_feature_importance()
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'asof_date': str(asof_date) if asof_date else 'all',
        'n_samples': len(X),
        'n_features': len(valid_features),
        'global_importance': importance_df.to_dict('records'),
        'top_10_features': importance_df.head(10).to_dict('records')
    }
    
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = EXPLANATIONS_DIR / f"shap_report_{timestamp}.json"
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"SHAP report saved to {save_path}")
    return report


def explain_top10_predictions(model, top10_df: pd.DataFrame, features_df: pd.DataFrame, 
                              feature_cols: List[str]) -> pd.DataFrame:
    """Add SHAP explanations to Top-10 predictions"""
    valid_features = [c for c in feature_cols if c in features_df.columns]
    
    top10_with_features = top10_df.merge(
        features_df[['date', 'symbol'] + valid_features],
        on=['date', 'symbol'],
        how='left'
    )
    
    X = top10_with_features[valid_features].values
    
    explainer = SHAPExplainer(model, valid_features)
    background = features_df[valid_features].dropna().values
    explainer.initialize_explainer(background)
    
    explanations = []
    top_drivers = []
    
    for i in range(len(X)):
        exp = explainer.explain_single_prediction(X[i])
        explanations.append(exp)
        
        drivers = []
        if 'top_positive_features' in exp:
            for item in exp['top_positive_features'][:2]:
                drivers.append(f"+{item['feature'].replace('_z', '')}")
        if 'top_negative_features' in exp:
            for item in exp['top_negative_features'][:1]:
                drivers.append(f"-{item['feature'].replace('_z', '')}")
        
        top_drivers.append(", ".join(drivers) if drivers else "model ranking")
    
    result = top10_df.copy()
    result['shap_explanation'] = top_drivers
    result['shap_details'] = [json.dumps(e) for e in explanations]
    
    return result


def plot_shap_summary(model, X: np.ndarray, feature_names: List[str], 
                      save_path: str = None, show: bool = True) -> Optional[str]:
    """Generate SHAP summary plot"""
    if not check_shap_available():
        logger.warning("SHAP not installed. Cannot generate plot.")
        return None
    
    import shap
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=20)
    
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = EXPLANATIONS_DIR / f"shap_summary_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"SHAP summary plot saved to {save_path}")
    return str(save_path)


if __name__ == "__main__":
    from models.train import load_model
    from features.build_features import load_features
    
    print("Testing SHAP Explainability...\n")
    
    if check_shap_available():
        print("✅ SHAP is installed\n")
    else:
        print("⚠️ SHAP not installed. Install with: pip install shap\n")
    
    try:
        ranker, regressor, schema = load_model()
        features = load_features()
        feature_cols = schema.get('feature_cols', get_z_feature_cols())
        
        if not features.empty:
            print(f"Loaded model and {len(features)} feature rows")
            
            print("\nGenerating SHAP report...")
            report = generate_shap_report(ranker, features, feature_cols)
            
            if report:
                print("\nTop 10 Most Important Features:")
                for i, feat in enumerate(report.get('top_10_features', []), 1):
                    print(f"  {i}. {feat['feature']}: {feat['importance_pct']:.1f}%")
        else:
            print("No features found")
            
    except FileNotFoundError:
        print("No trained model found. Train a model first.")
