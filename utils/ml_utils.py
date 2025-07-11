"""
Machine Learning utilities for Credit Scoring System
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Advanced ML
try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

# Model Interpretation
try:
    import shap
    import lime.lime_tabular
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False

from config.settings import settings

class FeatureEngineer:
    """Advanced feature engineering for credit scoring"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features from raw data"""
        df_features = df.copy()
        
        # Financial ratios
        df_features['income_to_age_ratio'] = df_features['income'] / df_features['age']
        df_features['debt_per_account'] = (df_features['debt_to_income'] * df_features['income']) / df_features['number_of_accounts']
        df_features['credit_density'] = df_features['number_of_accounts'] / df_features['credit_history_length'].replace(0, 1)
        
        # Risk indicators
        df_features['high_utilization'] = (df_features['credit_utilization'] > 0.7).astype(int)
        df_features['young_borrower'] = (df_features['age'] < 25).astype(int)
        df_features['high_income'] = (df_features['income'] > df_features['income'].quantile(0.8)).astype(int)
        
        # Employment stability
        df_features['employment_stability'] = np.where(
            df_features['employment_years'] >= 5, 'Stable',
            np.where(df_features['employment_years'] >= 2, 'Moderate', 'Unstable')
        )
        
        # Credit behavior score
        df_features['credit_behavior_score'] = (
            df_features['payment_history'] * 0.4 +
            (1 - df_features['credit_utilization']) * 0.3 +
            (df_features['credit_history_length'] / 30) * 0.3
        )
        
        # Interaction features
        df_features['income_employment_interaction'] = df_features['income'] * df_features['employment_years']
        df_features['age_credit_history_interaction'] = df_features['age'] * df_features['credit_history_length']
        
        return df_features
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'default_risk') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training"""
        # Create advanced features
        df_processed = self.create_features(df)
        
        # Separate features and target
        if target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
        else:
            X = df_processed
            y = None
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = set(X[col].unique())
                known_values = set(self.encoders[col].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Add new categories to encoder
                    self.encoders[col].classes_ = np.append(
                        self.encoders[col].classes_, 
                        list(new_values)
                    )
                
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X[numerical_cols] = self.scalers['scaler'].fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scalers['scaler'].transform(X[numerical_cols])
        
        self.feature_names = X.columns.tolist()
        return X, y

class ModelEnsemble:
    """Ensemble of multiple ML models for credit scoring"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.feature_importance = {}
        self.model_weights = {}
        
    def get_base_models(self) -> Dict[str, Any]:
        """Get base models with optimized parameters"""
        models = {
            'gradient_boosting': GradientBoostingClassifier(
                **settings.MODEL_PARAMS['gradient_boosting']
            ),
            'random_forest': RandomForestClassifier(
                **settings.MODEL_PARAMS['random_forest']
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000
            )
        }
        
        if ADVANCED_ML_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                **settings.MODEL_PARAMS['xgboost'],
                eval_metric='logloss'
            )
            models['lightgbm'] = lgb.LGBMClassifier(
                **settings.MODEL_PARAMS['xgboost'],
                verbose=-1
            )
        
        return models
    
    def train_models(self, df: pd.DataFrame, credit_type: str, target_col: str = 'default_risk') -> Dict[str, Dict]:
        """Train ensemble of models"""
        # Prepare features
        X, y = self.feature_engineer.prepare_features(df, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        models = self.get_base_models()
        results = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name} for {credit_type}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'auc_score': roc_auc_score(y_test, y_pred_proba),
                    'cv_score': cross_val_score(model, X_train, y_train, cv=5).mean()
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(
                        self.feature_engineer.feature_names,
                        model.feature_importances_
                    ))
                    self.feature_importance[model_name] = importance
                
                # Store model
                self.models[f"{credit_type}_{model_name}"] = model
                results[model_name] = metrics
                
                print(f"✅ {model_name}: AUC = {metrics['auc_score']:.3f}")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        # Calculate ensemble weights based on AUC scores
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            total_auc = sum([metrics['auc_score'] for metrics in valid_results.values()])
            self.model_weights[credit_type] = {
                model_name: metrics['auc_score'] / total_auc 
                for model_name, metrics in valid_results.items()
            }
        
        return results
    
    def predict_ensemble(self, features: Dict, credit_type: str) -> Dict[str, Any]:
        """Make ensemble prediction"""
        # Prepare features
        df_features = pd.DataFrame([features])
        X, _ = self.feature_engineer.prepare_features(df_features)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name in self.model_weights.get(credit_type, {}):
            model_key = f"{credit_type}_{model_name}"
            if model_key in self.models:
                model = self.models[model_key]
                pred_proba = model.predict_proba(X)[0, 1]
                probabilities[model_name] = pred_proba
        
        if not probabilities:
            return {"error": "No trained models available"}
        
        # Weighted ensemble prediction
        weights = self.model_weights[credit_type]
        ensemble_probability = sum([
            prob * weights.get(model_name, 0) 
            for model_name, prob in probabilities.items()
        ])
        
        # Convert to credit score
        credit_score = int(300 + (1 - ensemble_probability) * 550)
        
        # Risk assessment
        risk_level = self._get_risk_level(ensemble_probability)
        decision = self._get_decision(credit_score)
        interest_rate = self._get_interest_rate(credit_score)
        
        return {
            'credit_score': credit_score,
            'risk_probability': ensemble_probability,
            'risk_level': risk_level,
            'decision': decision,
            'interest_rate': interest_rate,
            'model_predictions': probabilities,
            'confidence': self._calculate_confidence(probabilities)
        }
    
    def _get_risk_level(self, risk_prob: float) -> str:
        """Convert risk probability to level"""
        thresholds = settings.RISK_THRESHOLDS
        if risk_prob < thresholds['very_low']:
            return "Sangat Rendah"
        elif risk_prob < thresholds['low']:
            return "Rendah"
        elif risk_prob < thresholds['medium']:
            return "Sedang"
        elif risk_prob < thresholds['high']:
            return "Tinggi"
        else:
            return "Sangat Tinggi"
    
    def _get_decision(self, credit_score: int) -> str:
        """Get lending decision based on credit score"""
        if credit_score >= 740:
            return "Approved"
        elif credit_score >= 670:
            return "Approved"
        elif credit_score >= 580:
            return "Review"
        else:
            return "Rejected"
    
    def _get_interest_rate(self, credit_score: int) -> str:
        """Get interest rate based on credit score"""
        if credit_score >= 740:
            return "5.5% - 7.0%"
        elif credit_score >= 670:
            return "7.0% - 9.5%"
        elif credit_score >= 580:
            return "9.5% - 15.0%"
        else:
            return "N/A"
    
    def _calculate_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate prediction confidence"""
        if len(predictions) < 2:
            return 0.5
        
        probs = list(predictions.values())
        std_dev = np.std(probs)
        # Higher confidence when predictions are more consistent
        confidence = max(0.0, 1.0 - (std_dev * 2))
        return confidence
    
    def save_models(self, credit_type: str, model_dir: Path = None):
        """Save trained models"""
        if model_dir is None:
            model_dir = settings.MODEL_DIR
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name in self.model_weights.get(credit_type, {}):
            model_key = f"{credit_type}_{model_name}"
            if model_key in self.models:
                model_path = model_dir / f"{model_key}.joblib"
                joblib.dump(self.models[model_key], model_path)
        
        # Save feature engineer
        fe_path = model_dir / f"{credit_type}_feature_engineer.joblib"
        joblib.dump(self.feature_engineer, fe_path)
        
        # Save metadata
        metadata = {
            'credit_type': credit_type,
            'model_weights': self.model_weights.get(credit_type, {}),
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_engineer.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = model_dir / f"{credit_type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_models(self, credit_type: str, model_dir: Path = None):
        """Load trained models"""
        if model_dir is None:
            model_dir = settings.MODEL_DIR
        
        # Load metadata
        metadata_path = model_dir / f"{credit_type}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_weights[credit_type] = metadata.get('model_weights', {})
            self.feature_importance.update(metadata.get('feature_importance', {}))
        
        # Load feature engineer
        fe_path = model_dir / f"{credit_type}_feature_engineer.joblib"
        if fe_path.exists():
            self.feature_engineer = joblib.load(fe_path)
        
        # Load models
        for model_name in self.model_weights.get(credit_type, {}):
            model_key = f"{credit_type}_{model_name}"
            model_path = model_dir / f"{model_key}.joblib"
            if model_path.exists():
                self.models[model_key] = joblib.load(model_path)

class ModelInterpreter:
    """Model interpretation and explanation"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        if INTERPRETABILITY_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(model)
            except:
                pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def explain_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Explain individual prediction"""
        explanation = {
            'feature_contributions': {},
            'top_factors': [],
            'recommendation': ''
        }
        
        if self.explainer and INTERPRETABILITY_AVAILABLE:
            try:
                shap_values = self.explainer.shap_values(features)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                contributions = dict(zip(self.feature_names, shap_values[0]))
                explanation['feature_contributions'] = contributions
                
                # Top positive and negative factors
                sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                explanation['top_factors'] = sorted_contrib[:5]
                
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
        
        return explanation

def hyperparameter_tuning(model_class, param_grid: Dict, X_train, y_train, cv: int = 5):
    """Perform hyperparameter tuning"""
    grid_search = GridSearchCV(
        model_class(),
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model_stability(model, X, y, n_iterations: int = 10) -> Dict[str, float]:
    """Evaluate model stability through multiple train-test splits"""
    scores = {'accuracy': [], 'auc': [], 'precision': [], 'recall': []}
    
    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['auc'].append(roc_auc_score(y_test, y_pred_proba))
        scores['precision'].append(precision_score(y_test, y_pred))
        scores['recall'].append(recall_score(y_test, y_pred))
    
    stability_metrics = {}
    for metric, values in scores.items():
        stability_metrics[f'{metric}_mean'] = np.mean(values)
        stability_metrics[f'{metric}_std'] = np.std(values)
        stability_metrics[f'{metric}_cv'] = np.std(values) / np.mean(values)  # Coefficient of variation
    
    return stability_metrics

# Global model ensemble instance
model_ensemble = ModelEnsemble()