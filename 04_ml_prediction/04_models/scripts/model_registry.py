#!/usr/bin/env python3
"""
Model Registry - Central registry for all prediction models.

Provides a consistent interface for training, prediction, and saving models.
Makes it easy to swap between different model types.

Usage:
    from model_registry import MODEL_REGISTRY
    
    model = MODEL_REGISTRY['logistic'](config)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    model.save(output_dir)
"""

import json
import joblib
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb


# ============================================================================
# Base Model Class
# ============================================================================

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.training_metadata = {}
        
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Predict probabilities."""
        pass
    
    def predict(self, X, threshold=0.5):
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def save(self, output_dir: Path):
        """
        Save model and metadata.
        
        Args:
            output_dir: Directory to save model files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        metadata = {
            'model_type': self.__class__.__name__,
            'feature_names': self.feature_names,
            'training_metadata': self.training_metadata,
            'saved_at': datetime.now().isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to {output_dir}")
    
    @classmethod
    def load(cls, output_dir: Path):
        """Load model from directory."""
        output_dir = Path(output_dir)
        
        # Load config
        with open(output_dir / "config.json") as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(config)
        
        # Load model
        instance.model = joblib.load(output_dir / "model.pkl")
        
        # Load metadata
        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)
            instance.feature_names = metadata['feature_names']
            instance.training_metadata = metadata['training_metadata']
        
        return instance
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        return None


# ============================================================================
# Baseline Models
# ============================================================================

class MajorityBaseline(BaseModel):
    """Always predict the majority class (non-burst)."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """'Train' by computing majority class."""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Compute majority class
        self.majority_class = int(y_train.mode()[0])
        self.majority_prob = float(y_train.mean())
        
        self.training_metadata = {
            'n_samples': len(y_train),
            'n_bursts': int(y_train.sum()),
            'burst_rate': float(y_train.mean()),
            'majority_class': self.majority_class
        }
        
        # Create dummy model for consistency
        self.model = DummyClassifier(strategy='most_frequent')
        self.model.fit(X_train, y_train)
        
        print(f"✓ Majority baseline: Always predict class {self.majority_class} "
              f"(burst_rate={self.majority_prob:.1%})")
    
    def predict_proba(self, X):
        """Return constant probabilities."""
        n_samples = len(X)
        # Return [P(class=0), P(class=1)]
        prob_burst = self.majority_prob
        return np.array([[1 - prob_burst, prob_burst]] * n_samples)


class RandomBaseline(BaseModel):
    """Random predictions stratified by class distribution."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """'Train' by computing class distribution."""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        self.burst_rate = float(y_train.mean())
        
        self.training_metadata = {
            'n_samples': len(y_train),
            'n_bursts': int(y_train.sum()),
            'burst_rate': self.burst_rate
        }
        
        # Create dummy model
        self.model = DummyClassifier(strategy='stratified', random_state=42)
        self.model.fit(X_train, y_train)
        
        print(f"✓ Random baseline: Stratified random with burst_rate={self.burst_rate:.1%}")
    
    def predict_proba(self, X):
        """Return stratified random probabilities."""
        return self.model.predict_proba(X)


# ============================================================================
# Simple ML Models
# ============================================================================

class LogisticRegressionModel(BaseModel):
    """Logistic Regression with L2 regularization."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train logistic regression."""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Get hyperparameters from config
        C = self.config.get('C', 1.0)
        class_weight = self.config.get('class_weight', 'balanced')
        max_iter = self.config.get('max_iter', 1000)
        random_state = self.config.get('random_state', 42)
        
        # Initialize model
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'
        )
        
        # Train
        print(f"Training Logistic Regression (C={C}, class_weight={class_weight})...")
        self.model.fit(X_train, y_train)
        
        # Compute training metrics
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        
        self.training_metadata = {
            'n_samples': len(y_train),
            'n_features': X_train.shape[1],
            'n_bursts': int(y_train.sum()),
            'burst_rate': float(y_train.mean()),
            'train_auc': float(train_auc),
            'n_iterations': int(self.model.n_iter_[0]) if hasattr(self.model, 'n_iter_') else None
        }
        
        print(f"✓ Training complete: train_auc={train_auc:.4f}")
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature coefficients as importance."""
        if self.feature_names is None:
            return None
        
        coefs = self.model.coef_[0]
        return dict(zip(self.feature_names, np.abs(coefs)))


class DecisionTreeModel(BaseModel):
    """Simple Decision Tree for interpretability."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train decision tree."""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Get hyperparameters
        max_depth = self.config.get('max_depth', 5)
        min_samples_split = self.config.get('min_samples_split', 20)
        min_samples_leaf = self.config.get('min_samples_leaf', 10)
        class_weight = self.config.get('class_weight', 'balanced')
        random_state = self.config.get('random_state', 42)
        
        # Initialize model
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state
        )
        
        # Train
        print(f"Training Decision Tree (max_depth={max_depth})...")
        self.model.fit(X_train, y_train)
        
        # Compute training metrics
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        
        self.training_metadata = {
            'n_samples': len(y_train),
            'n_features': X_train.shape[1],
            'train_auc': float(train_auc),
            'tree_depth': int(self.model.get_depth()),
            'n_leaves': int(self.model.get_n_leaves())
        }
        
        print(f"✓ Training complete: train_auc={train_auc:.4f}, depth={self.model.get_depth()}")
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from tree."""
        if self.feature_names is None:
            return None
        
        return dict(zip(self.feature_names, self.model.feature_importances_))


# ============================================================================
# Ensemble Models
# ============================================================================

class RandomForestModel(BaseModel):
    """Random Forest ensemble."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train random forest."""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Get hyperparameters
        n_estimators = self.config.get('n_estimators', 100)
        max_depth = self.config.get('max_depth', 10)
        min_samples_split = self.config.get('min_samples_split', 10)
        min_samples_leaf = self.config.get('min_samples_leaf', 5)
        class_weight = self.config.get('class_weight', 'balanced')
        random_state = self.config.get('random_state', 42)
        n_jobs = self.config.get('n_jobs', -1)
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        # Train
        print(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        self.model.fit(X_train, y_train)
        
        # Compute training metrics
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        
        self.training_metadata = {
            'n_samples': len(y_train),
            'n_features': X_train.shape[1],
            'train_auc': float(train_auc),
            'n_estimators': n_estimators
        }
        
        print(f"✓ Training complete: train_auc={train_auc:.4f}")
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from forest."""
        if self.feature_names is None:
            return None
        
        return dict(zip(self.feature_names, self.model.feature_importances_))


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost with optional early stopping."""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Get hyperparameters
        n_estimators = self.config.get('n_estimators', 100)
        max_depth = self.config.get('max_depth', 6)
        learning_rate = self.config.get('learning_rate', 0.1)
        subsample = self.config.get('subsample', 0.8)
        colsample_bytree = self.config.get('colsample_bytree', 0.8)
        random_state = self.config.get('random_state', 42)
        
        # Compute scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='auc'
        )
        
        # Train with optional early stopping
        print(f"Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth})...")
        
        # Train (without early stopping for compatibility)
        self.model.fit(X_train, y_train)
        best_iteration = n_estimators
        
        # Compute training metrics
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        
        self.training_metadata = {
            'n_samples': len(y_train),
            'n_features': X_train.shape[1],
            'train_auc': float(train_auc),
            'best_iteration': int(best_iteration),
            'scale_pos_weight': float(scale_pos_weight)
        }
        
        print(f"✓ Training complete: train_auc={train_auc:.4f}")
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost."""
        if self.feature_names is None:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {
    'majority_baseline': MajorityBaseline,
    'random_baseline': RandomBaseline,
    'logistic': LogisticRegressionModel,
    'decision_tree': DecisionTreeModel,
    'random_forest': RandomForestModel,
    'xgboost': XGBoostModel,
}


def get_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
    """
    Factory function to create model from registry.
    
    Args:
        model_type: Type of model (key in MODEL_REGISTRY)
        config: Model configuration
        
    Returns:
        Initialized model instance
    """
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    
    return MODEL_REGISTRY[model_type](config)