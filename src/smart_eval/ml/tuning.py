"""Hyperparameter tuning utilities with GridSearchCV."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)
try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.debug("XGBoost not installed; multi-model tuning will skip XGBoost.")


def tune_model(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any],
    cv: int = 5,
    scoring: str = "r2",
    n_jobs: int = -1,
) -> tuple[object, Dict[str, Any]]:
    """Tune model hyperparams using GridSearchCV on training data only.
    
    Args:
        model: Sklearn-compatible estimator.
        X_train: Training feature matrix.
        y_train: Training target vector.
        param_grid: Parameter grid dict for GridSearchCV.
        cv: Number of cross-validation folds.
        scoring: Scoring metric (e.g., 'r2').
        n_jobs: Number of parallel jobs (-1 = all CPUs).
        
    Returns:
        Tuple of (best_estimator, best_params_dict).
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_

    print(f"[TUNE] Best params: {best_params}  CV score: {cv_score:.4f}")

    return grid_search.best_estimator_, best_params


def build_model(model_name: str, random_state: int = 42) -> object:
    """Build a model instance by name.
    
    Args:
        model_name: Name of model class (e.g., 'RandomForest', 'Ridge', 'XGBoost').
        random_state: Random seed.
        
    Returns:
        Sklearn-compatible model instance.
        
    Raises:
        ValueError: If model name is not recognized or dependency is missing.
    """
    if model_name == "Ridge":
        return Ridge(random_state=random_state, solver="auto")
    if model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise ValueError("XGBoost requested but not installed. Install with: pip install xgboost")
        return XGBRegressor(random_state=random_state, verbosity=0)
    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=random_state)

    raise ValueError(f"Unknown model name: {model_name}")
