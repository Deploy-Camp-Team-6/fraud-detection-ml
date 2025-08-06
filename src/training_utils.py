import mlflow
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np

def get_model(model_name: str, params: dict, random_state: int):
    """Returns a model instance based on the name."""
    static_params = params.get('static_params', {})

    if model_name == 'xgboost':
        return XGBClassifier(random_state=random_state, **static_params)
    elif model_name == 'lightgbm':
        return LGBMClassifier(random_state=random_state, **static_params)
    elif model_name == 'logistic_regression':
        return LogisticRegression(random_state=random_state, **static_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train_and_evaluate(X_train, y_train, model_name, params, cv_folds, imbalance_handler, random_state):
    """Performs cross-validated training and evaluation."""

    model = get_model(model_name, params, random_state)
    param_grid = params.get('param_grid', {})

    # Create a pipeline with SMOTE if specified
    if imbalance_handler == 'smote':
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('model', model)
        ])
        # Adjust param_grid keys for the pipeline
        param_grid = {f'model__{k}': v for k, v in param_grid.items()}
    else:
        pipeline = model
        # Adjust model params for class_weight if specified
        if imbalance_handler == 'class_weight':
            if model_name in ['lightgbm', 'logistic_regression']:
                pipeline.set_params(class_weight='balanced')
            elif model_name == 'xgboost':
                scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
                pipeline.set_params(scale_pos_weight=scale_pos_weight)


    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1', # Optimize for F1-score
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log best params and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)

    return best_model

def evaluate_on_test_set(model, X_test, y_test):
    """Evaluates the final model on the test set."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }

    mlflow.log_metrics(metrics)
    return metrics
