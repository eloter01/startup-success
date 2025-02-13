from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def get_logistic_regression(random_state=42):
    """
    Create logistic regression model with default parameters.
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state
    )

def get_random_forest(random_state=42):
    """
    Create random forest model with default parameters.
    """
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state
    )

def get_xgboost(random_state=42):
    """
    Create XGBoost model with default parameters.
    """
    return xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
        eval_metric='mlogloss'
    )

def get_all_models(random_state=42):
    """
    Get dictionary of all available models.
    """
    return {
        'logistic_regression': get_logistic_regression(random_state),
        'random_forest': get_random_forest(random_state),
        'xgboost': get_xgboost(random_state)
    }