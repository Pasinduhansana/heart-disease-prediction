import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit scaler on training data and transform both training and test sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train, solver: str = 'lbfgs', max_iter: int = 1000, random_state: int = 42):
    """Train a logistic regression model with a solver that supports multinomial loss."""
    model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def get_sorted_coefficients(model, feature_names):
    """Return feature coefficients sorted by absolute importance."""
    coefficients = model.coef_.flatten()
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': abs(coefficients)
    })
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    return coef_df
