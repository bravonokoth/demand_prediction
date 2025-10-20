import numpy as np

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, epsilon=1e-10):
    """Mean Absolute Percentage Error"""
    return 100 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE and MAPE"""
    return {
        'rmse': rmse(y_true, y_pred),
        'mape': mape(y_true, y_pred)
    }