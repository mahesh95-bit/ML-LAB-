import numpy as np
def calculate_regression_metrics(true_values, predicted_values):
    mse = np.mean((true_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    r2 = 1 - mse / np.var(true_values)