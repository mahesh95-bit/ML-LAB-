import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def calculate_regression_metrics(true_values, predicted_values):
    mse = np.mean((true_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    r2 = 1 - mse / np.var(true_values)
    
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

def generate_training_data(n_samples=20):
    X = np.random.uniform(1, 10, (n_samples, 2))
    # Classify: X+Y > 10 -> class 1, else class 0
    y = np.where(X[:, 0] + X[:, 1] > 10, 1, 0)
    return X, y

def plot_training_data(X, y, title="Training Data"):
    #Create scatter plot of training data colored by class.
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=80, label='Class0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', s=80, label='Class1')
    plt.plot([0, 10], [10, 0], 'k--', alpha=0.5, label='True Boundary')
    plt.xlim(0, 11); plt.ylim(0, 11)
    plt.xlabel('Feature X'); plt.ylabel('Feature Y')
    plt.title(title); plt.legend()
    plt.grid(True, alpha=0.3)

def generate_test_grid():
    x_range = np.arange(0, 10.1, 0.1)
    X1, X2 = np.meshgrid(x_range, x_range)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    return X_grid, X1, X2

def train_knn_classifier(X_train, y_train, k=3):
    X_grid, X1, X2 = generate_test_grid()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_grid_pred = knn.predict(X_grid).reshape(X1.shape)
    return knn, y_grid_pred, X1, X2

def plot_knn_results(X_train, y_train, X1, X2, y_grid_pred, k=3, title=""):
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, y_grid_pred, alpha=0.3, colors=['blue', 'red'], levels=[-0.5, 0.5, 1.5])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=['blue' if c==0 else 'red' for c in y_train], 
                s=80, edgecolors='black')
    plt.xlim(0, 10); plt.ylim(0, 10)
    plt.xlabel('Feature X'); plt.ylabel('Feature Y')
    plt.title(f'{title} (k={k})')
    plt.grid(True, alpha=0.3)

def compare_k_values(X_train, y_train, k_values):
    #Create subplots showing kNN results for different k values.
    
    X_grid, X1, X2 = generate_test_grid()
    n_plots = len(k_values)
    n_rows = 2
    n_cols = 3 if n_plots > 3 else n_plots
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    for idx, k in enumerate(k_values):
        ax = axes[idx//n_cols, idx%n_cols] if n_plots > 1 else axes
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_grid_pred = knn.predict(X_grid).reshape(X1.shape)
        
        ax.contourf(X1, X2, y_grid_pred, alpha=0.3, colors=['blue', 'red'], levels=[-0.5, 0.5, 1.5])
        ax.scatter(X_train[:, 0], X_train[:, 1], c=['blue' if c==0 else 'red' for c in y_train], s=40)
        ax.set_title(f'k = {k}')
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    #Main program execution.
    np.random.seed(42)
    
    # A2: Calculate regression metrics
    true_prices = 1000 + np.random.randn(100) * 200
    pred_prices = true_prices + np.random.randn(100) * 50
    metrics = calculate_regression_metrics(true_prices, pred_prices)
    
    print("A2: Price Prediction Metrics")
    print(f"  MSE: {metrics['MSE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R2: {metrics['R2']:.3f}")
    
    # A3: Generate training data
    X_train, y_train = generate_training_data(20)
    plot_training_data(X_train, y_train, "A3: Training Data")
    plt.show()
    print(f"\nA3: Generated {len(X_train)} training samples")
    print(f"  Class 0: {sum(y_train==0)} samples, Class 1: {sum(y_train==1)} samples")
    
    # A4: kNN classification with k=3
    knn_model, y_grid_pred, X1, X2 = train_knn_classifier(X_train, y_train, k=3)
    plot_knn_results(X_train, y_train, X1, X2, y_grid_pred, k=3, title="A4: kNN Classification")
    plt.show()
    print("\nA4: kNN classifier trained with k=3")
    print(f"  Test grid points: {X1.size} points (100x100 grid)")
    
    # A5: Compare different k values
    k_values_to_test = [1, 3, 5, 7, 10, 15]
    compare_k_values(X_train, y_train, k_values_to_test)
    print("\nA5: Compared k values:", k_values_to_test)
    print("  Small k (1,3): More complex boundaries")
    print("  Large k (10,15): Smoother boundaries")

if __name__ == "__main__":
    main()
