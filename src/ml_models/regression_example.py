"""
Example implementation of a regression model using scikit-learn.
This module demonstrates how to build, train, and evaluate a regression model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sample_data(n_samples=1000):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, 3)  # 3 features
    # Generate target with some noise
    y = 2 * X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    return X, y

def train_model(X, y):
    """Train a linear regression model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, X_test, y_test, y_pred, mse, r2

def plot_results(y_test, y_pred):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate the regression model."""
    # Generate sample data
    X, y = generate_sample_data()
    
    # Train the model
    model, scaler, X_test, y_test, y_pred, mse, r2 = train_model(X, y)
    
    # Print results
    print(f"Model Coefficients: {model.coef_}")
    print(f"Model Intercept: {model.intercept_}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot results
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main() 