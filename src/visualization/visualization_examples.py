"""
Example implementations of various data visualization techniques using popular Python libraries.
This module demonstrates different ways to create informative and visually appealing plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_sample_data():
    """Create sample data for visualization examples."""
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    time_series = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(100, 15, 100).cumsum()
    })
    
    # Create categorical data
    categories = ['A', 'B', 'C', 'D']
    categorical_data = pd.DataFrame({
        'category': np.random.choice(categories, 100),
        'value': np.random.normal(50, 10, 100)
    })
    
    # Create correlation data
    correlation_data = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'z': np.random.normal(0, 1, 100)
    })
    
    return time_series, categorical_data, correlation_data

def plot_time_series(time_series):
    """Create an interactive time series plot using Plotly."""
    fig = px.line(time_series, x='date', y='value',
                  title='Interactive Time Series Plot',
                  labels={'value': 'Value', 'date': 'Date'})
    
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified'
    )
    
    fig.show()

def plot_categorical_data(categorical_data):
    """Create various categorical plots using Seaborn."""
    # Set style
    plt.style.use('seaborn')
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Box plot
    sns.boxplot(data=categorical_data, x='category', y='value', ax=axes[0,0])
    axes[0,0].set_title('Box Plot')
    
    # Violin plot
    sns.violinplot(data=categorical_data, x='category', y='value', ax=axes[0,1])
    axes[0,1].set_title('Violin Plot')
    
    # Swarm plot
    sns.swarmplot(data=categorical_data, x='category', y='value', ax=axes[1,0])
    axes[1,0].set_title('Swarm Plot')
    
    # Bar plot with error bars
    sns.barplot(data=categorical_data, x='category', y='value', ax=axes[1,1])
    axes[1,1].set_title('Bar Plot with Error Bars')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(correlation_data):
    """Create an interactive correlation matrix using Plotly."""
    # Calculate correlation matrix
    corr_matrix = correlation_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='Interactive Correlation Matrix',
        template='plotly_dark'
    )
    
    fig.show()

def plot_3d_scatter(correlation_data):
    """Create an interactive 3D scatter plot using Plotly."""
    fig = px.scatter_3d(correlation_data,
                        x='x', y='y', z='z',
                        title='Interactive 3D Scatter Plot')
    
    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()

def main():
    """Main function to demonstrate various visualization techniques."""
    # Create sample data
    time_series, categorical_data, correlation_data = create_sample_data()
    
    # Create visualizations
    print("Creating time series plot...")
    plot_time_series(time_series)
    
    print("\nCreating categorical plots...")
    plot_categorical_data(categorical_data)
    
    print("\nCreating correlation matrix...")
    plot_correlation_matrix(correlation_data)
    
    print("\nCreating 3D scatter plot...")
    plot_3d_scatter(correlation_data)

if __name__ == "__main__":
    main() 