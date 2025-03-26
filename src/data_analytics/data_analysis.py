"""
Example implementations of data analysis and preprocessing techniques.
This module demonstrates various data cleaning, preprocessing, and analysis methods.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_data():
    """Create sample data with various characteristics for analysis."""
    np.random.seed(42)
    
    # Create a dataset with missing values, outliers, and different scales
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.normal(35, 15, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education_years': np.random.normal(16, 4, n_samples),
        'satisfaction': np.random.normal(7, 2, n_samples),
        'experience': np.random.normal(10, 5, n_samples)
    })
    
    # Add some missing values
    data.loc[np.random.choice(n_samples, 50), 'income'] = np.nan
    data.loc[np.random.choice(n_samples, 30), 'education_years'] = np.nan
    
    # Add some outliers
    data.loc[np.random.choice(n_samples, 20), 'income'] *= 3
    data.loc[np.random.choice(n_samples, 15), 'age'] *= 2
    
    return data

def handle_missing_values(data, strategy='mean'):
    """Handle missing values in the dataset."""
    imputer = SimpleImputer(strategy=strategy)
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data),
        columns=data.columns
    )
    return data_imputed

def handle_outliers(data, columns, method='zscore', threshold=3):
    """Handle outliers in the dataset."""
    data_cleaned = data.copy()
    
    for column in columns:
        if method == 'zscore':
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            data_cleaned = data_cleaned[z_scores < threshold]
        elif method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            data_cleaned = data_cleaned[
                (data[column] >= Q1 - 1.5 * IQR) & 
                (data[column] <= Q3 + 1.5 * IQR)
            ]
    
    return data_cleaned

def scale_features(data, method='standard'):
    """Scale features using different scaling methods."""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    
    scaled_data = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns
    )
    return scaled_data

def perform_pca(data, n_components=2):
    """Perform Principal Component Analysis."""
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    return pca_df, explained_variance, cumulative_variance

def analyze_correlations(data):
    """Analyze correlations between features."""
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def generate_summary_statistics(data):
    """Generate comprehensive summary statistics."""
    summary = pd.DataFrame({
        'count': data.count(),
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        '25%': data.quantile(0.25),
        '50%': data.quantile(0.50),
        '75%': data.quantile(0.75),
        'max': data.max(),
        'missing_values': data.isnull().sum(),
        'missing_percentage': (data.isnull().sum() / len(data) * 100).round(2)
    })
    
    return summary

def main():
    """Main function to demonstrate data analysis techniques."""
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data()
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_stats = generate_summary_statistics(data)
    print(summary_stats)
    
    # Handle missing values
    print("\nHandling missing values...")
    data_imputed = handle_missing_values(data)
    
    # Handle outliers
    print("\nHandling outliers...")
    data_cleaned = handle_outliers(data_imputed, ['age', 'income'])
    
    # Scale features
    print("\nScaling features...")
    data_scaled = scale_features(data_cleaned)
    
    # Perform PCA
    print("\nPerforming PCA...")
    pca_result, explained_variance, cumulative_variance = perform_pca(data_scaled)
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Cumulative variance: {cumulative_variance}")
    
    # Analyze correlations
    print("\nAnalyzing correlations...")
    correlation_matrix = analyze_correlations(data_scaled)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

if __name__ == "__main__":
    main() 