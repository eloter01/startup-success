import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import joblib

def plot_feature_importance(model, feature_names, n_features=20, save_path=None):
    """
    Plot the most important features using built-in feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        n_features: Number of top features to show
        save_path: Path to save the figure
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create dataframe of features and their importance scores
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort and get top features
    top_features = feature_importance.sort_values('importance', ascending=False).head(n_features)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    
    # Add value labels on bars
    for i, v in enumerate(top_features['importance']):
        plt.text(v, i, f' {v:.3f}', va='center')
    
    plt.title(f'Feature Importance Rankings (Top {n_features})', pad=20)
    plt.xlabel('Relative Importance')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed rankings
    print(f"\nFeature Importance Rankings (Top {n_features}):")
    for idx, row in top_features.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return top_features['feature'].tolist()

def plot_feature_correlations(X, selected_features, save_path=None):
    """
    Plot correlation matrix for selected features.
    
    Args:
        X: Feature matrix
        selected_features: List of features to analyze
        save_path: Path to save the figure
    """
    # Calculate correlation matrix for selected features
    corr_matrix = X[selected_features].corr()
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True)
    
    plt.title('Feature Correlation Matrix', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print strong correlations
    print("\nStrong Feature Correlations (|r| > 0.5):")
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                print(f"{selected_features[i]} - {selected_features[j]}: {corr:.3f}")

def analyze_feature_distributions(X, y, selected_features, n_display=6, save_path=None):
    """
    Analyze distribution of features across different status categories.
    
    Args:
        X: Feature matrix
        y: Target variable
        selected_features: List of features to analyze
        n_display: Number of features to display in plots
        save_path: Path to save the figure
    """
    # Combine features with target
    df = X[selected_features].copy()
    df['status'] = y
    
    # Create subplot for each feature
    n_features = min(n_display, len(selected_features))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(selected_features[:n_features]):
        sns.boxplot(data=df, x='status', y=feature, ax=axes[idx])
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
        axes[idx].set_title(feature)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nFeature Statistics by Status:")
    for feature in selected_features[:n_features]:
        print(f"\n{feature}:")
        print(df.groupby('status')[feature].describe().round(3))

def main():
    try:
        # Load model and data
        model = joblib.load('models/random_forest_model.pkl')
        X_test = joblib.load('models/X_test.pkl')
        y_test = joblib.load('models/y_test.pkl')
        
        # Get feature names
        feature_names = X_test.columns.tolist()
        
        # Create figures directory
        import os
        os.makedirs('figures', exist_ok=True)
        
        # Set style
        sns.set_theme(style="whitegrid")
        print("Analyzing feature importance...")
        
        # Plot feature importance
        selected_features = plot_feature_importance(model, feature_names, 
                                                  save_path='figures/feature_importance.png')
        
        # Plot correlations between features
        plot_feature_correlations(X_test, selected_features, 
                                save_path='figures/feature_correlations.png')
        
        # Analyze feature distributions by status
        analyze_feature_distributions(X_test, y_test, selected_features,
                                    save_path='figures/feature_distributions.png')
        
    except FileNotFoundError:
        print("Error: Required model files not found. Please ensure model training has been completed.")

if __name__ == "__main__":
    main()