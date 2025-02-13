import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """
    Calculate Cramer's V statistic for categorical variables.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def point_biserial_correlation(binary_var, continuous_var):
    """
    Calculate point-biserial correlation between binary and continuous variables.
    """
    return np.corrcoef(binary_var.astype(float), continuous_var)[0, 1]

def calculate_vif(X):
    """
    Calculate VIF for numerical features.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

def analyze_categorical_correlations(df, categorical_features):
    """
    Analyze correlations between categorical variables using Cramer's V.
    """
    n_features = len(categorical_features)
    cramer_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            v = cramers_v(df[categorical_features[i]], df[categorical_features[j]])
            cramer_matrix[i, j] = v
            cramer_matrix[j, i] = v
    
    return pd.DataFrame(cramer_matrix, 
                       index=categorical_features, 
                       columns=categorical_features)

def analyze_mixed_correlations(df, numerical_features, categorical_features):
    """
    Analyze correlations between numerical and categorical variables.
    """
    correlations = []
    
    for cat_feat in categorical_features:
        if df[cat_feat].nunique() == 2:  # Binary categorical
            for num_feat in numerical_features:
                corr = point_biserial_correlation(df[cat_feat], df[num_feat])
                correlations.append({
                    'categorical': cat_feat,
                    'numerical': num_feat,
                    'correlation': corr
                })
    
    return pd.DataFrame(correlations)

def plot_correlation_matrices(num_corr, cat_corr, mixed_corr, save_path=None):
    """
    Plot correlation matrices for all types of features.
    """
    fig = plt.figure(figsize=(20, 6))
    
    # Numerical correlations
    ax1 = plt.subplot(131)
    sns.heatmap(num_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax1)
    ax1.set_title('Numerical Features Correlation')
    plt.xticks(rotation=45, ha='right')
    
    # Categorical correlations
    ax2 = plt.subplot(132)
    sns.heatmap(cat_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax2)
    ax2.set_title("Categorical Features (Cramer's V)")
    plt.xticks(rotation=45, ha='right')
    
    # Mixed correlations (for binary categorical)
    if not mixed_corr.empty:
        mixed_pivot = mixed_corr.pivot(index='categorical', 
                                     columns='numerical', 
                                     values='correlation')
        ax3 = plt.subplot(133)
        sns.heatmap(mixed_pivot, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax3)
        ax3.set_title('Binary Categorical vs Numerical')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Read data
    df = pd.read_csv('data/processed/featured_startups.csv')
    
    # Define features
    numerical_features = [
        'funding_total_usd',
        'funding_rounds',
        'days_to_first_funding',
        'funding_duration',
        'company_age_at_last_funding',
        'founded_year',
        'founded_quarter',
        'first_funding_year',
        'last_funding_year',
        'category_count'
    ]
    
    categorical_features = [
        'country_code',
        'region',
        'primary_sector',
        'is_founded_recession_2008',
        'is_founded_covid'
    ]
    
    # Create figures directory
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    print("Analyzing multicollinearity...")
    
    # Numerical correlations
    num_corr = df[numerical_features].corr()
    
    # Categorical correlations
    cat_corr = analyze_categorical_correlations(df, categorical_features)
    
    # Mixed correlations
    mixed_corr = analyze_mixed_correlations(df, numerical_features, categorical_features)
    
    # Plot correlation matrices
    plot_correlation_matrices(num_corr, cat_corr, mixed_corr, 
                            'figures/all_correlations.png')
    
    # Calculate VIF for numerical features
    X = df[numerical_features].copy()
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=numerical_features)
    vif_df = calculate_vif(X)
    
    # Print results
    print("\nVariance Inflation Factors (Numerical Features):")
    print(vif_df)
    
    print("\nStrong Categorical Associations (Cramer's V > 0.3):")
    for i in range(len(categorical_features)):
        for j in range(i+1, len(categorical_features)):
            v = cat_corr.iloc[i, j]
            if v > 0.3:
                print(f"{categorical_features[i]} - {categorical_features[j]}: {v:.3f}")
    
    print("\nStrong Mixed Correlations (|r| > 0.3):")
    strong_mixed = mixed_corr[abs(mixed_corr['correlation']) > 0.3]
    if not strong_mixed.empty:
        for _, row in strong_mixed.iterrows():
            print(f"{row['categorical']} - {row['numerical']}: {row['correlation']:.3f}")
    
    # Print recommendations
    print("\nRecommendations for handling multicollinearity:")
    
    # Numerical features
    high_vif_features = vif_df[vif_df['VIF'] > 5]['Feature'].tolist()
    if high_vif_features:
        print("\nConsider addressing these numerical features (VIF > 5):")
        for feature in high_vif_features:
            print(f"- {feature}")
    
    # Categorical features
    print("\nConsider addressing these categorical feature relationships:")
    for i in range(len(categorical_features)):
        for j in range(i+1, len(categorical_features)):
            v = cat_corr.iloc[i, j]
            if v > 0.5:
                print(f"- {categorical_features[i]} and {categorical_features[j]}")

if __name__ == "__main__":
    main()