import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

def prepare_features(df):
    """
    Prepare numerical features including calculated time-based features.
    """
    # Convert date columns to datetime
    date_columns = ['founded_at', 'first_funding_at', 'last_funding_at']
    df_prep = df.copy()
    
    for col in date_columns:
        df_prep[col] = pd.to_datetime(df_prep[col])
    
    # Calculate time-based features
    df_prep['days_to_first_funding'] = (df_prep['first_funding_at'] - 
                                       df_prep['founded_at']).dt.days
    df_prep['funding_duration'] = (df_prep['last_funding_at'] - 
                                 df_prep['first_funding_at']).dt.days
    
    return df_prep

def analyze_smote_effect(df, save_path=None):
    """
    Analyze and visualize the effect of SMOTE on feature distributions.
    """
    # Prepare features
    df_prep = prepare_features(df)
    
    # Select numerical features
    numerical_features = [
        'funding_total_usd',
        'funding_rounds',
        'days_to_first_funding',
        'funding_duration'
    ]
    
    # Remove any rows with NaN values
    df_clean = df_prep.dropna(subset=numerical_features + ['status'])
    
    # Prepare data
    X = df_clean[numerical_features]
    y = df_clean['status']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=numerical_features)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_resampled = pd.DataFrame(X_resampled, columns=numerical_features)
    
    # Create figure
    n_features = len(numerical_features)
    fig, axes = plt.subplots(n_features, 2, figsize=(15, 5*n_features))
    
    # Plot distributions for each feature
    for idx, feature in enumerate(numerical_features):
        # Before SMOTE
        for status in y.unique():
            sns.kdeplot(data=X_scaled[y == status], x=feature, 
                       label=status, ax=axes[idx, 0])
        axes[idx, 0].set_title(f'{feature} Distribution Before SMOTE')
        axes[idx, 0].legend()
        
        # After SMOTE
        for status in y_resampled.unique():
            sns.kdeplot(data=X_resampled[y_resampled == status], x=feature, 
                       label=status, ax=axes[idx, 1])
        axes[idx, 1].set_title(f'{feature} Distribution After SMOTE')
        axes[idx, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print class distribution before and after SMOTE
    print("\nClass Distribution Before SMOTE:")
    print(y.value_counts(normalize=True).round(3) * 100)
    
    print("\nClass Distribution After SMOTE:")
    print(pd.Series(y_resampled).value_counts(normalize=True).round(3) * 100)
    
    # Calculate and print feature statistics
    print("\nFeature Statistics Before SMOTE:")
    for status in y.unique():
        print(f"\nStatus: {status}")
        status_data = X_scaled[y == status]
        for feature in numerical_features:
            print(f"\n{feature}:")
            print(f"Mean: {status_data[feature].mean():.2f}")
            print(f"Std: {status_data[feature].std():.2f}")
    
    print("\nFeature Statistics After SMOTE:")
    for status in y_resampled.unique():
        print(f"\nStatus: {status}")
        status_data = X_resampled[y_resampled == status]
        for feature in numerical_features:
            print(f"\n{feature}:")
            print(f"Mean: {status_data[feature].mean():.2f}")
            print(f"Std: {status_data[feature].std():.2f}")

if __name__ == "__main__":
    # Read data
    df = pd.read_csv('C:\\Users\\elote\\Repositories\\startup-success\\data\\interim\\cleaned_startups.csv')
    
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Generate visualization
    analyze_smote_effect(df, save_path='figures/smote_effect.png')