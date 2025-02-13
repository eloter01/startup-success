import pandas as pd
import numpy as np

def reduce_multicollinearity(df):
    """
    Reduce multicollinearity in the dataset by removing or combining redundant features.
    
    Args:
        df: Input DataFrame with all features
    
    Returns:
        DataFrame with reduced multicollinearity
    """
    # Create copy of DataFrame
    df_reduced = df.copy()
    
    # 1. Remove redundant temporal features
    features_to_drop = [
        'company_age_at_last_funding',
        'first_funding_year',
        'founded_year'
    ]
    
    # 2. Remove redundant geographic features
    features_to_drop.append('region')
    
    # Drop the identified features
    df_reduced = df_reduced.drop(columns=features_to_drop)
    
    print("Removed the following features to reduce multicollinearity:")
    for feature in features_to_drop:
        print(f"- {feature}")
    
    return df_reduced

def main():
    # Read the data
    input_file = 'data/processed/featured_startups.csv'
    output_file = 'data/processed/startups_reduced_multicollinearity.csv'
    
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Apply multicollinearity reduction
    df_reduced = reduce_multicollinearity(df)
    
    # Save the reduced dataset
    df_reduced.to_csv(output_file, index=False)
    print(f"\nSaved reduced dataset to {output_file}")
    print(f"Original number of features: {len(df.columns)}")
    print(f"Reduced number of features: {len(df_reduced.columns)}")
    
    # Print remaining features
    print("\nRemaining features:")
    for col in df_reduced.columns:
        print(f"- {col}")

if __name__ == "__main__":
    main()