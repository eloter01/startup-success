# src/data/clean_data.py
import pandas as pd
from datetime import datetime

def validate_date_sequence(df, date_cols):
    """
    Filter out records where:
    1. Any date is NULL
    2. Dates don't follow the sequence: founded_at <= first_funding_at <= last_funding_at
    """
 
    return df.loc[
        df[date_cols].notna().all(axis=1) &  # Remove NULL dates (COMMENT OUT IF NECESSARY!)
        (df['founded_at'] <= df['first_funding_at']) & 
        (df['first_funding_at'] <= df['last_funding_at'])
    ].copy()

def type_conversion_datetime(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df
        
def type_conversion_float(df, float_columns):
    for col in float_columns:
        df[col] = df[col].replace('-', float('nan'))
        df[col] = df[col].astype(float)
    return df

def validate_dates(df, date_columns):
    """
    Remove rows where any specified date column falls outside pandas datetime range
    (1677-09-22 to 2262-04-11) or contains invalid date format
    """
    min_date = datetime(1677, 9, 22)
    max_date = datetime(2262, 4, 11)
    
    valid_mask = pd.Series(True, index=df.index)
    
    for col in date_columns:
        col_mask = df[col].apply(lambda x: True if pd.isna(x) else False)
        
        def check_date(x):
            if pd.isna(x):
                return True
            try:
                date = datetime.strptime(x, '%Y-%m-%d')
                return min_date <= date <= max_date
            except ValueError:
                return False
        
        valid_dates = df[col].apply(check_date)
        valid_mask &= valid_dates
    
    return df[valid_mask]

def remove_outliers(df, column_names):
    for col in column_names:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        df = df.loc[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def clean_data(df):
    date_cols = ['founded_at', 'first_funding_at', 'last_funding_at']
    
    # First validate and convert dates so comparisons work properly
    df = validate_dates(df, date_cols)
    df = type_conversion_datetime(df, date_cols)
    
    # Now validate date sequence
    df = validate_date_sequence(df, date_cols)
    
    # Continue with other cleaning steps
    df = type_conversion_float(df, ['funding_total_usd'])
    df = remove_outliers(df, ['funding_total_usd'])
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/raw/startups.csv')
    cleaned_df = clean_data(df)
    cleaned_df.to_csv('data/interim/cleaned_startups.csv', index=False)