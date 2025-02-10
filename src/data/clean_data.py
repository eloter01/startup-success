# src/data/clean_data.py
import pandas as pd
from datetime import datetime

def type_conversion_datetime(df, date_columns):
    for col in date_columns:
        # change data type of founded_at, first_funding_at, last_funding_at to datetime
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
    
    # Create a boolean mask for valid dates
    valid_mask = pd.Series(True, index=df.index)
    
    for col in date_columns:
        col_mask = df[col].apply(lambda x: True if pd.isna(x) else False)  # Keep NaN values
        
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
    
    # Return filtered dataframe and number of dropped rows
    df = df[valid_mask]
    #dropped_count = len(df) - len(filtered_df)
    
    return df #, dropped_count

def remove_outliers(df, column_names):
    for col in column_names:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        df.loc[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def clean_data(df):
    # All your cleaning steps
    df = validate_dates(df, ['founded_at', 'first_funding_at', 'last_funding_at'])
    df = type_conversion_datetime(df, ['founded_at', 'first_funding_at', 'last_funding_at'])
    df = type_conversion_float(df, ['funding_total_usd'])
    df = remove_outliers(df, ['funding_total_usd'])
    # Add other cleaning steps
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/raw/startups.csv')
    cleaned_df = clean_data(df)
    cleaned_df.to_csv('data/interim/cleaned_startups.csv', index=False)