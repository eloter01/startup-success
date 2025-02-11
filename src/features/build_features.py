# # src/features/build_features.py
# import pandas as pd

# # Define category groups
# top_tier = ['Software', 'Mobile', 'E-Commerce', 'Social Media', 'Curated Web',
#             'Biotechnology', 'Enterprise Software', 'Advertising', 'SaaS']

# # Define sector keyword groups
# # sector_keywords = {
# #     'Tech': ['Software', 'Enterprise Software', 'SaaS', 'Cloud Computing',
# #              'Hardware + Software', 'Information Technology', 'Technology'],
# #     'Health': ['Biotechnology', 'Health and Wellness', 'Health Care',
# #                'Medical', 'Medical Devices', 'Pharmaceuticals', 'Life Sciences'],
# #     'Mobile': ['Mobile', 'Apps', 'Android', 'iOS', 'iPhone', 'iPad',
# #                'Mobile Commerce', 'Mobile Payments'],
# #     'Social': ['Social Media', 'Social Network Media', 'Social Commerce',
# #                'Social Games', 'Social Media Marketing'],
# #     'Data': ['Analytics', 'Big Data', 'Big Data Analytics',
# #              'Predictive Analytics', 'Data Mining', 'Data Visualization'],
# #     'Finance': ['Finance', 'FinTech', 'Financial Services', 'Payments',
# #                 'Banking', 'Insurance', 'Investment Management']
# # }

# primary_sectors = {
#     'Tech': ['Software', 'Enterprise Software', 'SaaS', 'Hardware + Software', 
#              'Technology', 'Cloud Computing', 'Security', 'Internet'],
    
#     'Mobile': ['Mobile', 'Apps'],
    
#     'Commerce': ['E-Commerce', 'Marketplaces', 'Fashion', 'Travel', 'Sales and Marketing'],
    
#     'Health': ['Biotechnology', 'Health and Wellness', 'Health Care'],
    
#     'Media': ['Social Media', 'Curated Web', 'Advertising', 'Video'],
    
#     'Analytics': ['Analytics', 'Big Data'],
    
#     'Consumer': ['Games', 'Services', 'Manufacturing'],
    
#     'Education': ['Education']
# }

# def create_time_features(df):
#     """
#     Create time-based features from clean date columns.
#     Assumes dates are already validated and follow: founded_at <= first_funding_at <= last_funding_at
#     """
#     # Time intervals
#     df['days_to_first_funding'] = (df['first_funding_at'] - df['founded_at']).dt.days
#     df['funding_duration'] = (df['last_funding_at'] - df['first_funding_at']).dt.days
#     df['company_age_at_last_funding'] = (df['last_funding_at'] - df['founded_at']).dt.days

#     # Economic period indicators
#     df['is_founded_recession_2008'] = (
#         ((df['founded_at'] >= '2008-09-01') & (df['founded_at'] <= '2009-06-30'))
#     ).astype(int)
#     df['is_founded_covid'] = (df['founded_at'] >= '2020-01-01').astype(int)

#     # Extract year and quarter for time-based analysis
#     df['founded_year'] = df['founded_at'].dt.year
#     df['founded_quarter'] = df['founded_at'].dt.quarter
    
#     # Funding round timing
#     df['first_funding_year'] = df['first_funding_at'].dt.year
#     df['last_funding_year'] = df['last_funding_at'].dt.year
    
#     return df

# def create_category_features(df):
#     """
#     Create category-based features from the category_list column.
#     Returns consolidated category columns that can be one-hot encoded later if needed.
#     """
#     df = df.copy()
    
#     # Initialize new columns with empty strings
#     df['sectors'] = ''
#     df['top_categories'] = ''
    
#     # Vectorized approach for categories using str.contains
#     categories = df['category_list'].fillna('')
    
#     # Add sectors (using str.contains instead of loops)
#     for sector, keywords in sector_keywords.items():
#         pattern = '|'.join(keywords)
#         mask = categories.str.contains(pattern, na=False)
#         df.loc[mask, 'sectors'] = df.loc[mask, 'sectors'].str.cat(
#             [sector] * mask.sum(), 
#             sep='|'
#         ).str.strip('|')
    
#     # Add top categories
#     for category in top_tier:
#         mask = categories.str.contains(category, na=False)
#         df.loc[mask, 'top_categories'] = df.loc[mask, 'top_categories'].str.cat(
#             [category] * mask.sum(), 
#             sep='|'
#         ).str.strip('|')
    
#     # Fill empty strings with 'Other'
#     df['sectors'] = df['sectors'].replace('', 'Other')
#     df['top_categories'] = df['top_categories'].replace('', 'Other')
    
#     # Add counts
#     df['category_count'] = categories.str.count(r'\|').fillna(0) + 1
#     df['sector_count'] = df['sectors'].str.count(r'\|').fillna(0) + 1
    
#     return df

# if __name__ == "__main__":
#     # Read the cleaned data
#     df = pd.read_csv('data/interim/cleaned_startups.csv')
    
#     # Convert date columns individually
#     date_cols = ['founded_at', 'first_funding_at', 'last_funding_at']
#     for col in date_cols:
#         df[col] = pd.to_datetime(df[col])
    
#     # Create features
#     featured_df = create_time_features(df)
#     featured_df = create_category_features(df)
    
#     # Save to processed data
#     featured_df.to_csv('data/processed/featured_startups.csv', index=False)

import pandas as pd

def create_time_features(df):
    """
    Create time-based features from clean date columns.
    Assumes dates are already validated and follow: founded_at <= first_funding_at <= last_funding_at
    """
    # Time intervals
    df['days_to_first_funding'] = (df['first_funding_at'] - df['founded_at']).dt.days
    df['funding_duration'] = (df['last_funding_at'] - df['first_funding_at']).dt.days
    df['company_age_at_last_funding'] = (df['last_funding_at'] - df['founded_at']).dt.days

    # Economic period indicators
    df['is_founded_recession_2008'] = (
        ((df['founded_at'] >= '2008-09-01') & (df['founded_at'] <= '2009-06-30'))
    ).astype(int)
    df['is_founded_covid'] = (df['founded_at'] >= '2020-01-01').astype(int)

    # Extract year and quarter for time-based analysis
    df['founded_year'] = df['founded_at'].dt.year
    df['founded_quarter'] = df['founded_at'].dt.quarter
    
    # Funding round timing
    df['first_funding_year'] = df['first_funding_at'].dt.year
    df['last_funding_year'] = df['last_funding_at'].dt.year
    
    return df

# Define sectors based on actual data distribution
primary_sectors = {
    'Tech': ['Software', 'Enterprise Software', 'SaaS', 'Hardware + Software', 
             'Technology', 'Cloud Computing', 'Security', 'Internet'],
    
    'Mobile': ['Mobile', 'Apps'],
    
    'Commerce': ['E-Commerce', 'Marketplaces', 'Fashion', 'Travel', 'Sales and Marketing'],
    
    'Health': ['Biotechnology', 'Health and Wellness', 'Health Care'],
    
    'Media': ['Social Media', 'Curated Web', 'Advertising', 'Video'],
    
    'Analytics': ['Analytics', 'Big Data'],
    
    'Consumer': ['Games', 'Services', 'Manufacturing'],
    
    'Education': ['Education'],
    
    'Financial': ['Finance', 'FinTech', 'Financial Services', 'Payments', 
                 'Banking', 'Insurance', 'Investment']
}

def create_primary_sector_features(df):
   """
   Create simplified sector-based features based on actual data distribution
   """
   df = df.copy()
   
   # Initialize primary_sector as 'Other'  
   df['primary_sector'] = 'Other'
   
   # Fill NaN values in category_list
   categories = df['category_list'].fillna('')
   
   # For each sector, check if any of its keywords are in the category list
   for sector, keywords in primary_sectors.items():
       pattern = '|'.join(keywords)
       mask = categories.str.contains(pattern, case=False, na=False)
       df.loc[mask, 'primary_sector'] = sector
   
   # Keep the original category count as it might be useful
   df['category_count'] = categories.str.count(r'\|').fillna(0) + 1
   
   return df

if __name__ == "__main__":
   # Read the cleaned data
   df = pd.read_csv('data/interim/cleaned_startups.csv')
   
   # Convert date columns individually
   date_cols = ['founded_at', 'first_funding_at', 'last_funding_at']
   for col in date_cols:
       df[col] = pd.to_datetime(df[col])
   
   # Create features
   featured_df = create_time_features(df)
   featured_df = create_primary_sector_features(featured_df)
   
   # Print distribution of primary sectors
   print("\nDistribution of Primary Sectors:")
   print(featured_df['primary_sector'].value_counts(normalize=True))
   
   # Save to processed data
   featured_df.to_csv('data/processed/featured_startups.csv', index=False)