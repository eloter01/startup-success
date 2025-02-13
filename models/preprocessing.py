from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

# # Define constants
# NUMERICAL_COLS = [
#     'funding_total_usd', 'funding_rounds', 'days_to_first_funding',
#     'funding_duration', 'company_age_at_last_funding', 'founded_year',
#     'founded_quarter', 'first_funding_year', 'last_funding_year',
#     'category_count'
# ]

# CATEGORICAL_COLS = ['country_code', 'region', 'primary_sector']

NUMERICAL_COLS = [
    'funding_total_usd', 'funding_rounds', 'days_to_first_funding',
    'funding_duration', 
    'founded_quarter', 'last_funding_year',
    'category_count'
]

CATEGORICAL_COLS = ['country_code', 'primary_sector']

def prepare_features(df, target_col='status'):
    """
    Prepare features by selecting and encoding columns.
    """
    # Create copy of selected features
    feature_columns = NUMERICAL_COLS + CATEGORICAL_COLS
    df_model = df[feature_columns + [target_col]].copy()
    
    # Handle categorical variables
    df_encoded = pd.get_dummies(df_model, columns=CATEGORICAL_COLS)
    
    return df_encoded

def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to balance classes.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    
    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit and transform training data
    X_train_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    
    # Transform test data
    X_test_scaled[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])
    
    return X_train_scaled, X_test_scaled, scaler

def prepare_data(df, target_col='status', test_size=0.2, random_state=42):
    """
    Main function to prepare data for training.
    """
    # Prepare features
    df_encoded = prepare_features(df, target_col)
    
    # Split features and target
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    # Apply SMOTE
    X_resampled, y_resampled = apply_smote(X, y, random_state)
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_resampled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_encoded,
        test_size=test_size,
        random_state=random_state
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler