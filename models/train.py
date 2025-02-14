import pandas as pd
import joblib
from preprocessing import prepare_data
from models import get_all_models
from evaluation import evaluate_models

def save_artifacts(models_dict, best_model, label_encoder, scaler, X_test, y_test):
    """
    Save all models, preprocessing objects, and test data.
    """
    # Save all models
    for name, result in models_dict.items():
        joblib.dump(result['model'], f'models/{name}_model.pkl')
    
    # Save best model separately
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Save preprocessors
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save test data
    joblib.dump(X_test, 'models/X_test.pkl')
    joblib.dump(y_test, 'models/y_test.pkl')

def main():
    # Read data
    df = pd.read_csv('data/processed/startups_reduced_multicollinearity.csv')
    
    # Optional: Filter for specific countries
    # df = df[df['country_code'].isin(['USA', 'GBR', 'CAN'])]
    
    # Prepare data
    X_train, X_test, y_train, y_test, label_encoder, scaler = prepare_data(df)
    
    # Get models
    models = get_all_models()
    
    # Train and evaluate all models
    results, best_model = evaluate_models(
        models, X_train, X_test, y_train, y_test, label_encoder
    )
    
    # Save all models, preprocessors, and test data
    save_artifacts(results, best_model, label_encoder, scaler, X_test, y_test)

if __name__ == "__main__":
    main()