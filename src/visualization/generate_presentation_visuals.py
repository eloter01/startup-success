import os
import joblib
import pandas as pd
import numpy as np
from visualize import (
    set_style, plot_model_comparison, plot_confusion_matrices,
    plot_class_performance_radar, create_summary_figure
)

def load_results():
    """
    Load model results and create results dictionary.
    """
    results = {
        'logistic_regression': {
            'balanced_accuracy': 0.789,
            'f1_score': 0.784,
            'cv_mean': 0.790,
            'cv_std': 0.004
        },
        'random_forest': {
            'balanced_accuracy': 0.928,
            'f1_score': 0.927,
            'cv_mean': 0.918,
            'cv_std': 0.002
        },
        'xgboost': {
            'balanced_accuracy': 0.788,
            'f1_score': 0.785,
            'cv_mean': 0.790,
            'cv_std': 0.004
        }
    }
    return results

def load_predictions():
    """
    Load models and generate predictions.
    """
    try:
        # Load test data and true labels
        X_test = joblib.load('models/X_test.pkl')
        y_test = joblib.load('models/y_test.pkl')
        
        # Generate predictions for each model
        predictions = {}
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            model = joblib.load(f'models/{model_name}_model.pkl')
            predictions[model_name] = model.predict(X_test)
        
        return y_test, predictions
    except FileNotFoundError:
        print("Warning: Test data or models not found. Skipping confusion matrices.")
        return None, None

def get_class_names():
    """
    Get class names from label encoder.
    """
    try:
        label_encoder = joblib.load('models/label_encoder.pkl')
        return list(label_encoder.classes_)
    except FileNotFoundError:
        return ['acquired', 'closed', 'ipo', 'operating']

def main():
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Set style for all plots
    set_style()
    
    # Load results and class names
    results = load_results()
    class_names = get_class_names()
    
    # Plot model comparison
    plot_model_comparison(results, 'figures/model_comparison.png')
    
    # Load predictions and plot confusion matrices if available
    y_test, predictions = load_predictions()
    if y_test is not None and predictions is not None:
        plot_confusion_matrices(y_test, predictions, class_names, 
                              'figures/confusion_matrices.png')
    
    # Performance metrics for Random Forest (best model)
    best_model_results = {
        'acquired': {'precision': 0.90, 'recall': 0.94, 'f1-score': 0.92},
        'closed': {'precision': 0.93, 'recall': 0.89, 'f1-score': 0.91},
        'ipo': {'precision': 0.98, 'recall': 0.99, 'f1-score': 0.98},
        'operating': {'precision': 0.90, 'recall': 0.90, 'f1-score': 0.90}
    }
    
    # Plot radar chart
    plot_class_performance_radar(best_model_results, class_names, 
                               'figures/radar_plot.png')
    
    # Generate summary figure if all data is available
    if y_test is not None and predictions is not None:
        create_summary_figure(results, y_test, predictions, class_names,
                            'figures/summary.png')

if __name__ == "__main__":
    main()