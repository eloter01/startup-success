from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
import pandas as pd

def evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder, model_name):
    """
    Evaluate a single model and return metrics.
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Store results
    results = {
        'model': model,
        'balanced_accuracy': bal_accuracy,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    # Print results
    print(f"\n{model_name.upper()} Results:")
    print(classification_report(y_test, y_pred, 
                             target_names=label_encoder.classes_))
    print(f"Balanced Accuracy: {bal_accuracy:.3f}")
    print(f"Weighted F1-Score: {f1:.3f}")
    print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return results

def evaluate_models(models, X_train, X_test, y_train, y_test, label_encoder):
    """
    Evaluate multiple models and return results with best model.
    """
    results = {}
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        results[name] = evaluate_model(
            model, X_train, X_test, y_train, y_test, label_encoder, name
        )
        
        # Track best model
        if results[name]['balanced_accuracy'] > best_score:
            best_score = results[name]['balanced_accuracy']
            best_model = model
    
    print_summary(results, best_model)
    
    return results, best_model

def print_summary(results, best_model):
    """
    Print summary of model comparison.
    """
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Model': name,
            'Balanced Accuracy': f"{res['balanced_accuracy']:.3f}",
            'F1 Score': f"{res['f1_score']:.3f}",
            'CV Score': f"{res['cv_mean']:.3f} (+/- {res['cv_std'] * 2:.3f})"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))
    
    best_name = [name for name, res in results.items() 
                if res['model'] == best_model][0]
    print(f"\nBest Model: {best_name}")
    print(f"Balanced Accuracy: {results[best_name]['balanced_accuracy']:.3f}")
    print(f"F1 Score: {results[best_name]['f1_score']:.3f}")