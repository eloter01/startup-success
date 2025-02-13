import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def set_style():
    """Set consistent style for all plots"""
    sns.set_theme(style="whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_model_comparison(results_dict, save_path=None):
    """
    Create bar plot comparing model performances.
    """
    # Prepare data
    models = []
    metrics = []
    values = []
    
    for model_name, result in results_dict.items():
        models.extend([model_name] * 3)
        metrics.extend(['Balanced Accuracy', 'F1 Score', 'CV Score'])
        values.extend([
            result['balanced_accuracy'] * 100,
            result['f1_score'] * 100,
            result['cv_mean'] * 100
        ])
    
    df = pd.DataFrame({
        'Model': models,
        'Metric': metrics,
        'Value': values
    })
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df)
    
    # Customize plot
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Performance (%)')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(y_test, predictions_dict, class_names, save_path=None):
    """
    Create confusion matrix plots for all models.
    """
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, predictions) in zip(axes, predictions_dict.items()):
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        cm_percentage = cm / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Plot confusion matrix
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{model_name}\nConfusion Matrix (%)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_performance_radar(best_model_results, class_names, save_path=None):
    """
    Create radar plot for best model's performance across classes.
    """
    # Set figure size
    plt.figure(figsize=(10, 10))
    
    # Calculate number of variables
    num_vars = len(class_names)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the spider plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], class_names)
    
    # Plot data
    metrics = {
        'Precision': [best_model_results[cn]['precision'] * 100 for cn in class_names],
        'Recall': [best_model_results[cn]['recall'] * 100 for cn in class_names],
        'F1-Score': [best_model_results[cn]['f1-score'] * 100 for cn in class_names]
    }
    
    # Add the first value to each list to close the circular plot
    for metric in metrics:
        metrics[metric].append(metrics[metric][0])
    
    # Plot each metric
    for metric_name, metric_values in metrics.items():
        ax.plot(angles, metric_values, 'o-', linewidth=2, label=metric_name)
        ax.fill(angles, metric_values, alpha=0.25)
    
    # Set chart title and legend
    plt.title("Random Forest Performance by Class", size=20, y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set axis limits
    ax.set_ylim(0, 100)
    
    # Add gridlines
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_figure(results_dict, y_test, predictions_dict, class_names, save_path=None):
    """
    Create a comprehensive summary figure with all visualizations.
    """
    set_style()
    
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2)
    
    # Model Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Confusion Matrix for Best Model (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Class Performance Radar (bottom)
    ax3 = fig.add_subplot(gs[1, :], projection='polar')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    set_style()
    
    # Example data
    results_dict = {
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
    
    # # Create test plots
    # plot_model_comparison(results_dict, 'figures/model_comparison.png')