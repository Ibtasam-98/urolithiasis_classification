import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from config import config


# Configure matplotlib for research paper quality
def setup_matplotlib():
    """Setup matplotlib for HD research paper quality"""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3


setup_matplotlib()


def plot_single_combined_confusion_matrix(results_dict, class_names, save_dir):
    """Plot a single confusion matrix containing all models in one graph with HD quality"""
    model_names = list(results_dict.keys())
    n_models = len(model_names)

    # Create a single large figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define color palette for confusion matrix
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", "#2E86AB", "#1B5E80"])

    # Calculate combined confusion matrix
    combined_cm = None
    combined_cm_normalized = None

    for model_name in model_names:
        # Get true labels and predictions
        y_true = results_dict[model_name]['y_true']
        y_pred = results_dict[model_name]['y_pred']

        # Compute confusion matrix for current model
        cm = confusion_matrix(y_true, y_pred)

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Initialize or add to combined matrices
        if combined_cm is None:
            combined_cm = cm
            combined_cm_normalized = cm_normalized
        else:
            combined_cm += cm
            combined_cm_normalized += cm_normalized

    # Average the normalized matrices
    combined_cm_normalized = combined_cm_normalized / n_models

    # Create heatmap
    im = ax.imshow(combined_cm_normalized, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    # Add text annotations
    thresh = combined_cm_normalized.max() / 2.
    for i in range(combined_cm_normalized.shape[0]):
        for j in range(combined_cm_normalized.shape[1]):
            ax.text(j, i,
                    f"{combined_cm_normalized[i, j]:.3f}\n({combined_cm[i, j]})",
                    ha="center", va="center",
                    color="white" if combined_cm_normalized[i, j] > thresh else "black",
                    fontsize=18, fontweight='bold')

    # Customize plot
    model_list = ", ".join(model_names)
    # ax.set_title(f'COMBINED CONFUSION MATRIX\nModels: {model_list}',
    #              fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=18, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=18, fontweight='bold')

    # Set tick labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(class_names, fontsize=18)

    # Add grid
    ax.grid(False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label('Normalized Value (Average across models)', fontsize=12, fontweight='bold')

    # Add model information as text
    model_info = f"Models included ({n_models}): {model_list}"
    fig.text(0.5, 0.01, model_info, ha='center', fontsize=18, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make space for the model info text

    # Save the figure
    plt.savefig(os.path.join(save_dir, "combined_confusion_matrix.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "combined_confusion_matrix.pdf"), bbox_inches='tight')
    plt.close()

    print("✓ Single combined confusion matrix saved with HD quality")


def plot_accuracy_comparison(results_dict, save_dir):
    """Create comprehensive accuracy comparison visualization with HD quality"""
    model_names = list(results_dict.keys())

    # Extract accuracy values
    accuracies = [results_dict[model]['metrics']['accuracy'] for model in model_names]

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies
    })

    # Sort by accuracy for better visualization
    df = df.sort_values('Accuracy', ascending=False)

    # Custom color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3C91E6']

    # Create the main accuracy comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 1. Bar chart for accuracy comparison
    bars = ax1.bar(df['Model'], df['Accuracy'],
                   color=colors[:len(model_names)],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_title('MODEL ACCURACY COMPARISON', fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels for better readability
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right', fontweight='bold')

    # Add value annotations on bars
    for bar, accuracy in zip(bars, df['Accuracy']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{accuracy:.4f}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    # 2. Enhanced performance metrics comparison (focus on accuracy-related metrics)
    accuracy_metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Prepare data for grouped bar chart
    x = np.arange(len(accuracy_metrics))
    width = 0.15

    for i, model in enumerate(model_names):
        metrics_values = [results_dict[model]['metrics'][metric] for metric in accuracy_metrics]
        offset = width * i
        bars = ax2.bar(x + offset, metrics_values, width, label=model,
                       alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, value in zip(bars, metrics_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax2.set_title('DETAILED PERFORMANCE METRICS COMPARISON',
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax2.set_xticklabels([m.upper() for m in accuracy_metrics], fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.15)

    plt.tight_layout()

    # Save the figures
    plt.savefig(os.path.join(save_dir, "accuracy_comparison_comprehensive.png"),
                dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "accuracy_comparison_comprehensive.pdf"),
                bbox_inches='tight')
    plt.close()

    # Create a separate simple accuracy comparison for quick reference
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Model'], df['Accuracy'],
                   color=colors[:len(model_names)],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    plt.title('MODEL ACCURACY COMPARISON\n(Higher is Better)',
              fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('Accuracy Score', fontsize=16, fontweight='bold')
    plt.xlabel('Machine Learning Models', fontsize=16, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontweight='bold')

    # Add value annotations
    for bar, accuracy in zip(bars, df['Accuracy']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.015,
                 f'{accuracy:.4f}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_comparison_simple.png"),
                dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "accuracy_comparison_simple.pdf"),
                bbox_inches='tight')
    plt.close()

    print("✓ Accuracy comparison visualizations saved with HD quality")


def create_comparison_visualizations(results_dict, class_names, save_dir):
    """Main function to create both single confusion matrix and accuracy comparison visualizations"""

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # 1. Create single combined confusion matrix for all models
    plot_single_combined_confusion_matrix(results_dict, class_names, save_dir)

    # 2. Create accuracy comparison visualizations
    plot_accuracy_comparison(results_dict, save_dir)

    print("✓ All comparison visualizations completed successfully!")


# Additional utility function for quick results summary
def print_results_summary(results_dict):
    """Print a clean summary of model accuracies"""
    print("\n" + "=" * 60)
    print("MODEL ACCURACY SUMMARY")
    print("=" * 60)

    for model_name, results in results_dict.items():
        accuracy = results['metrics']['accuracy']
        print(f"📊 {model_name:<20}: {accuracy:.4f}")

    print("=" * 60)

    # Find best model
    best_model = max(results_dict.keys(),
                     key=lambda x: results_dict[x]['metrics']['accuracy'])
    best_accuracy = results_dict[best_model]['metrics']['accuracy']

    print(f"🏆 BEST MODEL: {best_model} (Accuracy: {best_accuracy:.4f})")
    print("=" * 60)


print("Simplified HD Visualization module loaded successfully!")
print("Available visualizations: Single Combined Confusion Matrix & Accuracy Comparison")