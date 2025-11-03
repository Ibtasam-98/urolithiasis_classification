import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve
from tensorflow.keras.utils import plot_model
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


def plot_unified_training_history(histories, model_names, save_dir):
    """Plot training history for all models in a single figure with HD quality"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Define colors for each model
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3C91E6']

    # Accuracy plot
    for i, (history, model_name) in enumerate(zip(histories, model_names)):
        ax1.plot(history['accuracy'], label=f'{model_name} Train', linewidth=2.5, color=colors[i])
        ax1.plot(history['val_accuracy'], '--', label=f'{model_name} Validation',
                 linewidth=2.5, color=colors[i], alpha=0.8)

    ax1.set_title('Model Accuracy Comparison', fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Loss plot
    for i, (history, model_name) in enumerate(zip(histories, model_names)):
        ax2.plot(history['loss'], label=f'{model_name} Train', linewidth=2.5, color=colors[i])
        ax2.plot(history['val_loss'], '--', label=f'{model_name} Validation',
                 linewidth=2.5, color=colors[i], alpha=0.8)

    ax2.set_title('Model Loss Comparison', fontsize=18, fontweight='bold', pad=20)
    ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "unified_training_history.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "unified_training_history.pdf"), bbox_inches='tight')
    plt.close()


def create_comparison_visualizations(results_dict, save_dir):
    """Create enhanced comparison visualizations with HD quality"""
    model_names = list(results_dict.keys())

    # Define the metrics to visualize
    metrics_to_plot = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'specificity', 'npv', 'jaccard', 'kappa', 'fmi'
    ]

    # Error metrics (lower is better)
    error_metrics = ['fnr', 'fpr']

    # Create a DataFrame for easier plotting
    df_data = []
    for model_name, results in results_dict.items():
        row = {'Model': model_name}
        row.update(results['metrics'])
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Custom color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3C91E6']

    # 1. Enhanced Radar Chart for Key Metrics
    plt.figure(figsize=(12, 10))
    radar_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'npv']
    radar_df = df.set_index('Model')[radar_metrics]

    # Normalize values for radar chart (0-1)
    radar_df_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    for idx, (model, color) in enumerate(zip(radar_df_norm.index, colors)):
        values = radar_df_norm.loc[model].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=3, label=model, color=color, marker='o', markersize=6)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), [m.upper() for m in radar_metrics], fontsize=13)
    ax.set_title('COMPREHENSIVE MODEL PERFORMANCE RADAR CHART\n(Normalized Key Metrics)',
                 fontsize=18, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12,
              frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "radar_comparison_chart.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "radar_comparison_chart.pdf"), bbox_inches='tight')
    plt.close()

    # 2. Enhanced Heatmap for all metrics
    plt.figure(figsize=(16, 10))
    metrics_df = df.set_index('Model')[metrics_to_plot]

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#D32F2F", "#FFFFFF", "#1976D2"])

    # Create heatmap with annotations
    ax = sns.heatmap(metrics_df, annot=True, cmap=cmap, fmt='.4f', annot_kws={'size': 11, 'weight': 'bold'},
                     cbar_kws={'label': 'Metric Value', 'shrink': 0.8}, center=0.5, vmin=0, vmax=1,
                     linewidths=1, linecolor='white')

    plt.title('COMPREHENSIVE MODEL METRICS COMPARISON HEATMAP\n(Higher Values Indicate Better Performance)',
              fontsize=18, fontweight='bold', pad=25)
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Metric Value', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison_heatmap.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "metrics_comparison_heatmap.pdf"), bbox_inches='tight')
    plt.close()

    # 3. Enhanced Error Rates Visualization
    plt.figure(figsize=(14, 8))
    error_df = df.set_index('Model')[error_metrics]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width / 2, error_df['fnr'], width, label='False Negative Rate (FNR)',
                   color='#D32F2F', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width / 2, error_df['fpr'], width, label='False Positive Rate (FPR)',
                   color='#1976D2', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error Rate', fontsize=14, fontweight='bold')
    ax.set_title('FALSE POSITIVE AND FALSE NEGATIVE RATES COMPARISON\n(Lower Values Indicate Better Performance)',
                 fontsize=16, fontweight='bold', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_rates_comparison.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "error_rates_comparison.pdf"), bbox_inches='tight')
    plt.close()

    # 4. Performance Metrics Bar Chart (Accuracy, Precision, Recall, F1)
    plt.figure(figsize=(16, 10))
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    performance_df = df.set_index('Model')[performance_metrics]

    x = np.arange(len(performance_metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 10))
    for i, model in enumerate(model_names):
        ax.bar(x + i * width, performance_df.loc[model], width, label=model,
               color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('CORE PERFORMANCE METRICS COMPARISON ACROSS MODELS',
                 fontsize=18, fontweight='bold', pad=25)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in performance_metrics], fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for i, model in enumerate(model_names):
        for j, metric in enumerate(performance_metrics):
            height = performance_df.loc[model, metric]
            ax.text(x[j] + i * width, height + 0.01, f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_metrics_bar_chart.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "performance_metrics_bar_chart.pdf"), bbox_inches='tight')
    plt.close()

    # 5. ROC Curve Comparison
    plt.figure(figsize=(12, 10))
    for model_name, color in zip(model_names, colors):
        metrics = results_dict[model_name]['metrics']
        fpr, tpr, _ = roc_curve(results_dict[model_name]['y_true'], results_dict[model_name]['y_probs'])
        roc_auc = metrics['roc_auc']
        plt.plot(fpr, tpr, color=color, lw=3, label=f'{model_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('RECEIVER OPERATING CHARACTERISTIC (ROC) CURVES\n(Area Under Curve Comparison)',
              fontsize=16, fontweight='bold', pad=25)
    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve_comparison.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "roc_curve_comparison.pdf"), bbox_inches='tight')
    plt.close()

    # 6. Precision-Recall Curve Comparison
    plt.figure(figsize=(12, 10))
    for model_name, color in zip(model_names, colors):
        metrics = results_dict[model_name]['metrics']
        precision, recall, _ = precision_recall_curve(results_dict[model_name]['y_true'],
                                                      results_dict[model_name]['y_probs'])
        pr_auc = metrics['pr_auc']
        plt.plot(recall, precision, color=color, lw=3, label=f'{model_name} (AUC = {pr_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('PRECISION-RECALL CURVES COMPARISON\n(Balanced Performance Assessment)',
              fontsize=16, fontweight='bold', pad=25)
    plt.legend(loc="lower left", frameon=True, fancybox=True, shadow=True, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_curve_comparison.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "pr_curve_comparison.pdf"), bbox_inches='tight')
    plt.close()


def plot_epoch_comparison(histories, model_names, save_dir):
    """Create visualization comparing model performance across epochs with HD quality"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Define colors for each model
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3C91E6']

    # Final epoch values
    final_acc = [history['val_accuracy'][-1] for history in histories]
    final_loss = [history['val_loss'][-1] for history in histories]

    # Convergence speed (epochs to reach 90% of final accuracy)
    convergence_epochs = []
    for history in histories:
        target_acc = 0.9 * history['val_accuracy'][-1]
        for epoch, acc in enumerate(history['val_accuracy']):
            if acc >= target_acc:
                convergence_epochs.append(epoch)
                break
        else:
            convergence_epochs.append(len(history['val_accuracy']) - 1)

    # Plot final accuracy and loss
    x = np.arange(len(model_names))
    width = 0.4

    bars1 = ax1.bar(x - width / 2, final_acc, width, label='Final Validation Accuracy',
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('FINAL VALIDATION ACCURACY BY MODEL', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    bars2 = ax2.bar(x - width / 2, final_loss, width, label='Final Validation Loss',
                    color='#C73E1D', alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax2.set_title('FINAL VALIDATION LOSS BY MODEL', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "epoch_comparison_final.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "epoch_comparison_final.pdf"), bbox_inches='tight')
    plt.close()

    # Create convergence speed visualization
    plt.figure(figsize=(12, 8))
    bars = plt.bar(x, convergence_epochs, color=colors[:len(model_names)],
                   alpha=0.8, edgecolor='black', linewidth=1)
    plt.xlabel('Machine Learning Models', fontsize=14, fontweight='bold')
    plt.ylabel('Epochs to Reach 90% of Final Accuracy', fontsize=14, fontweight='bold')
    plt.title('MODEL CONVERGENCE SPEED ANALYSIS\n(Lower Values Indicate Faster Training)',
              fontsize=16, fontweight='bold', pad=25)
    plt.xticks(x, model_names, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_speed.png"), dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "convergence_speed.pdf"), bbox_inches='tight')
    plt.close()


def create_model_architecture_diagrams(models_dict, save_dir):
    """Create architecture diagrams for all models with HD quality"""
    arch_dir = os.path.join(save_dir, "architecture_diagrams")
    os.makedirs(arch_dir, exist_ok=True)

    for model_name, model in models_dict.items():
        try:
            # Create the architecture diagram
            plot_model(
                model,
                to_file=os.path.join(arch_dir, f"{model_name}_architecture.png"),
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=400,
                show_layer_activations=True,
                show_trainable=True
            )
            print(f"✓ HD Architecture diagram saved for {model_name}")
        except Exception as e:
            print(f"✗ Error creating architecture diagram for {model_name}: {e}")


# Additional function for comprehensive summary visualization
def create_comprehensive_summary(results_dict, save_dir):
    """Create a comprehensive summary visualization"""
    model_names = list(results_dict.keys())

    # Extract key metrics for summary
    summary_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    fig, ax = plt.subplots(figsize=(16, 10))

    # Prepare data
    data = []
    for model in model_names:
        row = [results_dict[model]['metrics'][metric] for metric in summary_metrics]
        data.append(row)

    # Create grouped bar chart
    x = np.arange(len(summary_metrics))
    width = 0.15

    for i, model in enumerate(model_names):
        offset = width * i
        bars = ax.bar(x + offset, data[i], width, label=model,
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, value in zip(bars, data[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('COMPREHENSIVE MODEL PERFORMANCE SUMMARY\n(All Key Metrics Comparison)',
                 fontsize=18, fontweight='bold', pad=25)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in summary_metrics], fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comprehensive_performance_summary.png"),
                dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "comprehensive_performance_summary.pdf"),
                bbox_inches='tight')
    plt.close()


print("HD Visualization module loaded with Times New Roman font and research paper quality settings.")