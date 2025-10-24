import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve
from tensorflow.keras.utils import plot_model
from config import config

def plot_unified_training_history(histories, model_names, save_dir):
    """Plot training history for all models in a single figure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors for each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']

    # Accuracy plot
    for i, (history, model_name) in enumerate(zip(histories, model_names)):
        ax1.plot(history['accuracy'], label=f'{model_name} Train', linewidth=2, color=colors[i])
        ax1.plot(history['val_accuracy'], '--', label=f'{model_name} Val', linewidth=2, color=colors[i], alpha=0.7)

    ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    for i, (history, model_name) in enumerate(zip(histories, model_names)):
        ax2.plot(history['loss'], label=f'{model_name} Train', linewidth=2, color=colors[i])
        ax2.plot(history['val_loss'], '--', label=f'{model_name} Val', linewidth=2, color=colors[i], alpha=0.7)

    ax2.set_title('Model Loss Comparison', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "unified_training_history.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_visualizations(results_dict, save_dir):
    """Create enhanced comparison visualizations"""
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
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']

    # 1. Enhanced Radar Chart for Key Metrics
    plt.figure(figsize=(10, 8))
    radar_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'npv']
    radar_df = df.set_index('Model')[radar_metrics]

    # Normalize values for radar chart (0-1)
    radar_df_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for idx, (model, color) in enumerate(zip(radar_df_norm.index, colors)):
        values = radar_df_norm.loc[model].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=3, label=model, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=12)
    ax.set_title('Model Performance Radar Chart\n(Normalized Key Metrics)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "radar_comparison_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Enhanced Heatmap for all metrics
    plt.figure(figsize=(14, 8))
    metrics_df = df.set_index('Model')[metrics_to_plot]

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom", ["#FF6B6B", "#FFFFFF", '#4ECDC4'])

    # Create heatmap with annotations
    sns.heatmap(metrics_df, annot=True, cmap=cmap, fmt='.3f',
                cbar_kws={'label': 'Metric Value'}, center=0.5, vmin=0, vmax=1)
    plt.title('Comprehensive Model Metrics Comparison\n(Higher is Better for Most Metrics)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Enhanced Error Rates Visualization - Stacked Area Chart
    plt.figure(figsize=(12, 8))
    error_df = df.set_index('Model')[error_metrics]

    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot FNR and FPR as stacked areas
    x = range(len(model_names))
    fnr_values = error_df['fnr'].values
    fpr_values = error_df['fpr'].values

    ax.fill_between(x, 0, fnr_values, label='FNR', color='#FF6B6B', alpha=0.8)
    ax.fill_between(x, fnr_values, fnr_values + fpr_values, label='FPR', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Error Rates Comparison (Stacked Area Chart)\n(Lower is Better)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for i, model in enumerate(model_names):
        ax.text(i, fnr_values[i] / 2, f'{fnr_values[i]:.3f}', ha='center', va='center', fontsize=10, color='white',
                fontweight='bold')
        ax.text(i, fnr_values[i] + fpr_values[i] / 2, f'{fpr_values[i]:.3f}', ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_rates_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Performance Metrics Bar Chart (Accuracy, Precision, Recall, F1)
    plt.figure(figsize=(14, 8))
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    performance_df = df.set_index('Model')[performance_metrics]

    x = np.arange(len(performance_metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, model in enumerate(model_names):
        ax.bar(x + i * width, performance_df.loc[model], width, label=model, color=colors[i])

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(performance_metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, model in enumerate(model_names):
        for j, metric in enumerate(performance_metrics):
            height = performance_df.loc[model, metric]
            ax.text(x[j] + i * width, height + 0.01, f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_metrics_bar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. ROC Curve Comparison
    plt.figure(figsize=(10, 8))
    for model_name, color in zip(model_names, colors):
        metrics = results_dict[model_name]['metrics']
        fpr, tpr, _ = roc_curve(results_dict[model_name]['y_true'], results_dict[model_name]['y_probs'])
        roc_auc = metrics['roc_auc']
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Precision-Recall Curve Comparison
    plt.figure(figsize=(10, 8))
    for model_name, color in zip(model_names, colors):
        metrics = results_dict[model_name]['metrics']
        precision, recall, _ = precision_recall_curve(results_dict[model_name]['y_true'],
                                                      results_dict[model_name]['y_probs'])
        pr_auc = metrics['pr_auc']
        plt.plot(recall, precision, color=color, lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_curve_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_epoch_comparison(histories, model_names, save_dir):
    """Create visualization comparing model performance across epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors for each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

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
    width = 0.35

    bars1 = ax1.bar(x - width / 2, final_acc, width, label='Final Accuracy', color='#4ECDC4')
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Final Validation Accuracy by Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(x - width / 2, final_loss, width, label='Final Loss', color='#FF6B6B')
    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Final Validation Loss by Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "epoch_comparison_final.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create convergence speed visualization
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, convergence_epochs, color=colors[:len(model_names)])
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Epochs to 90% of Final Accuracy', fontsize=12)
    plt.title('Model Convergence Speed\n(Lower is Better)', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_speed.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_model_architecture_diagrams(models_dict, save_dir):
    """Create architecture diagrams for all models"""
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
                dpi=300
            )
            print(f"Architecture diagram saved for {model_name}")
        except Exception as e:
            print(f"Error creating architecture diagram for {model_name}: {e}")