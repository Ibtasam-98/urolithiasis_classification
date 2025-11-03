import os
import numpy as np
import random
import tensorflow as tf
import pandas as pd
from config import config

def set_seeds():
    """Set all random seeds for reproducibility"""
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

def save_models(models_dict):
    """Save all models to disk"""
    for model_name, model in models_dict.items():
        model_path = os.path.join(config.MODELS_DIR, f"{model_name.lower()}_model.keras")
        model.save(model_path)
        print(f"{model_name} model saved to {model_path}")

def print_final_comparison(results_dict, model_names):
    """Print final comparison of all models"""
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    # Create a comprehensive comparison table
    comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                          'npv', 'fnr', 'fpr', 'jaccard', 'kappa', 'fmi']


    print(f"{'Metric':<15}", end="")
    for model in model_names:
        print(f"{model:>12}", end="")
    print()

    print("-" * (15 + 12 * len(model_names)))
    for metric in comparison_metrics:
        print(f"{metric:<15}", end="")
        for model in model_names:
            value = results_dict[model]['metrics'][metric]
            print(f"{value:>12.4f}", end="")
        print()

    print("\n" + "=" * 60)

    print(f"\nVisualizations saved to: {config.SAVE_FIGS_DIR}/")
    print("Training history plots:")
    print("  - unified_training_history.png")

    print("\nComparison visualizations:")
    print("  - radar_comparison_chart.png")
    print("  - metrics_comparison_heatmap.png")
    print("  - error_rates_comparison.png")
    print("  - performance_metrics_bar_chart.png")
    print("  - roc_curve_comparison.png")
    print("  - pr_curve_comparison.png")

    print("\nEpoch comparison visualizations:")
    print("  - epoch_comparison_final.png")
    print("  - convergence_speed.png")

    print("\nArchitecture diagrams:")
    print("  - architecture_diagrams/DNN_architecture.png")
    print("  - architecture_diagrams/MLP_architecture.png")
    print("  - architecture_diagrams/AE-DNN_architecture.png")

    print("\nAll visualizations and models saved successfully!")