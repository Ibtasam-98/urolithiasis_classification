import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, jaccard_score, fowlkes_mallows_score
)

def calculate_all_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive evaluation metrics"""
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # NPV (Negative Predictive Value)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # FDR (False Discovery Rate)
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0

    # FNR (False Negative Rate)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

    # FPR (False Positive Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Error rate
    error_rate = (fp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # FOR (False Omission Rate)
    for_rate = fn / (fn + tn) if (fn + tn) > 0 else 0

    # Markedness
    markedness = (prec + npv - 1)

    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Jaccard index
    jaccard = jaccard_score(y_true, y_pred, average='weighted')

    # Fowlkes-Mallows index
    fmi = fowlkes_mallows_score(y_true, y_pred)

    # Informedness
    informedness = rec + specificity - 1

    # Negative Likelihood Ratio
    nlr = fnr / specificity if specificity > 0 else 0

    # ROC AUC
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr_roc, tpr_roc)

    # PR AUC
    precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_pr, precision_pr)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'specificity': specificity,
        'npv': npv,
        'fdr': fdr,
        'fnr': fnr,
        'fpr': fpr,
        'error': error_rate,
        'for': for_rate,
        'markedness': markedness,
        'mcc': mcc,
        'kappa': kappa,
        'jaccard': jaccard,
        'fmi': fmi,
        'informedness': informedness,
        'nlr': nlr,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def evaluate_model(model, train_ds, val_ds, test_ds, model_name, class_names, history=None):
    """Evaluate a model and return comprehensive metrics"""
    # Evaluate metrics
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # Get predictions
    stone_idx = class_names.index('stone') if 'stone' in class_names else 1
    y_true, y_probs = [], []
    for x, y in val_ds:
        y_true.append(y.numpy())
        y_probs.append(model.predict(x, verbose=0)[:, stone_idx])
    y_true, y_probs = np.concatenate(y_true), np.concatenate(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    # Calculate all metrics
    metrics = calculate_all_metrics(y_true, y_pred, y_probs)

    # Print report
    print("\n" + "=" * 60)
    print(f"MODEL EVALUATION: {model_name.upper()}")
    print("=" * 60)
    print(f"\nTrain Accuracy:    {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Test Accuracy:     {test_acc * 100:.2f}%")

    print("\nKey Metrics:")
    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'npv', 'fnr', 'fpr', 'jaccard',
                   'kappa', 'fmi']
    for metric in key_metrics:
        value = metrics[metric]
        print(f"{metric:15}: {value:.4f}")

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "metrics": metrics,
        "history": history.history if history else None,
        "y_true": y_true,
        "y_probs": y_probs
    }