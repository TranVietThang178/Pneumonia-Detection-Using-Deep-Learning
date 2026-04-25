"""
evaluate.py
-----------
Evaluation utilities for the chest X-ray pneumonia classification project.

Functions
---------
evaluate(model, test_loader, device, threshold)
    Run inference on a test set and return a dictionary of evaluation metrics.

print_results(metrics, model_name)
    Print a formatted summary of evaluation metrics to stdout.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, device, threshold=0.5):
    """
    Evaluate a trained model on a test set and compute classification metrics.

    Inference is performed without gradient computation. Predicted probabilities
    are obtained by applying sigmoid to the raw model output, then thresholded
    to produce binary class predictions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    test_loader : DataLoader
        DataLoader for the test set.
    device : torch.device
        Device to run inference on (CPU or CUDA).
    threshold : float, optional
        Decision threshold for converting probabilities to class labels.
        Default is 0.5.

    Returns
    -------
    metrics : dict
        Dictionary containing the following keys:
            'accuracy'         : float
            'precision'        : float
            'recall'           : float
            'f1_score'         : float
            'roc_auc'          : float
            'fpr'              : np.ndarray  (False Positive Rate)
            'tpr'              : np.ndarray  (True Positive Rate)
            'confusion_matrix' : np.ndarray  (shape 2x2)
    """
    model.eval()

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs   = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_imgs)
            probs   = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    all_probs  = np.array(all_probs).squeeze()
    all_labels = np.array(all_labels).squeeze()
    all_preds  = (all_probs >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    metrics = {
        'accuracy':         accuracy_score(all_labels, all_preds),
        'precision':        precision_score(all_labels, all_preds, average='binary'),
        'recall':           recall_score(all_labels, all_preds, average='binary'),
        'f1_score':         f1_score(all_labels, all_preds, average='binary'),
        'roc_auc':          roc_auc_score(all_labels, all_probs),
        'fpr':              fpr,
        'tpr':              tpr,
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
    }

    return metrics


# ---------------------------------------------------------------------------
# Results Printer
# ---------------------------------------------------------------------------

def print_results(metrics, model_name):
    """
    Print a formatted summary of evaluation metrics to stdout.

    Parameters
    ----------
    metrics : dict
        Output of the evaluate() function.
    model_name : str
        Display name for the model (used in the header).
    """
    separator = '=' * 40
    print(f"\n{separator}")
    print(f"  {model_name} Results")
    print(separator)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"{separator}\n")