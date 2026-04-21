import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def evaluate(model, test_loader, device, threshold=0.5):
    model.eval()

    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            
            outputs = model(batch_imgs)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_probs = np.array(all_probs).squeeze()
    all_labels = np.array(all_labels).squeeze()
    all_preds = (all_probs >= threshold).astype(int)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # Calculate evaluation metrics (e.g., accuracy, precision, recall, F1-score)
    evaluation_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary'),
        'f1_score': f1_score(all_labels, all_preds, average='binary'),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'fpr': fpr,  # False Positive Rate
        'tpr': tpr,  # True Positive Rate
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

    return evaluation_metrics

def print_results(metrics, model_name):
    print(f"\n{'='*40}")
    print(f"  {model_name} Results")
    print(f"{'='*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"{'='*40}\n")