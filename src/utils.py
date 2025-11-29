"""
Utility functions for training, evaluation, and visualization
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import shutil


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=15, delta=0.001):
        """
        Args:
            patience: How many epochs to wait after last improvement
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_checkpoint(state, filepath, is_best=False, model_dir=None):
    """Save model checkpoint"""
    torch.save(state, filepath)
    print(f"✓ Checkpoint saved: {filepath}")
    
    if is_best and model_dir is not None:
        best_path = Path(model_dir) / 'best_model.pth'
        shutil.copyfile(filepath, best_path)
        print(f"✓ Best model saved: {best_path}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Checkpoint loaded from: {filepath}")
    return checkpoint


def compute_metrics(labels, predictions, class_names):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        labels: Ground truth labels [N, num_classes]
        predictions: Model predictions [N, num_classes]
        class_names: List of class names
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Compute AUROC for each class
    class_auroc = []
    for i in range(labels.shape[1]):
        try:
            auroc = roc_auc_score(labels[:, i], predictions[:, i])
            class_auroc.append(auroc)
        except:
            class_auroc.append(0.0)
    
    metrics['class_auroc'] = class_auroc
    metrics['mean_auroc'] = np.mean(class_auroc)
    
    # Compute optimal thresholds using Youden's index
    optimal_thresholds = []
    for i in range(labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(labels[:, i], predictions[:, i])
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)
        optimal_thresholds.append(thresholds[optimal_idx])
    
    metrics['optimal_thresholds'] = optimal_thresholds
    
    # Apply thresholds to get binary predictions
    binary_preds = np.zeros_like(predictions)
    for i in range(labels.shape[1]):
        binary_preds[:, i] = (predictions[:, i] >= optimal_thresholds[i]).astype(int)
    
    # Compute sensitivity and specificity for each class
    sensitivities = []
    specificities = []
    f1_scores = []
    
    for i in range(labels.shape[1]):
        tn, fp, fn, tp = confusion_matrix(labels[:, i], binary_preds[:, i]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(labels[:, i], binary_preds[:, i])
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_scores.append(f1)
    
    metrics['sensitivities'] = sensitivities
    metrics['specificities'] = specificities
    metrics['f1_scores'] = f1_scores
    metrics['mean_sensitivity'] = np.mean(sensitivities)
    metrics['mean_specificity'] = np.mean(specificities)
    metrics['mean_f1'] = np.mean(f1_scores)
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(labels.ravel(), binary_preds.ravel())
    
    return metrics


def plot_training_history(history, save_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUROC curve
    axes[0, 1].plot(history['val_auroc'], label='Val AUROC', linewidth=2, color='green')
    axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='Target (0.85)')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('AUROC', fontsize=12)
    axes[0, 1].set_title('Validation AUROC', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rates'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    max_auroc = max(history['val_auroc'])
    min_val_loss = min(history['val_loss'])
    
    summary_text = f"""
    Training Summary:
    
    Best Val AUROC: {max_auroc:.4f}
    Min Val Loss: {min_val_loss:.4f}
    Total Epochs: {len(history['train_loss'])}
    
    Final Metrics:
    Train Loss: {history['train_loss'][-1]:.4f}
    Val Loss: {history['val_loss'][-1]:.4f}
    Val AUROC: {history['val_auroc'][-1]:.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved: {save_path}")


def plot_roc_curves(labels, predictions, class_names, save_path):
    """Plot ROC curves for all classes"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
        auroc = roc_auc_score(labels[:, i], predictions[:, i])
        
        axes[i].plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[i].set_xlabel('False Positive Rate', fontsize=10)
        axes[i].set_ylabel('True Positive Rate', fontsize=10)
        axes[i].set_title(f'{class_name}', fontsize=11, fontweight='bold')
        axes[i].legend(loc='lower right', fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curves saved: {save_path}")


def plot_confusion_matrices(labels, predictions, class_names, save_path, threshold=0.5):
    """Plot confusion matrices for all classes"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, class_name in enumerate(class_names):
        binary_preds = (predictions[:, i] >= threshold).astype(int)
        cm = confusion_matrix(labels[:, i], binary_preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[i].set_title(f'{class_name}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('True Label', fontsize=10)
        axes[i].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrices saved: {save_path}")


def plot_metrics_comparison(metrics, class_names, save_path):
    """Plot comparison of metrics across classes"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(class_names))
    width = 0.6
    
    # AUROC
    axes[0, 0].bar(x, metrics['class_auroc'], width, color='steelblue')
    axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='Target')
    axes[0, 0].set_ylabel('AUROC', fontsize=12)
    axes[0, 0].set_title('AUROC by Class', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Sensitivity
    axes[0, 1].bar(x, metrics['sensitivities'], width, color='green')
    axes[0, 1].axhline(y=0.90, color='r', linestyle='--', label='Target')
    axes[0, 1].set_ylabel('Sensitivity', fontsize=12)
    axes[0, 1].set_title('Sensitivity by Class', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Specificity
    axes[1, 0].bar(x, metrics['specificities'], width, color='orange')
    axes[1, 0].axhline(y=0.85, color='r', linestyle='--', label='Target')
    axes[1, 0].set_ylabel('Specificity', fontsize=12)
    axes[1, 0].set_title('Specificity by Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # F1 Score
    axes[1, 1].bar(x, metrics['f1_scores'], width, color='purple')
    axes[1, 1].set_ylabel('F1 Score', fontsize=12)
    axes[1, 1].set_title('F1 Score by Class', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics comparison saved: {save_path}")


def save_metrics_report(metrics, class_names, save_path):
    """Save detailed metrics report as JSON and text"""
    report = {
        'summary': {
            'mean_auroc': float(metrics['mean_auroc']),
            'mean_sensitivity': float(metrics['mean_sensitivity']),
            'mean_specificity': float(metrics['mean_specificity']),
            'mean_f1': float(metrics['mean_f1']),
            'accuracy': float(metrics['accuracy'])
        },
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        report['per_class'][class_name] = {
            'auroc': float(metrics['class_auroc'][i]),
            'sensitivity': float(metrics['sensitivities'][i]),
            'specificity': float(metrics['specificities'][i]),
            'f1_score': float(metrics['f1_scores'][i]),
            'optimal_threshold': float(metrics['optimal_thresholds'][i])
        }
    
    # Save JSON
    json_path = save_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save text report
    txt_path = save_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CHEST X-RAY CLASSIFICATION - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("SUMMARY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean AUROC:        {metrics['mean_auroc']:.4f}\n")
        f.write(f"Mean Sensitivity:  {metrics['mean_sensitivity']:.4f}\n")
        f.write(f"Mean Specificity:  {metrics['mean_specificity']:.4f}\n")
        f.write(f"Mean F1 Score:     {metrics['mean_f1']:.4f}\n")
        f.write(f"Overall Accuracy:  {metrics['accuracy']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<20} {'AUROC':<10} {'Sens':<10} {'Spec':<10} {'F1':<10}\n")
        f.write("-"*70 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<20} "
                   f"{metrics['class_auroc'][i]:<10.4f} "
                   f"{metrics['sensitivities'][i]:<10.4f} "
                   f"{metrics['specificities'][i]:<10.4f} "
                   f"{metrics['f1_scores'][i]:<10.4f}\n")
        
        f.write("="*70 + "\n")
    
    print(f"✓ Metrics report saved: {json_path} and {txt_path}")


if __name__ == "__main__":
    print("✓ Utils module loaded successfully")