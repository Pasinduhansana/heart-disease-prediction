import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _ensure_folder(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def evaluate_model(model, X_test, y_test) -> Dict[str, object]:
    """Compute evaluation metrics for a binary classifier."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_score),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, digits=4)
    }
    return metrics


def save_metrics(metrics: Dict[str, object], file_path: str):
    """Write evaluation metrics and the classification report to a text file."""
    _ensure_folder(file_path)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('Logistic Regression Evaluation Metrics\n')
        file.write('===============================\n')
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"Precision: {metrics['precision']:.4f}\n")
        file.write(f"Recall: {metrics['recall']:.4f}\n")
        file.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        file.write(f"ROC AUC Score: {metrics['roc_auc']:.4f}\n\n")
        file.write('Classification Report:\n')
        file.write(metrics['classification_report'])


def plot_confusion_matrix(confusion_matrix_data: np.ndarray, save_path: Optional[str] = None):
    """Plot and optionally save the confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(confusion_matrix_data, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Disease', 'Disease'])
    ax.set_yticklabels(['No Disease', 'Disease'])

    for i in range(confusion_matrix_data.shape[0]):
        for j in range(confusion_matrix_data.shape[1]):
            ax.text(j, i, int(confusion_matrix_data[i, j]), ha='center', va='center', color='black')

    plt.tight_layout()
    if save_path:
        _ensure_folder(save_path)
        plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig


def plot_roc_curve(y_test, y_score, save_path: Optional[str] = None):
    """Plot and optionally save the ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_value = roc_auc_score(y_test, y_score)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        _ensure_folder(save_path)
        plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig
