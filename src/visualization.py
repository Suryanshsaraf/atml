import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve

def plot_training_history(history, save_path=None):
    """
    Plots training and validation loss over epochs.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], label='Training Loss (MSE)', color='dodgerblue', lw=2)
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='darkorange', lw=2)
    ax.set_title('LSTM Autoencoder Training History', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_error_distribution_and_threshold(normal_errors, anomaly_errors, threshold, save_path=None):
    """
    Plots reconstruction error distributions for normal and anomalous sequences with threshold line.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(normal_errors, label='Normal Sequences', color='teal', fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(anomaly_errors, label='Anomalous Sequences', color='crimson', fill=True, alpha=0.4, ax=ax)
    ax.axvline(threshold, color='black', linestyle='--', lw=2, label=f'Threshold ({threshold:.4f})')
    ax.set_title('Reconstruction Error Distribution & Threshold', fontsize=12, fontweight='bold')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_roc_comparison(roc_dict, save_path=None):
    """
    Plots comparative ROC curves for all benchmarked models.
    roc_dict: dict of {model_name: (y_true, anomaly_scores, auc_score)}
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['crimson', 'forestgreen', 'darkorange', 'dodgerblue', 'purple']
    
    for (model_name, (y_true, scores, auc_val)), color in zip(roc_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, scores)
        label_text = f'{model_name} (AUC = {auc_val:.4f})' if auc_val is not None else model_name
        ax.plot(fpr, tpr, label=label_text, color=color, lw=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5000)')
    ax.set_title('Comparative ROC Curves — AWS Anomaly Detection', fontsize=13, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_anomaly_timeline(y_true, y_pred, errors, threshold, sample_range=(0, 1000), save_path=None):
    """
    Visualizes a segment of the time-series with reconstruction error, threshold, and ground truth.
    """
    start, end = sample_range
    sub_errors = errors[start:end]
    sub_true = y_true[start:end]
    sub_pred = y_pred[start:end]
    x_axis = np.arange(start, end)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    ax1.plot(x_axis, sub_errors, color='dodgerblue', lw=1.2, label='Reconstruction Error')
    ax1.axhline(threshold, color='crimson', linestyle='--', lw=1.5, label=f'Threshold ({threshold:.3f})')
    ax1.set_ylabel('MSE Error')
    ax1.set_title(f'AWS Timeline (Steps {start} to {end}) — Errors & Detection', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_axis, sub_true, color='black', lw=1.5, label='Ground Truth Anomaly', alpha=0.8)
    ax2.scatter(x_axis[sub_pred == 1], sub_pred[sub_pred == 1], color='red', s=20, label='Model Detected (Flagged)', zorder=5)
    ax2.set_ylabel('Anomaly State')
    ax2.set_xlabel('Time Step Index')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_xai_feature_contributions(explanation, save_path=None):
    """
    Plots horizontal bar chart of feature contributions to a detected anomaly.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    contrib = explanation['contributions_percent']
    sorted_items = sorted(contrib.items(), key=lambda x: x[1])
    features = [x[0] for x in sorted_items]
    percentages = [x[1] for x in sorted_items]
    
    colors = ['crimson' if f == explanation['top_feature'] else 'steelblue' for f in features]
    bars = ax.barh(features, percentages, color=colors)
    ax.set_xlabel('Reconstruction Error Contribution (%)')
    ax.set_title(f'XAI Attribution: Sequence #{explanation["sequence_index"]} (Top: {explanation["top_feature"]})', 
                 fontsize=11, fontweight='bold')
    
    for bar in bars:
        width = bar.get_width()
        if width > 0.5:
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                    va='center', fontsize=9, fontweight='bold')
            
    ax.set_xlim(0, max(percentages) + 10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig
