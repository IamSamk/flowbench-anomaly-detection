import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd

# --- Config ---
RESULTS_FILE = 'results/roberta_text_improved/results.json'
SUMMARY_DIR = 'results/roberta_text_summary'
os.makedirs(SUMMARY_DIR, exist_ok=True)

# --- Load improved results ---
with open(RESULTS_FILE, 'r') as fp:
    res = json.load(fp)

# Use dataset name 'montage' for improved results (or customize if needed)
dataset_name = 'montage'

# --- 1. ROC Curve ---
plt.figure(figsize=(8, 6))
if 'fpr' in res and 'tpr' in res and 'roc_auc' in res:
    plt.plot(res['fpr'], res['tpr'], label=f'{dataset_name} (AUC = {res["roc_auc"]:.3f})')
elif 'roc_curve' in res and 'roc_auc' in res:
    fpr, tpr, _ = res['roc_curve']
    plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {res["roc_auc"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RoBERTa Text (Montage)')
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SUMMARY_DIR, 'roberta_text_roc_curve.png'), dpi=300)
plt.close()

# --- 2. Confusion Matrix ---
if 'confusion_matrix' in res:
    cm = np.array(res['confusion_matrix'])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - RoBERTa Text ({dataset_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0.5, 1.5], ['Normal', 'Anomaly'])
    plt.yticks([0.5, 1.5], ['Normal', 'Anomaly'])
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'roberta_text_confusion_matrix.png'), dpi=300)
    plt.close()

# --- 3. Training/Validation Loss Curves ---
plt.figure(figsize=(8, 6))
if 'train_loss_curve' in res:
    plt.plot(res['train_loss_curve'], label='Train Loss', linestyle='-')
if 'val_loss_curve' in res:
    plt.plot(res['val_loss_curve'], label='Val Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training/Validation Loss Curves - RoBERTa Text (Montage)')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SUMMARY_DIR, 'roberta_text_loss_curves.png'), dpi=300)
plt.close()

# --- 4. Metrics Table ---
metrics = ['roc_auc', 'f1_score', 'accuracy', 'precision', 'recall']
metric_values = [res.get(m, 0) for m in metrics]
df = pd.DataFrame([metric_values], columns=metrics, index=[dataset_name])
df.to_csv(os.path.join(SUMMARY_DIR, 'roberta_text_benchmark_table.csv'))

print(f'All summary plots and table for improved RoBERTa saved in {SUMMARY_DIR}/') 