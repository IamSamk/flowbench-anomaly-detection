import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Find all roberta_text_* result directories (exclude summary and improved)
base_dir = 'results_roberta_text'
pattern = os.path.join(base_dir, 'roberta_text_*')
all_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d) and not d.endswith(('summary', 'improved'))]
all_dirs = sorted(all_dirs)

plt.figure(figsize=(12, 8))

# Collect ROC curves from all datasets
for d in all_dirs:
    results_file = os.path.join(d, 'results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            res = json.load(f)
        
        dataset_name = os.path.basename(d).replace('roberta_text_', '')
        
        # Extract ROC curve data
        if 'roc_curve' in res and 'roc_auc' in res:
            fpr, tpr, _ = res['roc_curve']
            auc = res['roc_auc']
            plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {auc:.3f})', linewidth=2)

# Add diagonal line for random classifier
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Datasets', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'roberta_text_summary', 'all_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved all ROC curves plot to {os.path.join(base_dir, 'roberta_text_summary', 'all_roc_curves.png')}") 