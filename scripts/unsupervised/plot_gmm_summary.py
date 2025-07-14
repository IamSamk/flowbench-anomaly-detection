#!/usr/bin/env python3
"""
GMM Results Summary and Visualization
- Compiles results from all GMM datasets
- Creates comprehensive visualizations
- Generates performance comparison plots
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

# Find all GMM results files
results_files = glob.glob('results_gmm/gmm_*/results.json')

metrics = ['roc_auc', 'f1_score', 'accuracy', 'precision', 'recall']
all_results = {m: [] for m in metrics}
dataset_names = []

for f in sorted(results_files):
    try:
        with open(f, 'r') as fp:
            res = json.load(fp)
        dataset = os.path.basename(os.path.dirname(f)).replace('gmm_', '')
        dataset_names.append(dataset)
        for m in metrics:
            all_results[m].append(res.get(m, 0))
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue

# Create summary directory
os.makedirs('results_gmm/gmm_summary', exist_ok=True)

# 1. Performance comparison bar chart
x = np.arange(len(dataset_names))
bar_width = 0.15
plt.figure(figsize=(16, 8))
for i, m in enumerate(metrics):
    plt.bar(x + i*bar_width, all_results[m], width=bar_width, label=m.capitalize())
plt.xticks(x + bar_width*2, dataset_names, rotation=45, ha='right')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('GMM Unsupervised Model Performance Across Datasets')
plt.legend()
plt.tight_layout()
plt.savefig('results_gmm/gmm_summary/gmm_benchmark_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. ROC curves compilation
plt.figure(figsize=(12, 8))
for f in sorted(results_files):
    try:
        with open(f, 'r') as fp:
            res = json.load(fp)
        dataset = os.path.basename(os.path.dirname(f)).replace('gmm_', '')
        
        if 'roc_curve' in res and 'roc_auc' in res:
            fpr, tpr, _ = res['roc_curve']
            auc = res['roc_auc']
            plt.plot(fpr, tpr, label=f'{dataset} (AUC = {auc:.3f})', linewidth=2)

    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('GMM ROC Curves - All Datasets', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results_gmm/gmm_summary/gmm_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Confusion matrices grid
import glob
from PIL import Image

# Find all confusion matrix images
confusion_files = glob.glob('results_gmm/gmm_*/confusion_matrix.png')
confusion_files = sorted(confusion_files)

if confusion_files:
    images = []
    dataset_names_cm = []
    for f in confusion_files:
        if os.path.exists(f):
            images.append(Image.open(f))
            dataset_name = os.path.basename(os.path.dirname(f)).replace('gmm_', '')
            dataset_names_cm.append(dataset_name)
    
    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (img, name) in enumerate(zip(images, dataset_names_cm)):
        axes[i].imshow(img)
        axes[i].set_title(name, fontsize=12)
        axes[i].axis('off')
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_gmm/gmm_summary/gmm_all_confusion_matrices.png', dpi=200, bbox_inches='tight')
    plt.close()

# 4. Save summary table
df = pd.DataFrame(all_results, index=dataset_names)
df.to_csv('results_gmm/gmm_summary/gmm_benchmark_table.csv')
print('GMM Summary plots and table saved in results_gmm/gmm_summary/')

# 5. Performance heatmap
plt.figure(figsize=(12, 8))
heatmap_data = df.T  # Transpose for better visualization
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5)
plt.title('GMM Performance Heatmap Across Datasets')
plt.xlabel('Datasets')
plt.ylabel('Metrics')
plt.tight_layout()
plt.savefig('results_gmm/gmm_summary/gmm_performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("All GMM summary visualizations completed!") 