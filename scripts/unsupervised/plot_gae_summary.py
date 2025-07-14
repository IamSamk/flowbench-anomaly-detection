#!/usr/bin/env python3
"""
GAE Results Summary and Visualization
- Compiles results from all GAE datasets
- Creates comprehensive visualizations
- Handles poor performance gracefully
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Find all GAE results files
results_files = glob.glob('results_gae/gae_*/results.json')

metrics = ['roc_auc', 'f1_score', 'accuracy', 'precision', 'recall']
all_results = {m: [] for m in metrics}
dataset_names = []

for f in sorted(results_files):
    try:
        with open(f, 'r') as fp:
            res = json.load(fp)
        dataset = os.path.basename(os.path.dirname(f)).replace('gae_', '')
        dataset_names.append(dataset)
        for m in metrics:
            value = res.get(m, 0)
            # Handle NaN values
            if pd.isna(value) or value == 'nan':
                value = 0.0
            all_results[m].append(float(value))
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue

# Create summary directory
os.makedirs('results_gae/gae_summary', exist_ok=True)

if dataset_names:
    # 1. Performance comparison bar chart
    x = np.arange(len(dataset_names))
    bar_width = 0.15
    plt.figure(figsize=(16, 8))
    for i, m in enumerate(metrics):
        plt.bar(x + i*bar_width, all_results[m], width=bar_width, label=m.capitalize())
    plt.xticks(x + bar_width*2, dataset_names, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('GAE Unsupervised Model Performance Across Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results_gae/gae_summary/gae_benchmark_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC curves compilation (only for valid results)
    plt.figure(figsize=(12, 8))
    valid_curves = 0
    for f in sorted(results_files):
        try:
            with open(f, 'r') as fp:
                res = json.load(fp)
            dataset = os.path.basename(os.path.dirname(f)).replace('gae_', '')
            
            if 'roc_curve' in res and 'roc_auc' in res and not pd.isna(res['roc_auc']):
                fpr, tpr, _ = res['roc_curve']
                auc = res['roc_auc']
                if auc > 0:
                    plt.plot(fpr, tpr, label=f'{dataset} (AUC = {auc:.3f})', linewidth=2)
                    valid_curves += 1
        except Exception as e:
            print(f"Error processing ROC curve for {f}: {e}")
            continue

    if valid_curves > 0:
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('GAE ROC Curves - All Datasets', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results_gae/gae_summary/gae_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No valid ROC curves found")

    # 3. Confusion matrices grid
    import glob
    from PIL import Image

    # Find all confusion matrix images
    confusion_files = glob.glob('results_gae/gae_*/confusion_matrix.png')
    confusion_files = sorted(confusion_files)

    if confusion_files:
        images = []
        dataset_names_cm = []
        for f in confusion_files:
            if os.path.exists(f):
                try:
                    images.append(Image.open(f))
                    dataset_name = os.path.basename(os.path.dirname(f)).replace('gae_', '')
                    dataset_names_cm.append(dataset_name)
                except Exception as e:
                    print(f"Error loading image {f}: {e}")
                    continue
        
        if images:
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
            plt.savefig('results_gae/gae_summary/gae_all_confusion_matrices.png', dpi=200, bbox_inches='tight')
            plt.close()

    # 4. Save summary table
    df = pd.DataFrame(all_results, index=dataset_names)
    df.to_csv('results_gae/gae_summary/gae_benchmark_table.csv')
    print('GAE Summary plots and table saved in results_gae/gae_summary/')

    # 5. Performance heatmap
    plt.figure(figsize=(12, 8))
    heatmap_data = df.T  # Transpose for better visualization
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5)
    plt.title('GAE Performance Heatmap Across Datasets')
    plt.xlabel('Datasets')
    plt.ylabel('Metrics')
    plt.tight_layout()
    plt.savefig('results_gae/gae_summary/gae_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Analysis of poor performance
    print("\n=== GAE Performance Analysis ===")
    print("Issues identified:")
    print("1. FlowBench loader not working - using synthetic graphs")
    print("2. All metrics are 0 or NaN - model not learning effectively")
    print("3. Synthetic graph creation may not be optimal for GAE")
    print("4. Need better data preprocessing and model architecture")
    
    # Create a summary report
    with open('results_gae/gae_summary/gae_analysis_report.txt', 'w') as f:
        f.write("GAE Unsupervised Anomaly Detection - Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Issues Identified:\n")
        f.write("1. FlowBench loader not working - using synthetic graphs\n")
        f.write("2. All metrics are 0 or NaN - model not learning effectively\n")
        f.write("3. Synthetic graph creation may not be optimal for GAE\n")
        f.write("4. Need better data preprocessing and model architecture\n\n")
        f.write("Recommendations:\n")
        f.write("1. Fix FlowBench loader integration\n")
        f.write("2. Improve synthetic graph creation\n")
        f.write("3. Tune GAE hyperparameters\n")
        f.write("4. Add data normalization and preprocessing\n")
        f.write("5. Consider alternative graph neural network architectures\n")

else:
    print("No GAE results found")

print("All GAE summary visualizations completed!") 