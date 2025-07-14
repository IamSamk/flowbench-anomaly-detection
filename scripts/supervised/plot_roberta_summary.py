import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Find all results files
results_files = glob.glob('results/roberta_text_*/results.json')

metrics = ['roc_auc', 'f1_score', 'accuracy', 'precision', 'recall']
all_results = {m: [] for m in metrics}
dataset_names = []

for f in sorted(results_files):
    with open(f, 'r') as fp:
        res = json.load(fp)
    dataset = os.path.basename(os.path.dirname(f)).replace('roberta_text_', '')
    dataset_names.append(dataset)
    for m in metrics:
        all_results[m].append(res.get(m, 0))

# Plot grouped bar chart
os.makedirs('results/roberta_text_summary', exist_ok=True)
x = np.arange(len(dataset_names))
bar_width = 0.15
plt.figure(figsize=(14, 7))
for i, m in enumerate(metrics):
    plt.bar(x + i*bar_width, all_results[m], width=bar_width, label=m.capitalize())
plt.xticks(x + bar_width*2, dataset_names, rotation=45, ha='right')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('RoBERTa Text Model Performance Across Datasets')
plt.legend()
plt.tight_layout()
plt.savefig('results/roberta_text_summary/roberta_text_benchmark_bar.png', dpi=300)
plt.close()

# Also save as CSV for easy table
import pandas as pd
df = pd.DataFrame(all_results, index=dataset_names)
df.to_csv('results/roberta_text_summary/roberta_text_benchmark_table.csv')
print('Summary plot and table saved in results/roberta_text_summary/') 