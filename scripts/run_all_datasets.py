import os
import subprocess
from pathlib import Path

data_root = Path('data')
exclude = {'adjacency_list_dags'}

datasets = [d.name for d in data_root.iterdir() if d.is_dir() and d.name not in exclude]
print(f"Found datasets: {datasets}")

for ds in datasets:
    data_dir = str(data_root / ds)
    output_dir = f"results/roberta_text_{ds}"
    print(f"\n=== Running on dataset: {ds} ===")
    subprocess.run([
        'python', 'scripts/train_roberta_text.py',
        '--data_dir', data_dir,
        '--output_dir', output_dir
    ]) 