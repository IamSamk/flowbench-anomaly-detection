import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

# Find all roberta_text_* result directories (exclude summary and improved)
base_dir = 'results'
pattern = os.path.join(base_dir, 'roberta_text_*')
all_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d) and not d.endswith(('summary', 'improved'))]
all_dirs = sorted(all_dirs)

# Collect confusion matrix images and dataset names
images = []
dataset_names = []
for d in all_dirs:
    img_path = os.path.join(d, 'confusion_matrix.png')
    if os.path.exists(img_path):
        images.append(Image.open(img_path))
        dataset_names.append(os.path.basename(d).replace('roberta_text_', ''))

n = len(images)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten()
for i, (img, name) in enumerate(zip(images, dataset_names)):
    axes[i].imshow(img)
    axes[i].set_title(name, fontsize=12)
    axes[i].axis('off')
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'roberta_text_summary', 'all_confusion_matrices.png'), dpi=200)
plt.close()
print(f"Saved all confusion matrices grid to {os.path.join(base_dir, 'roberta_text_summary', 'all_confusion_matrices.png')}") 