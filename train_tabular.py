import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from flowbench.dataset import FlowDataset
    print("FlowBench imported successfully!")
    # Print the expected zip path
    import flowbench.dataset as fbd
    real_dir = os.path.realpath(os.path.dirname(fbd.__file__))
    relative_path = os.path.join(real_dir, '..', 'data')
    expected_zip = os.path.abspath(os.path.join(relative_path, 'montage.zip'))
    print(f"FlowBench expects montage.zip at: {expected_zip}")
    print(f"File exists? {os.path.exists(expected_zip)}")
except ImportError as e:
    print(f"Error importing FlowBench: {e}")
    sys.exit(1)

# Set root to the folder containing your zipped datasets
root_dir = r"C:\Users\Samarth Kadam\MLpaper\data"
print(f"Using root directory: {root_dir}")

try:
    # Load the 'montage' workflow dataset
    print("Loading 'montage' dataset...")
    dataset = FlowDataset(root=root_dir, name="montage", force_reprocess=False)
    data = dataset[0]
    X = data.x.numpy()
    y = data.y.numpy()

    print("\n--- Dataset Summary ---")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of anomalies: {y.sum()} / {len(y)} ({y.sum() / len(y):.2%})")

    # Basic statistics
    print(f"Feature statistics:")
    print(f"  Mean: {X.mean():.3f}")
    print(f"  Std: {X.std():.3f}")
    print(f"  Min: {X.min():.3f}")
    print(f"  Max: {X.max():.3f}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 2. Preprocess and Split Data ---
print("\n--- Preprocessing and Splitting Data ---")
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# --- 3. Train Random Forest Classifier ---
print("\n--- Training Random Forest Classifier ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
print("Training complete.")

# --- 4. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = rf_clf.predict(X_test)
y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

# Metrics
accuracy = rf_clf.score(X_test, y_test)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

# --- 5. Visualize Results ---
print("\n--- Visualizing Results ---")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to 'confusion_matrix.png'")

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("ROC curve saved to 'roc_curve.png'")

# --- 6. Feature Importance ---
print("\n--- Feature Importance ---")
# Define the correct feature names based on FlowBench's "v1" option
feature_names = [
    "wms_delay", "queue_delay", "runtime", "post_script_delay", 
    "stage_in_delay", "stage_out_delay", "node_hop"
]
feature_importances = pd.Series(rf_clf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(feature_importances)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved to 'feature_importance.png'")
