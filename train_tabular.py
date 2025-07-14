import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import json
import time
import warnings
warnings.filterwarnings('ignore')

try:
from flowbench.dataset import FlowDataset
except ImportError as e:
    print(f"Error importing FlowBench: {e}")
    sys.exit(1)

def create_visualizations(y_test, y_pred, y_pred_proba, rf_clf, X_train, save_plots=True, results_dir="results", dataset_name="montage"):
    """Create and save visualizations in a dedicated results directory."""
    if not save_plots:
        return
    out_dir = os.path.join(results_dir, dataset_name, "random_forest")
    os.makedirs(out_dir, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved to '{os.path.join(out_dir, 'confusion_matrix.png')}'")

    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    plt.close()
    print(f"ROC curve saved to '{os.path.join(out_dir, 'roc_curve.png')}'")

    # Feature Importance
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
    plt.savefig(os.path.join(out_dir, 'feature_importance.png'))
    plt.close()
    print(f"Feature importance plot saved to '{os.path.join(out_dir, 'feature_importance.png')}'")

def train_random_forest(dataset_name, root_dir, save_plots=True, results_dir="results"):
    """Train and evaluate Random Forest on a dataset."""
    print(f"🔄 Loading '{dataset_name}' dataset...")
    dataset = FlowDataset(root=root_dir, name=dataset_name, force_reprocess=False)
    data = dataset[0]
    X = data.x.numpy()
    y = data.y.numpy()

    print(f"📊 Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📈 Anomaly ratio: {y.sum() / len(y):.2%}")

    # Standardize features
    print("🔧 Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📋 Training set size: {X_train.shape[0]}")
    print(f"📋 Test set size: {X_test.shape[0]}")

    # Train Random Forest
    print("🌳 Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Simulate training progress
    with tqdm(total=100, desc="Training RF", ncols=80) as pbar:
        rf_clf.fit(X_train, y_train)
        pbar.update(100)
    
    print("✅ Training complete.")

    # Make predictions
    print("🔮 Making predictions...")
    y_pred = rf_clf.predict(X_test)
    y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📊 F1-score: {f1:.4f}")
    print(f"📊 ROC-AUC Score: {roc_auc:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

    if save_plots:
        print("📈 Creating visualizations...")
        create_visualizations(y_test, y_pred, y_pred_proba, rf_clf, X_train, save_plots, results_dir, dataset_name)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_true': y_test
    }

def train_gmm_model(X_train, X_test, y_test, dataset_name, save_dir):
    """Train GMM model and evaluate performance"""
    print(f"\n🔄 Training GMM model for {dataset_name}...")
    
    # Scale features
    print("  📊 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train GMM with progress bar
    print("  🤖 Training Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    
    # Simulate training progress (GMM training is fast, so we'll show fitting progress)
    with tqdm(total=100, desc="  Training GMM", ncols=80) as pbar:
        gmm.fit(X_train_scaled)
        pbar.update(100)
    
    # Predict anomalies
    print("  🔍 Detecting anomalies...")
    with tqdm(total=len(X_test_scaled), desc="  Predicting", ncols=80) as pbar:
        log_likelihood = gmm.score_samples(X_test_scaled)
        pbar.update(len(X_test_scaled))
    
    # Use negative log-likelihood as anomaly score
    anomaly_scores = -log_likelihood
    
    # Calculate metrics
    print("  📈 Calculating metrics...")
    roc_auc = roc_auc_score(y_test, anomaly_scores)
    
    # Convert to binary predictions using threshold
    threshold = np.percentile(anomaly_scores, 90)  # Top 10% as anomalies
    y_pred = (anomaly_scores > threshold).astype(int)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save results
    print(f"  💾 Saving results to {save_dir}...")
    results = {
        'dataset': dataset_name,
        'model': 'GMM',
        'roc_auc': float(roc_auc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'anomaly_rate': float(np.mean(y_pred))
    }
    
    # Save metrics
    with open(os.path.join(save_dir, f'{dataset_name}_gmm_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} (GMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_gmm_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix - {dataset_name} (GMM)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_gmm_confusion.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ GMM training completed for {dataset_name}")
    print(f"     ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    
    return results

def prepare_features_and_labels(df, dataset_name):
    """
    Prepare features and labels for GMM model.
    This function is a placeholder and needs to be implemented
    based on the actual dataset structure.
    """
    # Example: Assuming 'label' column is the target
    # and other columns are features.
    # For FlowBench datasets, the label is 'y' and features are 'x'.
    # This function needs to be adapted to the specific dataset.
    
    # For now, let's assume 'y' is the label and 'x' are features
    # and we need to handle the case where 'y' might not be present.
    if 'y' in df.columns:
        y = df['y'].values
        X = df.drop(columns=['y']).values
        return X, y
    elif 'label' in df.columns:
        y = df['label'].values
        X = df.drop(columns=['label']).values
        return X, y
    else:
        print(f"  ❌ No 'y' or 'label' column found in {dataset_name} dataset.")
        return None, None

def main():
    """Main function to run Random Forest experiment on montage dataset."""
    print("🚀 Starting Random Forest Anomaly Detection Experiment")
    
    # Dataset configuration
    dataset_name = "montage"
    root_dir = "data"
    results_dir = "results"
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Run Random Forest experiment
    print(f"\n{'='*60}")
    print(f"📊 Running Random Forest on {dataset_name} dataset")
    print(f"{'='*60}")
    
    results = train_random_forest(dataset_name, root_dir, save_plots=True, results_dir=results_dir)
    
    print(f"\n🎯 Final Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    
    print(f"\n✅ Random Forest experiment completed! Results saved in '{results_dir}' directory")

if __name__ == "__main__":
    main()
