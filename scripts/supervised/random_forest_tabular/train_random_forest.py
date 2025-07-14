import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Patch sys.path to import FlowBench if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from flowbench.dataset import FlowDataset
except ImportError as e:
    print(f"Error importing FlowBench: {e}")
    sys.exit(1)

DATASETS = [
    "1000genome",
    "casa_nowcast",
    "casa_wind_speed",
    "eht_difmap",
    "eht_imaging",
    "eht_smili",
    "montage",
    "predict_future_sales",
    "pycbc_inference",
    "pycbc_search",
    "somospie",
    "variant_calling"
]

RESULTS_DIR = "results_random_forest"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Global storage for summary plots
all_results = []
all_roc_data = []
all_feature_importances = []

def create_summary_plots():
    """Create comprehensive summary plots for all datasets"""
    print("\nCreating summary plots...")
    
    # 1. ROC Curves for all datasets
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(DATASETS)))
    
    for i, (dataset_name, fpr, tpr, roc_auc) in enumerate(all_roc_data):
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{dataset_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Random Forest (All Datasets)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance comparison bar chart
    df_results = pd.DataFrame(all_results)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(range(len(df_results)), df_results['accuracy'], 
                    color='skyblue', alpha=0.7)
    ax1.set_xlabel('Datasets', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(df_results)))
    ax1.set_xticklabels(df_results['dataset'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.75, 0.95)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, df_results['accuracy']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # F1-score comparison
    bars2 = ax2.bar(range(len(df_results)), df_results['f1_score'], 
                    color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Datasets', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(df_results)))
    ax2.set_xticklabels(df_results['dataset'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 0.8)
    
    # Add value labels on bars
    for bar, f1 in zip(bars2, df_results['f1_score']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ROC-AUC comparison
    bars3 = ax3.bar(range(len(df_results)), df_results['roc_auc'], 
                    color='lightgreen', alpha=0.7)
    ax3.set_xlabel('Datasets', fontsize=12)
    ax3.set_ylabel('ROC-AUC', fontsize=12)
    ax3.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(df_results)))
    ax3.set_xticklabels(df_results['dataset'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.7, 0.95)
    
    # Add value labels on bars
    for bar, auc in zip(bars3, df_results['roc_auc']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance heatmap across all datasets
    if all_feature_importances:
        feature_names = [
            "wms_delay", "queue_delay", "runtime", "post_script_delay", 
            "stage_in_delay", "stage_out_delay", "node_hop"
        ]
        
        # Create feature importance matrix
        importance_matrix = np.array(all_feature_importances)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(importance_matrix, 
                   xticklabels=feature_names,
                   yticklabels=[result['dataset'] for result in all_results],
                   annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Feature Importance'})
        plt.title('Feature Importance Heatmap - Random Forest (All Datasets)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Datasets', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Confusion matrices in a grid
    n_datasets = len(all_results)
    n_cols = 4
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (dataset_name, cm) in enumerate([(result['dataset'], result['confusion_matrix']) for result in all_results]):
        row = i // n_cols
        col = i % n_cols
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'],
                   ax=axes[row, col])
        axes[row, col].set_title(f'{dataset_name}', fontsize=10, fontweight='bold')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_datasets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Confusion Matrices - Random Forest (All Datasets)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plots saved to {RESULTS_DIR}/")

def run_random_forest(dataset_name, root_dir):
    print(f"\n=== Random Forest on {dataset_name} ===")
    try:
        dataset = FlowDataset(root=root_dir, name=dataset_name, force_reprocess=False)
        data = dataset[0]
        X = data.x.numpy()
        y = data.y.numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)
        y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
        
        # Store results for summary plots
        all_results.append({
            'dataset': dataset_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        })
        
        # Store ROC data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        all_roc_data.append((dataset_name, fpr, tpr, roc_auc))
        
        # Store feature importance
        all_feature_importances.append(rf_clf.feature_importances_)
        
        return {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    except Exception as e:
        print(f"Error running Random Forest on {dataset_name}: {e}")
        return None

def main():
    root_dir = r"C:/Users/Samarth Kadam/MLpaper/data"
    results = []
    
    for dataset in DATASETS:
        res = run_random_forest(dataset, root_dir)
        if res:
            results.append(res)
    
    # Create summary plots
    create_summary_plots()
    
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "rf_results_summary.csv"), index=False)
    print(f"\nSummary saved to {os.path.join(RESULTS_DIR, 'rf_results_summary.csv')}")

if __name__ == "__main__":
    main() 