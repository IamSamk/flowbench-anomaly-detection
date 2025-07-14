import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
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

RESULTS_DIR = "results_gcn"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Global storage for summary plots
all_results = []
all_roc_data = []
all_training_losses = []

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
    plt.title('ROC Curves - GCN (All Datasets)', fontsize=14, fontweight='bold')
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
    
    # 3. Training loss comparison
    if all_training_losses:
        plt.figure(figsize=(15, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(DATASETS)))
        
        for i, (dataset_name, train_losses) in enumerate(all_training_losses):
            plt.plot(train_losses, color=colors[i], lw=2, 
                    label=f'{dataset_name}', alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Training Loss Comparison - GCN (All Datasets)', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'training_loss_comparison.png'), dpi=300, bbox_inches='tight')
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
    
    plt.suptitle('Confusion Matrices - GCN (All Datasets)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plots saved to {RESULTS_DIR}/")

def train_gcn(model, data, optimizer, criterion, train_mask, val_mask=None, epochs=100):
    model.train()
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask].float())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_mask], data.y[val_mask].float())
                val_losses.append(val_loss.item())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            model.train()
        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}: Train Loss: {loss:.4f}', end='')
            if val_mask is not None:
                print(f', Val Loss: {val_loss:.4f}')
            else:
                print()
    return train_losses, val_losses

def evaluate_gcn(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred = (out[test_mask] > 0.5).float()
        y_pred_proba = out[test_mask].cpu().numpy()
        y_true = data.y[test_mask].cpu().numpy()
        accuracy = (y_pred == data.y[test_mask]).float().mean().item()
        f1 = f1_score(y_true, y_pred.cpu().numpy())
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        return accuracy, f1, roc_auc, y_pred.cpu().numpy(), y_pred_proba, y_true

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=64, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze()

def run_gcn(dataset_name, root_dir):
    print(f"\n=== GCN on {dataset_name} ===")
    try:
        dataset = FlowDataset(root=root_dir, name=dataset_name, force_reprocess=False)
        data = dataset[0]
        num_nodes = data.x.shape[0]
        indices = torch.randperm(num_nodes)
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        num_features = data.x.shape[1]
        model = GCN(num_features=num_features, hidden_channels=64, num_layers=2, dropout=0.5)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        train_losses, val_losses = train_gcn(
            model, data, optimizer, criterion, 
            train_mask, val_mask, epochs=100
        )
        accuracy, f1, roc_auc, y_pred, y_pred_proba, y_true = evaluate_gcn(model, data, test_mask)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
        
        # Store results for summary plots
        all_results.append({
            'dataset': dataset_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        })
        
        # Store ROC data
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        all_roc_data.append((dataset_name, fpr, tpr, roc_auc))
        
        # Store training losses
        all_training_losses.append((dataset_name, train_losses))
        
        return {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    except Exception as e:
        print(f"Error running GCN on {dataset_name}: {e}")
        return None

def main():
    root_dir = r"C:/Users/Samarth Kadam/MLpaper/data"
    results = []
    
    for dataset in DATASETS:
        res = run_gcn(dataset, root_dir)
        if res:
            results.append(res)
    
    # Create summary plots
    create_summary_plots()
    
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "gcn_results_summary.csv"), index=False)
    print(f"\nSummary saved to {os.path.join(RESULTS_DIR, 'gcn_results_summary.csv')}")

if __name__ == "__main__":
    main() 