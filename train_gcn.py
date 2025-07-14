import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from flowbench.dataset import FlowDataset
except ImportError as e:
    print(f"Error importing FlowBench: {e}")
    sys.exit(1)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class GCN(torch.nn.Module):
    """Graph Convolutional Network for node-level anomaly detection."""
    
    def __init__(self, num_features, hidden_channels=64, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        
        # GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Classification layer
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # GCN layers with ReLU activation and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply dropout after last conv
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze()

def train_gcn(model, data, optimizer, criterion, train_mask, val_mask=None, epochs=200):
    """Train the GCN model."""
    model.train()
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Calculate loss
        loss = criterion(out[train_mask], data.y[train_mask].float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_mask], data.y[val_mask].float())
                val_losses.append(val_loss.item())
            
            # Early stopping
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
    """Evaluate the GCN model."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        # Get predictions
        y_pred = (out[test_mask] > 0.5).float()
        y_pred_proba = out[test_mask].cpu().numpy()
        y_true = data.y[test_mask].cpu().numpy()
        
        # Calculate metrics
        accuracy = (y_pred == data.y[test_mask]).float().mean().item()
        f1 = f1_score(y_true, y_pred.cpu().numpy())
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        return accuracy, f1, roc_auc, y_pred.cpu().numpy(), y_pred_proba, y_true

def save_gcn_plots(dataset_name, train_losses, val_losses, y_true, y_pred, y_pred_proba, roc_auc):
    out_dir = os.path.join('results', dataset_name, 'gcn')
    os.makedirs(out_dir, exist_ok=True)
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title('GCN Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'GCN ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('GCN Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Training Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if val_losses:
        plt.subplot(1, 2, 2)
        plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== GCN Node-Level Anomaly Detection ===")
    
    # --- 1. Load Dataset ---
    root_dir = r"C:\Users\Samarth Kadam\MLpaper\data"
    print("Loading 'montage' dataset...")
    dataset = FlowDataset(root=root_dir, name="montage", force_reprocess=False)
    data = dataset[0]
    
    print("\n--- Dataset Summary ---")
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Node features: {data.x.shape[1]}")
    print(f"Number of anomalies: {data.y.sum().item()} / {len(data.y)} ({data.y.sum().item() / len(data.y):.2%})")
    
    # --- 2. Prepare Data ---
    print("\n--- Preparing Data ---")
    
    # Create train/val/test masks
    num_nodes = data.x.shape[0]
    indices = torch.randperm(num_nodes)
    
    # 70% train, 15% val, 15% test
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    print(f"Training nodes: {train_mask.sum().item()}")
    print(f"Validation nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")
    
    # --- 3. Initialize Model ---
    print("\n--- Initializing GCN Model ---")
    num_features = data.x.shape[1]
    model = GCN(num_features=num_features, hidden_channels=64, num_layers=2, dropout=0.5)
    
    # Loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # --- 4. Train Model ---
    print("\n--- Training GCN Model ---")
    train_losses, val_losses = train_gcn(
        model, data, optimizer, criterion, 
        train_mask, val_mask, epochs=200
    )
    
    # --- 5. Evaluate Model ---
    print("\n--- Model Evaluation ---")
    accuracy, f1, roc_auc, y_pred, y_pred_proba, y_true = evaluate_gcn(model, data, test_mask)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    # --- 6. Visualize Results ---
    print("\n--- Visualizing Results ---")
    save_gcn_plots("montage", train_losses, val_losses, y_true, y_pred, y_pred_proba, roc_auc)
    
    # --- 7. Save Results ---
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_true': y_true,
        'model_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    print(f"\n=== GCN Results Summary ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Model Parameters: {results['model_params']:,}")
    
    return results

if __name__ == "__main__":
    results = main() 