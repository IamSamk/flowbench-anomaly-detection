#!/usr/bin/env python3
"""
Graph Autoencoder (GAE) for Unsupervised Anomaly Detection
- Loads workflow graphs using FlowBench
- Trains GAE model unsupervised
- Uses reconstruction error for anomaly scoring
- Comprehensive evaluation and visualization
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GraphAutoencoder(nn.Module):
    """Graph Autoencoder for anomaly detection."""
    
    def __init__(self, num_features, hidden_dim=64, latent_dim=32):
        super(GraphAutoencoder, self).__init__()
        
        # Encoder
        self.encoder1 = GCNConv(num_features, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, hidden_dim)
        self.encoder3 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, hidden_dim)
        self.decoder3 = GCNConv(hidden_dim, num_features)
        
        # Node-level MLP for final reconstruction
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features)
        )
        
    def encode(self, x, edge_index):
        """Encode graph to latent representation."""
        h = F.relu(self.encoder1(x, edge_index))
        h = F.relu(self.encoder2(h, edge_index))
        h = self.encoder3(h, edge_index)
        return h
    
    def decode(self, z, edge_index):
        """Decode latent representation back to graph."""
        h = F.relu(self.decoder1(z, edge_index))
        h = F.relu(self.decoder2(h, edge_index))
        h = self.decoder3(h, edge_index)
        return h
    
    def forward(self, x, edge_index):
        """Forward pass through encoder and decoder."""
        z = self.encode(x, edge_index)
        x_recon = self.decode(z, edge_index)
        return x_recon, z
    
    def get_node_embeddings(self, x, edge_index):
        """Get node embeddings for anomaly scoring."""
        return self.encode(x, edge_index)

def load_graph_dataset(data_dir, dataset_name):
    """Load graph dataset using FlowBench loader."""
    print(f"Loading {dataset_name} graph dataset...")
    
    try:
        # Import FlowBench loader
        sys.path.append('flowbench-env/Lib/site-packages')
        from flowbench import load_dataset
        
        # Load the dataset
        dataset = load_dataset(dataset_name, data_dir=data_dir)
        
        # Convert to PyTorch Geometric format
        if hasattr(dataset, 'graphs'):
            # Multiple graphs
            graphs = []
            for graph in dataset.graphs[:20]:  # Limit for speed
                if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
                    data = Data(
                        x=torch.FloatTensor(graph.x),
                        edge_index=torch.LongTensor(graph.edge_index),
                        y=torch.LongTensor(graph.y) if hasattr(graph, 'y') else None
                    )
                    graphs.append(data)
            return graphs
        else:
            # Single graph
            if hasattr(dataset, 'x') and hasattr(dataset, 'edge_index'):
                data = Data(
                    x=torch.FloatTensor(dataset.x),
                    edge_index=torch.LongTensor(dataset.edge_index),
                    y=torch.LongTensor(dataset.y) if hasattr(dataset, 'y') else None
                )
                return [data]
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        # Fallback: create synthetic graph from CSV data
        return create_synthetic_graph(data_dir, dataset_name)

def create_synthetic_graph(data_dir, dataset_name):
    """Create synthetic graph from CSV data as fallback."""
    print(f"Creating synthetic graph for {dataset_name}...")
    
    import glob
    csv_files = glob.glob(os.path.join(data_dir, dataset_name, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for {dataset_name}")
    
    # Use first CSV file to create graph
    df = pd.read_csv(csv_files[0])
    
    # Create node features from numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric columns found in {dataset_name}")
    
    # Sample data for graph creation
    sample_size = min(1000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Create node features
    node_features = df_sample[numeric_cols].values
    node_features = np.nan_to_num(node_features, nan=0.0)
    
    # Create edges (connect each node to k nearest neighbors)
    from sklearn.neighbors import NearestNeighbors
    k = min(5, len(node_features) - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(node_features)
    distances, indices = nbrs.kneighbors(node_features)
    
    # Create edge list
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip self
            edges.append([i, j])
            edges.append([j, i])  # Undirected graph
    
    # Create anomaly labels (synthetic)
    from scipy import stats
    z_scores = np.abs(stats.zscore(node_features, axis=0))
    outlier_scores = np.mean(z_scores, axis=1)
    threshold = np.percentile(outlier_scores, 90)
    labels = (outlier_scores > threshold).astype(int)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.FloatTensor(node_features),
        edge_index=torch.LongTensor(np.array(edges).T),
        y=torch.LongTensor(labels)
    )
    
    print(f"Created synthetic graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    return [data]

def train_gae_model(model, train_loader, device, epochs=100, lr=0.001):
    """Train the Graph Autoencoder."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training GAE"):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, _ = model(batch.x, batch.edge_index)
            
            # Reconstruction loss
            loss = criterion(x_recon, batch.x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return train_losses

def compute_anomaly_scores(model, data, device):
    """Compute anomaly scores using reconstruction error."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        x_recon, z = model(data.x, data.edge_index)
        
        # Node-level reconstruction error
        recon_error = F.mse_loss(x_recon, data.x, reduction='none')
        node_scores = torch.mean(recon_error, dim=1).cpu().numpy()
        
        return node_scores, z.cpu().numpy()

def evaluate_anomaly_detection(y_true, scores):
    """Evaluate anomaly detection performance."""
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    
    # Find optimal threshold for F1 score
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else thresholds[np.argmax(tpr - fpr)]
    
    # Make predictions
    y_pred = (scores > best_threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'roc_auc': auc,
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_curve': (fpr.tolist(), tpr.tolist(), thresholds.tolist()),
        'best_threshold': float(best_threshold)
    }

def plot_results(results, scores, y_true, output_dir):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, _ = results['roc_curve']
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {results['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - GAE Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    y_pred = (scores > results['best_threshold']).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - GAE Anomaly Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Anomaly Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores[y_true == 0], bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(scores[y_true == 1], bins=50, alpha=0.7, label='Anomaly', density=True)
    plt.axvline(results['best_threshold'], color='red', linestyle='--', label='Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='GAE Unsupervised Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results_gae', help='Output directory')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Find all datasets
    data_root = Path(args.data_dir)
    exclude = {'adjacency_list_dags'}
    datasets = [d.name for d in data_root.iterdir() if d.is_dir() and d.name not in exclude]
    print(f"Found datasets: {datasets}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    for dataset_name in datasets:
        print(f"\n=== Training GAE on {dataset_name} ===")
        
        try:
            # Load graph dataset
            graphs = load_graph_dataset(args.data_dir, dataset_name)
            
            if not graphs:
                print(f"No valid graphs found for {dataset_name}")
                continue
            
            # Use the largest graph for training
            graph = max(graphs, key=lambda g: g.x.shape[0])
            print(f"Using graph with {graph.x.shape[0]} nodes and {graph.edge_index.shape[1]} edges")
            
            # Split data (80% train, 20% test)
            num_nodes = graph.x.shape[0]
            train_size = int(0.8 * num_nodes)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[:train_size] = True
            
            # Create model
            model = GraphAutoencoder(
                num_features=graph.x.shape[1],
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim
            ).to(device)
            
            # Train model
            train_losses = train_gae_model(model, [graph], device, args.epochs, args.lr)
            
            # Compute anomaly scores
            scores, embeddings = compute_anomaly_scores(model, graph, device)
            
            # Get labels for evaluation (if available)
            if hasattr(graph, 'y') and graph.y is not None:
                y_true = graph.y.cpu().numpy()
            else:
                # Create synthetic labels for evaluation
                from scipy import stats
                z_scores = np.abs(stats.zscore(graph.x.cpu().numpy(), axis=0))
                outlier_scores = np.mean(z_scores, axis=1)
                threshold = np.percentile(outlier_scores, 90)
                y_true = (outlier_scores > threshold).astype(int)
            
            # Evaluate
            results = evaluate_anomaly_detection(y_true, scores)
            
            print(f"\n=== Results for {dataset_name} ===")
            print(f"Accuracy:  {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall:    {results['recall']:.4f}")
            print(f"F1-Score:  {results['f1_score']:.4f}")
            print(f"ROC-AUC:   {results['roc_auc']:.4f}")
            
            # Save results
            output_path = os.path.join(args.output_dir, f'gae_{dataset_name}')
            os.makedirs(output_path, exist_ok=True)
            
            # Save results
            with open(os.path.join(output_path, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create plots
            plot_results(results, scores, y_true, output_path)
            
            # Save embeddings
            np.save(os.path.join(output_path, 'embeddings.npy'), embeddings)
            np.save(os.path.join(output_path, 'scores.npy'), scores)
            
            all_results.append({
                'dataset': dataset_name,
                **results
            })
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(args.output_dir, 'gae_results_summary.csv'), index=False)
        print(f"\nSummary saved to {args.output_dir}/gae_results_summary.csv")
    
    print("\n=== Training Complete ===")

if __name__ == '__main__':
    main() 