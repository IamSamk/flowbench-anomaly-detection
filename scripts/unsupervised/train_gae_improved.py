#!/usr/bin/env python3
"""
Improved Graph Autoencoder (GAE) for Unsupervised Anomaly Detection
- Enhanced data preprocessing and normalization
- Better model architecture with attention mechanisms
- Improved training parameters and loss functions
- More robust anomaly scoring
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
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ImprovedGraphAutoencoder(nn.Module):
    """Improved Graph Autoencoder with attention and better architecture."""
    
    def __init__(self, num_features, hidden_dim=128, latent_dim=64, dropout=0.2):
        super(ImprovedGraphAutoencoder, self).__init__()
        
        # Encoder with residual connections
        self.encoder1 = GCNConv(num_features, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, hidden_dim)
        self.encoder3 = GCNConv(hidden_dim, hidden_dim)
        self.encoder4 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder with symmetric architecture
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, hidden_dim)
        self.decoder3 = GCNConv(hidden_dim, hidden_dim)
        self.decoder4 = GCNConv(hidden_dim, num_features)
        
        # Attention mechanism for better reconstruction
        self.attention = nn.MultiheadAttention(latent_dim, num_heads=4, dropout=dropout)
        
        # Node-level MLP for final reconstruction
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_features)
        )
        
        # Global pooling for graph-level features
        self.global_pool = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.dropout = dropout
        
    def encode(self, x, edge_index):
        """Encode graph to latent representation with residual connections."""
        h1 = F.relu(self.encoder1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        h2 = F.relu(self.encoder2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + h1  # Residual connection
        
        h3 = F.relu(self.encoder3(h2, edge_index))
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        h3 = h3 + h2  # Residual connection
        
        z = self.encoder4(h3, edge_index)
        return z
    
    def decode(self, z, edge_index):
        """Decode latent representation back to graph with attention."""
        # Apply self-attention to latent representations
        z_reshaped = z.unsqueeze(0)  # Add batch dimension
        z_attended, _ = self.attention(z_reshaped, z_reshaped, z_reshaped)
        z_attended = z_attended.squeeze(0)  # Remove batch dimension
        
        h1 = F.relu(self.decoder1(z_attended, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        h2 = F.relu(self.decoder2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + h1  # Residual connection
        
        h3 = F.relu(self.decoder3(h2, edge_index))
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        h3 = h3 + h2  # Residual connection
        
        x_recon = self.decoder4(h3, edge_index)
        return x_recon
    
    def forward(self, x, edge_index):
        """Forward pass through encoder and decoder."""
        z = self.encode(x, edge_index)
        x_recon = self.decode(z, edge_index)
        return x_recon, z
    
    def get_node_embeddings(self, x, edge_index):
        """Get node embeddings for anomaly scoring."""
        return self.encode(x, edge_index)

def create_improved_graph(data_dir, dataset_name):
    """Create improved synthetic graph with better preprocessing."""
    print(f"Creating improved synthetic graph for {dataset_name}...")
    
    import glob
    csv_files = glob.glob(os.path.join(data_dir, dataset_name, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for {dataset_name}")
    
    # Use multiple CSV files for better graph construction
    all_data = []
    for csv_file in csv_files[:5]:  # Use first 5 files
        try:
            df = pd.read_csv(csv_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                features = df[numeric_cols].values
                features = np.nan_to_num(features, nan=0.0)
                all_data.append(features)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No valid data found for {dataset_name}")
    
    # Combine all data
    X = np.vstack(all_data)
    print(f"Combined data shape: {X.shape}")
    
    # Better preprocessing
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    
    # Handle outliers with RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction if too many features
    if X_scaled.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"Reduced to {X_scaled.shape[1]} features using PCA")
    
    # Sample data for graph creation (larger sample)
    sample_size = min(2000, len(X_scaled))
    indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    node_features = X_scaled[indices]
    
    # Create edges using multiple methods
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import pdist, squareform
    
    edges = []
    
    # Method 1: K-nearest neighbors
    k = min(8, len(node_features) - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(node_features)
    distances, indices = nbrs.kneighbors(node_features)
    
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip self
            edges.append([i, j])
            edges.append([j, i])  # Undirected graph
    
    # Method 2: Distance-based edges (connect nodes within threshold)
    distances_matrix = squareform(pdist(node_features))
    threshold = np.percentile(distances_matrix, 75)  # Connect 25% closest pairs
    
    for i in range(len(node_features)):
        for j in range(i+1, len(node_features)):
            if distances_matrix[i, j] < threshold:
                edges.append([i, j])
                edges.append([j, i])
    
    # Remove duplicates
    edges = list(set(map(tuple, edges)))
    edges = [list(edge) for edge in edges]
    
    # Create improved anomaly labels using multiple methods
    from scipy import stats
    
    # Method 1: Z-score based
    z_scores = np.abs(stats.zscore(node_features, axis=0))
    outlier_scores_z = np.mean(z_scores, axis=1)
    
    # Method 2: IQR based
    Q1 = np.percentile(node_features, 25, axis=0)
    Q3 = np.percentile(node_features, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = ((node_features < (Q1 - 1.5 * IQR)) | (node_features > (Q3 + 1.5 * IQR)))
    outlier_scores_iqr = np.sum(outlier_mask, axis=1)
    
    # Method 3: Isolation Forest
    try:
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_scores_iso = -iso_forest.fit_predict(node_features)
    except:
        outlier_scores_iso = np.zeros(len(node_features))
    
    # Combine all outlier scores
    combined_scores = (0.4 * outlier_scores_z + 0.4 * outlier_scores_iqr + 0.2 * outlier_scores_iso)
    
    # Ensure reasonable anomaly rate (5-15%)
    target_rate = 0.1  # 10% anomaly rate
    threshold = np.percentile(combined_scores, (1 - target_rate) * 100)
    labels = (combined_scores > threshold).astype(int)
    
    # If we still have too few anomalies, adjust threshold
    if labels.sum() / len(labels) < 0.05:
        threshold = np.percentile(combined_scores, 85)  # Force 15% anomaly rate
        labels = (combined_scores > threshold).astype(int)
    
    print(f"Created improved graph: {len(node_features)} nodes, {len(edges)} edges")
    print(f"Anomaly rate: {labels.sum() / len(labels):.2%}")
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.FloatTensor(node_features),
        edge_index=torch.LongTensor(np.array(edges).T),
        y=torch.LongTensor(labels)
    )
    
    return [data]

def train_improved_gae_model(model, train_loader, device, epochs=200, lr=0.001):
    """Train the improved Graph Autoencoder with better parameters."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()
    
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training Improved GAE"):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, z = model(batch.x, batch.edge_index)
            
            # Reconstruction loss
            recon_loss = criterion(x_recon, batch.x)
            
            # Add regularization loss
            reg_loss = torch.mean(torch.norm(z, dim=1))
            
            # Total loss
            loss = recon_loss + 0.01 * reg_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 30:  # Early stopping patience
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return train_losses

def compute_improved_anomaly_scores(model, data, device):
    """Compute improved anomaly scores using multiple methods."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        x_recon, z = model(data.x, data.edge_index)
        
        # Method 1: Node-level reconstruction error
        recon_error = F.mse_loss(x_recon, data.x, reduction='none')
        node_scores_recon = torch.mean(recon_error, dim=1).cpu().numpy()
        
        # Method 2: Embedding distance from mean
        z_mean = torch.mean(z, dim=0, keepdim=True)
        embedding_dist = torch.norm(z - z_mean, dim=1).cpu().numpy()
        
        # Method 3: Local outlier factor approximation
        z_np = z.cpu().numpy()
        from sklearn.neighbors import LocalOutlierFactor
        try:
            lof = LocalOutlierFactor(n_neighbors=min(20, len(z_np)//10), contamination=0.1)
            lof_scores = -lof.fit_predict(z_np)
        except:
            lof_scores = np.zeros(len(z_np))
        
        # Combine scores with weights
        combined_scores = (0.5 * node_scores_recon + 0.3 * embedding_dist + 0.2 * lof_scores)
        
        return combined_scores, z.cpu().numpy()

def evaluate_anomaly_detection(y_true, scores):
    """Evaluate anomaly detection results."""
    y_pred = (scores > np.percentile(scores, 95)).astype(int) # Use 95th percentile as threshold
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # ROC-AUC is not directly applicable to binary classification
    # as it requires continuous scores.
    # For binary classification, we can use PR-AUC or F1-Score.
    # Here, we'll use a placeholder or a simple threshold-based AUC.
    # A common approach is to use a threshold that maximizes F1-Score.
    # Let's find the threshold that maximizes F1-Score.
    f1_scores = []
    thresholds = np.linspace(0, 1, 100)
    for thresh in thresholds:
        y_pred_thresh = (scores > thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))
    
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    
    y_pred_best_f1 = (scores > best_threshold).astype(int)
    roc_auc = roc_auc_score(y_true, scores) # This will be 0.5 for binary classification
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'best_threshold': best_threshold
    }

def plot_results(results, scores, y_true, output_dir):
    """Plot anomaly detection results."""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, label='True Anomalies', alpha=0.6, c='red')
    plt.scatter(range(len(y_true)), scores, label='Predicted Scores', alpha=0.6, c='blue')
    plt.axhline(y=results['best_threshold'], color='green', linestyle='--', label=f'Best F1 Threshold: {results["best_threshold"]:.4f}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection Results')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'anomaly_detection_plot.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Improved GAE Unsupervised Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results_gae_improved', help='Output directory')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
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
        print(f"\n=== Training Improved GAE on {dataset_name} ===")
        
        try:
            # Create improved graph
            graphs = create_improved_graph(args.data_dir, dataset_name)
            
            if not graphs:
                print(f"No valid graphs found for {dataset_name}")
                continue
            
            # Use the largest graph for training
            graph = max(graphs, key=lambda g: g.x.shape[0])
            print(f"Using graph with {graph.x.shape[0]} nodes and {graph.edge_index.shape[1]} edges")
            
            # Create improved model
            model = ImprovedGraphAutoencoder(
                num_features=graph.x.shape[1],
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                dropout=0.2
            ).to(device)
            
            # Train model
            train_losses = train_improved_gae_model(model, [graph], device, args.epochs, args.lr)
            
            # Compute improved anomaly scores
            scores, embeddings = compute_improved_anomaly_scores(model, graph, device)
            
            # Get labels for evaluation
            y_true = graph.y.cpu().numpy()
            
            # Evaluate
            results = evaluate_anomaly_detection(y_true, scores)
            
            print(f"\n=== Results for {dataset_name} ===")
            print(f"Accuracy:  {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall:    {results['recall']:.4f}")
            print(f"F1-Score:  {results['f1_score']:.4f}")
            print(f"ROC-AUC:   {results['roc_auc']:.4f}")
            
            # Save results
            output_path = os.path.join(args.output_dir, f'gae_improved_{dataset_name}')
            os.makedirs(output_path, exist_ok=True)
            
            # Save results
            with open(os.path.join(output_path, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create plots
            plot_results(results, scores, y_true, output_path)
            
            # Save embeddings and training losses
            np.save(os.path.join(output_path, 'embeddings.npy'), embeddings)
            np.save(os.path.join(output_path, 'scores.npy'), scores)
            np.save(os.path.join(output_path, 'train_losses.npy'), train_losses)
            
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
        df.to_csv(os.path.join(args.output_dir, 'gae_improved_results_summary.csv'), index=False)
        print(f"\nSummary saved to {args.output_dir}/gae_improved_results_summary.csv")
    
    print("\n=== Improved GAE Training Complete ===")

if __name__ == '__main__':
    main() 