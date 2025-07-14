#!/usr/bin/env python3
"""
Fixed Graph Autoencoder (GAE) for Unsupervised Anomaly Detection
- Proper error handling and imports
- Better model architecture with skip connections
- Improved training with better weights and learning rates
- Fixed anomaly detection logic
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
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class ImprovedGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, dropout=0.2):
        super(ImprovedGAE, self).__init__()
        
        # Encoder with skip connections
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, latent_dim)
        
        # Skip connection layers
        self.skip1 = nn.Linear(input_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, latent_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x, edge_index):
        # First layer with skip connection
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.bn1(h1)
        h1 = h1 + self.skip1(x)  # Skip connection
        h1 = self.dropout(h1)
        
        # Second layer
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = self.bn2(h2)
        h2 = self.dropout(h2)
        
        # Third layer with skip connection
        z = self.conv3(h2, edge_index)
        z = z + self.skip2(h1)  # Skip connection
        
        return z
    
    def decode(self, z, edge_index):
        # Simple inner product decoder
        return torch.sigmoid(torch.mm(z, z.t()))
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decode(z, edge_index)
        return adj_pred, z

def create_graph_from_features(features, k=5):
    """Create graph from features using k-nearest neighbors"""
    print(f"  🔗 Creating graph with k={k} nearest neighbors...")
    
    from sklearn.neighbors import NearestNeighbors
    
    # Use k-NN to create edges
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
    
    with tqdm(total=100, desc="  Building graph", ncols=80) as pbar:
        nbrs.fit(features)
        pbar.update(50)
        
        distances, indices = nbrs.kneighbors(features)
        pbar.update(50)
    
    # Create edge list
    edge_list = []
    for i in range(len(features)):
        for j in range(1, k+1):  # Skip self (index 0)
            neighbor = indices[i][j]
            edge_list.append([i, neighbor])
            edge_list.append([neighbor, i])  # Add reverse edge
    
    # Convert to tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    
    print(f"  📊 Created graph with {edge_index.shape[1]} edges")
    
    return edge_index

def create_synthetic_anomaly_labels(features, anomaly_rate=0.1):
    """Create synthetic anomaly labels using statistical methods"""
    print(f"  🎯 Creating synthetic anomaly labels (rate={anomaly_rate})...")
    
    from sklearn.ensemble import IsolationForest
    
    # Use Isolation Forest to identify anomalies
    iso_forest = IsolationForest(contamination=anomaly_rate, random_state=42)
    
    with tqdm(total=100, desc="  Creating labels", ncols=80) as pbar:
        anomaly_labels = iso_forest.fit_predict(features)
        pbar.update(100)
    
    # Convert to binary (1 for anomaly, 0 for normal)
    binary_labels = (anomaly_labels == -1).astype(int)
    
    print(f"  📊 Created {np.sum(binary_labels)} anomalies out of {len(binary_labels)} samples ({np.mean(binary_labels):.2%})")
    
    return binary_labels

def train_gae_model(features, dataset_name, save_dir):
    """Train GAE model with improved architecture"""
    print(f"\n🔄 Training GAE model for {dataset_name}...")
    
    # Create synthetic labels
    labels = create_synthetic_anomaly_labels(features)
    
    # Scale features
    print("  📏 Scaling features...")
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create graph
    edge_index = create_graph_from_features(features_scaled, k=5)
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(features_scaled)
    y = torch.LongTensor(labels)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Split data
    print("  ✂️ Splitting data...")
    train_idx, test_idx = train_test_split(
        range(len(features)), test_size=0.3, random_state=42, stratify=labels
    )
    
    # Initialize model
    print("  🤖 Initializing GAE model...")
    input_dim = features_scaled.shape[1]
    model = ImprovedGAE(input_dim=input_dim, hidden_dim=64, latent_dim=32, dropout=0.2)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    # Training
    print("  🏋️ Training GAE model...")
    model.train()
    
    num_epochs = 200
    train_losses = []
    
    with tqdm(total=num_epochs, desc="  Training", ncols=80) as pbar:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            adj_pred, z = model(data.x, data.edge_index)
            
            # Create positive and negative samples
            pos_edge_index = data.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.x.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
            
            # Calculate loss
            pos_loss = F.binary_cross_entropy(
                adj_pred[pos_edge_index[0], pos_edge_index[1]],
                torch.ones(pos_edge_index.size(1))
            )
            neg_loss = F.binary_cross_entropy(
                adj_pred[neg_edge_index[0], neg_edge_index[1]],
                torch.zeros(neg_edge_index.size(1))
            )
            
            loss = pos_loss + neg_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Update progress bar
            if epoch % 10 == 0:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            pbar.update(1)
    
    # Evaluation
    print("  📊 Evaluating model...")
    model.eval()
    
    with torch.no_grad():
        adj_pred, z = model(data.x, data.edge_index)
        
        # Calculate reconstruction error as anomaly score
        reconstruction_errors = []
        
        with tqdm(total=len(data.x), desc="  Calculating scores", ncols=80) as pbar:
            for i in range(len(data.x)):
                # Get neighbors
                neighbors = data.edge_index[1][data.edge_index[0] == i]
                if len(neighbors) > 0:
                    # Calculate reconstruction error
                    pred_adj = adj_pred[i, neighbors]
                    true_adj = torch.ones_like(pred_adj)
                    error = F.mse_loss(pred_adj, true_adj).item()
                else:
                    error = 1.0  # High error for isolated nodes
                
                reconstruction_errors.append(error)
                pbar.update(1)
    
    anomaly_scores = np.array(reconstruction_errors)
    
    # Get test predictions
    test_scores = anomaly_scores[test_idx]
    test_labels = labels[test_idx]
    
    # Calculate metrics
    print("  📈 Calculating metrics...")
    roc_auc = roc_auc_score(test_labels, test_scores)
    
    # Convert to binary predictions using threshold
    threshold = np.percentile(test_scores, 90)  # Top 10% as anomalies
    test_pred = (test_scores > threshold).astype(int)
    
    f1 = f1_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, test_pred)
    
    # Save results
    print(f"  💾 Saving results to {save_dir}...")
    results = {
        'dataset': dataset_name,
        'model': 'GAE_Fixed',
        'roc_auc': float(roc_auc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'anomaly_rate': float(np.mean(test_pred))
    }
    
    # Save metrics
    with open(os.path.join(save_dir, f'{dataset_name}_gae_fixed_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training losses
    np.save(os.path.join(save_dir, f'{dataset_name}_train_losses.npy'), train_losses)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} (GAE Fixed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_gae_fixed_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix - {dataset_name} (GAE Fixed)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_gae_fixed_confusion.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses)
    plt.title(f'Training Loss - {dataset_name} (GAE Fixed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_gae_fixed_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ GAE Fixed training completed for {dataset_name}")
    print(f"     ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    
    return results

def load_and_prepare_data(dataset_name):
    """Load and prepare data for training"""
    print(f"  📥 Loading data for {dataset_name}...")
    
    data_path = f'data/{dataset_name}/{dataset_name}.csv'
    if not os.path.exists(data_path):
        print(f"  ❌ Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"  📊 Dataset shape: {df.shape}")
    
    # Select numeric columns for features
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target columns if they exist
    target_columns = ['y', 'label', 'anomaly', 'target']
    numeric_columns = [col for col in numeric_columns if col not in target_columns]
    
    if not numeric_columns:
        print(f"  ❌ No numeric features found for {dataset_name}")
        return None
    
    print(f"  🔢 Using {len(numeric_columns)} numeric features")
    
    # Extract features
    features = df[numeric_columns].values
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)
    
    print(f"  ✅ Prepared features shape: {features.shape}")
    
    return features

def main():
    # Define datasets
    datasets = [
        '1000genome', 'casa_nowcast', 'casa_wind_speed', 'eht_difmap', 
        'eht_imaging', 'eht_smili', 'montage', 'predict_future_sales',
        'pycbc_inference', 'pycbc_search', 'somospie', 'variant_calling'
    ]
    
    results_dir = 'results_gae_fixed'
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    print("🚀 Starting GAE Fixed Anomaly Detection Training")
    print(f"📊 Processing {len(datasets)} datasets...")
    
    # Process each dataset with progress bar
    for dataset_name in tqdm(datasets, desc="Overall Progress", ncols=100):
        print(f"\n{'='*60}")
        print(f"📂 Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Load and prepare data
            features = load_and_prepare_data(dataset_name)
            
            if features is None:
                print(f"  ❌ Failed to load data for {dataset_name}")
                continue
            
            # Train model
            result = train_gae_model(features, dataset_name, results_dir)
            all_results.append(result)
            
            # Small delay to show progress
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {str(e)}")
            continue
    
    # Save summary results
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(results_dir, 'gae_fixed_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"💾 Summary saved to {summary_path}")
        
        # Print summary statistics
        print(f"\n📈 Performance Summary:")
        print(f"  Average ROC-AUC: {summary_df['roc_auc'].mean():.4f} (±{summary_df['roc_auc'].std():.4f})")
        print(f"  Average F1-Score: {summary_df['f1_score'].mean():.4f} (±{summary_df['f1_score'].std():.4f})")
        print(f"  Best ROC-AUC: {summary_df['roc_auc'].max():.4f} ({summary_df.loc[summary_df['roc_auc'].idxmax(), 'dataset']})")
        print(f"  Best F1-Score: {summary_df['f1_score'].max():.4f} ({summary_df.loc[summary_df['f1_score'].idxmax(), 'dataset']})")
        
        # Show top 5 performers
        print(f"\n🏆 Top 5 Datasets by ROC-AUC:")
        top_5 = summary_df.nlargest(5, 'roc_auc')
        for i, row in top_5.iterrows():
            print(f"  {row['dataset']:20} ROC-AUC: {row['roc_auc']:.4f}, F1: {row['f1_score']:.4f}")
    
    print(f"\n✅ GAE Fixed training completed! Results saved in '{results_dir}' directory")

if __name__ == "__main__":
    main() 