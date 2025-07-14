#!/usr/bin/env python3
"""
Text-based Unsupervised Anomaly Detection
- Uses pre-trained language models (Sentence-BERT) for embeddings
- Applies classic outlier detectors (Isolation Forest, GMM, kNN)
- Comprehensive evaluation and comparison
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence transformers, fallback to simpler approach if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("SentenceTransformers not available, using TF-IDF fallback")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer

def load_text_dataset(data_dir, dataset_name):
    """Load text dataset from CSV files."""
    print(f"Loading text dataset for {dataset_name}...")
    
    import glob
    csv_files = glob.glob(os.path.join(data_dir, dataset_name, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for {dataset_name}")
    
    # Use first 5 files for text extraction
    all_texts = []
    all_labels = []
    
    for csv_file in csv_files[:5]:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract text from various columns
            text_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['name', 'description', 'command', 'task', 'job', 'workflow']):
                    text_columns.append(col)
            
            # If no specific text columns, use all string columns
            if not text_columns:
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # If still no text columns, create synthetic text from numeric data
            if not text_columns:
                # Create text descriptions from numeric features
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for idx, row in df.iterrows():
                    text = f"Job with features: "
                    for col in numeric_cols[:5]:  # Use first 5 features
                        text += f"{col}={row[col]:.2f}, "
                    all_texts.append(text)
                    
                    # Create synthetic anomaly labels
                    from scipy import stats
                    if len(numeric_cols) > 0:
                        features = row[numeric_cols].values
                        z_score = np.abs(stats.zscore(features)) if len(features) > 1 else 0
                        is_anomaly = np.mean(z_score) > 2.0
                        all_labels.append(1 if is_anomaly else 0)
            else:
                # Use actual text columns
                for idx, row in df.iterrows():
                    text_parts = []
                    for col in text_columns:
                        if pd.notna(row[col]):
                            text_parts.append(str(row[col]))
                    
                    if text_parts:
                        text = " ".join(text_parts)
                        all_texts.append(text)
                        
                        # Create synthetic anomaly labels based on text length and content
                        is_anomaly = len(text) > np.percentile([len(t) for t in all_texts], 90) if all_texts else False
                        all_labels.append(1 if is_anomaly else 0)
                        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not all_texts:
        raise ValueError(f"No valid text data found for {dataset_name}")
    
    print(f"Loaded {len(all_texts)} text samples")
    print(f"Anomaly rate: {sum(all_labels) / len(all_labels):.2%}")
    
    return all_texts, all_labels

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings using pre-trained language model."""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"Using SentenceTransformer: {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
    else:
        print("Using TF-IDF vectorization as fallback")
        vectorizer = TfidfVectorizer(max_features=768, stop_words='english')
        embeddings = vectorizer.fit_transform(texts).toarray()
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings

def train_isolation_forest(embeddings, contamination=0.1):
    """Train Isolation Forest outlier detector."""
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(embeddings)
    scores = -iso_forest.decision_function(embeddings)  # Convert to positive scores
    return scores, iso_forest

def train_gmm_outlier(embeddings, n_components=5):
    """Train GMM-based outlier detector."""
    print("Training GMM outlier detector...")
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(embeddings)
    scores = -gmm.score_samples(embeddings)  # Negative log-likelihood as anomaly score
    return scores, gmm

def train_lof_outlier(embeddings, n_neighbors=20):
    """Train Local Outlier Factor detector."""
    print("Training Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(embeddings)//10), contamination=0.1)
    lof.fit(embeddings)
    scores = -lof.negative_outlier_factor_  # Convert to positive scores
    return scores, lof

def train_knn_outlier(embeddings, n_neighbors=5):
    """Train k-NN based outlier detector."""
    print("Training k-NN outlier detector...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    scores = np.mean(distances, axis=1)  # Average distance to neighbors
    return scores, nbrs

def evaluate_anomaly_detection(y_true, scores):
    """Evaluate anomaly detection performance."""
    try:
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
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            'roc_auc': 0.5,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'roc_curve': ([0, 1], [0, 1], [0, 1]),
            'best_threshold': 0.0
        }

def plot_text_results(results, scores, y_true, output_dir, method_name):
    """Create visualization plots for text-based anomaly detection."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. ROC Curve
        fpr, tpr, _ = results['roc_curve']
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {results['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {method_name} Text Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        y_pred = (scores > results['best_threshold']).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {method_name} Text Anomaly Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Anomaly Score Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(scores[y_true == 0], bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(scores[y_true == 1], bins=50, alpha=0.7, label='Anomaly', density=True)
        plt.axvline(results['best_threshold'], color='red', linestyle='--', label='Threshold')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title(f'Anomaly Score Distribution - {method_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating plots for {method_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Text-based Unsupervised Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results_text_unsupervised', help='Output directory')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='Sentence transformer model')
    
    args = parser.parse_args()
    
    # Find all datasets
    data_root = Path(args.data_dir)
    exclude = {'adjacency_list_dags'}
    datasets = [d.name for d in data_root.iterdir() if d.is_dir() and d.name not in exclude]
    print(f"Found datasets: {datasets}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    for dataset_name in datasets:
        print(f"\n=== Training Text-based Anomaly Detection on {dataset_name} ===")
        
        try:
            # Load text dataset
            texts, labels = load_text_dataset(args.data_dir, dataset_name)
            
            # Generate embeddings
            embeddings = generate_embeddings(texts, args.model_name)
            
            # Normalize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Train multiple outlier detectors
            methods = {
                'IsolationForest': train_isolation_forest,
                'GMM': train_gmm_outlier,
                'LOF': train_lof_outlier,
                'KNN': train_knn_outlier
            }
            
            dataset_results = {'dataset': dataset_name}
            
            for method_name, method_func in methods.items():
                print(f"\n--- Training {method_name} ---")
                
                try:
                    scores, model = method_func(embeddings_scaled)
                    results = evaluate_anomaly_detection(labels, scores)
                    
                    print(f"{method_name} Results:")
                    print(f"  Accuracy:  {results['accuracy']:.4f}")
                    print(f"  Precision: {results['precision']:.4f}")
                    print(f"  Recall:    {results['recall']:.4f}")
                    print(f"  F1-Score:  {results['f1_score']:.4f}")
                    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
                    
                    # Save results
                    output_path = os.path.join(args.output_dir, f'text_{dataset_name}_{method_name.lower()}')
                    os.makedirs(output_path, exist_ok=True)
                    
                    with open(os.path.join(output_path, 'results.json'), 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Create plots
                    plot_text_results(results, scores, labels, output_path, method_name)
                    
                    # Save scores
                    np.save(os.path.join(output_path, 'scores.npy'), scores)
                    
                    # Store results for summary
                    for metric, value in results.items():
                        if metric != 'roc_curve':
                            dataset_results[f'{method_name.lower()}_{metric}'] = value
                    
                except Exception as e:
                    print(f"Error with {method_name}: {e}")
                    continue
            
            all_results.append(dataset_results)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(args.output_dir, 'text_unsupervised_results_summary.csv'), index=False)
        print(f"\nSummary saved to {args.output_dir}/text_unsupervised_results_summary.csv")
    
    print("\n=== Text-based Unsupervised Training Complete ===")

if __name__ == '__main__':
    main() 