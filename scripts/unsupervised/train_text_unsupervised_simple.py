#!/usr/bin/env python3
"""
Simple Text-based Unsupervised Anomaly Detection
- Uses pre-trained language models (Sentence-BERT) for embeddings
- Applies Isolation Forest outlier detector
- Simple and consistent with other model approaches
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
    
    # Use first 3 files for text extraction
    all_texts = []
    all_labels = []
    
    for csv_file in csv_files[:3]:
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

def train_isolation_forest_outlier(embeddings, contamination=0.1):
    """Train Isolation Forest outlier detector."""
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(embeddings)
    scores = -iso_forest.decision_function(embeddings)  # Convert to positive scores
    return scores, iso_forest

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

def plot_text_results(results, scores, y_true, output_dir):
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
        plt.title('ROC Curve - Text-based Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        y_pred = (scores > results['best_threshold']).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Text-based Anomaly Detection')
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
        plt.title('Anomaly Score Distribution - Text-based')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating plots: {e}")

def create_synthetic_anomaly_labels(text_data, anomaly_rate=0.1):
    """Create synthetic anomaly labels based on text characteristics"""
    print("  🎯 Creating synthetic anomaly labels...")
    
    # Calculate text features for anomaly detection
    text_lengths = [len(str(text)) for text in text_data]
    unique_words = [len(set(str(text).split())) for text in text_data]
    
    # Combine features
    features = np.column_stack([text_lengths, unique_words])
    
    # Use isolation forest to identify unusual texts
    iso_forest = IsolationForest(contamination=anomaly_rate, random_state=42)
    
    with tqdm(total=100, desc="  Creating labels", ncols=80) as pbar:
        anomaly_labels = iso_forest.fit_predict(features)
        pbar.update(100)
    
    # Convert to binary (1 for anomaly, 0 for normal)
    binary_labels = (anomaly_labels == -1).astype(int)
    
    print(f"  📊 Created {np.sum(binary_labels)} anomalies out of {len(binary_labels)} samples ({np.mean(binary_labels):.2%})")
    
    return binary_labels

def extract_text_features(df, dataset_name):
    """Extract text features from dataset"""
    print(f"  📝 Extracting text features for {dataset_name}...")
    
    # Text columns to consider
    text_columns = ['name', 'description', 'command', 'task', 'job', 'workflow']
    
    # Find available text columns
    available_columns = [col for col in text_columns if col in df.columns]
    
    if not available_columns:
        print(f"  ❌ No text columns found in {dataset_name}")
        return None
    
    print(f"  📋 Found text columns: {available_columns}")
    
    # Combine text from all available columns
    text_data = []
    
    with tqdm(total=len(df), desc="  Processing rows", ncols=80) as pbar:
        for _, row in df.iterrows():
            combined_text = []
            for col in available_columns:
                if pd.notna(row[col]):
                    combined_text.append(str(row[col]))
            
            # Join all text with spaces
            full_text = ' '.join(combined_text) if combined_text else ''
            text_data.append(full_text)
            pbar.update(1)
    
    print(f"  ✅ Extracted {len(text_data)} text samples")
    
    return text_data

def train_text_anomaly_model(text_data, dataset_name, save_dir):
    """Train text-based anomaly detection model"""
    print(f"\n🔄 Training text anomaly model for {dataset_name}...")
    
    # Create synthetic labels
    y = create_synthetic_anomaly_labels(text_data)
    
    # Load sentence transformer model
    print("  🤖 Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    print("  🔤 Generating text embeddings...")
    with tqdm(total=len(text_data), desc="  Embedding", ncols=80) as pbar:
        embeddings = model.encode(text_data, show_progress_bar=False)
        pbar.update(len(text_data))
    
    print(f"  📊 Generated embeddings shape: {embeddings.shape}")
    
    # Scale embeddings
    print("  📏 Scaling embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Split data
    print("  ✂️ Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train Isolation Forest
    print("  🌲 Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    
    with tqdm(total=100, desc="  Training IF", ncols=80) as pbar:
        iso_forest.fit(X_train)
        pbar.update(100)
    
    # Predict anomalies
    print("  🔍 Detecting anomalies...")
    with tqdm(total=len(X_test), desc="  Predicting", ncols=80) as pbar:
        anomaly_scores = iso_forest.decision_function(X_test)
        y_pred = iso_forest.predict(X_test)
        pbar.update(len(X_test))
    
    # Convert predictions to binary (1 for anomaly, 0 for normal)
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Calculate metrics
    print("  📈 Calculating metrics...")
    roc_auc = roc_auc_score(y_test, -anomaly_scores)  # Negative because lower scores indicate anomalies
    f1 = f1_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    # Save results
    print(f"  💾 Saving results to {save_dir}...")
    results = {
        'dataset': dataset_name,
        'model': 'Text_Unsupervised_Simple',
        'roc_auc': float(roc_auc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'anomaly_rate': float(np.mean(y_pred_binary))
    }
    
    # Save metrics
    with open(os.path.join(save_dir, f'{dataset_name}_text_unsupervised_simple_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, -anomaly_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} (Text Unsupervised Simple)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_text_unsupervised_simple_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix - {dataset_name} (Text Unsupervised Simple)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_text_unsupervised_simple_confusion.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Text unsupervised simple training completed for {dataset_name}")
    print(f"     ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    
    return results

def main():
    # Define datasets
    datasets = [
        '1000genome', 'casa_nowcast', 'casa_wind_speed', 'eht_difmap', 
        'eht_imaging', 'eht_smili', 'montage', 'predict_future_sales',
        'pycbc_inference', 'pycbc_search', 'somospie', 'variant_calling'
    ]
    
    results_dir = 'results_text_unsupervised_simple'
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    print("🚀 Starting Text Unsupervised Simple Anomaly Detection Training")
    print(f"📊 Processing {len(datasets)} datasets...")
    
    # Process each dataset with progress bar
    for dataset_name in tqdm(datasets, desc="Overall Progress", ncols=100):
        print(f"\n{'='*60}")
        print(f"📂 Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Load data
            data_path = f'data/{dataset_name}/{dataset_name}.csv'
            if not os.path.exists(data_path):
                print(f"  ❌ Data file not found: {data_path}")
                continue
            
            print(f"  📥 Loading data from {data_path}...")
            df = pd.read_csv(data_path)
            print(f"  📊 Dataset shape: {df.shape}")
            
            # Extract text features
            text_data = extract_text_features(df, dataset_name)
            
            if text_data is None or len(text_data) == 0:
                print(f"  ❌ No text data found for {dataset_name}")
                continue
            
            print(f"  📝 Text samples: {len(text_data)}")
            
            # Train model
            result = train_text_anomaly_model(text_data, dataset_name, results_dir)
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
        summary_path = os.path.join(results_dir, 'text_unsupervised_simple_summary.csv')
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
    
    print(f"\n✅ Text unsupervised simple training completed! Results saved in '{results_dir}' directory")

if __name__ == "__main__":
    main() 