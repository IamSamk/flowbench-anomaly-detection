#!/usr/bin/env python3
"""
Fixed test of enhanced text-based unsupervised anomaly detection
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import flowbench
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available, using TF-IDF fallback")

from sklearn.feature_extraction.text import TfidfVectorizer

def load_text_dataset(data_dir, dataset_name):
    """Load text dataset from CSV files."""
    print(f"Loading text dataset for {dataset_name}...")
    
    import glob
    csv_files = glob.glob(os.path.join(data_dir, dataset_name, '*.csv'))
    if not csv_files:
        print(f"No CSV files found for {dataset_name}")
        return None, None
    
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
        print(f"No valid text data found for {dataset_name}")
        return None, None
    
    print(f"Loaded {len(all_texts)} text samples")
    
    # Create synthetic anomalies (10% of data)
    n_samples = len(all_texts)
    n_anomalies = max(1, int(0.1 * n_samples))
    
    # Mark last n_anomalies as anomalies
    labels = np.zeros(n_samples)
    labels[-n_anomalies:] = 1
    
    anomaly_rate = (n_anomalies / n_samples) * 100
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    
    return all_texts, labels

def get_embeddings(texts, model_name):
    """Generate embeddings using specified model"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Using TF-IDF vectorization as fallback")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings, f"TF-IDF-{embeddings.shape[1]}"
    
    print(f"Using SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings, model_name

def preprocess_embeddings(embeddings):
    """Preprocess embeddings: handle missing values, outliers, and scale"""
    # Convert to numpy array if needed
    embeddings = np.array(embeddings)
    
    # Handle missing values (replace with median)
    if np.any(np.isnan(embeddings)):
        print("Handling missing values...")
        median_vals = np.nanmedian(embeddings, axis=0)
        for i in range(embeddings.shape[1]):
            embeddings[np.isnan(embeddings[:, i]), i] = median_vals[i]
    
    # Handle infinite values
    if np.any(np.isinf(embeddings)):
        print("Handling infinite values...")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    return embeddings_scaled

def train_isolation_forest(embeddings, labels):
    """Train Isolation Forest and evaluate"""
    print("Training IsolationForest...")
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(embeddings)
    
    # Get anomaly scores
    scores = -iso_forest.decision_function(embeddings)  # Convert to positive scores
    
    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else np.median(scores)
    
    # Make predictions
    predictions = (scores > best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(labels, scores)
    
    return {
        'detector': 'IsolationForest',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': predictions,
        'scores': scores
    }

def train_lof(embeddings, labels):
    """Train Local Outlier Factor and evaluate"""
    print("Training LOF...")
    
    # LOF needs to be used differently - it doesn't have a predict method
    # We'll use it for scoring only
    lof = LocalOutlierFactor(contamination=0.1, novelty=False)
    scores = -lof.fit_predict(embeddings)  # Convert to positive scores
    
    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else np.median(scores)
    
    # Make predictions
    predictions = (scores > best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(labels, scores)
    
    return {
        'detector': 'LOF',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': predictions,
        'scores': scores
    }

def train_oneclass_svm(embeddings, labels):
    """Train One-Class SVM and evaluate"""
    print("Training OneClassSVM...")
    
    # Use normal samples for training
    normal_mask = labels == 0
    if np.sum(normal_mask) > 0:
        svm = OneClassSVM(kernel='rbf', nu=0.1)
        svm.fit(embeddings[normal_mask])
    else:
        svm = OneClassSVM(kernel='rbf', nu=0.1)
        svm.fit(embeddings)
    
    # Get scores
    scores = -svm.decision_function(embeddings)  # Convert to positive scores
    
    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else np.median(scores)
    
    # Make predictions
    predictions = (scores > best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(labels, scores)
    
    return {
        'detector': 'OneClassSVM',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': predictions,
        'scores': scores
    }

def main():
    """Test on one dataset"""
    print("Testing Fixed Enhanced Text-based Unsupervised Anomaly Detection...")
    
    # Test on montage dataset
    dataset_name = "montage"
    data_dir = "data"
    
    # Load dataset
    texts, labels = load_text_dataset(data_dir, dataset_name)
    if texts is None or labels is None:
        print("Failed to load dataset")
        return
    
    # Define models to try (focus on smaller, faster models)
    models = [
        ('all-MiniLM-L6-v2', 'all-MiniLM-L6-v2'),
        ('all-mpnet-base-v2', 'all-mpnet-base-v2')
    ]
    
    all_results = []
    
    for model_name, model_path in models:
        try:
            # Generate embeddings
            embeddings, actual_model = get_embeddings(texts, model_path)
            
            # Preprocess embeddings
            embeddings_processed = preprocess_embeddings(embeddings)
            print(f"Generated embeddings shape: {embeddings_processed.shape}")
            
            # Test detectors
            try:
                result_if = train_isolation_forest(embeddings_processed, labels)
                result_if['model'] = model_name
                result_if['actual_model'] = actual_model
                result_if['dataset'] = dataset_name
                all_results.append(result_if)
                print(f"  {model_name} + IsolationForest: ROC-AUC = {result_if['roc_auc']:.4f}")
            except Exception as e:
                print(f"  Error with {model_name} + IsolationForest: {e}")
            
            try:
                result_lof = train_lof(embeddings_processed, labels)
                result_lof['model'] = model_name
                result_lof['actual_model'] = actual_model
                result_lof['dataset'] = dataset_name
                all_results.append(result_lof)
                print(f"  {model_name} + LOF: ROC-AUC = {result_lof['roc_auc']:.4f}")
            except Exception as e:
                print(f"  Error with {model_name} + LOF: {e}")
            
            try:
                result_svm = train_oneclass_svm(embeddings_processed, labels)
                result_svm['model'] = model_name
                result_svm['actual_model'] = actual_model
                result_svm['dataset'] = dataset_name
                all_results.append(result_svm)
                print(f"  {model_name} + OneClassSVM: ROC-AUC = {result_svm['roc_auc']:.4f}")
            except Exception as e:
                print(f"  Error with {model_name} + OneClassSVM: {e}")
                    
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue
    
    # Show results
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n=== Results Summary ===")
        print(df[['model', 'detector', 'roc_auc', 'f1_score', 'accuracy']].to_string(index=False))
        
        # Find best result
        best_idx = df['roc_auc'].idxmax()
        best_result = df.loc[best_idx]
        
        print(f"\n=== Best Result ===")
        print(f"Model: {best_result['model']} + {best_result['detector']}")
        print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
        print(f"F1-Score: {best_result['f1_score']:.4f}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        
        # Compare with simple results
        print(f"\n=== Comparison with Simple Results ===")
        print("Simple (all-MiniLM-L6-v2 + IsolationForest): ROC-AUC = 0.9391")
        print(f"Enhanced Best: ROC-AUC = {best_result['roc_auc']:.4f}")
        
        if best_result['roc_auc'] > 0.9391:
            print("✅ Enhanced approach shows improvement!")
        else:
            print("❌ No improvement with enhanced approach")
            print("Recommendation: Stick with the simple approach (all-MiniLM-L6-v2 + IsolationForest)")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 