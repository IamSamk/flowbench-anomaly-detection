#!/usr/bin/env python3
"""
Quick test of enhanced text-based unsupervised anomaly detection
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
from sklearn.covariance import EllipticEnvelope
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
    
    # Remove outliers using IQR method
    print("Removing outliers from embeddings...")
    Q1 = np.percentile(embeddings, 25, axis=0)
    Q3 = np.percentile(embeddings, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of removing them
    embeddings = np.clip(embeddings, lower_bound, upper_bound)
    
    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    return embeddings_scaled

def train_outlier_detector(embeddings, labels, detector_name, detector):
    """Train outlier detector and evaluate"""
    print(f"Training {detector_name}...")
    
    # Fit the detector (use normal samples for training)
    normal_mask = labels == 0
    if np.sum(normal_mask) > 0:
        detector.fit(embeddings[normal_mask])
    else:
        detector.fit(embeddings)
    
    # Predict anomalies
    predictions = detector.predict(embeddings)
    
    # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
    predictions = (predictions == -1).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Calculate ROC-AUC using decision function if available
    try:
        if hasattr(detector, 'decision_function'):
            scores = detector.decision_function(embeddings)
        elif hasattr(detector, 'score_samples'):
            scores = detector.score_samples(embeddings)
        else:
            scores = predictions.astype(float)
        
        # Normalize scores to [0, 1] range
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        roc_auc = roc_auc_score(labels, scores)
    except:
        roc_auc = 0.5
    
    return {
        'detector': detector_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': predictions,
        'scores': scores if 'scores' in locals() else predictions.astype(float)
    }

def create_detectors():
    """Create different outlier detection algorithms"""
    return [
        ('IsolationForest', IsolationForest(contamination=0.1, random_state=42)),
        ('LOF', LocalOutlierFactor(contamination=0.1, novelty=False)),
        ('OneClassSVM', OneClassSVM(kernel='rbf', nu=0.1)),
        ('EllipticEnvelope', EllipticEnvelope(contamination=0.1, random_state=42))
    ]

def main():
    """Test on one dataset"""
    print("Testing Enhanced Text-based Unsupervised Anomaly Detection...")
    
    # Test on montage dataset
    dataset_name = "montage"
    data_dir = "data"
    
    # Load dataset
    texts, labels = load_text_dataset(data_dir, dataset_name)
    if texts is None or labels is None:
        print("Failed to load dataset")
        return
    
    # Define models to try
    models = [
        ('all-MiniLM-L6-v2', 'all-MiniLM-L6-v2'),
        ('all-mpnet-base-v2', 'all-mpnet-base-v2'),
        ('all-roberta-large-v1', 'all-roberta-large-v1')
    ]
    
    all_results = []
    
    for model_name, model_path in models:
        try:
            # Generate embeddings
            embeddings, actual_model = get_embeddings(texts, model_path)
            
            # Preprocess embeddings
            embeddings_processed = preprocess_embeddings(embeddings)
            print(f"Generated embeddings shape: {embeddings_processed.shape}")
            
            # Test all detectors
            detectors = create_detectors()
            
            for detector_name, detector in detectors:
                try:
                    result = train_outlier_detector(embeddings_processed, labels, detector_name, detector)
                    result['model'] = model_name
                    result['actual_model'] = actual_model
                    result['dataset'] = dataset_name
                    all_results.append(result)
                    
                    print(f"  {model_name} + {detector_name}: ROC-AUC = {result['roc_auc']:.4f}")
                    
                except Exception as e:
                    print(f"  Error with {model_name} + {detector_name}: {e}")
                    continue
                    
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
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 