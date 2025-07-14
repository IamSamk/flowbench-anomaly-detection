#!/usr/bin/env python3
"""
Enhanced Text-based Unsupervised Anomaly Detection
Uses multiple Sentence-BERT models and multiple outlier detection algorithms
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

def evaluate_dataset(dataset_name, texts, labels):
    """Evaluate multiple models and detectors on a dataset"""
    print(f"\n=== Training Enhanced Text-based Anomaly Detection on {dataset_name} ===")
    
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
    
    return all_results

def save_results(results, dataset_name):
    """Save results for a dataset"""
    if not results:
        return
    
    # Create results directory
    results_dir = f"results_text_unsupervised_enhanced/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save detailed results
    df.to_csv(f"{results_dir}/detailed_results.csv", index=False)
    
    # Find best result
    best_idx = df['roc_auc'].idxmax()
    best_result = df.loc[best_idx]
    
    print(f"\n=== Best Results for {dataset_name} ===")
    print(f"Model: {best_result['model']} + {best_result['detector']}")
    print(f"Accuracy:  {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1-Score:  {best_result['f1_score']:.4f}")
    print(f"ROC-AUC:   {best_result['roc_auc']:.4f}")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC-AUC comparison
    ax1 = axes[0, 0]
    model_detector = df['model'] + ' + ' + df['detector']
    ax1.bar(range(len(df)), df['roc_auc'])
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(model_detector, rotation=45, ha='right')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('ROC-AUC Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Metrics heatmap
    ax2 = axes[0, 1]
    metrics_df = df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax2)
    ax2.set_title('Metrics Heatmap')
    ax2.set_xticklabels(model_detector, rotation=45, ha='right')
    
    # Confusion matrix for best model
    ax3 = axes[1, 0]
    best_predictions = results[best_idx]['predictions']
    cm = confusion_matrix(labels, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'Confusion Matrix\n{best_result["model"]} + {best_result["detector"]}')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # F1 vs ROC-AUC scatter
    ax4 = axes[1, 1]
    ax4.scatter(df['roc_auc'], df['f1_score'], s=100, alpha=0.7)
    ax4.set_xlabel('ROC-AUC')
    ax4.set_ylabel('F1-Score')
    ax4.set_title('F1-Score vs ROC-AUC')
    ax4.grid(True, alpha=0.3)
    
    # Highlight best result
    ax4.scatter(best_result['roc_auc'], best_result['f1_score'], 
                color='red', s=200, marker='*', label='Best')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/comparison_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_result

def main():
    """Main function"""
    print("Running Enhanced Text-based Unsupervised Anomaly Detection...")
    
    # Set data directory
    data_dir = "data"
    
    # Define datasets
    datasets = ['1000genome', 'casa_nowcast', 'casa_wind_speed', 'eht_difmap', 
               'eht_imaging', 'eht_smili', 'montage', 'predict_future_sales', 
               'pycbc_inference', 'pycbc_search', 'somospie', 'variant_calling']
    
    print(f"Found datasets: {datasets}")
    
    all_best_results = []
    
    for dataset_name in datasets:
        try:
            # Load dataset
            texts, labels = load_text_dataset(data_dir, dataset_name)
            if texts is None or labels is None:
                continue
            
            # Evaluate with multiple models and detectors
            results = evaluate_dataset(dataset_name, texts, labels)
            
            if results:
                # Save results for this dataset
                best_result = save_results(results, dataset_name)
                all_best_results.append(best_result)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    # Create overall summary
    if all_best_results:
        summary_df = pd.DataFrame(all_best_results)
        summary_df = summary_df.sort_values('roc_auc', ascending=False)
        
        # Save overall summary
        os.makedirs("results_text_unsupervised_enhanced", exist_ok=True)
        summary_df.to_csv("results_text_unsupervised_enhanced/enhanced_text_results_summary.csv", index=False)
        
        print(f"\n=== Enhanced Text-based Unsupervised Training Complete ===")
        print(f"Processed {len(all_best_results)} datasets")
        print(f"Best overall ROC-AUC: {summary_df['roc_auc'].max():.4f}")
        print(f"Average ROC-AUC: {summary_df['roc_auc'].mean():.4f}")
        
        # Create overall comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ROC-AUC by dataset
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(summary_df)), summary_df['roc_auc'])
        ax1.set_xticks(range(len(summary_df)))
        ax1.set_xticklabels(summary_df['dataset'], rotation=45, ha='right')
        ax1.set_ylabel('ROC-AUC')
        ax1.set_title('Best ROC-AUC by Dataset')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            if summary_df.iloc[i]['roc_auc'] > 0.8:
                bar.set_color('green')
            elif summary_df.iloc[i]['roc_auc'] > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Model distribution
        ax2 = axes[0, 1]
        model_counts = summary_df['model'].value_counts()
        ax2.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
        ax2.set_title('Best Model Distribution')
        
        # Detector distribution
        ax3 = axes[1, 0]
        detector_counts = summary_df['detector'].value_counts()
        ax3.pie(detector_counts.values, labels=detector_counts.index, autopct='%1.1f%%')
        ax3.set_title('Best Detector Distribution')
        
        # ROC-AUC vs F1 scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(summary_df['roc_auc'], summary_df['f1_score'], 
                            c=summary_df['roc_auc'], cmap='RdYlBu', s=100)
        ax4.set_xlabel('ROC-AUC')
        ax4.set_ylabel('F1-Score')
        ax4.set_title('ROC-AUC vs F1-Score')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('ROC-AUC')
        
        plt.tight_layout()
        plt.savefig("results_text_unsupervised_enhanced/overall_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results saved to results_text_unsupervised_enhanced/")
    
    print("\nEnhanced text-based unsupervised training complete!")

if __name__ == "__main__":
    main() 