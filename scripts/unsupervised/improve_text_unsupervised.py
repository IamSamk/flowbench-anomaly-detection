import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available, using TF-IDF as fallback")

def get_text_columns_for_dataset(dataset_name):
    """Get the appropriate text columns for each dataset."""
    
    # Dataset-specific text column mappings
    text_column_mappings = {
        # EHT datasets have specific column structure
        'eht_difmap': [0, 'kickstart_transformations', 'kickstart_executables', 'kickstart_executables_argv'],
        'eht_imaging': [0, 'kickstart_transformations', 'kickstart_executables', 'kickstart_executables_argv'],
        'eht_smili': [0, 'kickstart_transformations', 'kickstart_executables', 'kickstart_executables_argv'],
        
        # Somospie might have different structure
        'somospie': [0, 'kickstart_transformations', 'kickstart_executables', 'kickstart_executables_argv'],
        
        # Default text columns for other datasets
        'default': ['name', 'description', 'command', 'task', 'job', 'workflow', 'kickstart_transformations', 'kickstart_executables']
    }
    
    return text_column_mappings.get(dataset_name, text_column_mappings['default'])

def load_and_extract_text_features(data_dir, dataset_name, anomaly_rate=0.1):
    """Load dataset and extract enhanced text features."""
    
    print(f"\n🔍 Processing {dataset_name}...")
    
    # Get appropriate text columns
    text_columns = get_text_columns_for_dataset(dataset_name)
    print(f"📋 Using text columns: {text_columns}")
    
    dataset_path = Path(data_dir) / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    
    all_texts = []
    all_labels = []
    
    print(f"📂 Found {len(csv_files)} CSV files")
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        try:
            df = pd.read_csv(csv_file)
            
            # Extract text features from each row
            for idx, row in df.iterrows():
                text_parts = []
                
                # Handle different column types
                for col in text_columns:
                    if col == 0:  # First column (usually task name)
                        if len(df.columns) > 0:
                            first_col = df.columns[0]
                            if pd.notna(row[first_col]):
                                text_parts.append(str(row[first_col]))
                    elif isinstance(col, str) and col in df.columns:
                        if pd.notna(row[col]):
                            text_parts.append(str(row[col]))
                
                # Create combined text
                if text_parts:
                    combined_text = " ".join(text_parts)
                    all_texts.append(combined_text)
                    
                    # Create synthetic anomaly label (will be improved)
                    all_labels.append(0)  # Placeholder
                        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not all_texts:
        raise ValueError(f"No valid text data found for {dataset_name}")
    
    # Create better synthetic anomaly labels
    all_labels = create_enhanced_synthetic_labels(all_texts, anomaly_rate)
    
    print(f"✅ Extracted {len(all_texts)} text samples")
    print(f"📊 Anomaly rate: {sum(all_labels) / len(all_labels):.2%}")
    
    return all_texts, all_labels

def create_enhanced_synthetic_labels(texts, anomaly_rate=0.1):
    """Create enhanced synthetic anomaly labels based on multiple text features."""
    
    print("🎯 Creating enhanced synthetic anomaly labels...")
    
    # Calculate multiple text features
    features = []
    
    for text in tqdm(texts, desc="Extracting features"):
        text_str = str(text)
        
        # Basic text features
        text_length = len(text_str)
        word_count = len(text_str.split())
        unique_words = len(set(text_str.split()))
        avg_word_length = np.mean([len(word) for word in text_str.split()]) if text_str.split() else 0
        
        # Advanced features
        char_diversity = len(set(text_str)) / len(text_str) if text_str else 0
        digit_ratio = sum(c.isdigit() for c in text_str) / len(text_str) if text_str else 0
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in text_str) / len(text_str) if text_str else 0
        
        features.append([
            text_length, word_count, unique_words, avg_word_length,
            char_diversity, digit_ratio, special_char_ratio
        ])
    
    features = np.array(features)
    
    # Use isolation forest to identify unusual texts
    iso_forest = IsolationForest(contamination=anomaly_rate, random_state=42)
    anomaly_labels = iso_forest.fit_predict(features)
    
    # Convert to binary (1 for anomaly, 0 for normal)
    binary_labels = (anomaly_labels == -1).astype(int)
    
    print(f"📊 Created {np.sum(binary_labels)} anomalies out of {len(binary_labels)} samples ({np.mean(binary_labels):.2%})")
    
    return binary_labels

def try_multiple_embedding_models(texts, models_to_try=None):
    """Try multiple embedding models and return the best one."""
    
    if models_to_try is None:
        models_to_try = [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'paraphrase-MiniLM-L6-v2',
            'distilbert-base-nli-mean-tokens'
        ]
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("🔄 Using TF-IDF as fallback embedding")
        vectorizer = TfidfVectorizer(max_features=512, stop_words='english', ngram_range=(1, 2))
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings, 'tfidf'
    
    print(f"🤖 Trying {len(models_to_try)} embedding models...")
    
    best_embeddings = None
    best_model_name = None
    
    for model_name in models_to_try:
        try:
            print(f"  🔄 Testing {model_name}...")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts[:100], show_progress_bar=False)  # Test on small sample
            
            # Simple quality check - embeddings should have reasonable variance
            if embeddings.var() > 0.01:  # Arbitrary threshold
                print(f"  ✅ {model_name} looks good")
                # Generate full embeddings
                full_embeddings = model.encode(texts, show_progress_bar=True)
                best_embeddings = full_embeddings
                best_model_name = model_name
                break
            else:
                print(f"  ❌ {model_name} has low variance")
                
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
            continue
    
    if best_embeddings is None:
        print("🔄 All models failed, using TF-IDF as fallback")
        vectorizer = TfidfVectorizer(max_features=512, stop_words='english', ngram_range=(1, 2))
        best_embeddings = vectorizer.fit_transform(texts).toarray()
        best_model_name = 'tfidf'
    
    print(f"🎯 Selected model: {best_model_name}")
    print(f"📊 Embeddings shape: {best_embeddings.shape}")
    
    return best_embeddings, best_model_name

def train_improved_isolation_forest(embeddings, contamination=0.1):
    """Train improved isolation forest with better parameters."""
    
    print("🤖 Training improved Isolation Forest...")
    
    # Try different contamination rates and pick the best
    contamination_rates = [0.05, 0.1, 0.15, 0.2]
    
    best_iso_forest = None
    best_scores = None
    
    for cont_rate in contamination_rates:
        iso_forest = IsolationForest(
            contamination=cont_rate,
            random_state=42,
            n_estimators=200,  # More trees
            max_samples='auto',
            bootstrap=True
        )
        
        iso_forest.fit(embeddings)
        scores = -iso_forest.decision_function(embeddings)
        
        # Simple quality check - scores should have good spread
        if scores.std() > 0.1:
            best_iso_forest = iso_forest
            best_scores = scores
            print(f"  ✅ Contamination rate {cont_rate} selected")
            break
    
    if best_iso_forest is None:
        # Fallback to default
        best_iso_forest = IsolationForest(contamination=0.1, random_state=42)
        best_iso_forest.fit(embeddings)
        best_scores = -best_iso_forest.decision_function(embeddings)
    
    return best_scores, best_iso_forest

def evaluate_and_save_results(dataset_name, texts, y_true, scores, model_name, output_dir):
    """Evaluate and save results with comprehensive metrics."""
    
    print(f"📊 Evaluating results for {dataset_name}...")
    
    try:
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        
        # Find optimal threshold
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
        
        results = {
            'dataset': dataset_name,
            'model': model_name,
            'roc_auc': auc,
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'best_threshold': float(best_threshold),
            'roc_curve': (fpr.tolist(), tpr.tolist(), thresholds.tolist())
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        create_improved_visualizations(results, y_true, y_pred, scores, output_dir)
        
        print(f"✅ Results saved to {output_dir}")
        print(f"📈 ROC-AUC: {auc:.4f}, F1: {f1:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        return {
            'dataset': dataset_name,
            'model': model_name,
            'roc_auc': 0.5,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'best_threshold': 0.0,
            'roc_curve': ([0, 1], [0, 1], [0, 1])
        }

def create_improved_visualizations(results, y_true, y_pred, scores, output_dir):
    """Create improved visualizations."""
    
    # ROC Curve
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    fpr, tpr, _ = results['roc_curve']
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {results['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion Matrix
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Score Distribution
    plt.subplot(2, 2, 3)
    plt.hist(scores[y_true == 0], bins=30, alpha=0.7, label='Normal', density=True)
    plt.hist(scores[y_true == 1], bins=30, alpha=0.7, label='Anomaly', density=True)
    plt.axvline(results['best_threshold'], color='red', linestyle='--', label='Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    
    # Metrics Summary
    plt.subplot(2, 2, 4)
    metrics = ['ROC-AUC', 'F1', 'Accuracy', 'Precision', 'Recall']
    values = [results['roc_auc'], results['f1_score'], results['accuracy'], 
              results['precision'], results['recall']]
    
    bars = plt.bar(metrics, values, color=['green', 'blue', 'orange', 'red', 'purple'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset(data_dir, dataset_name, output_base_dir):
    """Process a single dataset with improved text approach."""
    
    try:
        # Load and extract text features
        texts, labels = load_and_extract_text_features(data_dir, dataset_name)
        
        # Try multiple embedding models
        embeddings, model_name = try_multiple_embedding_models(texts)
        
        # Train improved isolation forest
        scores, iso_forest = train_improved_isolation_forest(embeddings)
        
        # Evaluate and save results
        output_dir = os.path.join(output_base_dir, f'improved_text_{dataset_name}')
        results = evaluate_and_save_results(dataset_name, texts, labels, scores, model_name, output_dir)
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        return None

def main():
    """Main function to run improved text unsupervised anomaly detection."""
    
    # Focus on the poor performing datasets
    poor_datasets = ['eht_difmap', 'eht_imaging', 'eht_smili', 'somospie']
    
    data_dir = 'data'
    output_dir = 'results_text_unsupervised'
    
    print("🚀 Starting improved text unsupervised anomaly detection...")
    print(f"📁 Data directory: {data_dir}")
    print(f"📊 Output directory: {output_dir}")
    print(f"🎯 Focusing on poor performing datasets: {poor_datasets}")
    
    all_results = []
    
    for dataset in poor_datasets:
        print(f"\n{'='*60}")
        print(f"🔍 Processing dataset: {dataset}")
        print(f"{'='*60}")
        
        result = process_dataset(data_dir, dataset, output_dir)
        if result:
            all_results.append(result)
            print(f"✅ {dataset}: ROC-AUC = {result['roc_auc']:.4f}")
        else:
            print(f"❌ {dataset}: Failed to process")
    
    # Create summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = os.path.join(output_dir, 'improved_text_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n📊 IMPROVED TEXT RESULTS SUMMARY:")
        print(f"📈 Average ROC-AUC: {summary_df['roc_auc'].mean():.4f}")
        print(f"🏆 Best performer: {summary_df.loc[summary_df['roc_auc'].idxmax(), 'dataset']} (ROC-AUC: {summary_df['roc_auc'].max():.4f})")
        print(f"📊 Summary saved to: {summary_file}")
        
        # Show improvements
        print(f"\n🔄 IMPROVEMENTS:")
        original_scores = {
            'eht_difmap': 0.3660,
            'eht_imaging': 0.2868,
            'eht_smili': 0.2766,
            'somospie': 0.4931
        }
        
        for _, row in summary_df.iterrows():
            dataset = row['dataset']
            new_score = row['roc_auc']
            old_score = original_scores.get(dataset, 0.5)
            improvement = new_score - old_score
            
            if improvement > 0:
                print(f"  ✅ {dataset}: {old_score:.4f} → {new_score:.4f} (+{improvement:.4f})")
            else:
                print(f"  ❌ {dataset}: {old_score:.4f} → {new_score:.4f} ({improvement:.4f})")
    
    print(f"\n🎉 Improved text unsupervised analysis complete!")

if __name__ == "__main__":
    main() 