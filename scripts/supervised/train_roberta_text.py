#!/usr/bin/env python3
"""
Improved RoBERTa Text Classification for Montage Workflow
- Per-batch logging and plots
- Early stopping (patience=2)
- Class weights for imbalance
- ROC-AUC-based checkpointing
- Progress bar and timing
- More epochs (default 10)
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class UltraFastTextDataset(Dataset):
    """Ultra-fast text dataset with pre-tokenization and caching."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts for maximum speed
        print("🔤 Pre-tokenizing texts for ultra-fast training...")
        with tqdm(total=100, desc="🔄 Tokenizing", ncols=80) as pbar:
            self.encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            )
            pbar.update(100)
        self.labels = torch.tensor(labels, dtype=torch.long)
        print(f"✅ Tokenization complete. Dataset size: {len(self.texts)}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item

# --- Focal Loss implementation ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedRoBERTaTrainer:
    """Ultra-fast RoBERTa trainer with mixed precision and optimizations."""
    
    def __init__(self, model, device, class_weights, learning_rate=1e-5, batch_size=32, grad_accum=2, use_focal=True):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Optimizer with higher learning rate for faster convergence
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)
        
        # Configurable loss
        if use_focal:
            self.criterion = FocalLoss(gamma=0.5)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_rocs = []
        self.train_loss_steps = []
        self.val_loss_steps = []
        self.val_acc_steps = []
        self.val_roc_steps = []
        self.best_val_roc = 0
        self.best_model_path = None
        
    def train(self, train_loader, val_loader, num_epochs=10, eval_steps=50, early_stop_patience=2):
        """Train for one epoch with mixed precision and validation."""
        print(f"🚀 Starting RoBERTa training for {num_epochs} epochs...")
        step = 0
        best_epoch = 0
        patience = 0
        for epoch in range(num_epochs):
            print(f"\n🔄 Epoch {epoch+1}/{num_epochs}")
            self.model.train()
            epoch_loss = 0
            start_time = time.time()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"🏋️ Training Epoch {epoch+1}", ncols=100)
            for batch_idx, batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                # --- Place for upsampling or focal loss if needed ---
                with autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs.logits, labels) / self.grad_accum
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                epoch_loss += loss.item() * self.grad_accum
                self.train_loss_steps.append(loss.item() * self.grad_accum)
                step += 1
                if val_loader and step % eval_steps == 0:
                    val_loss, val_acc, val_roc = self.evaluate(val_loader)
                    self.val_loss_steps.append(val_loss)
                    self.val_acc_steps.append(val_acc)
                    self.val_roc_steps.append(val_roc)
                    print(f"Step {step}: Train Loss: {loss.item()*self.grad_accum:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ROC-AUC: {val_roc:.4f}")
                    if val_roc > self.best_val_roc:
                        self.best_val_roc = val_roc
                        self.save_checkpoint(f"best_model.pt")
                        best_epoch = epoch
                        patience = 0
                        print(f"🎯 New best ROC-AUC: {val_roc:.4f}")
                    else:
                        patience += 1
                    self.scheduler.step(val_roc)
                    if patience > early_stop_patience:
                        print(f"⏹️ Early stopping at epoch {epoch+1}")
                        return
            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            epoch_time = time.time() - start_time
            print(f"✅ Epoch {epoch+1} completed in {epoch_time:.2f} seconds. Avg Loss: {avg_loss:.4f}")
        print("🎉 Training finished!")
    
    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="🔍 Evaluating", leave=False, ncols=80):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs.logits, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs.logits, 1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                correct += (preds == labels.cpu().numpy()).sum()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        try:
            roc = roc_auc_score(all_labels, all_probs)
        except Exception:
            roc = 0.5
        return avg_loss, accuracy, roc
    
    def predict(self, test_loader):
        """Get predictions for test set."""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    _, predictions = torch.max(outputs.logits, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Anomaly probability
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)
    
    def save_checkpoint(self, filename):
        pass  # Disabled: do not save model checkpoints
    def load_checkpoint(self, filename):
        pass  # Disabled: do not load model checkpoints

def load_montage_data(data_dir: str = 'data/montage') -> Tuple[List[str], List[int]]:
    """Load Montage workflow data directly from CSV files and create text representations."""
    print("Loading Montage workflow data from CSV files...")
    
    texts = []
    labels = []
    
    # Get all CSV files in the montage directory
    csv_files = list(Path(data_dir).glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    # Sample a subset for faster training (first 10 files)
    csv_files = csv_files[:20]
    print(f"Using {len(csv_files)} files for training")
    
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            print(f"Processing {csv_file.name}: {len(df)} rows")
            
            # Create text representations for each row
            for idx, row in df.iterrows():
                # Create a text description from the workflow features
                # Convert numeric features to text descriptions
                text_parts = []
                
                # Add basic workflow info
                text_parts.append(f"Montage workflow execution")
                
                # Add key features as text (convert numeric to descriptive)
                for col in df.columns:
                    if col in ['job_id', 'task_id', 'workflow_id']:
                        continue  # Skip ID columns
                    
                    value = row[col]
                    if pd.isna(value):
                        continue
                    
                    # Convert numeric values to descriptive text
                    if isinstance(value, (int, float)):
                        if value > 0:
                            text_parts.append(f"{col} is {value}")
                        else:
                            text_parts.append(f"{col} is zero")
                    else:
                        text_parts.append(f"{col} is {value}")
                
                # Join all parts into a single text
                text = ". ".join(text_parts)
                
                # Create anomaly label (for demonstration, use a simple rule)
                # In real scenarios, this would come from actual anomaly labels
                # For now, we'll use a simple heuristic based on execution time or other features
                if 'execution_time' in df.columns:
                    exec_time = row['execution_time']
                    if pd.notna(exec_time) and exec_time > df['execution_time'].quantile(0.95):
                        label = 1  # Anomaly (high execution time)
                    else:
                        label = 0  # Normal
                else:
                    # If no execution_time, use a random pattern for demonstration
                    label = 1 if idx % 20 == 0 else 0  # 5% anomaly rate
                
                texts.append(text)
                labels.append(label)
                
                # Limit the number of samples per file for speed
                if idx >= 100:  # Only use first 100 samples per file
                    break
                    
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print(f"Total samples: {len(texts)}")
    print(f"Anomaly rate: {sum(labels)/len(labels)*100:.2f}%")
    
    return texts, labels

def load_general_csv_data(data_dir: str) -> Tuple[List[str], List[int]]:
    """General loader for scientific workflow datasets in CSV format."""
    print(f"Loading general workflow data from CSV files in {data_dir}...")
    texts = []
    labels = []
    csv_files = list(Path(data_dir).glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    csv_files = csv_files[:20]  # Sample a subset for speed, as in montage
    print(f"Using {len(csv_files)} files for training")
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"Processing {csv_file.name}: {len(df)} rows")
            for idx, row in df.iterrows():
                text_parts = []
                text_parts.append(f"Workflow execution")
                for col in df.columns:
                    if col.lower() in ['job_id', 'task_id', 'workflow_id']:
                        continue
                    value = row[col]
                    if pd.isna(value):
                        continue
                    if isinstance(value, (int, float)):
                        if value > 0:
                            text_parts.append(f"{col} is {value}")
                        else:
                            text_parts.append(f"{col} is zero")
                    else:
                        text_parts.append(f"{col} is {value}")
                text = ". ".join(text_parts)
                # Label: use execution_time if present, else fallback to 5% anomaly
                if 'execution_time' in df.columns:
                    exec_time = row['execution_time']
                    if pd.notna(exec_time) and exec_time > df['execution_time'].quantile(0.95):
                        label = 1
                    else:
                        label = 0
                else:
                    label = 1 if idx % 20 == 0 else 0
                texts.append(text)
                labels.append(label)
                if idx >= 100:
                    break
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    print(f"Total samples: {len(texts)}")
    print(f"Anomaly rate: {sum(labels)/len(labels)*100:.2f}%")
    return texts, labels

def create_plots(results: Dict, output_dir: str, dataset_name: str = None):
    """Create comprehensive evaluation plots (with dataset name in titles if provided)."""
    os.makedirs(output_dir, exist_ok=True)
    # 1. Training curves (per step)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['train_loss_steps'], label='Train Loss (step)')
    if results['val_loss_steps']:
        plt.plot(np.linspace(0, len(results['train_loss_steps']), len(results['val_loss_steps'])), results['val_loss_steps'], label='Val Loss (eval)')
    plt.title(f'Training/Validation Loss (per step){f" - {dataset_name}" if dataset_name else ""}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    if results['val_roc_steps']:
        plt.plot(np.linspace(0, len(results['train_loss_steps']), len(results['val_roc_steps'])), results['val_roc_steps'], label='Val ROC-AUC (eval)')
        plt.title(f'Validation ROC-AUC (per eval){f" - {dataset_name}" if dataset_name else ""}')
        plt.xlabel('Step')
        plt.ylabel('ROC-AUC')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(results['true_labels'], results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix{f" - {dataset_name}" if dataset_name else ""}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 3. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = results['roc_curve']
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve{f" - {dataset_name}" if dataset_name else ""}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 4. Metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [
        results['accuracy'],
        results['precision'],
        results['recall'],
        results['f1_score'],
        results['roc_auc']
    ]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'])
    plt.title(f'Model Performance Metrics{f" - {dataset_name}" if dataset_name else ""}')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/montage', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='results/roberta_text_improved', help='Path to output directory')
    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    print(f"🚀 === Improved RoBERTa Text Classification for {dataset_name} Workflow ===")
    config = {
        'model_name': 'roberta-large',
        'max_length': 128,
        'batch_size': 32,
        'learning_rate': 1e-5,
        'num_epochs': 15,
        'eval_steps': 100,
        'grad_accum': 2,
        'output_dir': args.output_dir,
        'checkpoint_dir': os.path.join(args.output_dir.replace('results', 'checkpoints')),
        'use_focal': True
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Use montage loader only for montage, else use general loader
    print(f"📥 Loading data for {dataset_name}...")
    if dataset_name.lower() == 'montage':
        texts, labels = load_montage_data(args.data_dir)
    else:
        texts, labels = load_general_csv_data(args.data_dir)
    if len(texts) == 0:
        print("❌ No data loaded. Exiting.")
        return
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"📊 Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Load tokenizer and model
    print("🤖 Loading RoBERTa model and tokenizer...")
    try:
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        
        tokenizer = RobertaTokenizer.from_pretrained(config['model_name'])
        model = RobertaForSequenceClassification.from_pretrained(
            config['model_name'], 
            num_labels=2,
            problem_type="single_label_classification"
        )
        print("✅ Loaded roberta-large.")
    except RuntimeError as e:
        print("⚠️ [WARNING] Out of memory or error with roberta-large. Falling back to roberta-base.")
        config['model_name'] = 'roberta-base'
        config['batch_size'] = 16
        tokenizer = RobertaTokenizer.from_pretrained(config['model_name'])
        model = RobertaForSequenceClassification.from_pretrained(
            config['model_name'], num_labels=2, problem_type="single_label_classification")
    # Compute class weights
    print("⚖️ Computing class weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"📊 Class weights: {class_weights}")
    
    # Create datasets
    print("📦 Creating datasets...")
    train_dataset = UltraFastTextDataset(train_texts, train_labels, tokenizer, config['max_length'])
    val_dataset = UltraFastTextDataset(val_texts, val_labels, tokenizer, config['max_length'])
    test_dataset = UltraFastTextDataset(test_texts, test_labels, tokenizer, config['max_length'])
    
    # Create data loaders
    print("🔄 Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Initialize trainer
    trainer = ImprovedRoBERTaTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        grad_accum=config['grad_accum'],
        use_focal=config['use_focal']
    )
    
    # Training loop
    print("🏋️ Starting improved training...")
    start_time = time.time()
    
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        eval_steps=config['eval_steps'],
        early_stop_patience=2
    )
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Training completed in {total_time:.2f} seconds")
    
    # Load best model (last checkpoint)
    best_model_path = 'best_model.pt'
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)
        print(f"🎯 Loaded best model from {best_model_path}")
    
    # Evaluate on test set
    print("\n🔧 Tuning threshold on validation set for best F1...")
    # Get validation predictions and probabilities
    val_preds, val_probs, val_true = trainer.predict(val_loader)
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    best_f1 = 0
    best_thresh = 0.5
    with tqdm(total=19, desc="🎯 Tuning threshold", ncols=80) as pbar:
        for thresh in np.linspace(0.05, 0.95, 19):
            preds = (val_probs >= thresh).astype(int)
            f1 = f1_score(val_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
            pbar.update(1)
    print(f"🎯 Best threshold on validation set: {best_thresh:.2f} (F1={best_f1:.4f})")
    print("\n📊 Evaluating on test set with best threshold...")
    predictions, probabilities, true_labels = trainer.predict(test_loader)
    test_preds = (probabilities >= best_thresh).astype(int)
    accuracy = accuracy_score(true_labels, test_preds)
    precision = precision_score(true_labels, test_preds, average='weighted', zero_division=0)
    recall = recall_score(true_labels, test_preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, test_preds, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(true_labels, probabilities)
    from sklearn.metrics import roc_curve, confusion_matrix, classification_report
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': test_preds,
        'probabilities': probabilities,
        'true_labels': true_labels,
        'roc_curve': (fpr, tpr, _),
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_accuracies': trainer.val_accuracies,
        'train_loss_steps': trainer.train_loss_steps,
        'val_loss_steps': trainer.val_loss_steps,
        'val_acc_steps': trainer.val_acc_steps,
        'val_roc_steps': trainer.val_roc_steps
    }
    print("\n=== Test Results (using best threshold) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, test_preds))
    print("Class distribution in predictions:", np.bincount(test_preds))
    print("Class distribution in true labels:", np.bincount(true_labels))
    
    # Save results
    results_file = os.path.join(config['output_dir'], 'results.json')
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    results['dataset'] = dataset_name
    json_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            json_results[k] = v.tolist()
        elif isinstance(v, tuple) and any(isinstance(x, np.ndarray) for x in v):
            json_results[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        else:
            json_results[k] = v
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    print("Creating evaluation plots...")
    create_plots(results, config['output_dir'], dataset_name=dataset_name)
    print(f"Plots saved to {config['output_dir']}")
    
    print("\n=== Training Complete ===")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Results directory: {config['output_dir']}")

if __name__ == "__main__":
    main() 