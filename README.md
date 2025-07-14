# Machine Learning Paper - Anomaly Detection Across Workflow Datasets

## 🎯 Project Overview

This project implements and evaluates **6 anomaly detection models** across 12 workflow datasets using multiple modalities (text, tabular, graph) and approaches (3 supervised, 3 unsupervised).

## 📊 Datasets Analyzed

The following 12 workflow datasets were analyzed:
- `1000genome` - Genomics workflow
- `casa_nowcast` - Climate modeling  
- `casa_wind_speed` - Wind speed analysis
- `eht_difmap` - Event Horizon Telescope data processing
- `eht_imaging` - EHT imaging pipeline
- `eht_smili` - EHT SMILI processing
- `montage` - Astronomical image mosaicking
- `predict_future_sales` - Sales prediction workflow
- `pycbc_inference` - Gravitational wave inference
- `pycbc_search` - Gravitational wave search
- `somospie` - Solar dynamics analysis
- `variant_calling` - Genomic variant calling

## 🤖 Models Implemented

### Supervised Learning (3 models)
- **GCN Graph Classification**: Graph-based anomaly detection
- **Random Forest Tabular**: Tabular-based anomaly detection
- **RoBERTa Text Classification**: Text-based anomaly detection

### Unsupervised Learning (3 models)
- **Graph Autoencoder (Fixed)**: Graph-based anomaly detection
- **Gaussian Mixture Model**: Tabular-based anomaly detection
- **Sentence-BERT + Isolation Forest**: Text-based anomaly detection


## 📁 Project Structure

```
├── results/
│   ├── supervised/
│   │   └── gcn_graph/          # GCN Graph Classification
│   │   └── random_forest_tabular/          # Random Forest Tabular
│   │   └── roberta_text/          # RoBERTa Text Classification
│   └── unsupervised/
│       └── graph_gae/       # Graph Autoencoder (Fixed)
│       └── tabular_gmm/       # Gaussian Mixture Model
│       └── text_sentence_bert/       # Sentence-BERT + Isolation Forest
├── reports/
│   ├── supervised/                # Supervised learning reports
│   └── unsupervised/              # Unsupervised learning reports  
├── scripts/
│   ├── supervised/                # Supervised model scripts
│   ├── unsupervised/              # Unsupervised model scripts
│   └── utils/                     # Utility and organization scripts
└── data/                          # Dataset files
```

## 🚀 Usage

### Running Individual Models

```bash
# Supervised Models
python scripts/supervised/gcn_graph/train_*.py
python scripts/supervised/random_forest_tabular/train_*.py
python scripts/supervised/roberta_text/train_*.py

# Unsupervised Models
python scripts/unsupervised/graph_gae/train_*.py
python scripts/unsupervised/tabular_gmm/train_*.py
python scripts/unsupervised/text_sentence_bert/train_*.py
```

## 📊 Result Files

Each model directory contains:
- `*_results_summary.csv` - Performance metrics summary
- `*_confusion_matrices.png` - Confusion matrix visualizations  
- `*_roc_curves.png` - ROC curve plots
- `*_performance_summary.png` - Comprehensive performance analysis

## 🏆 Key Findings

This comprehensive evaluation across multiple modalities and approaches provides insights into:
- **Best performing models** for each data type
- **Cross-modal performance** comparisons
- **Supervised vs unsupervised** effectiveness
- **Dataset-specific** anomaly patterns

## 📚 Dependencies

```
torch>=1.9.0
transformers>=4.0.0
sentence-transformers>=2.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

---

**🎉 Project Status: Complete**  
All models trained, evaluated, and results organized successfully!
