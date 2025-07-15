git # Machine Learning Workflow Anomaly Detection - Project Guide

Welcome! This project is a comprehensive benchmark of anomaly detection models across scientific workflow datasets. It is organized for clarity, reproducibility, and ease of use.

---

## 📦 Project Structure: What is What?

- **`data/`**: All raw and processed datasets used for experiments.
- **`results/`**: Final, organized results for each model (metrics, plots, summaries).
  - **`supervised/`**: Results from models trained with labeled data.
    - `gcn_graph/`: Graph Convolutional Network (GCN) results
    - `random_forest_tabular/`: Random Forest (tabular) results
    - `roberta_text/`: RoBERTa (text) results
  - **`unsupervised/`**: Results from models trained without labels.
    - `graph_gae/`: Graph Autoencoder (GAE) results
    - `tabular_gmm/`: Gaussian Mixture Model (GMM) results
    - `text_sentence_bert/`: Sentence-BERT + Isolation Forest (text) results
- **`reports/`**: Written Markdown reports summarizing each model's findings.
  - **`supervised/`** and **`unsupervised/`**: Each contains a folder per model, with a single clear report file (e.g., `gcn_report.md`).
- **`scripts/`**: All code for training, evaluating, and organizing models/results.
  - `supervised/`, `unsupervised/`, `utils/`: Scripts grouped by model type or utility.
- **Other files**: This README, environment files, and helper scripts.

---

## 🤔 What are Supervised and Unsupervised Models?

- **Supervised models** learn from labeled data (they know which examples are normal or anomalous during training).
- **Unsupervised models** do not use labels; they try to find anomalies based on patterns in the data alone.

### Models Used:
| Model Name                | Modality | Supervision   | Report Location                                      | Results Location                                      |
|--------------------------|----------|---------------|------------------------------------------------------|-------------------------------------------------------|
| GCN Graph Classification | Graph    | Supervised    | `reports/supervised/gcn_graph/gcn_report.md`         | `results/supervised/gcn_graph/`                       |
| Random Forest Tabular     | Tabular  | Supervised    | `reports/supervised/random_forest_tabular/random_forest_report.md` | `results/supervised/random_forest_tabular/`           |
| RoBERTa Text Classification | Text   | Supervised    | `reports/supervised/roberta_text/roberta_text_report.md` | `results/supervised/roberta_text/`                    |
| Graph Autoencoder (GAE)   | Graph    | Unsupervised  | `reports/unsupervised/graph_gae/graph_gae_report.md` | `results/unsupervised/graph_gae/`                     |
| Gaussian Mixture Model (GMM) | Tabular | Unsupervised | `reports/unsupervised/tabular_gmm/tabular_gmm_report.md` | `results/unsupervised/tabular_gmm/`                   |
| Sentence-BERT + Isolation Forest | Text | Unsupervised | `reports/unsupervised/text_sentence_bert/text_sentence_bert_report.md` | `results/unsupervised/text_sentence_bert/`            |

- **Modality**: The type of data the model uses (Graph, Tabular, or Text).
- **Report Location**: Where to find the written summary and interpretation for each model.
- **Results Location**: Where to find all metrics, plots, and CSVs for each model.

---

## 🚀 How to Use This Project (Step-by-Step)

### 1. **Install Dependencies**
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### 2. **Download Datasets**
If not already present, use the provided scripts to download and extract datasets into the `data/` folder.

### 3. **Run Models**
Each model has its own training script. Example usage:
```bash
# Supervised Models
python scripts/supervised/gcn_graph/train_gcn.py
python scripts/supervised/random_forest_tabular/train_random_forest.py
python scripts/supervised/roberta_text/train_roberta_text.py

# Unsupervised Models
python scripts/unsupervised/graph_gae/train_gae_fixed.py
python scripts/unsupervised/tabular_gmm/train_gmm_unsupervised.py
python scripts/unsupervised/text_sentence_bert/train_text_unsupervised_simple.py
```

### 4. **Organize and Summarize Results**
To automatically organize all results and generate summary folders:
```bash
python scripts/organize_final_project_improved.py
```
To compile and visualize text model results:
```bash
python scripts/utils/compile_text_final_results.py
```

### 5. **Check Results and Reports**
- For each model, check the summary CSV and PNG files in the corresponding `results/` folder.
- For a written summary and interpretation, read the corresponding Markdown report in the `reports/` folder.

---

## 📚 What Do the Results and Reports Contain?
- **CSV files**: Summaries of model performance (accuracy, F1, ROC-AUC, etc.) for each dataset.
- **PNG images**: Visualizations like ROC curves, confusion matrices, and performance comparisons.
- **Markdown reports**: Human-readable summaries, interpretation, and comparison to prior work.

---

## 🧩 Need More Details?
- Each script is commented and organized by model type.
- Reports explain the strengths, weaknesses, and findings for each model.
- The project is ready for extension or reproduction of results.

---

**🎉 Project Status: Complete**
All models trained, evaluated, and results organized successfully!
