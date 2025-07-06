# FlowBench Anomaly Detection Research

This repository contains research on anomaly detection in computational workflows using the FlowBench dataset. The project implements and evaluates machine learning models for detecting anomalies in scientific workflow executions.

## Project Overview

This research focuses on:
- **Dataset**: FlowBench "montage" workflow dataset
- **Task**: Binary anomaly detection (normal vs. anomalous workflow executions)
- **Models**: Random Forest Classifier (baseline)
- **Features**: 7 delay-related features (v1 feature set)
- **Goal**: Benchmark and improve ML models for workflow anomaly detection

## Results

### Random Forest Performance (Baseline)
- **Accuracy**: 80.23%
- **F1-Score**: 40.14%
- **ROC-AUC**: 76.94%

### Dataset Statistics
- **Total samples**: 172,480
- **Anomaly ratio**: 20.44% (35,251 anomalies)
- **Features**: 7 (delay features + node_hop)
- **Train/Test split**: 70%/30%

### Feature Importance (Top 3)
1. **queue_delay**: 29.77%
2. **runtime**: 28.55%
3. **stage_in_delay**: 27.47%

## Setup Instructions

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for future GNN experiments)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd MLpaper
   ```

2. **Create virtual environment**
   ```bash
   python -m venv flowbench-env
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   .\flowbench-env\Scripts\activate
   
   # Linux/Mac
   source flowbench-env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install torch torch-geometric torchvision torchaudio
   pip install scikit-learn matplotlib seaborn pandas numpy
   pip install flowbench
   ```

5. **Download datasets**
   - Place the following zip files in the `data/` directory:
     - `montage.zip`
     - `1000genome.zip`
     - `predict_future_sales.zip`

## Usage

### Running the Baseline Model
```bash
python train_tabular.py
```

This will:
- Load the montage dataset
- Train a Random Forest classifier
- Evaluate performance metrics
- Generate visualizations:
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `feature_importance.png`

### Data Extraction
```bash
python extract_data.py
```

## Project Structure

```
MLpaper/
├── data/                          # Dataset directory
│   ├── montage.zip               # Montage workflow data
│   ├── 1000genome.zip           # 1000genome workflow data
│   ├── predict_future_sales.zip # Sales prediction workflow data
│   └── adjacency_list_dags/     # Workflow graph definitions
├── flowbench-env/                # Virtual environment
├── train_tabular.py             # Main training script
├── extract_data.py              # Data extraction utility
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## Technical Details

### FlowBench Patches
The repository includes patches to the FlowBench library to resolve compatibility issues:
- **Dataset loading**: Modified to use local data directory
- **PyTorch compatibility**: Fixed for PyTorch 2.6+ `weights_only` parameter
- **Adjacency files**: Updated path resolution for workflow graphs

### Feature Set (v1)
The model uses the following 7 features:
- `wms_delay`: Workflow Management System delay
- `queue_delay`: Queue waiting time
- `runtime`: Actual execution time
- `post_script_delay`: Post-script execution delay
- `stage_in_delay`: Input data staging delay
- `stage_out_delay`: Output data staging delay
- `node_hop`: Graph distance from root node

## Future Work

This baseline establishes a foundation for:
1. **Advanced Models**: GNNs, Transformers, LLMs
2. **Feature Engineering**: Exploring v2/v3 feature sets
3. **Multi-dataset Evaluation**: Testing on other FlowBench workflows
4. **Real-time Detection**: Deployment considerations

## Comparison with FlowBench Paper

*[To be added: Comparison with baseline results from the original FlowBench paper]*

## License

This project is for research purposes. The FlowBench dataset is licensed under Creative Commons Attribution 4.0 International License.

## Citation

If you use this code in your research, please cite:
```
@article{flowbench2024,
  title={cFlow-Bench: A Dataset and Benchmarks for Computational Workflow Anomaly Detection},
  author={PoSeiDon Team},
  journal={arXiv preprint},
  year={2024}
}
```

## Contact

For questions or contributions, please open an issue on GitHub. 