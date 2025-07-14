# GCN Benchmark Report (FlowBench Datasets)

## Experiment Summary
- **Model:** Graph Convolutional Network (GCN)
- **Datasets:** All 12 FlowBench datasets (graph structure + tabular features)
- **Metrics:** Accuracy, F1-score, ROC-AUC
- **Plots:** ROC curves, performance comparison, training loss, confusion matrices
- **Date:** [Fill in date]

## Results Table
| Dataset              | Accuracy | F1-score | ROC-AUC |
|----------------------|----------|----------|---------|
| 1000genome           | 0.673    | 0.000    | 0.500   |
| casa_nowcast         | 0.782    | 0.046    | 0.621   |
| casa_wind_speed      | 0.790    | 0.038    | 0.660   |
| eht_difmap           | 0.790    | 0.015    | 0.678   |
| eht_imaging          | 0.811    | 0.000    | 0.341   |
| eht_smili            | 0.803    | 0.040    | 0.751   |
| montage              | 0.792    | 0.021    | 0.678   |
| predict_future_sales | 0.815    | 0.001    | 0.751   |
| pycbc_inference      | 0.834    | 0.311    | 0.852   |
| pycbc_search         | 0.814    | 0.001    | 0.672   |
| somospie             | 0.813    | 0.009    | 0.743   |
| variant_calling      | 0.785    | 0.002    | 0.660   |

## Summary Plots
- **ROC Curves:** ![ROC Curves](../results_gcn/all_roc_curves.png)
- **Performance Comparison:** ![Performance Comparison](../results_gcn/performance_comparison.png)
- **Training Loss Comparison:** ![Training Loss](../results_gcn/training_loss_comparison.png)
- **Confusion Matrices:** ![Confusion Matrices](../results_gcn/all_confusion_matrices.png)

## Comparison to FlowBench Paper
- GCN performance is generally lower than Random Forest and the results reported in the FlowBench paper, especially in terms of F1-score and ROC-AUC for highly imbalanced datasets.
- The GCN model struggles to identify anomalies, often predicting the majority class, as reflected in the low F1-scores.
- Some datasets (e.g., `pycbc_inference`) show better GCN performance, but overall, the model may require further tuning or advanced techniques (e.g., class balancing, deeper architectures).

## Interpretation
- **Strengths:**
  - GCN leverages graph structure, which may be beneficial for certain workflow types.
  - Achieves reasonable accuracy on most datasets.
- **Weaknesses:**
  - Very low F1-scores and ROC-AUC on many datasets, indicating poor anomaly detection.
  - Model is likely biased toward the majority class due to class imbalance.
  - Early stopping and shallow architecture may limit performance.
- **Next Steps:**
  - Experiment with deeper GNNs, different architectures, or advanced imbalance handling.
  - Compare to Random Forest and BERT/LLM-based models for a comprehensive benchmark.

---
*Generated automatically. Please update the date and add any additional interpretation as needed.* 