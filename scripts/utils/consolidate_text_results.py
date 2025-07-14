import json
import os
import pandas as pd
from tqdm import tqdm

def find_best_text_results():
    """Find the best text unsupervised results for each dataset."""
    
    datasets = ['1000genome', 'casa_nowcast', 'casa_wind_speed', 'eht_difmap', 
                'eht_imaging', 'eht_smili', 'montage', 'predict_future_sales', 
                'pycbc_inference', 'pycbc_search', 'somospie', 'variant_calling']
    
    results_dir = 'results_text_unsupervised'
    best_results = []
    
    print("🔍 Finding best text unsupervised results for each dataset...")
    
    for dataset in tqdm(datasets, desc="Processing datasets"):
        best_roc_auc = 0
        best_result = None
        best_method = None
        
        # Check main text unsupervised results
        main_dir = os.path.join(results_dir, f'text_unsupervised_{dataset}')
        if os.path.exists(os.path.join(main_dir, 'results.json')):
            with open(os.path.join(main_dir, 'results.json'), 'r') as f:
                result = json.load(f)
                if result['roc_auc'] > best_roc_auc:
                    best_roc_auc = result['roc_auc']
                    best_result = result
                    best_method = 'sentence_bert_isolation_forest'
        
        # Check alternative methods
        for method in ['isolationforest', 'lof', 'knn', 'gmm']:
            alt_dir = os.path.join(results_dir, f'text_{dataset}_{method}')
            if os.path.exists(os.path.join(alt_dir, 'results.json')):
                with open(os.path.join(alt_dir, 'results.json'), 'r') as f:
                    result = json.load(f)
                    if result['roc_auc'] > best_roc_auc:
                        best_roc_auc = result['roc_auc']
                        best_result = result
                        best_method = f'sentence_bert_{method}'
        
        if best_result:
            best_results.append({
                'dataset': dataset,
                'method': best_method,
                'roc_auc': best_result['roc_auc'],
                'f1_score': best_result['f1_score'],
                'accuracy': best_result['accuracy'],
                'precision': best_result['precision'],
                'recall': best_result['recall'],
                'roc_curve': str(best_result['roc_curve']),
                'best_threshold': best_result.get('best_threshold', 0.0)
            })
            print(f"✅ {dataset}: ROC-AUC = {best_result['roc_auc']:.4f} ({best_method})")
        else:
            print(f"❌ {dataset}: No results found")
    
    # Save updated summary
    df = pd.DataFrame(best_results)
    summary_path = os.path.join(results_dir, 'text_unsupervised_best_results_summary.csv')
    df.to_csv(summary_path, index=False)
    
    print(f"\n📊 Summary saved to: {summary_path}")
    print(f"🎯 Average ROC-AUC: {df['roc_auc'].mean():.4f}")
    print(f"🏆 Best performer: {df.loc[df['roc_auc'].idxmax(), 'dataset']} (ROC-AUC: {df['roc_auc'].max():.4f})")
    
    return df

if __name__ == "__main__":
    results_df = find_best_text_results()
    print("\n🎉 Text unsupervised results consolidation complete!") 