import os
import shutil
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import glob

def discover_all_models():
    """Discover all trained models by scanning result directories."""
    
    print("🔍 Discovering all trained models...")
    
    # Scan for result directories
    result_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('results_')]
    
    models_found = {
        'supervised': [],
        'unsupervised': []
    }
    
    for result_dir in result_dirs:
        dir_name = result_dir.name
        print(f"  📁 Found: {dir_name}")
        
        # Classify models based on directory names and content
        if 'roberta' in dir_name or 'random_forest' in dir_name or 'gcn' in dir_name:
            if 'roberta' in dir_name:
                models_found['supervised'].append({
                    'name': 'roberta_text',
                    'type': 'text',
                    'dir': dir_name,
                    'description': 'RoBERTa Text Classification'
                })
            elif 'random_forest' in dir_name:
                models_found['supervised'].append({
                    'name': 'random_forest_tabular',
                    'type': 'tabular', 
                    'dir': dir_name,
                    'description': 'Random Forest Tabular'
                })
            elif 'gcn' in dir_name:
                models_found['supervised'].append({
                    'name': 'gcn_graph',
                    'type': 'graph',
                    'dir': dir_name,
                    'description': 'GCN Graph Classification'
                })
        else:
            # Unsupervised models
            if 'text_unsupervised' in dir_name:
                models_found['unsupervised'].append({
                    'name': 'text_sentence_bert',
                    'type': 'text',
                    'dir': dir_name,
                    'description': 'Sentence-BERT + Isolation Forest'
                })
            elif 'gmm' in dir_name:
                models_found['unsupervised'].append({
                    'name': 'tabular_gmm',
                    'type': 'tabular',
                    'dir': dir_name,
                    'description': 'Gaussian Mixture Model'
                })
            elif 'gae' in dir_name:
                # Choose the best GAE version
                if 'fixed' in dir_name:
                    models_found['unsupervised'].append({
                        'name': 'graph_gae',
                        'type': 'graph',
                        'dir': dir_name,
                        'description': 'Graph Autoencoder (Fixed)'
                    })
    
    # Remove duplicates and keep best versions
    for category in models_found:
        seen_names = set()
        unique_models = []
        for model in models_found[category]:
            if model['name'] not in seen_names:
                unique_models.append(model)
                seen_names.add(model['name'])
        models_found[category] = unique_models
    
    print(f"\n📊 Models discovered:")
    print(f"  🎯 Supervised: {len(models_found['supervised'])}")
    for model in models_found['supervised']:
        print(f"    - {model['description']} ({model['dir']})")
    print(f"  🎯 Unsupervised: {len(models_found['unsupervised'])}")
    for model in models_found['unsupervised']:
        print(f"    - {model['description']} ({model['dir']})")
    
    return models_found

def find_best_results_for_model(model_info):
    """Find the best/final result files for a specific model."""
    
    model_dir = Path(model_info['dir'])
    if not model_dir.exists():
        return []
    
    print(f"  🔍 Searching in {model_dir}...")
    
    # Priority order for file types
    file_patterns = [
        # Highest priority: FINAL files
        "*FINAL*results*.csv",
        "*FINAL*summary*.csv", 
        "*FINAL*.png",
        
        # High priority: compiled/summary files
        "*results_summary*.csv",
        "*summary*.csv",
        "*confusion_matrices*.png",
        "*roc_curves*.png", 
        "*performance_summary*.png",
        
        # Medium priority: specific visualizations
        "*confusion_matrix*.png",
        "*roc_curve*.png",
        "*performance*.png"
    ]
    
    found_files = []
    
    for pattern in file_patterns:
        matching_files = list(model_dir.glob(pattern))
        for file_path in matching_files:
            if file_path not in found_files:
                found_files.append(file_path)
                print(f"    ✅ Found: {file_path.name}")
    
    # If no summary files found, look in subdirectories for summary
    if not any('summary' in f.name for f in found_files):
        summary_dirs = [d for d in model_dir.iterdir() if d.is_dir() and 'summary' in d.name]
        for summary_dir in summary_dirs:
            for pattern in file_patterns:
                matching_files = list(summary_dir.glob(pattern))
                for file_path in matching_files:
                    if file_path not in found_files:
                        found_files.append(file_path)
                        print(f"    ✅ Found in summary: {file_path.name}")
    
    return found_files

def create_organized_structure():
    """Create the organized directory structure."""
    
    print("\n🏗️ Creating organized directory structure...")
    
    # Discover all models first
    models = discover_all_models()
    
    # Create directories based on discovered models
    directories = ["results", "reports", "scripts"]
    
    for base_dir in directories:
        Path(base_dir).mkdir(exist_ok=True)
        
        for category in ['supervised', 'unsupervised']:
            category_dir = Path(base_dir) / category
            category_dir.mkdir(exist_ok=True)
            
            # Create subdirectories for each model
            for model in models[category]:
                model_dir = category_dir / model['name']
                model_dir.mkdir(exist_ok=True)
                print(f"  ✅ Created: {model_dir}")
    
    return models

def organize_results_intelligently(models):
    """Intelligently organize results by finding the best files for each model."""
    
    print("\n📊 Organizing results intelligently...")
    
    for category in ['supervised', 'unsupervised']:
        print(f"\n🎯 Processing {category} models:")
        
        for model in models[category]:
            print(f"\n  📂 Organizing {model['description']}...")
            
            # Find best result files
            best_files = find_best_results_for_model(model)
            
            if not best_files:
                print(f"    ⚠️  No result files found for {model['name']}")
                continue
            
            # Create destination directory
            dest_dir = Path("results") / category / model['name']
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            copied_count = 0
            for source_file in best_files:
                dest_file = dest_dir / source_file.name
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"    ✅ Copied: {source_file.name}")
                    copied_count += 1
                except Exception as e:
                    print(f"    ❌ Error copying {source_file.name}: {e}")
            
            print(f"    📊 Total files copied: {copied_count}")

def organize_scripts_intelligently(models):
    """Organize scripts by model type."""
    
    print("\n📝 Organizing scripts intelligently...")
    
    # Script patterns for each model type
    script_mappings = {
        'supervised': {
            'roberta_text': ['*roberta*'],
            'random_forest_tabular': ['*random_forest*', '*rf_*'],
            'gcn_graph': ['*gcn*']
        },
        'unsupervised': {
            'text_sentence_bert': ['*text_unsupervised*', '*sentence*', '*improve_text*'],
            'tabular_gmm': ['*gmm*'],
            'graph_gae': ['*gae*']
        },
        'utils': ['*organize*', '*compile*', '*consolidate*', '*fix*', '*create*']
    }
    
    scripts_dir = Path("scripts")
    
    # Create utils directory
    utils_dir = scripts_dir / "utils"
    utils_dir.mkdir(exist_ok=True)
    
    # Move utility scripts first
    for pattern in script_mappings['utils']:
        matching_files = list(scripts_dir.glob(pattern))
        for script_file in matching_files:
            if script_file.is_file() and script_file.name != "organize_final_project_improved.py":
                dest_file = utils_dir / script_file.name
                try:
                    shutil.move(str(script_file), str(dest_file))
                    print(f"  ✅ Moved to utils: {script_file.name}")
                except Exception as e:
                    print(f"  ❌ Error moving {script_file.name}: {e}")
    
    # Move model-specific scripts
    for category in ['supervised', 'unsupervised']:
        print(f"\n🎯 Moving {category} scripts:")
        
        for model in models[category]:
            model_name = model['name']
            if model_name in script_mappings[category]:
                patterns = script_mappings[category][model_name]
                dest_dir = scripts_dir / category / model_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                for pattern in patterns:
                    matching_files = list(scripts_dir.glob(pattern))
                    for script_file in matching_files:
                        if script_file.is_file():
                            dest_file = dest_dir / script_file.name
                            try:
                                shutil.move(str(script_file), str(dest_file))
                                print(f"    ✅ Moved: {script_file.name}")
                            except Exception as e:
                                print(f"    ❌ Error moving {script_file.name}: {e}")

def create_model_reports(models):
    """Create detailed reports for each model category."""
    
    print("\n📄 Creating model reports...")
    
    for category in ['supervised', 'unsupervised']:
        report_dir = Path("reports") / category
        report_dir.mkdir(exist_ok=True)
        
        # Create main category README
        readme_content = f"# {category.title()} Learning Models\n\n"
        readme_content += f"This directory contains results and analysis for {category} anomaly detection models.\n\n"
        readme_content += "## Models Included:\n\n"
        
        for model in models[category]:
            readme_content += f"### {model['description']}\n"
            readme_content += f"- **Type**: {model['type'].title()} data\n"
            readme_content += f"- **Directory**: `../results/{category}/{model['name']}/`\n"
            readme_content += f"- **Source**: {model['dir']}\n\n"
        
        readme_content += "## Performance Summary\n\n"
        readme_content += "Please refer to the individual result directories for detailed performance metrics, "
        readme_content += "ROC curves, confusion matrices, and summary statistics.\n"
        
        with open(report_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"  ✅ Created: reports/{category}/README.md")

def create_comprehensive_readme(models):
    """Create a comprehensive project README."""
    
    print("\n📖 Creating comprehensive README...")
    
    # Count total models
    total_supervised = len(models['supervised'])
    total_unsupervised = len(models['unsupervised'])
    total_models = total_supervised + total_unsupervised
    
    readme_content = f'''# Machine Learning Paper - Anomaly Detection Across Workflow Datasets

## 🎯 Project Overview

This project implements and evaluates **{total_models} anomaly detection models** across 12 workflow datasets using multiple modalities (text, tabular, graph) and approaches ({total_supervised} supervised, {total_unsupervised} unsupervised).

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

### Supervised Learning ({total_supervised} models)
'''
    
    for model in models['supervised']:
        readme_content += f"- **{model['description']}**: {model['type'].title()}-based anomaly detection\n"
    
    readme_content += f"\n### Unsupervised Learning ({total_unsupervised} models)\n"
    
    for model in models['unsupervised']:
        readme_content += f"- **{model['description']}**: {model['type'].title()}-based anomaly detection\n"
    
    readme_content += '''

## 📁 Project Structure

```
├── results/
│   ├── supervised/
'''
    
    for model in models['supervised']:
        readme_content += f"│   │   └── {model['name']}/          # {model['description']}\n"
    
    readme_content += "│   └── unsupervised/\n"
    
    for model in models['unsupervised']:
        readme_content += f"│       └── {model['name']}/       # {model['description']}\n"
    
    readme_content += '''├── reports/
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
'''
    
    for model in models['supervised']:
        readme_content += f"python scripts/supervised/{model['name']}/train_*.py\n"
    
    readme_content += "\n# Unsupervised Models\n"
    
    for model in models['unsupervised']:
        readme_content += f"python scripts/unsupervised/{model['name']}/train_*.py\n"
    
    readme_content += '''```

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
'''

    with open("README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("  ✅ Created comprehensive README.md")

def main():
    """Main function to intelligently organize the entire project."""
    
    print("🚀 Starting intelligent project organization...")
    print("=" * 60)
    
    # Discover and organize
    models = create_organized_structure()
    organize_results_intelligently(models)
    organize_scripts_intelligently(models)
    create_model_reports(models)
    create_comprehensive_readme(models)
    
    print("\n" + "=" * 60)
    print("🎉 Intelligent project organization complete!")
    
    # Summary
    total_supervised = len(models['supervised'])
    total_unsupervised = len(models['unsupervised'])
    
    print(f"\n📊 Final Summary:")
    print(f"  🎯 Supervised Models: {total_supervised}")
    for model in models['supervised']:
        print(f"    - {model['description']}")
    print(f"  🎯 Unsupervised Models: {total_unsupervised}")
    for model in models['unsupervised']:
        print(f"    - {model['description']}")
    
    print(f"\n✅ Total models organized: {total_supervised + total_unsupervised}")
    print("🎯 Ready for paper submission!")

if __name__ == "__main__":
    main() 