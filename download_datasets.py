import os
import requests
import zipfile
from tqdm import tqdm

# FlowBench datasets and their URLs (from the paper and repository)
DATASET_URLS = {
    "casa_nowcast": "https://zenodo.org/record/1234567/files/casa_nowcast.zip",  # Placeholder
    "casa_wind_speed": "https://zenodo.org/record/1234567/files/casa_wind_speed.zip",  # Placeholder
    "eht_difmap": "https://zenodo.org/record/1234567/files/eht_difmap.zip",  # Placeholder
    "eht_imaging": "https://zenodo.org/record/1234567/files/eht_imaging.zip",  # Placeholder
    "eht_smili": "https://zenodo.org/record/1234567/files/eht_smili.zip",  # Placeholder
    "pycbc_inference": "https://zenodo.org/record/1234567/files/pycbc_inference.zip",  # Placeholder
    "pycbc_search": "https://zenodo.org/record/1234567/files/pycbc_search.zip",  # Placeholder
    "somospie": "https://zenodo.org/record/1234567/files/somospie.zip",  # Placeholder
    "variant_calling": "https://zenodo.org/record/1234567/files/variant_calling.zip",  # Placeholder
}

# Alternative: Try to find datasets from the FlowBench paper's supplementary materials
# Let me check if there are any public repositories or data sources

def download_file(url, filename):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def main():
    """Download missing FlowBench datasets."""
    data_dir = r"C:\Users\Samarth Kadam\MLpaper\data"
    
    # Check what datasets we already have
    existing_datasets = []
    for item in os.listdir(data_dir):
        if item.endswith('.zip'):
            existing_datasets.append(item.replace('.zip', ''))
    
    print(f"Existing datasets: {existing_datasets}")
    
    # Find missing datasets
    missing_datasets = []
    for dataset in DATASET_URLS.keys():
        if dataset not in existing_datasets:
            missing_datasets.append(dataset)
    
    print(f"Missing datasets: {missing_datasets}")
    
    if not missing_datasets:
        print("All datasets are already downloaded!")
        return
    
    print("\nNote: The URLs above are placeholders. The actual FlowBench datasets")
    print("are not publicly available through these URLs. You need to:")
    print("\n1. Contact the FlowBench authors for dataset access")
    print("2. Check the paper's supplementary materials")
    print("3. Look for the datasets in academic repositories")
    print("\nFor now, let's create a script to help you manually download them.")
    
    # Create a download script with instructions
    script_content = '''# FlowBench Dataset Download Instructions

# The following datasets need to be downloaded manually:

missing_datasets = [
    "casa_nowcast",
    "casa_wind_speed", 
    "eht_difmap",
    "eht_imaging",
    "eht_smili",
    "pycbc_inference",
    "pycbc_search",
    "somospie",
    "variant_calling"
]

# Steps to get the datasets:
# 1. Check the FlowBench paper: https://arxiv.org/abs/2306.09930
# 2. Look for supplementary materials or data availability statement
# 3. Contact the authors: poseidon-team (check paper for contact info)
# 4. Check academic repositories like Zenodo, Figshare, or institutional repositories
# 5. Look for the datasets in workflow management system repositories

# Once you have the datasets, place them as .zip files in the data/ directory
# with the exact names: casa_nowcast.zip, casa_wind_speed.zip, etc.

print("Please manually download the missing datasets and place them in the data/ directory")
'''
    
    with open('download_instructions.py', 'w') as f:
        f.write(script_content)
    
    print("\nCreated 'download_instructions.py' with detailed instructions.")
    print("\nFor now, let's run experiments on the datasets you have available.")

if __name__ == "__main__":
    main() 