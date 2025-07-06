import zipfile
import os

# Extract montage dataset
print("Extracting montage.zip...")
with zipfile.ZipFile('data/montage.zip', 'r') as zip_ref:
    zip_ref.extractall('data/montage/raw')

# Extract other datasets
print("Extracting 1000genome.zip...")
with zipfile.ZipFile('data/1000genome.zip', 'r') as zip_ref:
    zip_ref.extractall('data/1000genome/raw')

print("Extracting predict_future_sales.zip...")
with zipfile.ZipFile('data/predict_future_sales.zip', 'r') as zip_ref:
    zip_ref.extractall('data/predict_future_sales/raw')

print("Data extraction completed!") 