import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import gc  # Garbage collector

def process_large_dataset(zip_path, chunk_size=100000):
    print(f"Processing {zip_path}...")

    # Check if file exists
    if not os.path.exists(zip_path):
        print(f"Error: File {zip_path} does not exist")
        return []

    # Create directories to store processed chunks
    os.makedirs('processed_chunks', exist_ok=True)
    os.makedirs('temp_extract', exist_ok=True)

    chunk_files = []
    chunk_counter = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('.csv'):
                    print(f"  Processing {file}...")

                    # Extract to temporary location
                    zip_ref.extract(file, path='temp_extract')
                    extract_path = os.path.join('temp_extract', file)

                    # Process file in chunks
                    chunk_index = 0
                    for chunk in pd.read_csv(extract_path, chunksize=chunk_size):
                        # Basic cleaning for each chunk
                        chunk = chunk.dropna()

                        # Save processed chunk
                        chunk_filename = f'processed_chunks/chunk_{chunk_counter}.csv'
                        chunk.to_csv(chunk_filename, index=False)
                        print(f"    Saved chunk to {chunk_filename}")
                        chunk_files.append(chunk_filename)
                        chunk_counter += 1
                        chunk_index += 1

                        # Free memory
                        del chunk
                        gc.collect()

                    print(f"    Processed {chunk_index} chunks from {file}")

                    # Remove temporary file
                    try:
                        os.remove(extract_path)
                        print(f"    Removed temporary extract: {extract_path}")
                    except Exception as e:
                        print(f"    Warning: Could not remove temp file: {e}")
    except Exception as e:
        print(f"Error processing zip file {zip_path}: {e}")

    print(f"Processed {len(chunk_files)} chunks from {zip_path}")
    print(f"Chunk files: {chunk_files[:5]}{'...' if len(chunk_files) > 5 else ''}")

    return chunk_files

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Construct paths to the data files
data_file_1 = os.path.join(script_dir, "data", "CSV-01-12.zip")
data_file_2 = os.path.join(script_dir, "data", "CSV-03-11.zip")

print(f"Looking for data files at:")
print(f"  - {data_file_1}")
print(f"  - {data_file_2}")

# Process each zip file separately
chunk_files_1 = process_large_dataset(data_file_1)
chunk_files_2 = process_large_dataset(data_file_2)

# Combine all chunk file paths
all_chunk_files = chunk_files_1 + chunk_files_2
print(f"Total chunks: {len(all_chunk_files)}")

# Check if we have any chunks
if not all_chunk_files:
    print("No chunk files were created. Cannot proceed.")
    exit(1)

# Print the first few chunk files for debugging
print("First few chunk files:")
for i, file in enumerate(all_chunk_files[:5]):
    print(f"  {i}: {file}")
    if os.path.exists(file):
        print(f"    File exists with size: {os.path.getsize(file)} bytes")
    else:
        print(f"    File doesn't exist!")

# Identify the structure from the first chunk
print(f"Attempting to read first chunk file: {all_chunk_files[0]}")
try:
    sample_df = pd.read_csv(all_chunk_files[0], nrows=1000)
    print(f"Successfully read sample with shape: {sample_df.shape}")
    
    # Continue with your processing logic
    # ...
    
except Exception as e:
    print(f"Error reading first chunk file: {e}")
    print("Cannot proceed with feature selection without reading a sample.")
