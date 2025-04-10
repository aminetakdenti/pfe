import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import gc  # Garbage collector for memory management


# Process one zip file at a time and in chunks
def process_large_dataset(zip_path, chunk_size=100000):
    print(f"Processing {zip_path}...")

    # Create a directory to store processed chunks
    os.makedirs("processed_chunks", exist_ok=True)
    chunk_files = []
    chunk_counter = 0

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".csv"):
                print(f"  Processing {file}...")
                # Extract to temporary location
                zip_ref.extract(file, path="temp_extract")

                # Process file in chunks
                for chunk in pd.read_csv(f"temp_extract/{file}", chunksize=chunk_size):
                    # Basic cleaning for each chunk
                    chunk = chunk.dropna()

                    # Save processed chunk
                    chunk_filename = f"processed_chunks/chunk_{chunk_counter}.csv"
                    chunk.to_csv(chunk_filename, index=False)
                    chunk_files.append(chunk_filename)
                    chunk_counter += 1

                    # Free memory
                    del chunk
                    gc.collect()

                # Remove temporary file
                os.remove(f"temp_extract/{file}")

    return chunk_files


# Create temporary directories
os.makedirs("temp_extract", exist_ok=True)

# Process each zip file separately
chunk_files_1 = process_large_dataset("../data/CSV-01-12.zip")
chunk_files_2 = process_large_dataset("../data/CSV-03-11.zip")

# Combine all chunk file paths
all_chunk_files = chunk_files_1 + chunk_files_2

# Identify the structure from the first chunk
if all_chunk_files:
    sample_df = pd.read_csv(all_chunk_files[0], nrows=1000)

    # Find the label column
    label_column = None
    possible_label_columns = [
        "Label",
        "label",
        "class",
        "Class",
        "attack_type",
        "Attack_type",
    ]
    for col in possible_label_columns:
        if col in sample_df.columns:
            label_column = col
            break

    if label_column is None:
        print("Warning: Could not identify the label column")
        # You might want to examine the columns and set it manually
        print(f"Available columns: {sample_df.columns.tolist()}")
    else:
        print(f"Label column identified: {label_column}")

    # Now we'll do feature selection on a sample to identify important features
    print("Performing feature selection on a sample...")
    sample_size = min(
        500000, sum(pd.read_csv(file).shape[0] for file in all_chunk_files[:3])
    )

    # Load a sample for feature selection
    sample_dfs = []
    remaining_rows = sample_size
    for file in all_chunk_files:
        if remaining_rows <= 0:
            break
        df = pd.read_csv(file)
        rows_to_take = min(remaining_rows, df.shape[0])
        sample_dfs.append(df.iloc[:rows_to_take])
        remaining_rows -= rows_to_take

    sample_df = pd.concat(sample_dfs, ignore_index=True)
    print(f"Sample shape for feature selection: {sample_df.shape}")

    # Fix data types in the sample
    categorical_columns = sample_df.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        if column != label_column:
            le = LabelEncoder()
            sample_df[column] = le.fit_transform(sample_df[column].astype(str))

    # Encode the label column in the sample
    if label_column:
        le_label = LabelEncoder()
        sample_df[label_column] = le_label.fit_transform(
            sample_df[label_column].astype(str)
        )

        # Feature selection on sample
        X_sample = sample_df.drop(label_column, axis=1)
        y_sample = sample_df[label_column]

        # Remove any non-numeric columns
        X_sample = X_sample.select_dtypes(include=["number"])

        # Select top k features
        k = min(20, X_sample.shape[1])
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_sample, y_sample)

        # Get selected feature names
        selected_features = X_sample.columns[selector.get_support()].tolist()
        selected_features.append(label_column)
        print(f"Selected features: {selected_features}")

        # Now process each chunk keeping only the selected features
        print("Processing all chunks with selected features...")
        processed_files = []

        for i, chunk_file in enumerate(all_chunk_files):
            print(f"Processing chunk {i+1}/{len(all_chunk_files)}")
            df = pd.read_csv(chunk_file)

            # Handle categorical columns
            for column in df.select_dtypes(include=["object"]).columns:
                if column != label_column and column in selected_features:
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column].astype(str))

            # Encode label column
            if label_column in df.columns:
                le = LabelEncoder()
                df[label_column] = le.fit_transform(df[label_column].astype(str))

            # Keep only selected features
            df = df[[col for col in selected_features if col in df.columns]]

            # Save processed chunk
            processed_file = f"processed_chunks/processed_{i}.csv"
            df.to_csv(processed_file, index=False)
            processed_files.append(processed_file)

            # Free memory
            del df
            gc.collect()

        # Now we can either:
        # 1. Combine all processed files into one (if it fits in memory)
        # 2. Keep them separate and process them in batches during training

        print("Preprocessing complete. Data is ready for training.")
else:
    print("No chunks were processed.")
