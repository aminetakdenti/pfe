import pandas as pd
import os
import glob

features_to_remove = [
    "Unnamed: 0",
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
    "Fwd Header Length.1", # Duplicate feature
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "SimillarHTTP",
    "Inbound"
]

def preprocess_chunks(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    chunk_files = glob.glob(os.path.join(folder_path, "*.csv"))
    loaded_chunks = glob.glob(os.path.join(output_folder, "*.csv"))


    for file in chunk_files:
        output_file_path = os.path.join(output_folder, os.path.basename(file))


        if output_file_path in loaded_chunks:
            print("already loaded")
            continue

        df = pd.read_csv(file)

        existing_features_to_remove = [feature for feature in features_to_remove if feature in df.columns]

        print("existing_features_to_remove", existing_features_to_remove)

        if existing_features_to_remove:
            # Remove the identified features
            df_cleaned = df.drop(columns=existing_features_to_remove)
            print(f"\nRemoved the following features: {existing_features_to_remove}")
            print(f"\nOriginal shape: {df.shape}")
            print(f"Cleaned shape: {df_cleaned.shape}")
            print("\nRemaining columns:")
            print(df_cleaned.columns.tolist())

            df_cleaned.to_csv(output_file_path, index=False)
        else:
            print("\nNone of the default features to remove were found in the DataFrame.")
            df_cleaned = df.copy() # Create a copy even if no columns were removed

            df_cleaned.to_csv(output_file_path, index=False)



if __name__ == "__main__":
    processed_chunks = os.path.join(os.getcwd(), "processed_chunks")
    cleaned_chunks = os.path.join(os.getcwd(), "cleaned_chunks")

    preprocess_chunks(
        folder_path=processed_chunks,
        output_folder=cleaned_chunks
    )
