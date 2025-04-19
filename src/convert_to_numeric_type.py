import pandas as pd
import os
import glob
import numpy as np  # Import numpy for handling potential NaN values after conversion
import gc

# Define columns that should NOT be converted to numeric
# 'Label' is typically a target variable and should remain non-numeric (e.g., string or category)
# Add any other columns here that you know should not be numeric
COLUMNS_TO_EXCLUDE_FROM_NUMERIC_CONVERSION = ["Label"]


def preprocess_chunks(folder_path, output_folder):
    """
    Reads CSV files from a folder, converts applicable columns to numeric types,
    and saves the processed data to an output folder. Skips files already processed.

    Args:
        folder_path (str): The path to the folder containing the input CSV files.
        output_folder (str): The path to the folder where numeric CSV files will be saved.
    """
    print(f"Starting numeric conversion for chunks in: {folder_path}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Ensured output folder exists: {output_folder}")

    # Get list of input files
    chunk_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not chunk_files:
        print(f"No CSV files found in input folder: {folder_path}")
        return

    # Get list of already processed files in the output folder
    loaded_chunks = glob.glob(os.path.join(output_folder, "*.csv"))
    # Extract just the base filenames for easier comparison
    loaded_chunk_basenames = [os.path.basename(f) for f in loaded_chunks]

    print(f"Found {len(chunk_files)} input chunk files.")
    print(f"Found {len(loaded_chunks)} previously processed files in output folder.")

    processed_count = 0

    for file_path in chunk_files:
        file_basename = os.path.basename(file_path)
        output_file_path = os.path.join(output_folder, file_basename)

        # Check if the corresponding output file already exists
        if file_basename in loaded_chunk_basenames:
            print(f"Skipping '{file_basename}': Already processed.")
            continue

        print(f"Processing file: {file_basename}")

        try:
            # Read the CSV file
            # Use low_memory=False for potentially large files with mixed types
            df = pd.read_csv(file_path, low_memory=False)
            # print(f"  Successfully read '{file_basename}' with shape: {df.shape}")

            # --- Numeric Type Conversion ---
            print(f"  Attempting numeric conversion for columns...")
            for col in df.columns:
                # Skip columns that are in our exclusion list
                if col in COLUMNS_TO_EXCLUDE_FROM_NUMERIC_CONVERSION:
                    # print(f"    Skipping conversion for excluded column: '{col}'")
                    continue

                # Attempt to convert the column to numeric
                # errors='coerce' will turn any values that cannot be converted into NaN
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    # print(f"    Converted column '{col}' to numeric.")
                except Exception as e:
                    # This catch is mostly for unexpected issues, pd.to_numeric with coerce is robust
                    print(
                        f"    Warning: Could not convert column '{col}' to numeric: {e}"
                    )

            # Optional: After coercing non-numeric to NaN, you might want to handle NaNs again
            # depending on your modeling needs (e.g., impute or drop rows/columns)
            # Example: Drop rows that now have NaNs after conversion (be cautious!)
            # initial_rows_after_conversion = len(df)
            # df.dropna(inplace=True)
            # if len(df) < initial_rows_after_conversion:
            #     print(
            #         f"  Dropped {initial_rows_after_conversion - len(df)} rows with new NaN values after conversion."
            #     )

            # --- End Numeric Type Conversion ---

            # Save the DataFrame with numeric types to the output folder
            df.to_csv(output_file_path, index=False)
            print(f"  Successfully saved numeric data to: {output_file_path}")

            processed_count += 1
            # Free memory for the DataFrame
            del df
            gc.collect()

        except pd.errors.EmptyDataError:
            print(f"Skipping '{file_basename}': File is empty.")
        except FileNotFoundError:
            print(
                f"Error: Input file '{file_path}' not found (should not happen here)."
            )
        except Exception as e:
            print(f"An error occurred while processing '{file_basename}': {e}")

    print(f"\nFinished numeric conversion. Processed {processed_count} new files.")


if __name__ == "__main__":
    # Define input and output folders relative to the current working directory
    # Assuming 'cleaned_chunks' is the input folder from the previous step
    # and 'numeric_data' is the desired output folder for numeric data
    cleaned_chunks = os.path.join(
        os.getcwd(), "cleaned_chunks"
    )  # This should be the output of the previous step
    numeric_data_output = os.path.join(os.getcwd(), "numeric_data")

    # Note: Based on your previous script, the output of process_large_dataset
    # was saved into 'processed_chunks'. So 'processed_chunks' is the input
    # for this numeric conversion step.

    preprocess_chunks(folder_path=cleaned_chunks, output_folder=numeric_data_output)

    print(f"\nNumeric conversion process complete.")
    print(f"Cleaned and numeric data chunks are saved in: {numeric_data_output}")

    # Example of how to list the files in the output folder
    # print("\nFiles in the output folder:")
    # for f in glob.glob(os.path.join(numeric_data_output, "*.csv")):
    #     print(f"- {os.path.basename(f)}")
