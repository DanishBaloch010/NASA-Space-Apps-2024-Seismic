import os
import pandas as pd

# Folder containing the seismic result CSV files
data_folder = 'DataCSV'

# Load the catalog data
catalog_df = pd.read_csv('catalog.csv')

# Iterate through each unique event ID in the catalog
for _, row in catalog_df.iterrows():
    # Extract relevant fields from the catalog
    evid = row['evid']
    filename = row['filename']
    time_rel = row['time_rel(sec)']

    # Extract the 5-digit identifier for the CSV file
    event_id = filename.split("_evid")[1][:5]

    # Construct the corresponding CSV file name
    csv_filename = f"{event_id}_seismic_results.csv"
    csv_filepath = os.path.join(data_folder, csv_filename)

    # If the file exists, proceed with processing
    if os.path.exists(csv_filepath):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_filepath)

        # Ensure the new 'is_seismic' column is initialized to 0
        df['is_seismic'] = 0

        # Locate the range where `time_rel` falls in the CSV
        # We want to find the index where Relative Time is just greater or equal
        match_idx = df[df['Relative Time (s)'] >= time_rel].index.min()

        # If a valid index is found, mark it as seismic
        if not pd.isna(match_idx):
            df.at[match_idx, 'is_seismic'] = 1

        # Save the updated CSV back to the file
        df.to_csv(csv_filepath, index=False)

print("Catalog events have been matched and the CSV files updated successfully.")
