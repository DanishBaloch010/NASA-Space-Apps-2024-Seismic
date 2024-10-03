import pandas as pd
import numpy as np
import os

# Set the path to the directory containing the seismic data files and the catalog file
data_directory = "DataCSV"  # Replace with the path to your seismic files
catalog_file = "catalog.csv"  # Replace with the catalog file path

# Step 1: Read the catalog file
catalog = pd.read_csv(catalog_file)

# Process the catalog to extract event IDs from filenames and map them to the corresponding CSV filenames
catalog['File_ID'] = catalog['filename'].str.extract(r'evid(\d+)', expand=False)
catalog['File_ID'] = catalog['File_ID'].astype(str).str.zfill(5)  # Zero-pad to match CSV file naming
catalog['CSV_Filename'] = catalog['File_ID'] + "_seismic_results.csv"

# Step 2: Loop through all seismic files in the specified directory
for file in os.listdir(data_directory):
    if file.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(data_directory, file)
        
        # Read the seismic data file into a DataFrame
        data = pd.read_csv(file_path)
        
        # Check if the file has corresponding entries in the catalog
        file_events = catalog[catalog['CSV_Filename'] == file]
        
        # Step 3: Add a 'Seismic Event' column based on the catalog entries
        # Initialize all values as 0 (no event)
        data['Seismic Event'] = 0
        
        # Mark rows where a seismic event occurs (matching relative time from the catalog)
        for _, row in file_events.iterrows():
            event_time = row['time_rel(sec)']  # Extract the relative time in seconds
            
            # Mark the closest relative time as an event (within a small tolerance)
            data.loc[np.abs(data['Relative Time (s)'] - event_time) < 1, 'Seismic Event'] = 1
        
        # Save the updated DataFrame back to a CSV file (overwrite original file or create a new one)
        updated_file_path = os.path.join(data_directory, f"updated_{file}")
        data.to_csv(updated_file_path, index=False)
        print(f"Processed and saved: {updated_file_path}")

print("All files processed and saved with event labels.")
