import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set the path to the directory containing the seismic data files and the catalog file
data_directory = "path_to_seismic_files"  # Replace with the path to your seismic files
catalog_file = "path_to_catalog_file.csv"  # Replace with the catalog file path

# Step 1: Read the catalog file
catalog = pd.read_csv(catalog_file)

# Assuming the catalog has columns: ['File', 'Relative Time (s)', 'Seismic Event']
# 'File' - The filename associated with each event
# 'Relative Time (s)' - The time in seconds where the event occurred in each file
# 'Seismic Event' - The label indicating if the file has an event (1 for event, 0 for no event)

# Step 2: Read all seismic files and create a master DataFrame
all_data = pd.DataFrame()

# Loop through all files in the directory
for file in os.listdir(data_directory):
    if file.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(data_directory, file)
        
        # Read the file into a DataFrame
        data = pd.read_csv(file_path)
        
        # Check if the file has corresponding entries in the catalog
        file_events = catalog[catalog['File'] == file]
        
        # Step 3: Add a 'Seismic Event' column based on the catalog entries
        # Initialize all values as 0 (no event)
        data['Seismic Event'] = 0
        
        # Mark rows where a seismic event occurs (matching relative time from the catalog)
        for _, row in file_events.iterrows():
            event_time = row['Relative Time (s)']
            
            # Mark the closest relative time as an event (adjust threshold as needed)
            data.loc[np.abs(data['Relative Time (s)'] - event_time) < 1, 'Seismic Event'] = 1
        
        # Add the filename to distinguish files (optional)
        data['File'] = file

        # Append the processed DataFrame to the master DataFrame
        all_data = pd.concat([all_data, data], ignore_index=True)

# Step 4: Feature Selection
# Use only the necessary columns for training
features = all_data.drop(columns=['Relative Time (s)', 'Seismic Event', 'File'])  # Drop non-feature columns
target = all_data['Seismic Event']  # Use the 'Seismic Event' column as the target

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 6: Neural Network Model Definition
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (event vs. no event)
])

# Step 7: Model Compilation and Training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 8: Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Optional: Save the trained model
model.save("seismic_event_detection_model.h5")

# Step 9: Predicting New Data
# You can now load new data and make predictions using this trained model.
