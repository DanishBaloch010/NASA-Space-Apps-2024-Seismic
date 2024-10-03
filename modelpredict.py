import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model

# Step 1: Load the model
model = load_model('seismic_model.keras')

# Step 2: Load the CSV file
csv_file_path = 'TestingDataCSVs/xa.s12.00.mhz.1970-01-19HR00_evid00002_seismic_results.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file_path)

# Step 3: Preprocess the data
# Extracting features, omitting 'Relative Time (s)' as it's typically not a feature for prediction
features = data[['Velocity (m/s)', 'Average Power ((m/s)^2/Hz)', 
                 'Weighted Frequency (Hz)', 'Dominant Frequency (Hz)', 
                 'Velocity Upper Band', 'Velocity Lower Band', 
                 'Moving Average']].values  # Adjust this list based on actual features needed

# Reshape the data: model input should be (batch_size, timesteps, features)
# Here, we assume you want to use all rows as a single sequence for prediction
# Adjust 'timesteps' accordingly if your model was trained with a specific window size
features = features.reshape((1, features.shape[0], features.shape[1]))  

# Step 4: Make predictions
predictions = model.predict(features)

# Step 5: Process predictions
predicted_classes = (predictions > 0.5).astype(int)

# Step 6: Print results
print("Predictions:", predictions)
print("Predicted classes:", predicted_classes)