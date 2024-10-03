import numpy as np
import pandas as pd
from keras.models import load_model

# Step 1: Load the model
model = load_model('seismic_model.keras')

# Step 2: Load the testing CSV file
csv_file_path = 'TestingDataCSVs/xa.s12.00.mhz.1970-01-19HR00_evid00002_seismic_results.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file_path)

# Step 3: Preprocess the data
# Selecting the features used for prediction (7 features excluding 'is_seismic')
features = data[['Velocity (m/s)', 'Average Power ((m/s)^2/Hz)', 
                 'Weighted Frequency (Hz)', 'Dominant Frequency (Hz)', 
                 'Velocity Upper Band', 'Velocity Lower Band', 
                 'Moving Average']].values

# Step 4: Reshape the data
# Here, we assume that each sample has 2555 timesteps, and we are predicting on a single batch.
# Since we don't know the number of timesteps in your testing data, you may need to adjust accordingly.
# For example, if you are processing all the features at once and the shape should be (1, 2555, 7):
# Check the shape of features
timesteps = features.shape[0]  # Number of rows in the testing data
features = features.reshape((1, timesteps, features.shape[1]))  # Adjust if needed

# Step 5: Make predictions
predictions = model.predict(features)

# Step 6: Process predictions
predicted_classes = (predictions > 0.5).astype(int)

# Step 7: Print results
print("Predictions:", predictions)
print("Predicted classes:", predicted_classes)
