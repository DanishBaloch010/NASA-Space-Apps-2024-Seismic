import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Step 1: Load the model
model = load_model('seismic_event_model.keras')

# Step 2: Load the testing CSV file
test_file_path = 'TestingDataCSVs/xa.s15.00.mhz.1973-08-10HR00_evid00126_seismic_results.csv'  # Replace with your actual file path
test_data = pd.read_csv(test_file_path)

# Print the structure of the testing data
print(f"Testing Data Columns: {list(test_data.columns)}")
print(f"Shape of the test data: {test_data.shape}")

# Drop the 'Relative Time (s)' column, as it should not be included in features
X_test = test_data.drop(columns=['Relative Time (s)'])

# Scale the test data using the same scaler used for training data
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Make predictions
predicted_time = model.predict(X_test_scaled)

# Print the predicted relative time for the seismic event
print(f"Predicted relative time for the seismic event start: {predicted_time}")
