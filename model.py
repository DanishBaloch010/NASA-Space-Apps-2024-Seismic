import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# Load Training Data
def load_data_from_directory(directory):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Load and prepare the data
train_data = load_data_from_directory('DataCSV')

# Ensure we are checking the shape of the data
print(f"Shape of training data: {train_data.shape}")

# Separate features and target
X = train_data.drop(columns=['Relative Time (s)', 'is_seismic'])
y = train_data['Relative Time (s)'][train_data['is_seismic'] == 1]

# Check the number of samples
print(f"Number of samples in X: {X.shape[0]}")
print(f"Number of samples in y: {y.shape[0]}")

# If y is empty, we need to handle that case
if y.empty:
    raise ValueError("No seismic event found in the training data. Ensure there are entries with is_seismic = 1.")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ensure that y is a 1D array and the same length as X
y = y.values  # Convert y to numpy array

# Create class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),  # Increased layer size
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression (relative time)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Add Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor the training loss
    patience=5,      # Number of epochs to wait before stopping if no improvement
    restore_best_weights=True  # Restore the best weights after stopping
)

# Train the model with Early Stopping
model.fit(X_scaled, y, epochs=200, batch_size=128, callbacks=[early_stopping], class_weight=class_weights)

# Save the trained model
model.save('seismic_event_model.keras')

print("Model saved as 'seismic_event_model.keras'")
