import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

def load_data(directory):
    features, labels = [], []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            
            # Specify feature columns by name
            feature_columns = [
                'Velocity (m/s)', 
                'Average Power ((m/s)^2/Hz)', 
                'Weighted Frequency (Hz)', 
                'Dominant Frequency (Hz)', 
                'Velocity Upper Band', 
                'Velocity Lower Band', 
                'Moving Average'
            ]
            feature_data = data[feature_columns].values  # Extract features by column names
            features.append(feature_data)

            # Extract the label for the start of the seismic event
            if 1 in data['is_seismic'].values:
                # Get the relative time of the first occurrence of the seismic event
                label_data = data['Relative Time (s)'][data['is_seismic'] == 1].values[0]
            else:
                label_data = np.nan  # No event detected

            labels.append(label_data)  # Append label (relative time or NaN)

    # Convert features and labels to numpy arrays
    features = pad_sequences(features, padding='post', dtype='float32')  # Pad feature arrays
    labels = np.array(labels, dtype='float32')  # Convert labels to numpy array

    # Remove entries with NaN labels
    valid_indices = ~np.isnan(labels)
    features = features[valid_indices]
    labels = labels[valid_indices]

    print(f'Feature array shape: {features.shape}')  # e.g. (76, 2555, 7)
    print(f'Label array shape: {labels.shape}')      # e.g. (76,)

    return features, labels

# Define the model
def create_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=False, input_shape=input_shape),  # Output a single value
        layers.Dense(1, activation='relu')  # Outputs a single continuous value for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Use MSE for regression
    return model

# Main execution
if __name__ == "__main__":
    directory = "DataCSV"  # Update this to your directory
    features, labels = load_data(directory)

    # Reshape features to match the LSTM input shape
    features = features.reshape((features.shape[0], features.shape[1], features.shape[2]))

    print(f'Reshaped feature array shape: {features.shape}')  # e.g. (76, 2555, 7)
    print(f'Label array shape: {labels.shape}')              # e.g. (76,)

    # Create the model
    model = create_model((features.shape[1], features.shape[2]))

    # Set up early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(features, labels, epochs=1000, batch_size=128, callbacks=[early_stopping])

    # Save the model
    model.save('seismic_model.keras')