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
                'Relative Time (s)', 
                'Velocity (m/s)', 
                'Average Power ((m/s)²/Hz)', 
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
                label_data = 1  # The event starts
            else:
                label_data = 0  # No event
            labels.append(label_data)  # Append single label (0 or 1)

    # Pad sequences to the same length
    features = pad_sequences(features, padding='post', dtype='float32')  # Pad feature arrays
    labels = np.array(labels, dtype='float32')  # Convert labels to numpy array

    print(f'Feature array shape: {features.shape}')  # e.g. (76, 2555, 7)
    print(f'Label array shape: {labels.shape}')      # e.g. (76,)

    return features, labels

# Define the model
def create_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=False, input_shape=input_shape),  # Output a single value
        layers.Dense(1, activation='sigmoid')  # Outputs a single value (0 or 1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
    history = model.fit(features, labels, epochs=50, batch_size=32, callbacks=[early_stopping])

    # Save the model
    model.save('seismic_model.keras')
