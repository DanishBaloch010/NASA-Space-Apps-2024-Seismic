import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import keras

# Step 1: Data loading and preprocessing with padding
def load_data(directory):
    features = []  # Initialize features list
    labels = []    # Initialize labels list
    max_len = 0    # Track the maximum sequence length

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)

            # Extract the relevant features and the labels
            feature_columns = data.columns[1:-1]  # Exclude 'Relative Time' and 'is_seismic'
            features.append(data[feature_columns].values)

            # Track the maximum length for padding
            if len(data[feature_columns]) > max_len:
                max_len = len(data[feature_columns])

            # The label is a single value indicating if there is a seismic event in the sequence
            if 1 in data['is_seismic'].values:
                labels.append(1)  # Event detected in the sequence
            else:
                labels.append(0)  # No event in the sequence

    # Step 2: Pad all sequences to the same length
    features_padded = pad_sequences(features, padding='post', dtype='float32', maxlen=max_len)

    return np.array(features_padded), np.array(labels, dtype='float32')

# Step 2: Load the data
directory = 'DataCSV'  # Change to your directory
features, labels = load_data(directory)

print(f"Feature array shape: {features.shape}")
print(f"Label array shape: {labels.shape}")

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 4: Build the LSTM model
model = keras.Sequential([
    keras.layers.Input(shape=(features.shape[1], features.shape[2])),  # Adjusted input shape
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')  # Single output for binary classification
])

# Step 5: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

model.save('seismic_event_detection_model.keras')

# Step 8: Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()