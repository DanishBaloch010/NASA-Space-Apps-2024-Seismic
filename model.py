import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load your dataset
data = pd.read_csv('seismic_results.csv')  # Replace with your actual filename

# Example: Creating a target column
event_timestamps = [event_time_1, event_time_2, ...]  # Add your seismic event timestamps here
data['Seismic Event'] = data['Relative Time (s)'].isin(event_timestamps).astype(int)

from sklearn.model_selection import train_test_split

X = data.drop(columns=['Relative Time (s)', 'Seismic Event'])  # Features
y = data['Seismic Event']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output for seismic event detection
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')


# # Assuming new_data is a DataFrame with the same feature structure as X
# new_data_normalized = scaler.transform(new_data)  # Normalize new data
# predictions = model.predict(new_data_normalized)

# # Threshold for classification
# predicted_events = (predictions > 0.5).astype(int)

# # Output corresponding times for detected events
# detected_times = new_data['Relative Time (s)'][predicted_events.flatten() == 1]
# print("Detected seismic event times:", detected_times)



