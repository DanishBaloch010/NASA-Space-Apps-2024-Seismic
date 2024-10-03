import tensorflow as tf
import numpy as np
import keras

# Step 1: Load the model
model = keras.models.load_model('seismic_event_detection_model.keras')

# Step 2: Prepare your input data
# Assuming your input data has the same shape as the training data
# For example, let's create some dummy data for prediction
# Replace this with your actual data preprocessing
new_data = np.random.rand(1, 2555, 8)  # Shape should match (batch_size, timesteps, features)

# Step 3: Make predictions
predictions = model.predict(new_data)

# Step 4: Process predictions
# Since you're predicting the start of seismic events, the output will typically be probabilities
# You can convert these probabilities into binary predictions based on a threshold (e.g., 0.5)
predicted_classes = (predictions > 0.5).astype(int)

# Step 5: Print results
print("Predictions:", predictions)
print("Predicted classes:", predicted_classes)
