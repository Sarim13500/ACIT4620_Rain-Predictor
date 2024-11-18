import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
file_path = 'meteostat_hourly_data.csv'  # Replace with the correct file path
dataset = pd.read_csv(file_path)

# Select relevant features and handle missing values
# Using temperature (temp), precipitation (prcp), relative humidity (rhum), and wind speed (wspd)
selected_features = dataset[['temp', 'prcp', 'rhum', 'wspd']].dropna()
selected_features.fillna(selected_features.mean(), inplace=True)  # Impute missing values with mean

# Ensure precipitation values are non-negative
assert (selected_features['prcp'] >= 0).all(), "Precipitation values must be non-negative!"

print(f"Filtered data shape: {selected_features.shape}")

# Scale the data to a range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(selected_features)

# Generate input sequences and corresponding targets
sequence_length = 3  # Number of time steps in each sequence
inputs, targets = [], []

for i in range(len(normalized_data) - sequence_length):
    inputs.append(normalized_data[i:i + sequence_length])
    targets.append(normalized_data[i + sequence_length, 1])  # Precipitation (prcp) is the second column

inputs, targets = np.array(inputs), np.array(targets)
print(f"Input shape before reshaping: {inputs.shape}")
print(f"Target shape: {targets.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Reshape inputs to be compatible with LSTM layers (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Define the LSTM model architecture
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)  # Single output for Precipitation prediction
])

# Compile the model with optimizer and loss function
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the training data
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)  # Set epochs to 10

# Make predictions on the test data
y_pred = lstm_model.predict(X_test)

# Rescale predictions back to original scale
# Create placeholders for all columns (fill with zeros except for predictions in the precipitation column)
placeholder = np.zeros((y_pred.shape[0], selected_features.shape[1]))  # Match original feature count
placeholder[:, 1] = y_pred.flatten()  # Place predictions in the precipitation column (index 1)
rescaled_predictions = scaler.inverse_transform(placeholder)[:, 1]  # Extract only the rescaled precipitation column

# Clip negative values to ensure valid precipitation predictions
rescaled_predictions = np.clip(rescaled_predictions, 0, None)

# Display predictions
print("Predicted Precipitation values:")
for index, prediction in enumerate(rescaled_predictions[:10]):  # Display first 10 predictions
    print(f"Prediction {index + 1}: It is expected to rain approximately {prediction:.2f} mm.")

# Calculate average predicted rainfall if available
if len(rescaled_predictions) > 0:
    avg_rainfall = np.mean(rescaled_predictions)
    print(f"\nAverage predicted precipitation: {avg_rainfall:.2f} mm")
else:
    print("No predictions available.")
