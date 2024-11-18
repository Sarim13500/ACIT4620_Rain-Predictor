import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
file_path = 'meteostat_hourly_data.csv'  # Replace with the correct file path
dataset = pd.read_csv(file_path)

# Select relevant features and handle missing values
features = dataset[['temp', 'prcp', 'rhum', 'wspd']].dropna()
features.fillna(features.mean(), inplace=True)  # Impute missing values with mean
print(f"Filtered data shape: {features.shape}")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(features)

# Create input sequences and corresponding targets
sequence_length = 5  # Number of time steps for input
inputs, targets = [], []

for i in range(len(normalized_data) - sequence_length):
    inputs.append(normalized_data[i:i + sequence_length])
    targets.append(normalized_data[i + sequence_length, 1])  # Precipitation (prcp) as target

inputs, targets = np.array(inputs), np.array(targets)
print(f"Generated inputs shape: {inputs.shape}")
print(f"Generated targets shape: {targets.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Convert data to float32 for faster processing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Define the model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Single output for precipitation prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Add early stopping to halt training when validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(
    X_train, y_train,
    epochs=10,  # Set epochs to 10
    batch_size=64,  # Increased batch size
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Make predictions
y_pred = model.predict(X_test)

# Rescale predictions back to the original scale
# Create placeholders for all columns (fill with zeros except for predictions in the precipitation column)
placeholder = np.zeros((y_pred.shape[0], features.shape[1]))  # Match original feature count
placeholder[:, 1] = y_pred.flatten()  # Place predictions in the 'prcp' column (index 1)

rescaled_predictions = scaler.inverse_transform(placeholder)[:, 1]  # Extract only the rescaled precipitation column

# Print human-readable predictions
print("\nPredicted Rainfall Summary:")
for idx, prediction in enumerate(rescaled_predictions[:10]):  # Limit to the first 10 predictions for readability
    print(f"Prediction {idx + 1}: It is expected to rain approximately {prediction:.2f} mm.")
