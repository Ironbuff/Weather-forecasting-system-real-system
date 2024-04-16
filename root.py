import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv(r"D:\finale.csv")

# Select the features and target variable
features = dataset[['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'RH2M', 'T2MWET', 'TS', 'WS10M', 'WS50M']]
target = dataset['T2M']

# Normalize both the features and the target variable using Min-Max scaling
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Reshape the data for LSTM input (samples, time steps, features)
features_reshaped = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_reshaped, target_scaled, test_size=0.2, random_state=42)

# Define the LSTM model using TensorFlow's built-in layers
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(1, features_reshaped.shape[2])),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
epochs = 10
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.savefig('training_history_plot.png')
plt.close()

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Final Test Loss: {test_loss}')

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the scaled predictions and observed values
y_train_pred = scaler_target.inverse_transform(train_predictions)
y_train_observed = scaler_target.inverse_transform(y_train)

y_test_pred = scaler_target.inverse_transform(test_predictions)
y_test_observed = scaler_target.inverse_transform(y_test)

# Ensure that 'Actual' and 'Predicted' arrays have the same length
min_length_train = min(len(y_train_observed), len(y_train_pred))
y_train_observed = y_train_observed[:min_length_train]
y_train_pred = y_train_pred[:min_length_train]

min_length_test = min(len(y_test_observed), len(y_test_pred))
y_test_observed = y_test_observed[:min_length_test]
y_test_pred = y_test_pred[:min_length_test]

# Plot actual vs predicted for train dataset
plt.figure(figsize=(10, 6))
plt.scatter(y_train_observed, y_train_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_train_observed.min(), y_train_observed.max()], [y_train_observed.min(), y_train_observed.max()], color='red', linestyle='--', label='Regression Line')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Temperature (Train Dataset)')
plt.legend()
plt.savefig('actual_vs_predicted_train_plot.png')
plt.close()

# Plot actual vs predicted for test dataset
plt.figure(figsize=(10, 6))
plt.scatter(y_test_observed, y_test_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test_observed.min(), y_test_observed.max()], [y_test_observed.min(), y_test_observed.max()], color='red', linestyle='--', label='Regression Line')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Temperature (Test Dataset)')
plt.legend()
plt.savefig('actual_vs_predicted_test_plot.png')
plt.close()

# Plot MAE, RMSE, MSE
plt.figure(figsize=(10, 6))
plt.plot(np.abs(y_test_observed - y_test_pred), label='MAE')
plt.plot(np.sqrt((y_test_observed - y_test_pred) ** 2), label='RMSE')
plt.plot((y_test_observed - y_test_pred) ** 2, label='MSE')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Evaluation Metrics')
plt.legend()
plt.savefig('evaluation_metrics_plot.png')
plt.close()

# Create DataFrames for saving to CSV
train_data = pd.DataFrame({
    'Actual_train': y_train_observed.flatten(),
    'Predicted_train': y_train_pred.flatten()
})

test_data = pd.DataFrame({
    'Actual_test': y_test_observed.flatten(),
    'Predicted_test': y_test_pred.flatten()
})

evaluation_metrics_data = pd.DataFrame({
    'MAE': np.abs(y_test_observed - y_test_pred).flatten(),
    'RMSE': np.sqrt((y_test_observed - y_test_pred) ** 2).flatten(),
    'MSE': ((y_test_observed - y_test_pred) ** 2).flatten()
})

# Save DataFrames to CSV files
train_data.to_csv('actual_vs_predicted_train.csv', index=False)
test_data.to_csv('actual_vs_predicted_test.csv', index=False)
evaluation_metrics_data.to_csv('evaluation_metrics.csv', index=False)
