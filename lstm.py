import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
epochs = 1
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Final Test Loss: {test_loss}')

# Make predictions
test_predictions = model.predict(X_test)

# Inverse transform the scaled predictions and observed values
y_pred = scaler_target.inverse_transform(test_predictions)
y_observed = scaler_target.inverse_transform(y_test)

# Ensure that 'Actual' and 'Predicted' arrays have the same length
min_length = min(len(y_observed), len(y_pred))
y_observed = y_observed[:min_length]
y_pred = y_pred[:min_length]

# Print the shapes of y_observed and y_pred for diagnosis
print("Shape of y_observed:", y_observed.shape)
print("Shape of y_pred:", y_pred.shape)

# Compare the predicted and observed values
min_length = min(len(y_observed.flatten()), len(y_pred.flatten()))
comparison_df = pd.DataFrame({
    'Actual': y_observed.flatten()[:min_length],
    'Predicted': y_pred.flatten()[:min_length]
})

# Calculate RMSE, MSE
rmse = np.sqrt(mean_squared_error(y_observed, y_pred))
mse = mean_squared_error(y_observed, y_pred)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Squared Error (MSE): {mse}')

# Print the comparison DataFrame
print(comparison_df)
mae = mean_absolute_error(y_observed, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Save the comparison DataFrame and metrics to CSV
comparison_df.to_csv('actual_vs_predicted.csv', index=False)
pd.DataFrame({'RMSE': [rmse]}).to_csv('rmse.csv', index=False)
pd.DataFrame({'MSE': [mse]}).to_csv('mse.csv', index=False)
pd.DataFrame({'MAE': [mae]}).to_csv('mae.csv', index=False)

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Save the model
model.save("simple_lstm_model.h5")
