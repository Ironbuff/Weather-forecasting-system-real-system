import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = tf.keras.models.load_model('simple_lstm_model.h5')

# Load the feature scaler
scaler_features = MinMaxScaler()
scaler_features = scaler_features.fit(pd.read_csv(r"D:\finale.csv")[['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'RH2M', 'T2MWET', 'TS', 'WS10M', 'WS50M']])

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dash.html')
#
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dash():
    return render_template('dashboard.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')



@app.route('/predict', methods=['POST'])
def predict():
    # Get the input feature values from the request
    input_data = request.form.to_dict()

    # Preprocess the input data
    input_features = [float(input_data[feature]) for feature in ['LAT', 'LON', 'PRECTOT', 'PS', 'QV2M', 'RH2M', 'T2MWET', 'TS', 'WS10M', 'WS50M']]
    input_features = scaler_features.transform([input_features])
    input_features = np.reshape(input_features, (1, 1, input_features.shape[1]))

    # Make the prediction using the loaded model
    prediction = model.predict(input_features)

    # Inverse transform the prediction
    scaler_target = MinMaxScaler()
    scaler_target = scaler_target.fit(pd.read_csv(r"D:\finale.csv")['T2M'].values.reshape(-1, 1))
    prediction = scaler_target.inverse_transform(prediction)

    # Return the prediction as a JSON response
    return str(prediction[0][0])



if __name__ == '__main__':
    app.run(debug=False)

