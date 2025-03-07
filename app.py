from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load historical Bitcoin price data from CSV
bitcoin_data = pd.read_csv(r'C:\Users\JOHN PAUL\OneDrive\Desktop\hack1\Bitcoin (1).csv')

# Preprocess the data
bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
bitcoin_data.set_index('Date', inplace=True)

# Feature selection
X = bitcoin_data[['Open', 'High', 'Low', 'Volume']]
y = bitcoin_data['Close']

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        open_input = float(request.form['open'])
        high_input = float(request.form['high'])
        low_input = float(request.form['low'])
        volume_input = float(request.form['volume'])

        # Make prediction based on user inputs
        user_input = np.array([[open_input, high_input, low_input, volume_input]])
        predicted_price = model.predict(user_input)

        return render_template('result.html', predicted_price=predicted_price[0])
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
