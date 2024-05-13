from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
import time
import pickle
import datetime
from sklearn.preprocessing import MinMaxScaler
from joblib import dump,load
app = Flask(__name__)

# Load your bitcoin price prediction model
# Replace this with the code to load and initialize your model
with open('D:\\Data Science\\Final project\deployment\\bitcoin_venv\\Scripts\\lr_model.pkl', 'rb') as f:
    model = pickle.load(f)
loaded_scaler = load('D:\\Data Science\\Final project\\deployment\\bitcoin_venv\\Scripts\\scaler.joblib')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    hour = int(request.form['hour'])
    minute = int(request.form['minute'])
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    volume_btc = float(request.form['volume_btc'])
    volume_currency = float(request.form['volume_currency'])
    weighted_price = float(request.form['weighted_price'])
    dt = datetime.datetime(year, month, day, hour, minute, 0)
    timestamp = dt.timestamp()

    # Create a DataFrame with the input data
    data = pd.DataFrame({
        'Timestamp': [timestamp],
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Volume_(BTC)': [volume_btc],
        'Volume_(Currency)': [volume_currency],
        'Weighted_Price': [weighted_price]
    })
    
    scaled_data=loaded_scaler.transform(data)
    # Perform prediction using your model
    # Replace this with the code to make predictions using your model
    prediction = model.predict(scaled_data)  # Replace with your prediction results

    # Render the result in a new template or return as an API response
    return render_template('index.html', prediction=f'Predicted Close price: {prediction[0]:.2f}')

if __name__ =="__main__" :
    app.run(debug=True)
